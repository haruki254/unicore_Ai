"""
Adaptive Updater
================
Incrementally adjusts per-EA profile weights and flip/block thresholds
after each trade closes. Called by the ``/trade/update`` API endpoint
so the system learns in near-real-time without waiting for a full retrain.

Learning rule
-------------
For each profiled condition dimension that was *active* in the trade snapshot:

    new_weight = old_weight + learning_rate × (target − old_weight)

Where:
    target = 3.0  (WIN  → nudge weight toward HIGH)
    target = 0.0  (LOSS → nudge weight toward BLOCKED)
    learning_rate ≈ 0.05 by default (slow, stable)

This is an exponential moving average: weights converge slowly and
do not flip violently from a single trade result.

Adaptive thresholds
-------------------
Flip threshold:
    Tracks rolling flip win rate.
    High flip win rate → lower threshold (easier to flip; it's profitable).
    Low  flip win rate → raise threshold  (harder to flip; stop losing).

    threshold = clip(0.90 − flip_win_rate × 0.50, 0.55, 0.85)

Block threshold:
    Tracks allowed-trade win rate for the EA.
    Falling win rate → lower block threshold (block more aggressively).
    Rising win rate  → raise block threshold  (allow more through).

    threshold = clip(0.30 + allowed_win_rate × 0.50, 0.35, 0.75)

Usage (in /trade/update handler)
---------------------------------
    updater = AdaptiveUpdater()
    profile = EAProfile.from_dict(db.get_ea_profile(ea_id))
    flip_stats = db.get_flip_stats(ea_id)
    snapshot_features = db.get_snapshot_features(snapshot_id)

    result = updater.update(
        ea_id           = ea_id,
        outcome         = "WIN",           # "WIN" | "LOSS"
        was_flipped     = True,
        snapshot_features = snapshot_features,
        profile         = profile,
        flip_stats      = flip_stats or {},
    )

    db.update_ea_profile(ea_id, result.updated_profile.to_dict())
    db.update_flip_stats(ea_id, result.updated_flip_stats)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.profiles.ea_profile_builder import (
    EAProfile,
    EAProfileBuilder,
    SUPPORTED_DIMENSIONS,
    WEIGHT_DEFAULT,
)
from core.scoring.weighted_scorer import WeightedScorer

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
DEFAULT_LEARNING_RATE  = 0.05
MIN_WEIGHT             = 0.0
MAX_WEIGHT             = 3.0

# Flip threshold bounds
FLIP_THRESHOLD_MIN     = 0.50
FLIP_THRESHOLD_MAX     = 0.85
FLIP_THRESHOLD_DEFAULT = 0.65
RECENT_WINDOW          = 20   # length of the rolling outcome window

# Block threshold bounds
BLOCK_THRESHOLD_MIN    = 0.35
BLOCK_THRESHOLD_MAX    = 0.75
BLOCK_THRESHOLD_DEFAULT = 0.55


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class WeightChange:
    """Records a single weight adjustment for one (dimension, value, direction)."""
    dimension:  str
    value:      str
    direction:  str
    old_weight: float
    new_weight: float
    delta:      float
    old_label:  str
    new_label:  str


@dataclass
class UpdateResult:
    """
    Full output of one adaptive update step.

    Attributes
    ----------
    ea_id              : EA that was updated
    outcome            : "WIN" or "LOSS"
    was_flipped        : whether this trade was a flip
    weight_changes     : list of dimension weights that shifted
    updated_profile    : new EAProfile (ready to persist)
    updated_flip_stats : new flip stats dict (ready to persist)
    new_flip_threshold : updated flip confidence threshold
    new_block_threshold: updated block quality threshold
    """
    ea_id:              str
    outcome:            str
    was_flipped:        bool
    weight_changes:     List[WeightChange]
    updated_profile:    EAProfile
    updated_flip_stats: Dict[str, Any]
    new_flip_threshold: float
    new_block_threshold: float

    @property
    def n_changes(self) -> int:
        return len(self.weight_changes)

    def summary(self) -> str:
        parts = [f"[{self.ea_id}] {self.outcome}"]
        if self.was_flipped:
            parts.append("(FLIP)")
        parts.append(f"→ {self.n_changes} weights updated")
        parts.append(f"flip_thr={self.new_flip_threshold:.2f}")
        parts.append(f"block_thr={self.new_block_threshold:.2f}")
        return " | ".join(parts)


# ── AdaptiveUpdater ───────────────────────────────────────────────────────────

class AdaptiveUpdater:
    """
    Incrementally adjusts EA profile weights after a trade closes.

    Parameters
    ----------
    learning_rate : float
        Step size for the exponential moving average update (default 0.05).
        Increase for faster adaptation; decrease for more stability.
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE) -> None:
        self.learning_rate = learning_rate
        self._scorer = WeightedScorer()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        ea_id:             str,
        outcome:           str,               # "WIN" or "LOSS"
        snapshot_features: dict,
        profile:           EAProfile,
        flip_stats:        Dict[str, Any],
        was_flipped:       bool = False,
    ) -> UpdateResult:
        """
        Run one adaptive update step and return the new profile + flip stats.

        Parameters
        ----------
        ea_id             : EA identifier
        outcome           : "WIN" or "LOSS"
        snapshot_features : feature dict at the time of the trade signal
        profile           : current EAProfile (will NOT be mutated)
        flip_stats        : current flip stats dict from DB
        was_flipped       : True if the decision engine flipped the signal

        Returns
        -------
        UpdateResult — call .updated_profile.to_dict() and .updated_flip_stats
                       to get the values to persist.
        """
        outcome = outcome.upper().strip()
        if outcome not in ("WIN", "LOSS"):
            logger.warning("AdaptiveUpdater: unexpected outcome %r for %s — skipping", outcome, ea_id)
            return self._no_op_result(ea_id, outcome, was_flipped, profile, flip_stats)

        # Extract the active conditions from the snapshot
        conditions = self._scorer.extract_conditions(snapshot_features)

        # Determine which direction to update (the actual executed direction)
        direction = self._extract_direction(snapshot_features)

        # ── 1. Update dimension weights ───────────────────────────────────────
        import copy
        updated_profile = copy.deepcopy(profile)
        weight_changes:  List[WeightChange] = []

        target = MAX_WEIGHT if outcome == "WIN" else MIN_WEIGHT

        for dim in SUPPORTED_DIMENSIONS:
            value = conditions.get(dim)
            if value is None:
                continue
            if not value:
                continue

            # Ensure the path exists in the profile
            updated_profile.weights.setdefault(dim, {}).setdefault(value, {})

            for dir_ in ([direction] if direction else ["BUY", "SELL"]):
                old_w = updated_profile.weights[dim][value].get(dir_, WEIGHT_DEFAULT)
                new_w = self._ema_step(old_w, target)

                if abs(new_w - old_w) < 1e-6:
                    continue  # negligible change, skip

                updated_profile.weights[dim][value][dir_] = round(new_w, 4)

                from core.profiles.ea_profile_builder import discrete_label
                weight_changes.append(WeightChange(
                    dimension=dim,
                    value=value,
                    direction=dir_,
                    old_weight=round(old_w, 4),
                    new_weight=round(new_w, 4),
                    delta=round(new_w - old_w, 4),
                    old_label=discrete_label(old_w),
                    new_label=discrete_label(new_w),
                ))

        # ── 2. Update aggregate stats on the profile ──────────────────────────
        if outcome == "WIN":
            updated_profile.wins    += 1
        else:
            updated_profile.losses  += 1
        updated_profile.total_trades += 1
        total = updated_profile.wins + updated_profile.losses
        updated_profile.win_rate = round(
            updated_profile.wins / total if total > 0 else 0.5, 4
        )

        # ── 3. Update flip / block thresholds ────────────────────────────────
        updated_flip_stats = self._update_flip_stats(
            flip_stats, outcome, was_flipped
        )

        new_flip_threshold  = self._compute_flip_threshold(updated_flip_stats)
        new_block_threshold = self._compute_block_threshold(updated_profile)

        updated_profile.flip_threshold  = round(new_flip_threshold,  4)
        updated_profile.block_threshold = round(new_block_threshold, 4)
        updated_flip_stats["flip_threshold"]  = round(new_flip_threshold,  4)
        updated_flip_stats["block_threshold"] = round(new_block_threshold, 4)

        result = UpdateResult(
            ea_id=ea_id,
            outcome=outcome,
            was_flipped=was_flipped,
            weight_changes=weight_changes,
            updated_profile=updated_profile,
            updated_flip_stats=updated_flip_stats,
            new_flip_threshold=new_flip_threshold,
            new_block_threshold=new_block_threshold,
        )

        logger.info("AdaptiveUpdate: %s", result.summary())
        if weight_changes:
            for wc in weight_changes:
                logger.debug(
                    "  %s[%s][%s]: %.3f → %.3f (%s → %s)",
                    wc.dimension, wc.value, wc.direction,
                    wc.old_weight, wc.new_weight,
                    wc.old_label, wc.new_label,
                )

        return result

    # ── Threshold computations ────────────────────────────────────────────────

    def _compute_flip_threshold(self, flip_stats: dict) -> float:
        """
        Adaptive flip threshold based on rolling flip win rate.

        High flip WR → lower threshold (flip more readily; it works).
        Low  flip WR → raise threshold  (flip less; it's hurting us).

        Formula: threshold = clip(0.90 − flip_win_rate × 0.50, MIN, MAX)

        Examples:
          flip_wr = 0.80 → 0.90 − 0.40 = 0.50 (very easy to flip)
          flip_wr = 0.50 → 0.90 − 0.25 = 0.65 (neutral default)
          flip_wr = 0.20 → 0.90 − 0.10 = 0.80 (very hard to flip)
        """
        total = flip_stats.get("total_flips", 0)
        if total < 5:
            return FLIP_THRESHOLD_DEFAULT   # not enough data yet

        # Use recent window for faster reaction
        recent = flip_stats.get("recent_flip_outcomes", [])
        if len(recent) >= 5:
            recent_wins = sum(1 for o in recent if o == "WIN")
            flip_wr = recent_wins / len(recent)
        else:
            flip_wr = flip_stats.get("flip_win_rate", 0.5)

        threshold = 0.90 - float(flip_wr) * 0.50
        return round(max(FLIP_THRESHOLD_MIN, min(FLIP_THRESHOLD_MAX, threshold)), 4)

    def _compute_block_threshold(self, profile: EAProfile) -> float:
        """
        Adaptive block (min_risk_quality) threshold based on allowed-trade WR.

        High allowed-trade WR → raise threshold (block fewer trades; doing well).
        Low  allowed-trade WR → lower threshold (block more; too many losses through).

        Formula: threshold = clip(0.30 + win_rate × 0.50, MIN, MAX)

        Examples:
          win_rate = 0.70 → 0.30 + 0.35 = 0.65 (lenient; EA is doing well)
          win_rate = 0.50 → 0.30 + 0.25 = 0.55 (neutral default)
          win_rate = 0.30 → 0.30 + 0.15 = 0.45 (strict; EA is struggling)
        """
        if profile.total_trades < 10:
            return BLOCK_THRESHOLD_DEFAULT

        threshold = 0.30 + float(profile.win_rate) * 0.50
        return round(max(BLOCK_THRESHOLD_MIN, min(BLOCK_THRESHOLD_MAX, threshold)), 4)

    # ── Flip stats management ─────────────────────────────────────────────────

    def _update_flip_stats(
        self,
        flip_stats:  Dict[str, Any],
        outcome:     str,
        was_flipped: bool,
    ) -> Dict[str, Any]:
        """Return an updated copy of flip_stats without mutating the original."""
        fs = dict(flip_stats)

        if was_flipped:
            fs["total_flips"]  = fs.get("total_flips", 0) + 1
            if outcome == "WIN":
                fs["flip_wins"]    = fs.get("flip_wins", 0) + 1
            else:
                fs["flip_losses"]  = fs.get("flip_losses", 0) + 1

            # Update running win rate
            total = fs.get("total_flips", 1)
            wins  = fs.get("flip_wins", 0)
            fs["flip_win_rate"] = round(wins / total if total > 0 else 0.5, 4)

            # Update recent window (sliding RECENT_WINDOW)
            recent: List[str] = list(fs.get("recent_flip_outcomes", []))
            recent.append(outcome)
            if len(recent) > RECENT_WINDOW:
                recent = recent[-RECENT_WINDOW:]
            fs["recent_flip_outcomes"] = recent

        return fs

    # ── EMA step ─────────────────────────────────────────────────────────────

    def _ema_step(self, old_weight: float, target: float) -> float:
        """Single exponential moving average step."""
        new_w = old_weight + self.learning_rate * (target - old_weight)
        return max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_direction(features: dict) -> Optional[str]:
        """
        Try to recover the executed trade direction from the feature dict.
        The trade update request should embed it under 'direction' or 'ea_signal'.
        """
        for key in ("direction", "ea_signal", "executed_direction"):
            val = features.get(key)
            if val and str(val).upper() in ("BUY", "SELL"):
                return str(val).upper()
        return None

    @staticmethod
    def _no_op_result(
        ea_id:      str,
        outcome:    str,
        was_flipped: bool,
        profile:    EAProfile,
        flip_stats: Dict[str, Any],
    ) -> UpdateResult:
        """Return an UpdateResult that is effectively a no-op."""
        return UpdateResult(
            ea_id=ea_id,
            outcome=outcome,
            was_flipped=was_flipped,
            weight_changes=[],
            updated_profile=profile,
            updated_flip_stats=flip_stats,
            new_flip_threshold=profile.flip_threshold,
            new_block_threshold=profile.block_threshold,
        )

    # ── Bulk back-fill (for existing trade history without ea_id) ─────────────

    def bulk_update_from_history(
        self,
        ea_id:        str,
        trades:       List[dict],
        profile:      Optional[EAProfile] = None,
        flip_stats:   Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """
        Replay a sequence of historical trades through the adaptive updater
        to build up an initial profile state without a full batch rebuild.

        Useful when transitioning from a system that had no ea_id tracking —
        replay the assigned trades in chronological order.

        Parameters
        ----------
        ea_id      : EA to update
        trades     : list of trade dicts in chronological order (oldest first)
        profile    : starting EAProfile (defaults to blank profile)
        flip_stats : starting flip stats (defaults to empty)

        Returns
        -------
        UpdateResult of the LAST update (contains final profile state)
        """
        if profile is None:
            profile = EAProfile(ea_id=ea_id)
        if flip_stats is None:
            flip_stats = {}

        result: Optional[UpdateResult] = None

        for trade in trades:
            outcome   = str(trade.get("outcome", "")).upper()
            flipped   = bool(trade.get("was_flipped", False))
            features  = {**trade}   # the trade dict itself contains features

            if outcome not in ("WIN", "LOSS"):
                continue

            result = self.update(
                ea_id=ea_id,
                outcome=outcome,
                snapshot_features=features,
                profile=profile,
                flip_stats=flip_stats,
                was_flipped=flipped,
            )
            profile    = result.updated_profile
            flip_stats = result.updated_flip_stats

        if result is None:
            result = self._no_op_result(ea_id, "NONE", False, profile, flip_stats)

        return result