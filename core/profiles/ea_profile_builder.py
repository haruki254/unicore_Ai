"""
EA Profile Builder
==================
Analyses completed trade history grouped by EA and builds a set of
direction-aware, per-dimension weights that the WeightedScorer uses
to score incoming market snapshots.

Weight scale (continuous)
-------------------------
  0.0  →  BLOCKED   – EA loses consistently in this condition
  1.5  →  NEUTRAL   – EA's average performance (no edge)
  3.0  →  HIGH      – EA wins consistently in this condition

Formula: weight = win_rate * 3.0  (maps 0%→0.0, 50%→1.5, 100%→3.0)

Dimensions profiled
-------------------
  regime      : 9 MARKET_REGIMES values  (direct from trade record)
  session     : 6 SESSIONS values        (direct from trade record)
  volatility  : HIGH / NORMAL / LOW      (derived from atr_normalized / booleans)
  momentum    : BULLISH / NEUTRAL / BEARISH (derived from trend_alignment_score)
  level_prox  : FREE / NEAR / AT         (derived from dist_to_* / atr_14)

Each dimension weight is split by direction:
    weights[dim][value] = {"BUY": float, "SELL": float}

Usage
-----
    builder = EAProfileBuilder(min_samples=5)
    trades  = db.fetch_completed_trades_with_ea_id()  # List[dict]
    profiles = builder.build_from_trades(trades)
    # → {"EA4": EAProfile, "EA6": EAProfile, ...}

    # Persist
    for ea_id, profile in profiles.items():
        db.save_ea_profile(ea_id, profile.to_dict())
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Weight constants ─────────────────────────────────────────────────────────

WEIGHT_HIGH    = 3.0   # ≥ 65% win rate
WEIGHT_NEUTRAL = 1.5   # 50% win rate (EA average)
WEIGHT_BLOCKED = 0.0   # ≤ 25% win rate
WEIGHT_DEFAULT = 1.5   # used when sample count is below MIN_SAMPLES

# Discrete label thresholds (for display / logging)
# Applied to continuous weights: score → label
DISCRETE_LABELS: Dict[str, Tuple[float, float]] = {
    "BLOCKED":  (0.00, 0.75),
    "LOW":      (0.75, 1.50),
    "MEDIUM":   (1.50, 2.25),
    "HIGH":     (2.25, 3.01),
}

# ── Derived dimension categories ─────────────────────────────────────────────

VOLATILITY_VALUES  = ["HIGH", "NORMAL", "LOW"]
MOMENTUM_VALUES    = ["BULLISH", "NEUTRAL", "BEARISH"]
LEVEL_PROX_VALUES  = ["FREE", "NEAR", "AT"]


def discrete_label(weight: float) -> str:
    """Map a continuous weight [0, 3] to its human-readable label."""
    for label, (lo, hi) in DISCRETE_LABELS.items():
        if lo <= weight < hi:
            return label
    return "HIGH"


def categorize_volatility(features: dict) -> str:
    """
    Derive a volatility bucket from snapshot features.

    Priority order:
      1. range_expansion / range_contraction boolean flags
      2. atr_normalized threshold (>0.70 → HIGH, <0.30 → LOW)
      3. Default: NORMAL
    """
    if features.get("range_expansion"):
        return "HIGH"
    if features.get("range_contraction"):
        return "LOW"

    atr_norm = features.get("atr_normalized", 0.5)
    try:
        atr_norm = float(atr_norm)
    except (TypeError, ValueError):
        atr_norm = 0.5

    if atr_norm > 0.70:
        return "HIGH"
    if atr_norm < 0.30:
        return "LOW"
    return "NORMAL"


def categorize_momentum(features: dict) -> str:
    """
    Derive a momentum bucket from snapshot features.

    Uses trend_alignment_score (multi-timeframe trend consensus, -1 to +1)
    combined with momentum_5 sign confirmation.

    Thresholds:
      alignment > +0.40 AND momentum_5 > 0  →  BULLISH
      alignment < -0.40 AND momentum_5 < 0  →  BEARISH
      otherwise                              →  NEUTRAL
    """
    try:
        alignment = float(features.get("trend_alignment_score", 0.0))
        mom5      = float(features.get("momentum_5", 0.0))
    except (TypeError, ValueError):
        return "NEUTRAL"

    if alignment > 0.40 and mom5 > 0:
        return "BULLISH"
    if alignment < -0.40 and mom5 < 0:
        return "BEARISH"
    return "NEUTRAL"


def categorize_level_prox(features: dict) -> str:
    """
    Derive a key-level proximity bucket from snapshot features.

    Distances are normalised by ATR-14 so the thresholds are
    symbol- and volatility-independent.

    Thresholds (distance / ATR):
      < 0.30  →  AT     (price is sitting on a key level)
      < 1.50  →  NEAR   (within 1.5 ATRs of a key level)
      else    →  FREE
    """
    atr = features.get("atr_14") or 0.0
    try:
        atr = float(atr)
    except (TypeError, ValueError):
        atr = 0.0

    raw_dists = [
        features.get("dist_to_pdh"),
        features.get("dist_to_pdl"),
        features.get("dist_to_support"),
        features.get("dist_to_resistance"),
    ]

    # Filter missing / zero values
    valid_dists = []
    for d in raw_dists:
        try:
            fv = float(d)
            if fv > 0:
                valid_dists.append(fv)
        except (TypeError, ValueError):
            pass

    if not valid_dists:
        return "FREE"

    min_dist = min(valid_dists)

    if atr > 0:
        normalised = min_dist / atr
    else:
        normalised = min_dist  # fallback: treat raw distance

    if normalised < 0.30:
        return "AT"
    if normalised < 1.50:
        return "NEAR"
    return "FREE"


# ── Dimension extractors ─────────────────────────────────────────────────────

# Maps dimension name → callable(trade_dict) → Optional[str]
# Returns None when the required fields are absent from the trade record.
_DIMENSION_EXTRACTORS: Dict[str, Callable[[dict], Optional[str]]] = {
    "regime": lambda t: t.get("regime"),
    "session": lambda t: t.get("session"),
    "volatility": lambda t: categorize_volatility(t) if any(
        k in t for k in ("range_expansion", "range_contraction", "atr_normalized")
    ) else None,
    "momentum": lambda t: categorize_momentum(t) if any(
        k in t for k in ("trend_alignment_score", "momentum_5")
    ) else None,
    "level_prox": lambda t: categorize_level_prox(t) if any(
        k in t for k in ("dist_to_pdh", "dist_to_support", "atr_14")
    ) else None,
}

SUPPORTED_DIMENSIONS = list(_DIMENSION_EXTRACTORS.keys())


# ── EAProfile dataclass ──────────────────────────────────────────────────────

@dataclass
class EAProfile:
    """
    Immutable snapshot of one EA's profiled performance weights.

    Attributes
    ----------
    ea_id       : EA identifier string
    weights     : { dimension: { value: { "BUY": float, "SELL": float } } }
    sample_counts : { dimension: { value: { "BUY_WIN": int, "BUY_LOSS": int, ... } } }
    meta        : scalar statistics (total_trades, win_rate, etc.)
    flip_threshold  : current adaptive flip confidence threshold
    block_threshold : current adaptive block quality threshold
    """
    ea_id:           str
    weights:         Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    sample_counts:   Dict[str, Dict[str, Dict[str, int]]]   = field(default_factory=dict)
    total_trades:    int   = 0
    wins:            int   = 0
    losses:          int   = 0
    win_rate:        float = 0.5
    flip_threshold:  float = 0.65
    block_threshold: float = 0.55
    built_at:        str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Weight accessors ─────────────────────────────────────────────────────

    def get_weight(self, dimension: str, value: str, direction: str) -> float:
        """
        Return the continuous weight [0, 3] for a given condition.
        Falls back to WEIGHT_DEFAULT when unseen.
        """
        return (
            self.weights
            .get(dimension, {})
            .get(value, {})
            .get(direction, WEIGHT_DEFAULT)
        )

    def get_discrete_label(self, dimension: str, value: str, direction: str) -> str:
        """Return the human-readable label (BLOCKED / LOW / MEDIUM / HIGH)."""
        return discrete_label(self.get_weight(dimension, value, direction))

    def get_sample_count(
        self,
        dimension: str,
        value: str,
        direction: str,
        outcome: str,
    ) -> int:
        """Return the raw sample count for a (dimension, value, direction, outcome) cell."""
        key = f"{direction}_{outcome}"
        return (
            self.sample_counts
            .get(dimension, {})
            .get(value, {})
            .get(key, 0)
        )

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict suitable for DB storage."""
        return {
            "ea_id":           self.ea_id,
            "regime_weights":  self.weights.get("regime", {}),
            "session_weights": self.weights.get("session", {}),
            "volatility_weights": self.weights.get("volatility", {}),
            "momentum_weights":   self.weights.get("momentum", {}),
            "level_prox_weights": self.weights.get("level_prox", {}),
            "flip_threshold":  self.flip_threshold,
            "block_threshold": self.block_threshold,
            "total_trades":    self.total_trades,
            "wins":            self.wins,
            "losses":          self.losses,
            "win_rate":        round(self.win_rate, 4),
            "sample_counts":   self.sample_counts,
            "built_at":        self.built_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EAProfile":
        """
        Reconstruct an EAProfile from the dict returned by db.get_ea_profile().
        Handles the column-per-dimension storage layout used in the DB.
        """
        weights: Dict[str, Dict] = {}
        for dim in SUPPORTED_DIMENSIONS:
            col = f"{dim}_weights"
            raw = d.get(col) or d.get("weights", {}).get(dim, {})
            if raw:
                weights[dim] = raw

        return cls(
            ea_id=d.get("ea_id", "unknown"),
            weights=weights,
            sample_counts=d.get("sample_counts", {}),
            total_trades=d.get("total_trades", 0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            win_rate=float(d.get("win_rate") or 0.5),
            flip_threshold=float(d.get("flip_threshold") or 0.65),
            block_threshold=float(d.get("block_threshold") or 0.55),
            built_at=d.get("built_at", datetime.now(timezone.utc).isoformat()),
        )

    def __repr__(self) -> str:
        return (
            f"EAProfile(ea_id={self.ea_id!r}, trades={self.total_trades}, "
            f"win_rate={self.win_rate:.1%}, dims={list(self.weights.keys())})"
        )


# ── EAProfileBuilder ─────────────────────────────────────────────────────────

class EAProfileBuilder:
    """
    Builds EAProfile objects from a list of completed trade dicts.

    Each trade dict must have at minimum:
      - ``ea_id``   (str)  — added by Phase 1 schema migration
      - ``direction`` (str) — "BUY" or "SELL"
      - ``outcome``  (str) — "WIN" or "LOSS"
      - ``regime``   (str) — optional but highly recommended
      - ``session``  (str) — optional but highly recommended

    Richer trade dicts (containing raw feature values like
    ``atr_normalized``, ``trend_alignment_score``, ``dist_to_pdh`` etc.)
    will also enable volatility, momentum, and level_prox profiling.

    Parameters
    ----------
    min_samples : int
        Minimum number of trades in a cell before computing a real weight.
        Cells below this threshold fall back to ``WEIGHT_DEFAULT``.
    default_ea_id : str
        ea_id to assign when a trade dict has no ``ea_id`` field.
    """

    def __init__(
        self,
        min_samples: int = 5,
        default_ea_id: str = "default",
    ) -> None:
        self.min_samples    = min_samples
        self.default_ea_id  = default_ea_id

    # ── Public API ───────────────────────────────────────────────────────────

    def build_from_trades(self, trades: List[dict]) -> Dict[str, EAProfile]:
        """
        Build profiles for every EA present in the trade list.

        Parameters
        ----------
        trades : list of trade dicts (see class docstring)

        Returns
        -------
        dict mapping ea_id → EAProfile
        """
        # Group trades by ea_id
        ea_trades: Dict[str, List[dict]] = defaultdict(list)
        for t in trades:
            ea_id = str(t.get("ea_id") or self.default_ea_id).strip()
            if not ea_id:
                ea_id = self.default_ea_id
            ea_trades[ea_id].append(t)

        profiles: Dict[str, EAProfile] = {}
        for ea_id, ea_trade_list in ea_trades.items():
            try:
                profiles[ea_id] = self.build_for_ea(ea_id, ea_trade_list)
                logger.info(
                    "Built profile for %s: %d trades, %.1f%% WR, dims=%s",
                    ea_id,
                    profiles[ea_id].total_trades,
                    profiles[ea_id].win_rate * 100,
                    list(profiles[ea_id].weights.keys()),
                )
            except Exception as exc:
                logger.error("Failed to build profile for %s: %s", ea_id, exc, exc_info=True)

        return profiles

    def build_for_ea(self, ea_id: str, trades: List[dict]) -> EAProfile:
        """
        Build a single EAProfile from a list of trades for one EA.

        Parameters
        ----------
        ea_id  : EA identifier
        trades : list of trade dicts belonging to this EA

        Returns
        -------
        EAProfile
        """
        # Filter to WIN / LOSS only (skip PENDING / BREAKEVEN for weights)
        resolved = [
            t for t in trades
            if str(t.get("outcome", "")).upper() in ("WIN", "LOSS")
            and str(t.get("direction", "")).upper() in ("BUY", "SELL")
        ]

        total  = len(resolved)
        wins   = sum(1 for t in resolved if t["outcome"].upper() == "WIN")
        losses = total - wins
        ea_win_rate = wins / total if total > 0 else 0.5

        weights: Dict[str, Dict[str, Dict[str, float]]] = {}
        counts:  Dict[str, Dict[str, Dict[str, int]]]   = {}

        for dim, extractor in _DIMENSION_EXTRACTORS.items():
            dim_weights, dim_counts = self._build_dimension(
                resolved, extractor, ea_win_rate
            )
            if dim_weights:
                weights[dim] = dim_weights
                counts[dim]  = dim_counts

        return EAProfile(
            ea_id=ea_id,
            weights=weights,
            sample_counts=counts,
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=round(ea_win_rate, 4),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_dimension(
        self,
        trades:     List[dict],
        extractor:  Callable[[dict], Optional[str]],
        ea_win_rate: float,
    ) -> Tuple[Dict, Dict]:
        """
        For one dimension, compute per-(value, direction) weights.

        Returns
        -------
        (weights_dict, counts_dict)  — both empty if no data extracted.
        """
        # Accumulate: value → direction → {"WIN": n, "LOSS": n}
        buckets: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"WIN": 0, "LOSS": 0})
        )

        for trade in trades:
            value = extractor(trade)
            if value is None:
                continue
            value     = str(value).strip().lower()  # normalise
            direction = trade["direction"].upper()
            outcome   = trade["outcome"].upper()
            buckets[value][direction][outcome] += 1

        if not buckets:
            return {}, {}

        weights: Dict[str, Dict[str, float]] = {}
        counts:  Dict[str, Dict[str, int]]   = {}

        for value, dir_data in buckets.items():
            weights[value] = {}
            counts[value]  = {}

            for direction in ("BUY", "SELL"):
                cell = dir_data.get(direction, {"WIN": 0, "LOSS": 0})
                n    = cell["WIN"] + cell["LOSS"]
                wr   = cell["WIN"] / n if n >= self.min_samples else ea_win_rate

                weights[value][direction] = round(wr * 3.0, 4)

                # Store raw counts for transparency
                counts[value][f"{direction}_WIN"]  = cell["WIN"]
                counts[value][f"{direction}_LOSS"] = cell["LOSS"]

        return weights, counts

    # ── Utility: merge incremental updates ───────────────────────────────────

    @staticmethod
    def merge_weight_update(
        profile:    EAProfile,
        dimension:  str,
        value:      str,
        direction:  str,
        new_weight: float,
    ) -> EAProfile:
        """
        Return a new EAProfile with one weight cell updated.
        Used by AdaptiveUpdater to apply incremental changes.
        """
        import copy
        updated = copy.deepcopy(profile)
        updated.weights.setdefault(dimension, {}).setdefault(value, {})
        updated.weights[dimension][value][direction] = round(
            max(0.0, min(3.0, new_weight)), 4
        )
        return updated