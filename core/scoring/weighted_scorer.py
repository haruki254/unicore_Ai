"""
Weighted Scorer
===============
Scores a market snapshot against an EA's profile weights to produce
an explicit, field-by-field justification for ALLOW / BLOCK / FLIP
decisions — replacing the opaque ML probability with a transparent
"EA4 matches this setup 10/15 for SELL, only 3/15 for BUY".

Scoring model
-------------
  For each of the 5 profiled dimensions (regime, session, volatility,
  momentum, level_prox), look up the EA's continuous weight [0, 3]
  for the observed (dimension_value, direction) pair.

  score        = Σ weights across all available dimensions
  max_score    = n_active_dims × 3.0
  confidence   = score / max_score                  ∈ [0.0, 1.0]

Flip conditions (applied externally in DecisionEngine)
-------------------------------------------------------
  ONLY flip when BOTH of the following are true:
    - original_direction confidence  <  FLIP_LOW_CONFIDENCE  (profile says NO)
    - opposite_direction confidence  >  FLIP_HIGH_CONFIDENCE (profile says YES)

  This is a stricter dual-gate vs. the old single-threshold approach.

Usage
-----
    from core.profiles import EAProfile
    from core.scoring import WeightedScorer, DirectionComparison

    scorer  = WeightedScorer()
    profile = EAProfile.from_dict(db.get_ea_profile("EA4"))

    # Score both directions and decide
    cmp = scorer.compare_directions(snapshot_features, profile)
    print(cmp.buy.confidence)   # 0.23
    print(cmp.sell.confidence)  # 0.81

    # Single direction
    result = scorer.score_direction(snapshot_features, profile, "BUY")
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.profiles.ea_profile_builder import (
    EAProfile,
    WEIGHT_DEFAULT,
    SUPPORTED_DIMENSIONS,
    discrete_label,
    categorize_volatility,
    categorize_momentum,
    categorize_level_prox,
)

logger = logging.getLogger(__name__)

# ── Flip gate thresholds (used externally by DecisionEngine) ─────────────────
FLIP_LOW_CONFIDENCE  = 0.35   # original direction must be BELOW this to flip
FLIP_HIGH_CONFIDENCE = 0.65   # opposite direction must be ABOVE this to flip


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class DimensionResult:
    """Score detail for a single profiled dimension."""
    dimension:  str
    value:      str          # e.g. "strong_bull_trend", "BULLISH", "FREE"
    weight:     float        # continuous [0, 3]
    label:      str          # "BLOCKED" / "LOW" / "MEDIUM" / "HIGH"
    is_default: bool         # True when profile had no data → used WEIGHT_DEFAULT


@dataclass
class ScoringResult:
    """
    Full scoring outcome for one direction against one EA profile.

    Attributes
    ----------
    direction       : "BUY" or "SELL"
    score           : sum of dimension weights (0 – max_score)
    max_score       : n_active_dims × 3.0
    confidence      : score / max_score ∈ [0.0, 1.0]
    breakdown       : per-dimension detail list
    active_dims     : number of dimensions with observed values
    missing_dims    : dimensions that could not be extracted
    profile_has_data: False when the profile had NO data (all defaults used)
    """
    direction:       str
    score:           float
    max_score:       float
    confidence:      float
    breakdown:       List[DimensionResult]
    active_dims:     int
    missing_dims:    List[str] = field(default_factory=list)
    profile_has_data: bool = True

    def __str__(self) -> str:
        pct = f"{self.confidence:.0%}"
        detail = ", ".join(
            f"{d.dimension}={d.value}({d.label})"
            for d in self.breakdown
        )
        return f"ScoringResult({self.direction} {pct} [{self.score:.1f}/{self.max_score:.1f}] | {detail})"


@dataclass
class DirectionComparison:
    """
    Paired BUY + SELL scoring results for the same snapshot.
    Used by DecisionEngine to evaluate flip conditions.
    """
    buy:   ScoringResult
    sell:  ScoringResult

    @property
    def favoured_direction(self) -> str:
        """Return whichever direction has the higher confidence."""
        return "BUY" if self.buy.confidence >= self.sell.confidence else "SELL"

    @property
    def confidence_gap(self) -> float:
        """Absolute difference in confidence between BUY and SELL."""
        return abs(self.buy.confidence - self.sell.confidence)

    @property
    def should_flip_to_sell(self) -> bool:
        """True when the profile strongly favours SELL over an original BUY signal."""
        return (
            self.buy.confidence  < FLIP_LOW_CONFIDENCE and
            self.sell.confidence > FLIP_HIGH_CONFIDENCE
        )

    @property
    def should_flip_to_buy(self) -> bool:
        """True when the profile strongly favours BUY over an original SELL signal."""
        return (
            self.sell.confidence  < FLIP_LOW_CONFIDENCE and
            self.buy.confidence > FLIP_HIGH_CONFIDENCE
        )

    def flip_verdict(self, ea_signal: str) -> Optional[str]:
        """
        Return the recommended action string or None if no flip warranted.

        Returns
        -------
        "FLIP_TO_SELL", "FLIP_TO_BUY", or None
        """
        if ea_signal.upper() == "BUY" and self.should_flip_to_sell:
            return "FLIP_TO_SELL"
        if ea_signal.upper() == "SELL" and self.should_flip_to_buy:
            return "FLIP_TO_BUY"
        return None

    def to_dict(self) -> dict:
        return {
            "buy_confidence":  round(self.buy.confidence,  4),
            "sell_confidence": round(self.sell.confidence, 4),
            "buy_score":       round(self.buy.score,       4),
            "sell_score":      round(self.sell.score,      4),
            "favoured":        self.favoured_direction,
            "gap":             round(self.confidence_gap,  4),
        }


# ── WeightedScorer ───────────────────────────────────────────────────────────

class WeightedScorer:
    """
    Scores market snapshot features against an EAProfile.

    Parameters
    ----------
    dimensions : list of dimension names to include in scoring.
                 Defaults to all 5 supported dimensions.
    """

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
    ) -> None:
        self.dimensions = dimensions or SUPPORTED_DIMENSIONS

    # ── Condition extraction ─────────────────────────────────────────────────

    def extract_conditions(self, features: dict) -> Dict[str, Optional[str]]:
        """
        Convert a raw feature dict into categorised dimension values.

        Returns
        -------
        dict mapping dimension → value (or None when extraction fails)
        """
        conditions: Dict[str, Optional[str]] = {}

        # ── Regime (direct field) ─────────────────────────────────────────
        regime = features.get("regime") or features.get("market_regime")
        conditions["regime"] = str(regime).strip().lower() if regime else None

        # ── Session (direct field) ────────────────────────────────────────
        session = features.get("session") or features.get("session_type")
        conditions["session"] = str(session).strip().lower() if session else None

        # ── Volatility (derived) ──────────────────────────────────────────
        has_vol_fields = any(
            k in features
            for k in ("range_expansion", "range_contraction", "atr_normalized")
        )
        conditions["volatility"] = (
            categorize_volatility(features) if has_vol_fields else None
        )

        # ── Momentum (derived) ────────────────────────────────────────────
        has_mom_fields = any(
            k in features for k in ("trend_alignment_score", "momentum_5")
        )
        conditions["momentum"] = (
            categorize_momentum(features) if has_mom_fields else None
        )

        # ── Level proximity (derived) ─────────────────────────────────────
        has_lvl_fields = any(
            k in features
            for k in ("dist_to_pdh", "dist_to_pdl", "dist_to_support", "dist_to_resistance")
        )
        conditions["level_prox"] = (
            categorize_level_prox(features) if has_lvl_fields else None
        )

        return conditions

    # ── Core scoring ─────────────────────────────────────────────────────────

    def score_direction(
        self,
        features:    dict,
        ea_profile:  EAProfile,
        direction:   str,
    ) -> ScoringResult:
        """
        Score a snapshot for a specific direction against the EA profile.

        Parameters
        ----------
        features    : raw feature dict (output of FeaturePipeline)
        ea_profile  : the EA's loaded profile
        direction   : "BUY" or "SELL"

        Returns
        -------
        ScoringResult
        """
        direction = direction.upper()
        conditions = self.extract_conditions(features)

        breakdown:    List[DimensionResult] = []
        score:        float = 0.0
        active_dims:  int   = 0
        missing_dims: List[str] = []

        profile_has_data = bool(ea_profile.weights)

        for dim in self.dimensions:
            value = conditions.get(dim)
            if value is None:
                missing_dims.append(dim)
                continue

            active_dims += 1
            weight     = ea_profile.get_weight(dim, value, direction)
            is_default = (
                dim not in ea_profile.weights or
                value not in ea_profile.weights.get(dim, {}) or
                direction not in ea_profile.weights[dim].get(value, {})
            )

            score += weight
            breakdown.append(DimensionResult(
                dimension=dim,
                value=value,
                weight=weight,
                label=discrete_label(weight),
                is_default=is_default,
            ))

        max_score = active_dims * 3.0
        confidence = (score / max_score) if max_score > 0 else 0.5

        return ScoringResult(
            direction=direction,
            score=round(score, 4),
            max_score=round(max_score, 4),
            confidence=round(confidence, 4),
            breakdown=breakdown,
            active_dims=active_dims,
            missing_dims=missing_dims,
            profile_has_data=profile_has_data,
        )

    def compare_directions(
        self,
        features:    dict,
        ea_profile:  EAProfile,
    ) -> DirectionComparison:
        """
        Score both BUY and SELL for the same snapshot and return a comparison.

        Parameters
        ----------
        features   : raw feature dict
        ea_profile : the EA's loaded profile

        Returns
        -------
        DirectionComparison with .buy and .sell ScoringResults
        """
        buy_result  = self.score_direction(features, ea_profile, "BUY")
        sell_result = self.score_direction(features, ea_profile, "SELL")
        return DirectionComparison(buy=buy_result, sell=sell_result)

    # ── Audit / logging helper ────────────────────────────────────────────────

    def format_breakdown_log(
        self,
        cmp: DirectionComparison,
        ea_id: str,
        ea_signal: str,
    ) -> str:
        """
        Return a concise human-readable log string for audit trails.

        Example output:
            [EA4] BUY=23%(3.5/15.0) SELL=81%(12.2/15.0) | signal=BUY
            Breakdown: regime=strong_bull_trend[HIGH/HIGH], session=london[MED/MED],
                       volatility=HIGH[HIGH/MED], momentum=BULLISH[HIGH/LOW],
                       level_prox=FREE[HIGH/HIGH]
        """
        b, s = cmp.buy, cmp.sell
        rows = []
        for dr, sr in zip(b.breakdown, s.breakdown):
            rows.append(
                f"{dr.dimension}={dr.value}"
                f"[BUY:{dr.label[:3]}/SELL:{sr.label[:3]}]"
            )
        detail = ", ".join(rows)

        return (
            f"[{ea_id}] BUY={b.confidence:.0%}({b.score:.1f}/{b.max_score:.1f}) "
            f"SELL={s.confidence:.0%}({s.score:.1f}/{s.max_score:.1f}) "
            f"| signal={ea_signal} | {detail}"
        )