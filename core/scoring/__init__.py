"""
core.scoring — EA-profile-based weighted scoring engine.

Public API
----------
WeightedScorer   : Scores a market snapshot against an EAProfile.
ScoringResult    : Output dataclass from WeightedScorer.score_direction().
DirectionComparison : Paired BUY+SELL results from compare_directions().
"""
from .weighted_scorer import (
    WeightedScorer,
    ScoringResult,
    DirectionComparison,
    FLIP_LOW_CONFIDENCE,
    FLIP_HIGH_CONFIDENCE,
)

__all__ = [
    "WeightedScorer",
    "ScoringResult",
    "DirectionComparison",
    "FLIP_LOW_CONFIDENCE",
    "FLIP_HIGH_CONFIDENCE",
]