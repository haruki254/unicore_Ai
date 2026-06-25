"""
core.profiles — EA-specific performance profile management.

Public API
----------
EAProfile          : Dataclass representing one EA's weighted conditions.
EAProfileBuilder   : Builds / refreshes profiles from historical trade data.
categorize_*       : Feature categorisation helpers shared with the scorer.
"""
try:
    from .ea_profile_builder import (
        EAProfile,
        EAProfileBuilder,
        categorize_volatility,
        categorize_momentum,
        categorize_level_prox,
        WEIGHT_HIGH,
        WEIGHT_NEUTRAL,
        WEIGHT_BLOCKED,
        DISCRETE_LABELS,
    )
except Exception:
    # Fallback for environments where relative imports may not resolve
    # (editors/linters). Use absolute import path as a fallback.
    from core.profiles.ea_profile_builder import (
        EAProfile,
        EAProfileBuilder,
        categorize_volatility,
        categorize_momentum,
        categorize_level_prox,
        WEIGHT_HIGH,
        WEIGHT_NEUTRAL,
        WEIGHT_BLOCKED,
        DISCRETE_LABELS,
    )

__all__ = [
    "EAProfile",
    "EAProfileBuilder",
    "categorize_volatility",
    "categorize_momentum",
    "categorize_level_prox",
    "WEIGHT_HIGH",
    "WEIGHT_NEUTRAL",
    "WEIGHT_BLOCKED",
    "DISCRETE_LABELS",
]