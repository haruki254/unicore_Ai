"""
Volatility Feature Engineering

Computes:
  - ATR (Average True Range) — raw and normalised
  - Standard deviation of returns
  - Range expansion / contraction signals
  - Volatility regime classification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def _atr(
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute ATR using Wilder's smoothing."""
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_closes),
            np.abs(lows  - prev_closes),
        ),
    )

    atr_arr = np.zeros(len(tr))
    atr_arr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period

    return atr_arr


def compute_volatility(
    df: pd.DataFrame,
    atr_period: int = 14,
) -> Dict[str, float]:
    """
    Compute all volatility features.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: open, high, low, close

    Returns
    -------
    dict of feature_name -> value
    """
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    features: Dict[str, float] = {}

    # ── ATR ───────────────────────────────────────────────────
    atr_arr = _atr(highs, lows, closes, atr_period)
    current_atr = float(atr_arr[-1])
    features["atr_14"] = current_atr

    # ATR normalised by price
    features["atr_normalized"] = float(current_atr / (closes[-1] + 1e-10))

    # ATR percentile rank over last 100 periods (volatility rank)
    n = min(100, len(atr_arr))
    atr_window = atr_arr[-n:]
    atr_rank   = float(np.mean(atr_window < current_atr))
    features["atr_rank"] = atr_rank

    # ── Standard Deviation of Returns ─────────────────────────
    if len(closes) > 20:
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        features["std_dev_20"]  = float(np.std(returns[-20:]))
        features["std_dev_50"]  = float(np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns))
    else:
        features["std_dev_20"] = 0.001
        features["std_dev_50"] = 0.001

    # ── Range Expansion / Contraction ─────────────────────────
    daily_ranges = highs - lows

    if len(daily_ranges) > 10:
        avg_range_20 = np.mean(daily_ranges[-20:]) + 1e-10
        current_range = daily_ranges[-1]

        range_ratio = current_range / avg_range_20
        features["range_ratio"]       = float(range_ratio)
        features["range_expansion"]   = float(range_ratio > 1.3)
        features["range_contraction"] = float(range_ratio < 0.7)
    else:
        features["range_ratio"]       = 1.0
        features["range_expansion"]   = 0.0
        features["range_contraction"] = 0.0

    # ── ATR Trend (expanding or contracting volatility) ───────
    if len(atr_arr) > 10:
        atr_short = np.mean(atr_arr[-5:])
        atr_long  = np.mean(atr_arr[-20:]) if len(atr_arr) >= 20 else np.mean(atr_arr)
        atr_slope = (atr_short - atr_long) / (atr_long + 1e-10)
        features["atr_slope"]    = float(atr_slope)
        features["vol_expanding"] = float(atr_slope > 0.05)
        features["vol_contracting"] = float(atr_slope < -0.05)
    else:
        features["atr_slope"]     = 0.0
        features["vol_expanding"]  = 0.0
        features["vol_contracting"] = 0.0

    # ── Volatility Regime Encoding ─────────────────────────────
    # 0 = very low, 1 = low, 2 = normal, 3 = high, 4 = extreme
    if atr_rank < 0.20:
        vol_regime = 0
    elif atr_rank < 0.40:
        vol_regime = 1
    elif atr_rank < 0.70:
        vol_regime = 2
    elif atr_rank < 0.90:
        vol_regime = 3
    else:
        vol_regime = 4

    features["volatility_regime"] = float(vol_regime)

    # ── Bollinger Band-like width ──────────────────────────────
    if len(closes) >= 20:
        ma20  = np.mean(closes[-20:])
        std20 = np.std(closes[-20:]) + 1e-10
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / (ma20 + 1e-10)
        bb_pct   = (closes[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        features["bb_width"]   = float(bb_width)
        features["bb_pct"]     = float(np.clip(bb_pct, 0.0, 1.0))
        features["bb_squeeze"] = float(bb_width < 0.01)
    else:
        features["bb_width"]   = 0.02
        features["bb_pct"]     = 0.5
        features["bb_squeeze"] = 0.0

    return features
