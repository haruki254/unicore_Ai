"""
Price Action Feature Engineering

Computes:
  - Candle body sizes and ratios (last 50 and 100 candles)
  - Upper/lower wick analysis
  - Momentum at multiple lookbacks
  - Directional statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def compute_price_action(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute price action features from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: open, high, low, close, volume (volume optional)

    Returns
    -------
    dict of feature_name -> value
    """
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    # ── Candle components ──────────────────────────────────────
    body_sizes   = np.abs(closes - opens)
    upper_wicks  = highs - np.maximum(opens, closes)
    lower_wicks  = np.minimum(opens, closes) - lows
    total_ranges = highs - lows
    total_ranges = np.where(total_ranges == 0, 1e-10, total_ranges)

    body_ratios  = body_sizes / total_ranges
    upper_ratios = upper_wicks / total_ranges
    lower_ratios = lower_wicks / total_ranges

    # ── Averages over last 50 and 100 candles ─────────────────
    n50  = min(50,  len(df))
    n100 = min(100, len(df))

    features: Dict[str, float] = {}

    # Body sizes
    features["body_size_avg_50"]  = float(np.mean(body_sizes[-n50:]))
    features["body_size_avg_100"] = float(np.mean(body_sizes[-n100:]))
    features["body_ratio_avg_50"] = float(np.mean(body_ratios[-n50:]))

    # Wick sizes
    features["wick_upper_avg"] = float(np.mean(upper_wicks[-n50:]))
    features["wick_lower_avg"] = float(np.mean(lower_wicks[-n50:]))
    features["wick_upper_ratio"] = float(np.mean(upper_ratios[-n50:]))
    features["wick_lower_ratio"] = float(np.mean(lower_ratios[-n50:]))

    # ── Momentum (close-to-close rate of change) ───────────────
    for period in [5, 10, 20]:
        if len(closes) > period:
            mom = (closes[-1] - closes[-period - 1]) / (closes[-period - 1] + 1e-10)
            features[f"momentum_{period}"] = float(mom)
        else:
            features[f"momentum_{period}"] = 0.0

    # ── Directional features ───────────────────────────────────
    # Ratio of bullish candles in last 20
    n20 = min(20, len(df))
    bullish_mask = closes[-n20:] > opens[-n20:]
    features["candle_direction_ratio"] = float(np.mean(bullish_mask))

    # Close position within candle range (0 = at low, 1 = at high)
    last_range = highs[-1] - lows[-1]
    if last_range > 0:
        features["close_vs_open"] = float((closes[-1] - lows[-1]) / last_range)
    else:
        features["close_vs_open"] = 0.5

    # ── Large body candles (> 1.5x average body) ──────────────
    avg_body = np.mean(body_sizes[-n50:]) + 1e-10
    large_bodies = body_sizes[-n20:] > 1.5 * avg_body
    features["large_body_count"] = float(np.sum(large_bodies))

    # ── Doji detection (body < 10% of range) ──────────────────
    doji_mask = body_ratios[-n20:] < 0.1
    features["doji_count"] = float(np.sum(doji_mask))

    # ── Engulfing patterns ────────────────────────────────────
    if len(df) >= 2:
        c1_bull = closes[-2] > opens[-2]
        c2_bull = closes[-1] > opens[-1]

        # Bullish engulfing: prev bearish, current bullish, body engulfs
        if not c1_bull and c2_bull:
            if opens[-1] < closes[-2] and closes[-1] > opens[-2]:
                features["bullish_engulfing"] = 1.0
            else:
                features["bullish_engulfing"] = 0.0
        else:
            features["bullish_engulfing"] = 0.0

        # Bearish engulfing: prev bullish, current bearish, body engulfs
        if c1_bull and not c2_bull:
            if opens[-1] > closes[-2] and closes[-1] < opens[-2]:
                features["bearish_engulfing"] = 1.0
            else:
                features["bearish_engulfing"] = 0.0
        else:
            features["bearish_engulfing"] = 0.0
    else:
        features["bullish_engulfing"] = 0.0
        features["bearish_engulfing"] = 0.0

    # ── Pin bars ──────────────────────────────────────────────
    # Bullish pin: lower wick > 2x body, small upper wick
    last_body  = body_sizes[-1]
    last_upper = upper_wicks[-1]
    last_lower = lower_wicks[-1]
    features["bullish_pin_bar"] = float(
        last_lower > 2 * last_body and last_upper < 0.5 * last_body
    )
    features["bearish_pin_bar"] = float(
        last_upper > 2 * last_body and last_lower < 0.5 * last_body
    )

    # ── Volume (if available) ─────────────────────────────────
    if "volume" in df.columns:
        vols = df["volume"].values.astype(float)
        avg_vol = np.mean(vols[-n50:]) + 1e-10
        features["volume_ratio"] = float(vols[-1] / avg_vol)
        features["volume_surge"]  = float(vols[-1] > 2.0 * avg_vol)
    else:
        features["volume_ratio"] = 1.0
        features["volume_surge"]  = 0.0

    return features
