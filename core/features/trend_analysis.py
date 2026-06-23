"""
Trend Analysis Feature Engineering

Determines trend direction across M5, M15, H1, H4, Daily timeframes.
Computes trend alignment score and strength.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


def _determine_trend(closes: np.ndarray, period: int = 20) -> int:
    """
    Classify trend as -1 (bearish), 0 (neutral), or 1 (bullish).
    Uses SMA slope + price position relative to MA.
    """
    if len(closes) < period:
        return 0

    ma = np.mean(closes[-period:])
    price = closes[-1]

    # SMA slope (normalised)
    if len(closes) >= period + 5:
        ma_prev = np.mean(closes[-period - 5: -5])
        slope   = (ma - ma_prev) / (ma_prev + 1e-10)
    else:
        slope = 0.0

    price_above_ma = price > ma

    if price_above_ma and slope > 0.0002:
        return 1   # bullish
    elif not price_above_ma and slope < -0.0002:
        return -1  # bearish
    else:
        return 0   # neutral


def _trend_strength(closes: np.ndarray, period: int = 20) -> float:
    """
    Trend strength: how consistently price moves in one direction.
    Range [0, 1]. Higher = stronger trend.
    """
    if len(closes) < period:
        return 0.0

    returns = np.diff(closes[-period:])
    positive = np.sum(returns > 0)
    negative = np.sum(returns < 0)
    total    = len(returns) + 1e-10

    # Consistency of direction
    consistency = abs(positive - negative) / total

    # Magnitude (coefficient of variation of cumulative return)
    cum_return = abs((closes[-1] - closes[-period]) / (closes[-period] + 1e-10))

    return float(np.clip((consistency + cum_return) / 2.0, 0.0, 1.0))


def compute_trend_analysis(
    candles_by_tf: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    Compute trend features for each timeframe.

    Parameters
    ----------
    candles_by_tf : dict
        Keys: 'M5', 'M15', 'H1', 'H4', 'D1'
        Values: OHLCV DataFrames (sorted oldest → newest)

    Returns
    -------
    dict of feature_name -> value
    """
    features: Dict[str, float] = {}
    tf_map = {
        "M5":  ("trend_m5",  20),
        "M15": ("trend_m15", 20),
        "H1":  ("trend_h1",  20),
        "H4":  ("trend_h4",  20),
        "D1":  ("trend_d1",  14),
    }

    trend_values = []

    for tf, (feat_name, period) in tf_map.items():
        df = candles_by_tf.get(tf)
        if df is not None and len(df) >= 5:
            closes = df["close"].values.astype(float)
            trend  = _determine_trend(closes, period)
            strength = _trend_strength(closes, period)
        else:
            trend    = 0
            strength = 0.0

        features[feat_name] = float(trend)
        trend_values.append(trend)

    # ── Trend Alignment Score ──────────────────────────────────
    # How many timeframes agree? Range [-1, 1]
    if trend_values:
        alignment = np.mean(trend_values)
        features["trend_alignment_score"] = float(alignment)
    else:
        features["trend_alignment_score"] = 0.0

    # ── Overall Trend Strength ────────────────────────────────
    # Use H1 candles as primary, fallback to M15
    primary_tf = "H1" if "H1" in candles_by_tf else "M15"
    df_primary = candles_by_tf.get(primary_tf)
    if df_primary is not None and len(df_primary) >= 20:
        closes_p = df_primary["close"].values.astype(float)
        features["trend_strength"] = _trend_strength(closes_p, 20)
    else:
        features["trend_strength"] = 0.0

    # ── Higher-timeframe Agreement ─────────────────────────────
    # H4 and D1 agree on direction
    h4 = features.get("trend_h4", 0.0)
    d1 = features.get("trend_d1", 0.0)
    features["htf_agreement"]       = float(h4 == d1 and h4 != 0)
    features["htf_bullish"]         = float(h4 == 1 and d1 == 1)
    features["htf_bearish"]         = float(h4 == -1 and d1 == -1)

    # ── Trend Conflict (short TF opposing long TF) ─────────────
    m5  = features.get("trend_m5",  0.0)
    m15 = features.get("trend_m15", 0.0)
    h1  = features.get("trend_h1",  0.0)

    # Conflict: M5 opposes H4
    features["trend_conflict"] = float(m5 != 0 and h4 != 0 and m5 != h4)

    # ── Trend Maturity (how long has trend been running?) ──────
    # Check consecutive aligned closes on H1
    df_h1 = candles_by_tf.get("H1")
    if df_h1 is not None and len(df_h1) >= 10:
        h1_closes = df_h1["close"].values.astype(float)
        h1_direction = np.sign(np.diff(h1_closes))
        dominant = h1_direction[-1]
        if dominant != 0:
            streak = 1
            for i in range(len(h1_direction) - 2, -1, -1):
                if h1_direction[i] == dominant:
                    streak += 1
                else:
                    break
            features["trend_maturity"] = float(min(streak, 20) / 20.0)
        else:
            features["trend_maturity"] = 0.0
    else:
        features["trend_maturity"] = 0.0

    return features
