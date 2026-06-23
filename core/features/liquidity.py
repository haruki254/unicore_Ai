"""
Liquidity Feature Engineering

Computes:
  - Distance to nearest support/resistance
  - Distance to Previous Day High/Low (PDH/PDL)
  - Distance to Weekly High/Low
  - Liquidity pool proximity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def _identify_support_resistance(
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    window: int = 10,
    min_touches: int = 2,
) -> Tuple[List[float], List[float]]:
    """
    Identify significant support and resistance levels
    using price clustering around swing points.

    Returns (resistance_levels, support_levels)
    """
    n = len(highs)
    resistance_levels = []
    support_levels    = []

    # Swing highs → resistance
    for i in range(window, n - window):
        if highs[i] == np.max(highs[i - window: i + window + 1]):
            resistance_levels.append(float(highs[i]))

    # Swing lows → support
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window: i + window + 1]):
            support_levels.append(float(lows[i]))

    # Deduplicate levels within 0.1% tolerance
    def cluster(levels: List[float], tol: float = 0.001) -> List[float]:
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1]) / (clustered[-1] + 1e-10) > tol:
                clustered.append(lvl)
        return clustered

    return cluster(resistance_levels), cluster(support_levels)


def compute_liquidity(
    df_m5:  pd.DataFrame,
    df_h4:  pd.DataFrame,
    df_d1:  pd.DataFrame,
    current_price: float,
) -> Dict[str, float]:
    """
    Compute liquidity and key level features.

    Parameters
    ----------
    df_m5  : M5 candles (last 100)
    df_h4  : H4 candles (last 20)
    df_d1  : Daily candles (last 5+)
    current_price : current market price

    Returns
    -------
    dict of feature_name -> value
    """
    features: Dict[str, float] = {}
    price = current_price + 1e-10

    # ── S/R from H4 data ──────────────────────────────────────
    if df_h4 is not None and len(df_h4) >= 10:
        h4_highs  = df_h4["high"].values.astype(float)
        h4_lows   = df_h4["low"].values.astype(float)
        h4_closes = df_h4["close"].values.astype(float)
        resistance, support = _identify_support_resistance(h4_highs, h4_lows, h4_closes)
    else:
        resistance, support = [], []

    # Distance to nearest resistance (above price)
    above_res = [r for r in resistance if r > price]
    if above_res:
        nearest_res = min(above_res)
        features["dist_to_resistance"] = float((nearest_res - price) / price)
    else:
        features["dist_to_resistance"] = 0.05  # far away default

    # Distance to nearest support (below price)
    below_sup = [s for s in support if s < price]
    if below_sup:
        nearest_sup = max(below_sup)
        features["dist_to_support"] = float((price - nearest_sup) / price)
    else:
        features["dist_to_support"] = 0.05

    # S/R ratio (position within support-resistance range)
    if above_res and below_sup:
        nearest_res2 = min(above_res)
        nearest_sup2 = max(below_sup)
        rng = nearest_res2 - nearest_sup2 + 1e-10
        features["sr_ratio"] = float((price - nearest_sup2) / rng)
    else:
        features["sr_ratio"] = 0.5

    # ── Previous Day High/Low ─────────────────────────────────
    if df_d1 is not None and len(df_d1) >= 2:
        prev_day = df_d1.iloc[-2]
        pdh = float(prev_day["high"])
        pdl = float(prev_day["low"])
    elif df_h4 is not None and len(df_h4) >= 6:
        # Approximate PDH/PDL from last 6 H4 candles
        pdh = float(df_h4["high"].values[-6:].max())
        pdl = float(df_h4["low"].values[-6:].min())
    else:
        pdh = price * 1.01
        pdl = price * 0.99

    features["prev_day_high"] = pdh
    features["prev_day_low"]  = pdl
    features["dist_to_pdh"]   = float((pdh - price) / price)
    features["dist_to_pdl"]   = float((price - pdl) / price)
    features["near_pdh"]      = float(abs(price - pdh) / price < 0.001)
    features["near_pdl"]      = float(abs(price - pdl) / price < 0.001)
    features["above_pdh"]     = float(price > pdh)
    features["below_pdl"]     = float(price < pdl)

    # ── Weekly High/Low ───────────────────────────────────────
    if df_d1 is not None and len(df_d1) >= 5:
        week_highs = df_d1["high"].values.astype(float)[-5:]
        week_lows  = df_d1["low"].values.astype(float)[-5:]
        weekly_high = float(week_highs.max())
        weekly_low  = float(week_lows.min())
    elif df_h4 is not None and len(df_h4) >= 30:
        weekly_high = float(df_h4["high"].values[-30:].max())
        weekly_low  = float(df_h4["low"].values[-30:].min())
    else:
        weekly_high = price * 1.02
        weekly_low  = price * 0.98

    features["weekly_high"] = weekly_high
    features["weekly_low"]  = weekly_low
    features["dist_to_wh"]  = float((weekly_high - price) / price)
    features["dist_to_wl"]  = float((price - weekly_low) / price)
    features["near_weekly_high"] = float(abs(price - weekly_high) / price < 0.002)
    features["near_weekly_low"]  = float(abs(price - weekly_low) / price < 0.002)

    # ── Liquidity Pool Proximity ──────────────────────────────
    # Key liquidity zones are above recent highs / below recent lows
    if df_m5 is not None and len(df_m5) >= 20:
        m5_highs = df_m5["high"].values.astype(float)[-20:]
        m5_lows  = df_m5["low"].values.astype(float)[-20:]
        recent_high_pool = float(m5_highs.max())
        recent_low_pool  = float(m5_lows.min())
        features["near_buy_side_liquidity"]  = float(abs(price - recent_high_pool) / price < 0.0005)
        features["near_sell_side_liquidity"] = float(abs(price - recent_low_pool) / price < 0.0005)
    else:
        features["near_buy_side_liquidity"]  = 0.0
        features["near_sell_side_liquidity"] = 0.0

    return features
