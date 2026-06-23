"""
Market Structure Feature Engineering

Detects:
  - Higher Highs (HH), Higher Lows (HL)
  - Lower Highs (LH), Lower Lows (LL)
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify swing highs and swing lows using a rolling window.

    Returns boolean arrays: swing_high_mask, swing_low_mask
    """
    n = len(highs)
    swing_highs = np.zeros(n, dtype=bool)
    swing_lows  = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        left_h  = highs[i - window: i]
        right_h = highs[i + 1: i + window + 1]
        left_l  = lows[i - window: i]
        right_l = lows[i + 1: i + window + 1]

        if highs[i] >= np.max(left_h) and highs[i] >= np.max(right_h):
            swing_highs[i] = True
        if lows[i] <= np.min(left_l) and lows[i] <= np.min(right_l):
            swing_lows[i] = True

    return swing_highs, swing_lows


def compute_market_structure(
    df: pd.DataFrame,
    swing_window: int = 5,
    lookback: int = 20,
) -> Dict[str, float]:
    """
    Compute all market structure features from a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close
    swing_window : int
        Half-window for swing point detection
    lookback : int
        How many recent candles to count structure events within

    Returns
    -------
    dict of feature_name -> value
    """
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    swing_highs, swing_lows = _find_swing_points(highs, lows, swing_window)

    # ── Extract swing price levels ──────────────────────────────
    sh_prices = highs[swing_highs]
    sl_prices = lows[swing_lows]

    # Count structure patterns in lookback window
    recent_sh = highs[swing_highs][-lookback:] if len(highs[swing_highs]) >= 1 else np.array([])
    recent_sl = lows[swing_lows][-lookback:]  if len(lows[swing_lows])  >= 1 else np.array([])

    hh_count = _count_higher_highs(recent_sh)
    lh_count = _count_lower_highs(recent_sh)
    hl_count = _count_higher_lows(recent_sl)
    ll_count = _count_lower_lows(recent_sl)

    # ── BOS / CHoCH ────────────────────────────────────────────
    bos_bullish, bos_bearish    = _detect_bos(closes, highs, lows, swing_highs, swing_lows)
    choch_bullish, choch_bearish = _detect_choch(closes, highs, lows, swing_highs, swing_lows)

    # ── Structure Score (-1 strong bear → +1 strong bull) ──────
    structure_score = _compute_structure_score(
        hh_count, hl_count, lh_count, ll_count,
        bos_bullish, bos_bearish, choch_bullish, choch_bearish,
    )

    return {
        "hh_count":       float(hh_count),
        "hl_count":       float(hl_count),
        "lh_count":       float(lh_count),
        "ll_count":       float(ll_count),
        "bos_bullish":    float(bos_bullish),
        "bos_bearish":    float(bos_bearish),
        "choch_bullish":  float(choch_bullish),
        "choch_bearish":  float(choch_bearish),
        "structure_score": float(structure_score),
    }


def _count_higher_highs(swing_highs: np.ndarray) -> int:
    if len(swing_highs) < 2:
        return 0
    return int(np.sum(np.diff(swing_highs) > 0))


def _count_lower_highs(swing_highs: np.ndarray) -> int:
    if len(swing_highs) < 2:
        return 0
    return int(np.sum(np.diff(swing_highs) < 0))


def _count_higher_lows(swing_lows: np.ndarray) -> int:
    if len(swing_lows) < 2:
        return 0
    return int(np.sum(np.diff(swing_lows) > 0))


def _count_lower_lows(swing_lows: np.ndarray) -> int:
    if len(swing_lows) < 2:
        return 0
    return int(np.sum(np.diff(swing_lows) < 0))


def _detect_bos(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    swing_high_mask: np.ndarray,
    swing_low_mask:  np.ndarray,
) -> Tuple[bool, bool]:
    """
    Break of Structure: price closes beyond the most recent swing high/low.
    """
    if closes.shape[0] < 3:
        return False, False

    last_close = closes[-1]

    # Get most recent swing high/low BEFORE last candle
    sh_indices = np.where(swing_high_mask[:-1])[0]
    sl_indices = np.where(swing_low_mask[:-1])[0]

    bos_bullish = False
    bos_bearish = False

    if len(sh_indices) >= 1:
        recent_sh = highs[sh_indices[-1]]
        if last_close > recent_sh:
            bos_bullish = True

    if len(sl_indices) >= 1:
        recent_sl = lows[sl_indices[-1]]
        if last_close < recent_sl:
            bos_bearish = True

    return bos_bullish, bos_bearish


def _detect_choch(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    swing_high_mask: np.ndarray,
    swing_low_mask:  np.ndarray,
) -> Tuple[bool, bool]:
    """
    Change of Character: first BOS in the opposite direction
    after a sequence of same-direction BOS (trend reversal signal).
    """
    if closes.shape[0] < 10:
        return False, False

    sh_indices = np.where(swing_high_mask)[0]
    sl_indices = np.where(swing_low_mask)[0]

    choch_bullish = False
    choch_bearish = False

    # CHoCH Bullish: sequence of LLs then price breaks above last swing high
    if len(sl_indices) >= 3 and len(sh_indices) >= 1:
        last_three_sl = lows[sl_indices[-3:]]
        if np.all(np.diff(last_three_sl) < 0):   # downtrend (LLs)
            last_sh = highs[sh_indices[-1]]
            if closes[-1] > last_sh:
                choch_bullish = True

    # CHoCH Bearish: sequence of HHs then price breaks below last swing low
    if len(sh_indices) >= 3 and len(sl_indices) >= 1:
        last_three_sh = highs[sh_indices[-3:]]
        if np.all(np.diff(last_three_sh) > 0):   # uptrend (HHs)
            last_sl = lows[sl_indices[-1]]
            if closes[-1] < last_sl:
                choch_bearish = True

    return choch_bullish, choch_bearish


def _compute_structure_score(
    hh: int, hl: int, lh: int, ll: int,
    bos_bull: bool, bos_bear: bool,
    choch_bull: bool, choch_bear: bool,
) -> float:
    """
    Scalar market structure score in [-1, 1].
    Positive = bullish structure. Negative = bearish structure.
    """
    total_events = hh + hl + lh + ll + 1e-9

    bullish_weight = (hh + hl) / total_events
    bearish_weight = (lh + ll) / total_events
    net = bullish_weight - bearish_weight

    # Amplify for confirmed BOS / CHoCH
    if bos_bull:
        net = min(1.0, net + 0.25)
    if bos_bear:
        net = max(-1.0, net - 0.25)
    if choch_bull:
        net = min(1.0, net + 0.35)
    if choch_bear:
        net = max(-1.0, net - 0.35)

    return float(np.clip(net, -1.0, 1.0))
