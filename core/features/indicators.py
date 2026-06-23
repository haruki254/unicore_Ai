"""
Technical Indicators Feature Engineering

Computes:
  - RSI (Relative Strength Index)
  - ADX (Average Directional Index)
  - MACD (Moving Average Convergence Divergence)
  - Moving Averages: 20, 50, 200 SMA/EMA
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


# ── RSI ───────────────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# ── EMA ───────────────────────────────────────────────────────

def _ema(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period:
        return np.full(len(closes), closes.mean())

    ema    = np.zeros(len(closes))
    k      = 2.0 / (period + 1)
    ema[0] = closes[0]

    for i in range(1, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1.0 - k)

    return ema


# ── SMA ───────────────────────────────────────────────────────

def _sma(closes: np.ndarray, period: int) -> float:
    if len(closes) < period:
        return float(np.mean(closes))
    return float(np.mean(closes[-period:]))


# ── MACD ─────────────────────────────────────────────────────

def _macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Dict[str, float]:
    if len(closes) < slow + signal_period:
        return {"macd_line": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0}

    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal     = _ema(macd_line, signal_period)
    histogram  = macd_line - signal

    return {
        "macd_line":      float(macd_line[-1]),
        "macd_signal_line":    float(signal[-1]),
        "macd_histogram": float(histogram[-1]),
    }


# ── ADX ───────────────────────────────────────────────────────

def _adx(
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> Dict[str, float]:
    if len(closes) < period + 1:
        return {"adx_14": 25.0, "adx_trending": 0.0, "di_plus": 0.0, "di_minus": 0.0}

    # True Range
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_closes),
            np.abs(lows  - prev_closes),
        ),
    )

    # Directional movements
    up_move   = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    dm_plus  = np.where((up_move > down_move) & (up_move > 0), up_move,  0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = tr[1:]

    # Smooth with Wilder's method
    def wilder_smooth(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(len(arr))
        out[n - 1] = np.sum(arr[:n])
        for i in range(n, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / n + arr[i]
        return out

    tr_smooth  = wilder_smooth(tr,        period)
    dmp_smooth = wilder_smooth(dm_plus,   period)
    dmm_smooth = wilder_smooth(dm_minus,  period)

    di_plus  = 100.0 * dmp_smooth / (tr_smooth + 1e-10)
    di_minus = 100.0 * dmm_smooth / (tr_smooth + 1e-10)

    dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

    # ADX = smoothed DX
    adx_arr = wilder_smooth(dx, period)
    adx_val = float(adx_arr[-1])

    return {
        "adx_14":      adx_val,
        "adx_trending": float(adx_val > 25),
        "di_plus":     float(di_plus[-1]),
        "di_minus":    float(di_minus[-1]),
    }


# ── Main feature function ─────────────────────────────────────

def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    adx_period: int = 14,
    macd_fast:  int = 12,
    macd_slow:  int = 26,
    macd_signal: int = 9,
) -> Dict[str, float]:
    """
    Compute all technical indicators from OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: open, high, low, close

    Returns
    -------
    dict of feature_name -> value
    """
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    features: Dict[str, float] = {}

    # ── RSI ───────────────────────────────────────────────────
    rsi_val = _rsi(closes, rsi_period)
    features["rsi_14"]      = rsi_val
    features["rsi_overbought"] = float(rsi_val >= 70)
    features["rsi_oversold"]   = float(rsi_val <= 30)
    features["rsi_neutral"]    = float(40 <= rsi_val <= 60)

    # ── ADX ───────────────────────────────────────────────────
    adx_data = _adx(highs, lows, closes, adx_period)
    features.update(adx_data)

    # ── MACD ──────────────────────────────────────────────────
    macd_data = _macd(closes, macd_fast, macd_slow, macd_signal)
    features.update(macd_data)

    # Signal cross (histogram sign change)
    if len(closes) > macd_slow + macd_signal + 1:
        ema_fast_arr = _ema(closes, macd_fast)
        ema_slow_arr = _ema(closes, macd_slow)
        macd_arr     = ema_fast_arr - ema_slow_arr
        sig_arr      = _ema(macd_arr, macd_signal)
        hist_arr     = macd_arr - sig_arr

        features["macd_cross_bull"] = float(
            len(hist_arr) >= 2 and hist_arr[-2] < 0 and hist_arr[-1] > 0
        )
        features["macd_cross_bear"] = float(
            len(hist_arr) >= 2 and hist_arr[-2] > 0 and hist_arr[-1] < 0
        )
    else:
        features["macd_cross_bull"] = 0.0
        features["macd_cross_bear"] = 0.0

    # ── Moving Averages ───────────────────────────────────────
    price = closes[-1]

    for period in [20, 50, 200]:
        ma = _sma(closes, period)
        features[f"ma_{period}"] = ma
        # Price relative to MA (normalised by ATR equivalent)
        spread = np.std(closes[-50:]) + 1e-10
        features[f"price_vs_ma{period}"] = float((price - ma) / spread)

    # MA crossovers (20 vs 50)
    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)

    # Previous values
    if len(closes) > 50:
        ma20_prev = float(np.mean(closes[-21:-1]))
        ma50_prev = float(np.mean(closes[-51:-1]))
        features["ma_cross_bullish"] = float(ma20_prev <= ma50_prev and ma20 > ma50)
        features["ma_cross_bearish"] = float(ma20_prev >= ma50_prev and ma20 < ma50)
    else:
        features["ma_cross_bullish"] = 0.0
        features["ma_cross_bearish"] = 0.0

    features["ma20_above_ma50"]   = float(ma20 > ma50)
    features["price_above_ma20"]  = float(price > ma20)
    features["price_above_ma50"]  = float(price > ma50)

    return features
