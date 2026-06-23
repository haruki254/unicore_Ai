"""
Master Feature Pipeline

Orchestrates all feature engineering modules and produces
a single normalised feature vector for the ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.features.market_structure import compute_market_structure
from core.features.price_action     import compute_price_action
from core.features.indicators       import compute_indicators
from core.features.volatility       import compute_volatility
from core.features.trend_analysis   import compute_trend_analysis
from core.features.liquidity        import compute_liquidity
from core.features.sessions         import compute_sessions
from config.settings                import settings, ALL_FEATURE_NAMES
from monitoring.logger              import feature_logger


class FeaturePipeline:
    """
    Converts raw candle data + market context into a feature vector.

    Usage
    -----
    pipeline = FeaturePipeline()
    features_dict, feature_vector = pipeline.compute(snapshot)
    """

    def __init__(self):
        self._scaler_params: Optional[Dict[str, Dict[str, float]]] = None

    # ── Public API ─────────────────────────────────────────────

    def compute(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all features from a raw market snapshot.

        Parameters
        ----------
        snapshot : dict
            {
              "symbol":     "EURUSD",
              "timestamp":  datetime,
              "price":      1.08550,
              "candles_m5":  [ {open, high, low, close, volume}, ... ],
              "candles_m15": [ ... ],
              "candles_h1":  [ ... ],
              "candles_h4":  [ ... ],
              "candles_d1":  [ ... ],
              "spread_pips": 1.2,
            }

        Returns
        -------
        dict of all feature names -> float values
        """
        try:
            # ── Parse candle data ──────────────────────────────
            dfs = self._parse_candles(snapshot)
            df_m5  = dfs.get("M5",  pd.DataFrame())
            df_m15 = dfs.get("M15", pd.DataFrame())
            df_h1  = dfs.get("H1",  pd.DataFrame())
            df_h4  = dfs.get("H4",  pd.DataFrame())
            df_d1  = dfs.get("D1",  pd.DataFrame())

            primary_df = df_m5 if len(df_m5) > 10 else df_m15

            features: Dict[str, float] = {}

            # ── Market Structure ──────────────────────────────
            if len(primary_df) >= 15:
                ms_feats = compute_market_structure(primary_df)
                features.update(ms_feats)
            else:
                features.update(self._zero_market_structure())

            # ── Price Action ───────────────────────────────────
            if len(primary_df) >= 5:
                pa_feats = compute_price_action(primary_df)
                features.update(pa_feats)
            else:
                features.update(self._zero_price_action())

            # ── Technical Indicators ──────────────────────────
            if len(primary_df) >= 30:
                ind_feats = compute_indicators(
                    primary_df,
                    rsi_period=settings.rsi_period,
                    adx_period=settings.adx_period,
                    macd_fast=settings.macd_fast,
                    macd_slow=settings.macd_slow,
                    macd_signal=settings.macd_signal,
                )
                features.update(ind_feats)
            else:
                features.update(self._zero_indicators())

            # ── Volatility ─────────────────────────────────────
            if len(primary_df) >= 15:
                vol_feats = compute_volatility(primary_df, settings.atr_period)
                features.update(vol_feats)
            else:
                features.update(self._zero_volatility())

            # ── Trend Analysis ─────────────────────────────────
            trend_feats = compute_trend_analysis(dfs)
            features.update(trend_feats)

            # ── Liquidity ──────────────────────────────────────
            price = float(snapshot.get("price", 0.0))
            liq_feats = compute_liquidity(df_m5, df_h4, df_d1, price)
            features.update(liq_feats)

            # ── Sessions ───────────────────────────────────────
            ts = snapshot.get("timestamp", datetime.utcnow())
            sess_feats, session_label = compute_sessions(ts)
            features.update(sess_feats)
            features["_session_label"] = session_label  # keep for regime engine

            # ── Meta features ──────────────────────────────────
            features["spread_pips"] = float(snapshot.get("spread_pips", 0.0))
            features["price"]       = price

            feature_logger.debug(
                "Features computed | {} features for {symbol}",
                len(features),
                symbol=snapshot.get("symbol", "?"),
            )
            return features

        except Exception as exc:
            feature_logger.error("Feature computation failed: {}", exc)
            raise

    def to_vector(
        self,
        features: Dict[str, float],
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """
        Convert features dict to a numpy array in consistent order.
        Missing features are filled with 0.
        """
        names = feature_names or ALL_FEATURE_NAMES
        vec   = np.array([features.get(n, 0.0) for n in names], dtype=np.float32)
        return vec

    def to_dataframe_row(
        self,
        features: Dict[str, float],
        feature_names: List[str] = None,
    ) -> pd.DataFrame:
        """Convert features to a single-row DataFrame for model inference."""
        names = feature_names or ALL_FEATURE_NAMES
        data  = {n: [features.get(n, 0.0)] for n in names}
        return pd.DataFrame(data)

    # ── Private helpers ────────────────────────────────────────

    def _parse_candles(self, snapshot: Dict) -> Dict[str, pd.DataFrame]:
        """Convert raw candle lists to DataFrames keyed by timeframe."""
        dfs: Dict[str, pd.DataFrame] = {}
        tf_map = {
            "candles_m5":  "M5",
            "candles_m15": "M15",
            "candles_h1":  "H1",
            "candles_h4":  "H4",
            "candles_d1":  "D1",
        }
        for key, tf in tf_map.items():
            raw = snapshot.get(key)
            if raw and len(raw) > 0:
                df = pd.DataFrame(raw)
                # Normalise column names
                df.columns = [c.lower() for c in df.columns]
                # Ensure required columns exist
                for col in ["open", "high", "low", "close"]:
                    if col not in df.columns:
                        df[col] = 0.0
                dfs[tf] = df.sort_index()   # oldest → newest
        return dfs

    # ── Zero-fill fallbacks ────────────────────────────────────

    def _zero_market_structure(self) -> Dict[str, float]:
        return {
            "hh_count": 0, "hl_count": 0, "lh_count": 0, "ll_count": 0,
            "bos_bullish": 0, "bos_bearish": 0,
            "choch_bullish": 0, "choch_bearish": 0,
            "structure_score": 0.0,
        }

    def _zero_price_action(self) -> Dict[str, float]:
        return {
            "body_size_avg_50": 0, "body_size_avg_100": 0,
            "wick_upper_avg": 0, "wick_lower_avg": 0,
            "momentum_5": 0, "momentum_10": 0, "momentum_20": 0,
            "close_vs_open": 0.5, "candle_direction_ratio": 0.5,
            "large_body_count": 0,
        }

    def _zero_indicators(self) -> Dict[str, float]:
        return {
            "rsi_14": 50, "rsi_overbought": 0, "rsi_oversold": 0,
            "adx_14": 20, "adx_trending": 0,
            "di_plus": 0, "di_minus": 0,
            "macd_line": 0, "macd_signal": 0, "macd_histogram": 0,
            "ma_20": 0, "ma_50": 0, "ma_200": 0,
            "price_vs_ma20": 0, "price_vs_ma50": 0,
            "ma_cross_bullish": 0, "ma_cross_bearish": 0,
            "ma20_above_ma50": 0, "price_above_ma20": 0, "price_above_ma50": 0,
        }

    def _zero_volatility(self) -> Dict[str, float]:
        return {
            "atr_14": 0, "atr_normalized": 0, "atr_rank": 0.5,
            "std_dev_20": 0, "std_dev_50": 0,
            "range_expansion": 0, "range_contraction": 0, "range_ratio": 1.0,
            "volatility_regime": 2,
        }
