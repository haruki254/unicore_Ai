"""
Market Regime Engine

Classifies every market state into one of 9 regimes:
  1. strong_bull_trend
  2. weak_bull_trend
  3. strong_bear_trend
  4. weak_bear_trend
  5. sideways_range
  6. high_volatility
  7. low_volatility
  8. news_volatility
  9. liquidity_grab

Uses a rule-based scoring system layered over the feature vector.
Can also be trained as an ML classifier on labelled data.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from config.settings import MARKET_REGIMES
from monitoring.logger import regime_logger


class MarketRegimeEngine:
    """
    Classifies market regime from a feature dictionary.

    Usage
    -----
    engine = MarketRegimeEngine()
    regime, confidence, scores = engine.classify(features)
    """

    def classify(
        self,
        features: Dict[str, float],
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Determine the current market regime.

        Parameters
        ----------
        features : dict
            Full feature dictionary from FeaturePipeline.compute()

        Returns
        -------
        regime     : str   — regime name
        confidence : float — [0, 1]
        scores     : dict  — score for each regime
        """
        scores = self._score_all_regimes(features)

        # Softmax normalisation for clean probabilities
        raw_scores = np.array([scores[r] for r in MARKET_REGIMES], dtype=float)
        exp_scores = np.exp(raw_scores - raw_scores.max())
        probs      = exp_scores / exp_scores.sum()

        scored = {r: float(probs[i]) for i, r in enumerate(MARKET_REGIMES)}

        # Top regime
        best_regime = max(scored, key=scored.get)
        confidence  = scored[best_regime]

        regime_logger.log_regime(
            symbol=str(features.get("symbol", "?")),
            regime=best_regime,
            confidence=confidence,
        )

        return best_regime, confidence, scored

    # ── Regime Scoring ────────────────────────────────────────

    def _score_all_regimes(self, f: Dict[str, float]) -> Dict[str, float]:
        return {
            "strong_bull_trend":  self._score_strong_bull(f),
            "weak_bull_trend":    self._score_weak_bull(f),
            "strong_bear_trend":  self._score_strong_bear(f),
            "weak_bear_trend":    self._score_weak_bear(f),
            "sideways_range":     self._score_sideways(f),
            "high_volatility":    self._score_high_vol(f),
            "low_volatility":     self._score_low_vol(f),
            "news_volatility":    self._score_news_vol(f),
            "liquidity_grab":     self._score_liquidity_grab(f),
        }

    def _score_strong_bull(self, f: Dict[str, float]) -> float:
        score = 0.0
        # Structure
        score += f.get("hh_count", 0) * 0.15
        score += f.get("hl_count", 0) * 0.15
        score += f.get("bos_bullish", 0) * 0.25
        score += f.get("choch_bullish", 0) * 0.20
        score += max(0, f.get("structure_score", 0)) * 0.30

        # Trend alignment
        alignment = f.get("trend_alignment_score", 0)
        score += max(0, alignment) * 0.40

        # All HTFs bullish
        htf_bull = f.get("htf_bullish", 0)
        score += htf_bull * 0.30

        # ADX strong trend
        adx = f.get("adx_14", 20)
        if adx > 30:
            score += 0.25
        elif adx > 25:
            score += 0.10

        # Price above MAs
        score += f.get("price_above_ma20", 0) * 0.10
        score += f.get("price_above_ma50", 0) * 0.10

        # Momentum
        score += max(0, f.get("momentum_10", 0)) * 5.0

        return float(max(0.0, score))

    def _score_weak_bull(self, f: Dict[str, float]) -> float:
        score = 0.0
        alignment = f.get("trend_alignment_score", 0)
        score += max(0, alignment) * 0.25
        score += max(0, f.get("structure_score", 0)) * 0.15

        # Mixed signals
        adx = f.get("adx_14", 20)
        if 20 <= adx <= 30:
            score += 0.20

        score += f.get("price_above_ma20", 0) * 0.15
        score += f.get("trend_conflict", 0) * -0.10
        score += max(0, f.get("momentum_10", 0)) * 2.0

        return float(max(0.0, score))

    def _score_strong_bear(self, f: Dict[str, float]) -> float:
        score = 0.0
        score += f.get("lh_count", 0) * 0.15
        score += f.get("ll_count", 0) * 0.15
        score += f.get("bos_bearish", 0) * 0.25
        score += f.get("choch_bearish", 0) * 0.20
        score += max(0, -f.get("structure_score", 0)) * 0.30

        alignment = f.get("trend_alignment_score", 0)
        score += max(0, -alignment) * 0.40
        score += f.get("htf_bearish", 0) * 0.30

        adx = f.get("adx_14", 20)
        if adx > 30:
            score += 0.25

        score += (1 - f.get("price_above_ma20", 1)) * 0.10
        score += (1 - f.get("price_above_ma50", 1)) * 0.10
        score += max(0, -f.get("momentum_10", 0)) * 5.0

        return float(max(0.0, score))

    def _score_weak_bear(self, f: Dict[str, float]) -> float:
        score = 0.0
        alignment = f.get("trend_alignment_score", 0)
        score += max(0, -alignment) * 0.25
        score += max(0, -f.get("structure_score", 0)) * 0.15

        adx = f.get("adx_14", 20)
        if 20 <= adx <= 30:
            score += 0.20

        score += (1 - f.get("price_above_ma20", 1)) * 0.15
        score += f.get("trend_conflict", 0) * -0.10
        score += max(0, -f.get("momentum_10", 0)) * 2.0

        return float(max(0.0, score))

    def _score_sideways(self, f: Dict[str, float]) -> float:
        score = 0.0

        # Low ADX = no trend
        adx = f.get("adx_14", 20)
        if adx < 20:
            score += 0.50
        elif adx < 25:
            score += 0.25

        # Mixed structure
        alignment = abs(f.get("trend_alignment_score", 0))
        if alignment < 0.2:
            score += 0.40

        # Low momentum
        mom = abs(f.get("momentum_10", 0))
        if mom < 0.001:
            score += 0.20

        # Range contraction
        score += f.get("range_contraction", 0) * 0.20
        score += f.get("bb_squeeze", 0) * 0.20

        return float(max(0.0, score))

    def _score_high_vol(self, f: Dict[str, float]) -> float:
        score = 0.0
        vol_regime = f.get("volatility_regime", 2)
        if vol_regime >= 4:
            score += 0.70
        elif vol_regime >= 3:
            score += 0.40

        score += f.get("range_expansion", 0) * 0.30
        atr_rank = f.get("atr_rank", 0.5)
        if atr_rank > 0.80:
            score += 0.30

        return float(max(0.0, score))

    def _score_low_vol(self, f: Dict[str, float]) -> float:
        score = 0.0
        vol_regime = f.get("volatility_regime", 2)
        if vol_regime <= 0:
            score += 0.70
        elif vol_regime <= 1:
            score += 0.40

        score += f.get("range_contraction", 0) * 0.30
        score += f.get("bb_squeeze", 0) * 0.30

        return float(max(0.0, score))

    def _score_news_vol(self, f: Dict[str, float]) -> float:
        """
        News volatility: large range expansion + spiky wicks + dead session
        followed by extreme ATR spike.
        """
        score = 0.0
        # Large sudden wicks
        wick_ratio = f.get("wick_upper_ratio", 0) + f.get("wick_lower_ratio", 0)
        if wick_ratio > 0.6:
            score += 0.40

        # ATR spike
        atr_rank = f.get("atr_rank", 0.5)
        if atr_rank > 0.95:
            score += 0.40

        # Volume surge
        score += f.get("volume_surge", 0) * 0.20

        return float(max(0.0, score))

    def _score_liquidity_grab(self, f: Dict[str, float]) -> float:
        """
        Liquidity grab / stop hunt: price briefly spikes through a key level
        then reverses (pin bar near a key level).
        """
        score = 0.0

        # Pin bars near key levels
        near_pdh = f.get("near_pdh", 0)
        near_pdl = f.get("near_pdl", 0)
        near_wh  = f.get("near_weekly_high", 0)
        near_wl  = f.get("near_weekly_low", 0)

        score += (near_pdh + near_pdl + near_wh + near_wl) * 0.20

        # Pin bar pattern
        score += f.get("bullish_pin_bar", 0) * 0.30
        score += f.get("bearish_pin_bar", 0) * 0.30

        # Large wick
        if f.get("wick_upper_ratio", 0) > 0.5 or f.get("wick_lower_ratio", 0) > 0.5:
            score += 0.20

        # Near buy/sell side liquidity
        score += f.get("near_buy_side_liquidity", 0) * 0.20
        score += f.get("near_sell_side_liquidity", 0) * 0.20

        return float(max(0.0, score))

    # ── Helpers for external use ───────────────────────────────

    def regime_to_risk_multiplier(self, regime: str) -> float:
        """
        Return a risk multiplier for the regime.
        High-risk regimes get a lower multiplier (tighter position sizing).
        """
        multipliers = {
            "strong_bull_trend":  1.0,
            "weak_bull_trend":    0.8,
            "strong_bear_trend":  1.0,
            "weak_bear_trend":    0.8,
            "sideways_range":     0.7,
            "high_volatility":    0.5,
            "low_volatility":     0.9,
            "news_volatility":    0.2,
            "liquidity_grab":     0.6,
        }
        return multipliers.get(regime, 0.7)

    def is_tradeable_regime(self, regime: str) -> bool:
        """
        Returns False for regimes where trading is discouraged.
        """
        avoid = {"news_volatility", "high_volatility"}
        return regime not in avoid
