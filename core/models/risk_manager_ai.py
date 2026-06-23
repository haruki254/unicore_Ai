"""
Risk Manager AI

Sole responsibility: determine whether a trade should be ALLOWED or BLOCKED.
Receives everything the Trader AI sees PLUS risk context.

Label: 1 = trade should have been allowed (resulted in WIN)
       0 = trade should have been blocked (resulted in LOSS)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.models.base_model import BaseMLModel
from config.settings        import settings, ALL_FEATURE_NAMES
from monitoring.logger      import model_logger


# ── Extra risk features not in the Trader AI ─────────────────
RISK_EXTRA_FEATURES = [
    "trader_buy_prob",
    "trader_sell_prob",
    "trader_confidence",
    "spread_pips",
    "atr_14",
    "volatility_regime",
    "session_quality",
    "account_drawdown_pct",
    "recent_loss_streak",
    "recent_win_streak",
    "trades_today",
    "current_risk_exposure",
    "regime_risk_mult",
    "similar_win_rate",
    "similar_avg_pnl",
    "similar_count",
    "is_news_period",
    "spread_vs_atr_ratio",
]

RISK_FEATURE_NAMES = ALL_FEATURE_NAMES + RISK_EXTRA_FEATURES


class RiskManagerAI(BaseMLModel):
    """
    Risk Manager AI — Trade quality classifier.

    predict() → (quality_score, decision, block_reasons)
    """

    model_type = "risk_manager"

    # ── Hard rule thresholds ──────────────────────────────────
    MAX_SPREAD_PIPS   = settings.max_spread_pips
    MAX_DRAWDOWN_PCT  = settings.max_drawdown_pct
    MAX_DAILY_TRADES  = settings.max_daily_trades
    MIN_QUALITY_SCORE = settings.min_risk_quality

    # ── Label preparation ─────────────────────────────────────

    def _prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label = 1 if the trade SHOULD have been allowed (it was a WIN).
        Label = 0 if the trade should have been blocked (it was a LOSS).
        """
        if "outcome" not in df.columns:
            raise ValueError("DataFrame must contain 'outcome' column")
        return (df["outcome"] == "WIN").astype(int).values

    # ── Predict ───────────────────────────────────────────────

    def predict(
        self,
        features:          Dict[str, float],
        trader_buy_prob:   float,
        trader_sell_prob:  float,
        trader_confidence: float,
        risk_context:      Dict[str, float] = None,
        similar_result=None,
    ) -> Tuple[float, str, List[str]]:
        """
        Evaluate trade quality.

        Parameters
        ----------
        features          : full feature dict from FeaturePipeline
        trader_buy_prob   : Trader AI BUY probability
        trader_sell_prob  : Trader AI SELL probability
        trader_confidence : Trader AI confidence
        risk_context      : dict with live account/risk state
        similar_result    : SimilarityResult from TradeMemoryEngine

        Returns
        -------
        quality_score : float [0, 1]
        decision      : 'ALLOW' or 'BLOCK'
        block_reasons : list of reason strings
        """
        ctx = risk_context or {}
        block_reasons: List[str] = []

        # ── Hard-coded rule gates (always applied) ─────────────
        spread = features.get("spread_pips", 0.0)
        if spread > self.MAX_SPREAD_PIPS:
            block_reasons.append(f"spread_too_wide:{spread:.1f}pips")

        drawdown_pct = ctx.get("account_drawdown_pct", 0.0)
        if drawdown_pct >= self.MAX_DRAWDOWN_PCT:
            block_reasons.append(f"max_drawdown_reached:{drawdown_pct:.1%}")

        trades_today = int(ctx.get("trades_today", 0))
        if trades_today >= self.MAX_DAILY_TRADES:
            block_reasons.append(f"daily_trade_limit:{trades_today}")

        loss_streak = int(ctx.get("recent_loss_streak", 0))
        if loss_streak >= 5:
            block_reasons.append(f"loss_streak:{loss_streak}")

        regime = features.get("_regime", "unknown")
        if regime in ("news_volatility", "high_volatility"):
            if ctx.get("is_news_period", 0):
                block_reasons.append("news_period_block")

        # Dead zone (low liquidity hours)
        if features.get("dead_zone", 0) == 1.0:
            block_reasons.append("dead_zone_hours")

        # ── ML quality score ──────────────────────────────────
        if self._is_trained:
            risk_features = self._build_risk_features(
                features, trader_buy_prob, trader_sell_prob,
                trader_confidence, ctx, similar_result,
            )
            X = self._build_feature_df(risk_features)
            _, quality_score = self.predict_proba(X)
        else:
            # Heuristic quality score when model not yet trained
            quality_score = self._heuristic_quality(
                features, trader_confidence, ctx, similar_result
            )

        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        # ── ML-based soft block ───────────────────────────────
        if quality_score < self.MIN_QUALITY_SCORE:
            block_reasons.append(f"low_quality_score:{quality_score:.2f}")

        # ── Trader confidence too low ─────────────────────────
        if trader_confidence < (settings.min_trader_confidence - 0.5):
            block_reasons.append(f"low_trader_confidence:{trader_confidence:.2f}")

        # ── Memory evidence against trade ─────────────────────
        if similar_result and similar_result.count >= 10:
            if similar_result.win_rate < 0.35:
                block_reasons.append(
                    f"memory_low_winrate:{similar_result.win_rate:.0%}"
                )
            if similar_result.avg_pnl < -5.0:
                block_reasons.append(
                    f"memory_negative_pnl:{similar_result.avg_pnl:.1f}pips"
                )

        # ── Final decision ────────────────────────────────────
        decision = "BLOCK" if block_reasons else "ALLOW"

        if decision == "BLOCK":
            model_logger.log_block(
                symbol=str(features.get("symbol", "?")),
                reasons=block_reasons,
            )
        else:
            model_logger.info(
                "RiskManager | ALLOW | quality={:.0%}",
                quality_score,
            )

        return quality_score, decision, block_reasons

    # ── Training data builder ─────────────────────────────────

    def build_training_data(
        self,
        trades: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build training set from completed trade history."""
        rows     = []
        outcomes = []

        for trade in trades:
            conditions = trade.get("conditions", {})
            risk_ctx   = trade.get("risk_context", {})
            prediction = trade.get("prediction", {})

            if not conditions:
                continue

            outcome = trade.get("outcome", "LOSS")
            if outcome not in ("WIN", "LOSS"):
                continue

            row = dict(conditions)

            # Append trader AI outputs recorded at trade time
            row["trader_buy_prob"]   = prediction.get("trader_buy_prob",  0.5)
            row["trader_sell_prob"]  = prediction.get("trader_sell_prob", 0.5)
            row["trader_confidence"] = prediction.get("trader_confidence", 0.0)

            # Risk context at trade time
            row["account_drawdown_pct"]  = risk_ctx.get("account_drawdown_pct", 0.0)
            row["recent_loss_streak"]    = risk_ctx.get("recent_loss_streak",   0)
            row["recent_win_streak"]     = risk_ctx.get("recent_win_streak",    0)
            row["trades_today"]          = risk_ctx.get("trades_today",         0)
            row["current_risk_exposure"] = risk_ctx.get("current_risk_exposure",0.0)
            row["session_quality"]       = risk_ctx.get("session_quality",      0.5)
            row["spread_pips"]           = risk_ctx.get("spread_pips",          1.0)
            row["similar_win_rate"]      = risk_ctx.get("similar_win_rate",     0.5)
            row["similar_avg_pnl"]       = risk_ctx.get("similar_avg_pnl",      0.0)
            row["similar_count"]         = risk_ctx.get("similar_count",         0)

            rows.append(row)
            outcomes.append(outcome)

        if not rows:
            return pd.DataFrame(), pd.DataFrame()

        X      = pd.DataFrame(rows)
        df_raw = pd.DataFrame({"outcome": outcomes})

        for feat in RISK_FEATURE_NAMES:
            if feat not in X.columns:
                X[feat] = 0.0

        return X[RISK_FEATURE_NAMES], df_raw

    # ── Helpers ───────────────────────────────────────────────

    def _build_risk_features(
        self,
        features:          Dict[str, float],
        trader_buy_prob:   float,
        trader_sell_prob:  float,
        trader_confidence: float,
        ctx:               Dict[str, float],
        similar_result,
    ) -> Dict[str, float]:
        """Merge all feature sources into one dict for Risk Manager."""
        row = dict(features)

        row["trader_buy_prob"]      = trader_buy_prob
        row["trader_sell_prob"]     = trader_sell_prob
        row["trader_confidence"]    = trader_confidence
        row["account_drawdown_pct"] = ctx.get("account_drawdown_pct", 0.0)
        row["recent_loss_streak"]   = ctx.get("recent_loss_streak",   0)
        row["recent_win_streak"]    = ctx.get("recent_win_streak",    0)
        row["trades_today"]         = ctx.get("trades_today",         0)
        row["current_risk_exposure"]= ctx.get("current_risk_exposure",0.0)
        row["is_news_period"]       = ctx.get("is_news_period",       0)
        row["regime_risk_mult"]     = ctx.get("regime_risk_mult",     1.0)

        # Memory context
        if similar_result:
            row["similar_win_rate"] = similar_result.win_rate
            row["similar_avg_pnl"]  = similar_result.avg_pnl
            row["similar_count"]    = float(similar_result.count)
        else:
            row["similar_win_rate"] = 0.5
            row["similar_avg_pnl"]  = 0.0
            row["similar_count"]    = 0.0

        # Spread vs ATR ratio
        atr = features.get("atr_14", 0.001) + 1e-10
        spread = features.get("spread_pips", 0.0)
        row["spread_vs_atr_ratio"] = spread / (atr * 10000)

        return row

    def _heuristic_quality(
        self,
        features:          Dict[str, float],
        trader_confidence: float,
        ctx:               Dict[str, float],
        similar_result,
    ) -> float:
        """Heuristic quality score when ML model not yet trained."""
        score = 0.5

        # Trader confidence
        score += (trader_confidence - 0.0) * 0.3

        # Trend alignment
        alignment = features.get("trend_alignment_score", 0.0)
        score += alignment * 0.1

        # Session quality
        sess_q = features.get("session_quality", 0.5)
        score += (sess_q - 0.5) * 0.1

        # ADX trending
        if features.get("adx_trending", 0) == 1.0:
            score += 0.1

        # Memory evidence
        if similar_result and similar_result.count >= 5:
            score += (similar_result.win_rate - 0.5) * 0.3

        # Penalise loss streak
        loss_streak = ctx.get("recent_loss_streak", 0)
        score -= loss_streak * 0.02

        return float(np.clip(score, 0.0, 1.0))

    def _build_feature_df(self, features: Dict[str, float]) -> pd.DataFrame:
        names = self._feature_names or RISK_FEATURE_NAMES
        data  = {n: [features.get(n, 0.0)] for n in names}
        return pd.DataFrame(data)
