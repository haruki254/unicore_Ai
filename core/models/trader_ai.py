"""
Trader AI

Sole responsibility: identify the most profitable trade direction.
Outputs BUY probability and SELL probability.
Never considers risk — that's the Risk Manager's job.

Label: 1 = trade direction was profitable (WIN), 0 = not profitable (LOSS)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from core.models.base_model import BaseMLModel
from config.settings import settings, ALL_FEATURE_NAMES
from monitoring.logger import model_logger


class TraderAI(BaseMLModel):
    """
    Trader AI — Direction classifier.

    predict() → (buy_probability, sell_probability, direction, confidence)
    """

    model_type = "trader_ai"

    # ── Label preparation ─────────────────────────────────────

    def _prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Label = 1 if the trade direction was aligned with the profitable outcome.

        For BUY trades: label 1 if outcome == WIN
        For SELL trades: label 1 if outcome == WIN

        Combined: we train on "was the direction correct and profitable?"
        """
        if "outcome" not in df.columns:
            raise ValueError("DataFrame must contain 'outcome' column (WIN/LOSS/BREAKEVEN)")

        labels = (df["outcome"] == "WIN").astype(int).values
        return labels

    # ── Predict ───────────────────────────────────────────────

    def predict(
        self,
        features: Dict[str, float],
        ea_signal: str = "BUY",
    ) -> Tuple[float, float, str, float]:
        """
        Predict trade direction probabilities.

        Parameters
        ----------
        features  : feature dict from FeaturePipeline
        ea_signal : original EA signal ('BUY' or 'SELL')

        Returns
        -------
        buy_prob    : float [0, 1]
        sell_prob   : float [0, 1]
        direction   : 'BUY' or 'SELL'
        confidence  : float [0, 1] — abs difference from 0.5
        """
        if not self._is_trained:
            model_logger.warning("TraderAI not trained yet — returning 50/50")
            return 0.5, 0.5, ea_signal, 0.0

        # Build feature vector
        X = self._build_feature_df(features)
        prob_loss, prob_win = self.predict_proba(X)

        # ── Interpret probability ─────────────────────────────
        # prob_win = probability of a WIN in the predicted direction
        # We interpret this as directional confidence

        # Factor in structure_score and trend_alignment to bias direction
        structure = features.get("structure_score", 0.0)
        alignment = features.get("trend_alignment_score", 0.0)

        # Direction signal from features (-1 to +1)
        signal = 0.5 * structure + 0.5 * alignment

        if signal > 0:
            # Bullish bias → amplify buy probability
            buy_prob  = float(np.clip(0.5 + (prob_win - 0.5) + 0.1 * signal, 0.0, 1.0))
        elif signal < 0:
            # Bearish bias → amplify sell probability
            buy_prob  = float(np.clip(0.5 - (prob_win - 0.5) + 0.1 * signal, 0.0, 1.0))
        else:
            # Neutral — model drives direction
            buy_prob  = float(prob_win)

        # Ensure probabilities sum to ~1
        sell_prob = float(1.0 - buy_prob)
        buy_prob  = float(np.clip(buy_prob,  0.01, 0.99))
        sell_prob = float(np.clip(sell_prob, 0.01, 0.99))

        direction  = "BUY" if buy_prob >= sell_prob else "SELL"
        confidence = float(abs(max(buy_prob, sell_prob) - 0.5))

        model_logger.info(
            "TraderAI | BUY={:.0%} SELL={:.0%} → {} (conf={:.0%})",
            buy_prob, sell_prob, direction, confidence,
        )

        return buy_prob, sell_prob, direction, confidence

    # ── Training data builder ─────────────────────────────────

    def build_training_data(
        self,
        trades: list,
        feature_pipeline,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build training dataset from completed trade records.

        Parameters
        ----------
        trades           : list of trade dicts from database
        feature_pipeline : FeaturePipeline instance

        Returns
        -------
        X : feature DataFrame
        df_raw : DataFrame with outcome column for label generation
        """
        rows     = []
        outcomes = []

        for trade in trades:
            conditions = trade.get("conditions", {})
            if not conditions:
                continue

            outcome = trade.get("outcome", "LOSS")
            if outcome not in ("WIN", "LOSS"):
                continue

            rows.append(conditions)
            outcomes.append(outcome)

        if not rows:
            return pd.DataFrame(), pd.DataFrame()

        X      = pd.DataFrame(rows)
        df_raw = pd.DataFrame({"outcome": outcomes})

        # Fill missing features with 0
        for feat in ALL_FEATURE_NAMES:
            if feat not in X.columns:
                X[feat] = 0.0

        return X[ALL_FEATURE_NAMES], df_raw

    # ── Helpers ───────────────────────────────────────────────

    def _build_feature_df(self, features: Dict[str, float]) -> pd.DataFrame:
        names = self._feature_names or ALL_FEATURE_NAMES
        data  = {n: [features.get(n, 0.0)] for n in names}
        return pd.DataFrame(data)
