"""
Learning Pipeline

Orchestrates the full ML training cycle:
  1. Fetch completed trades from database
  2. Build feature matrices
  3. Walk-forward train all candidate models
  4. Compare and select best model per AI
  5. Persist to disk + log to database
  6. Schedule periodic retraining
"""

from __future__ import annotations

import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from core.models.trader_ai      import TraderAI
from core.models.risk_manager_ai import RiskManagerAI
from core.features.feature_pipeline import FeaturePipeline
from config.settings             import settings
from monitoring.logger           import model_logger


class LearningPipeline:
    """
    Full retraining pipeline for both AI models.

    Usage
    -----
    pipeline = LearningPipeline(trader_ai, risk_manager, db_client)
    summary  = pipeline.run()
    """

    MIN_SAMPLES   = settings.model_min_training_samples
    WF_SPLITS     = settings.walk_forward_splits

    def __init__(
        self,
        trader_ai:     TraderAI,
        risk_manager:  RiskManagerAI,
        db_client      = None,       # DatabaseClient or None for local-only
        feature_pipeline: FeaturePipeline = None,
    ):
        self.trader_ai       = trader_ai
        self.risk_manager    = risk_manager
        self.db              = db_client
        self.pipeline        = feature_pipeline or FeaturePipeline()
        self._last_train_ts: Optional[datetime] = None

    # ── Public API ─────────────────────────────────────────────

    def run(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute a full training cycle.

        Parameters
        ----------
        force : if True, train even if recently trained

        Returns
        -------
        summary dict with training results
        """
        if not force and not self._should_retrain():
            model_logger.info("Skipping retrain — trained recently")
            return {"status": "skipped", "reason": "recently_trained"}

        t0 = time.perf_counter()
        model_logger.info("=" * 50)
        model_logger.info("LEARNING PIPELINE STARTING")
        model_logger.info("=" * 50)

        # ── 1. Fetch training data ────────────────────────────
        trades = self._fetch_trades()
        if not trades:
            return {"status": "skipped", "reason": "no_trades_available"}

        model_logger.info("Fetched {} completed trades", len(trades))

        # ── 2. Train Trader AI ────────────────────────────────
        trader_summary = self._train_trader_ai(trades)

        # ── 3. Train Risk Manager AI ──────────────────────────
        risk_summary = self._train_risk_manager(trades)

        # ── 4. Save results to DB ────────────────────────────
        if self.db:
            self._save_model_results(trader_summary, "trader_ai")
            self._save_model_results(risk_summary, "risk_manager")

        self._last_train_ts = datetime.utcnow()
        elapsed = time.perf_counter() - t0

        # ── Build EA profiles from completed trades ───────────
        if self.db:
            try:
                from core.profiles import EAProfileBuilder
                model_logger.info("Building EA profiles from completed trades...")
                builder  = EAProfileBuilder(min_samples=5)
                profiles = builder.build_from_trades(trades)
                for ea_id, profile in profiles.items():
                    self.db.save_ea_profile(ea_id, profile.to_dict())
                    model_logger.info("Profile saved: {} ({} trades)", ea_id, profile.total_trades)
                model_logger.info("EA profiles updated: {} EAs", len(profiles))
            except Exception as e:
                model_logger.error("EA profile build failed (non-fatal): {}", e)

        summary = {
            "status":         "completed",
            "timestamp":      datetime.utcnow().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "trade_count":    len(trades),
            "trader_ai":      trader_summary,
            "risk_manager":   risk_summary,
        }

        model_logger.info("Training complete in {:.1f}s", elapsed)
        return summary

    def should_retrain(self) -> bool:
        return self._should_retrain()

    # ── Private ────────────────────────────────────────────────

    def _should_retrain(self) -> bool:
        if self._last_train_ts is None:
            return True
        interval = timedelta(hours=settings.model_retrain_interval_hours)
        return (datetime.utcnow() - self._last_train_ts) >= interval

    def _fetch_trades(self) -> List[Dict]:
        """Fetch completed trades from DB or local cache."""
        if self.db:
            try:
                return self.db.fetch_completed_trades()
            except Exception as e:
                model_logger.error("DB fetch failed: {} — using empty set", e)
                return []
        return []

    def _train_trader_ai(self, trades: List[Dict]) -> Dict[str, Any]:
        """Build dataset and train Trader AI."""
        model_logger.info("Training Trader AI...")
        try:
            X, df_raw = self.trader_ai.build_training_data(
                trades, self.pipeline
            )
            if X.empty or len(X) < self.MIN_SAMPLES:
                return {
                    "status":  "skipped",
                    "reason":  f"insufficient_samples:{len(X)}",
                    "samples": len(X),
                }

            metrics = self.trader_ai.train(X, df_raw, self.WF_SPLITS)
            best    = self.trader_ai.get_best_metrics()

            return {
                "status":      "trained",
                "algorithm":   self.trader_ai.best_algorithm,
                "samples":     len(X),
                "roc_auc":     round(best.roc_auc,   4) if best else 0,
                "wf_mean":     round(best.wf_mean,   4) if best else 0,
                "wf_std":      round(best.wf_std,    4) if best else 0,
                "feature_importance": self.trader_ai.get_feature_importance(),
            }
        except Exception as e:
            model_logger.error("Trader AI training failed: {}", e)
            return {"status": "error", "error": str(e)}

    def _train_risk_manager(self, trades: List[Dict]) -> Dict[str, Any]:
        """Build dataset and train Risk Manager AI."""
        model_logger.info("Training Risk Manager AI...")
        try:
            X, df_raw = self.risk_manager.build_training_data(trades)
            if X.empty or len(X) < self.MIN_SAMPLES:
                return {
                    "status":  "skipped",
                    "reason":  f"insufficient_samples:{len(X)}",
                    "samples": len(X),
                }

            metrics = self.risk_manager.train(X, df_raw, self.WF_SPLITS)
            best    = self.risk_manager.get_best_metrics()

            return {
                "status":    "trained",
                "algorithm": self.risk_manager.best_algorithm,
                "samples":   len(X),
                "roc_auc":   round(best.roc_auc,  4) if best else 0,
                "wf_mean":   round(best.wf_mean,  4) if best else 0,
                "wf_std":    round(best.wf_std,   4) if best else 0,
                "feature_importance": self.risk_manager.get_feature_importance(),
            }
        except Exception as e:
            model_logger.error("Risk Manager training failed: {}", e)
            return {"status": "error", "error": str(e)}

    def _save_model_results(
        self,
        summary: Dict[str, Any],
        model_type: str,
    ) -> None:
        """Persist model result metadata to Supabase."""
        if not self.db or summary.get("status") != "trained":
            return
        try:
            self.db.save_model_result({
                "id":            str(uuid.uuid4()),
                "model_type":    model_type,
                "algorithm":     summary.get("algorithm", "unknown"),
                "roc_auc":       summary.get("roc_auc",  0),
                "wf_mean_accuracy": summary.get("wf_mean", 0),
                "wf_std_accuracy":  summary.get("wf_std",  0),
                "is_active":     True,
                "feature_importance": json.dumps(
                    summary.get("feature_importance", {})
                ),
            })
        except Exception as e:
            model_logger.error("Failed saving model result to DB: {}", e)


# ── Walk-Forward Validator (standalone utility) ───────────────

class WalkForwardValidator:
    """
    Standalone walk-forward validation utility.

    Simulates realistic out-of-sample testing by training
    on a growing window and validating on the next period.
    Prevents any look-ahead bias.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap      = gap   # gap candles between train and val to prevent leakage

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/val index pairs (time-ordered, no shuffling).

        Returns list of (train_indices, val_indices).
        """
        n         = len(X)
        fold_size = n // (self.n_splits + 1)
        splits    = []

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.gap
            val_end   = val_start + fold_size

            if val_end > n:
                break

            train_idx = np.arange(0, train_end)
            val_idx   = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        return splits

    def validate(
        self,
        model_cls,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run walk-forward validation on any sklearn-compatible model.

        Returns averaged metrics across folds.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score
        import copy

        splits   = self.split(X, y)
        aucs     = []
        accs     = []

        for train_idx, val_idx in splits:
            X_tr  = X.iloc[train_idx].values
            X_val = X.iloc[val_idx].values
            y_tr  = y[train_idx]
            y_val = y[val_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue

            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            m = copy.deepcopy(model_cls)
            m.fit(X_tr_s, y_tr)

            proba  = m.predict_proba(X_val_s)[:, 1]
            preds  = (proba >= 0.5).astype(int)

            aucs.append(roc_auc_score(y_val, proba))
            accs.append(accuracy_score(y_val, preds))

        return {
            "wf_mean_auc": float(np.mean(aucs))   if aucs else 0.0,
            "wf_std_auc":  float(np.std(aucs))    if aucs else 0.0,
            "wf_mean_acc": float(np.mean(accs))   if accs else 0.0,
            "wf_folds":    len(aucs),
        }
