"""
Base ML Model — shared scaffold for Trader AI and Risk Manager AI.

Implements:
  - Multi-algorithm ensemble (RF, XGBoost, LightGBM, CatBoost, LR)
  - Walk-forward cross-validation
  - Automatic best-model selection by ROC-AUC
  - Feature importance extraction
  - Model persistence (joblib)
  - Incremental retraining
"""

from __future__ import annotations

import os
import time
import json
import joblib
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
)
from sklearn.calibration import CalibratedClassifierCV

# Optional heavy models
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

from config.settings import settings, ML_MODELS
from monitoring.logger import model_logger

warnings.filterwarnings("ignore", category=UserWarning)


class ModelMetrics:
    """Container for validation metrics."""
    def __init__(self):
        self.accuracy:  float = 0.0
        self.precision: float = 0.0
        self.recall:    float = 0.0
        self.f1:        float = 0.0
        self.roc_auc:   float = 0.0
        self.log_loss_val: float = 1.0
        self.wf_scores: List[float] = []
        self.wf_mean:   float = 0.0
        self.wf_std:    float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy":   self.accuracy,
            "precision":  self.precision,
            "recall":     self.recall,
            "f1":         self.f1,
            "roc_auc":    self.roc_auc,
            "log_loss":   self.log_loss_val,
            "wf_mean":    self.wf_mean,
            "wf_std":     self.wf_std,
            "wf_min":     min(self.wf_scores) if self.wf_scores else 0.0,
        }


class BaseMLModel(ABC):
    """
    Abstract base class for Trader AI and Risk Manager AI.

    Sub-classes must implement:
      - model_type: str         (e.g. 'trader_ai')
      - _prepare_labels()       converts raw data to binary labels
    """

    model_type: str = "base"

    def __init__(self):
        self._models:    Dict[str, Any] = {}     # algo_name → fitted model
        self._scalers:   Dict[str, StandardScaler] = {}
        self._best_algo: Optional[str]  = None
        self._metrics:   Dict[str, ModelMetrics] = {}
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._is_trained: bool = False
        self._save_dir = Path(settings.model_save_path)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    # ── Abstract ───────────────────────────────────────────────

    @abstractmethod
    def _prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to binary label array (0 or 1)."""
        ...

    # ── Public API ─────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        df_raw: pd.DataFrame,
        n_splits: int = None,
    ) -> Dict[str, ModelMetrics]:
        """
        Train all algorithms, run walk-forward validation,
        pick the best model, save to disk.

        Parameters
        ----------
        X      : feature matrix (rows = trades, cols = features)
        df_raw : raw DataFrame with outcome columns
        n_splits : walk-forward splits (default from settings)

        Returns
        -------
        dict of algo_name → ModelMetrics
        """
        n_splits = n_splits or settings.walk_forward_splits
        y = self._prepare_labels(df_raw)

        if len(X) < settings.model_min_training_samples:
            model_logger.warning(
                "{} needs ≥{} samples, got {}. Skipping train.",
                self.model_type, settings.model_min_training_samples, len(X)
            )
            return {}

        # TimeSeriesSplit requires n_samples > n_splits. On small bootstrap
        # datasets, scale n_splits down so folds stay non-degenerate rather
        # than erroring or silently producing tiny/empty validation folds.
        max_safe_splits = max(1, len(X) - 1)
        if n_splits > max_safe_splits:
            model_logger.warning(
                "{} requested {} splits but only {} samples available — "
                "reducing to {} splits.",
                self.model_type, n_splits, len(X), max_safe_splits
            )
            n_splits = max_safe_splits

        self._feature_names = list(X.columns)
        model_logger.info(
            "Training {} | {} samples | {} features | {} splits",
            self.model_type, len(X), len(self._feature_names), n_splits,
        )

        candidate_models = self._build_candidate_models()
        best_auc = -1.0

        for algo_name, model in candidate_models.items():
            try:
                metrics = self._walk_forward_train(
                    X.values, y, model, algo_name, n_splits
                )
                self._metrics[algo_name] = metrics

                model_logger.log_model_train(
                    self.model_type, algo_name,
                    metrics.accuracy, metrics.roc_auc, len(X)
                )

                if metrics.roc_auc > best_auc:
                    best_auc      = metrics.roc_auc
                    self._best_algo = algo_name

            except Exception as e:
                model_logger.error("Failed training {} / {}: {}", self.model_type, algo_name, e)

        model_logger.info(
            "{} best model: {} (AUC={:.4f})",
            self.model_type, self._best_algo, best_auc
        )

        # Final fit on full data with best algo
        if self._best_algo:
            self._fit_final(X.values, y, candidate_models[self._best_algo])
            self._extract_feature_importance()
            self._is_trained = True
            self.save()

        return self._metrics

    def predict_proba(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict class probabilities.

        Returns (prob_class_0, prob_class_1)
        """
        if not self._is_trained or self._best_algo is None:
            return 0.5, 0.5

        model  = self._models.get(self._best_algo)
        scaler = self._scalers.get(self._best_algo)
        if model is None:
            return 0.5, 0.5

        # Align features
        X_aligned = self._align_features(X)

        Xs = scaler.transform(X_aligned) if scaler else X_aligned
        proba = model.predict_proba(Xs)[0]
        return float(proba[0]), float(proba[1])

    def predict_proba_raw(self, feature_vec: np.ndarray) -> Tuple[float, float]:
        """Accept a raw numpy array directly."""
        if not self._is_trained:
            return 0.5, 0.5
        model  = self._models.get(self._best_algo)
        scaler = self._scalers.get(self._best_algo)
        X = feature_vec.reshape(1, -1)
        Xs = scaler.transform(X) if scaler else X
        proba = model.predict_proba(Xs)[0]
        return float(proba[0]), float(proba[1])

    def get_best_metrics(self) -> Optional[ModelMetrics]:
        if self._best_algo:
            return self._metrics.get(self._best_algo)
        return None

    def get_feature_importance(self) -> Dict[str, float]:
        return self._feature_importance

    def save(self) -> None:
        path = self._save_dir / f"{self.model_type}.joblib"
        joblib.dump({
            "models":       self._models,
            "scalers":      self._scalers,
            "best_algo":    self._best_algo,
            "metrics":      self._metrics,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
        }, path)
        model_logger.info("Saved {} → {}", self.model_type, path)

    def load(self) -> bool:
        path = self._save_dir / f"{self.model_type}.joblib"
        if not path.exists():
            return False
        try:
            data = joblib.load(path)
            self._models             = data["models"]
            self._scalers            = data["scalers"]
            self._best_algo          = data["best_algo"]
            self._metrics            = data.get("metrics", {})
            self._feature_names      = data.get("feature_names", [])
            self._feature_importance = data.get("feature_importance", {})
            self._is_trained         = True
            model_logger.info("Loaded {} from {}", self.model_type, path)
            return True
        except Exception as e:
            model_logger.error("Failed loading {}: {}", self.model_type, e)
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def best_algorithm(self) -> Optional[str]:
        return self._best_algo

    # ── Private helpers ────────────────────────────────────────

    def _build_candidate_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {}

        # Random Forest
        models["random_forest"] = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=20,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

        # Logistic Regression (fast baseline)
        models["logistic_regression"] = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", C=0.1, random_state=42,
        )

        # XGBoost
        if XGB_AVAILABLE:
            models["xgboost"] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )

        # LightGBM
        if LGB_AVAILABLE:
            models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8,
                class_weight="balanced", random_state=42,
                verbose=-1, n_jobs=-1,
            )

        # CatBoost
        if CAT_AVAILABLE:
            models["catboost"] = cb.CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.05,
                auto_class_weights="Balanced",
                verbose=0, random_seed=42,
            )

        return models

    def _walk_forward_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        algo_name: str,
        n_splits: int,
    ) -> ModelMetrics:
        """
        Time-series walk-forward cross-validation.
        NO shuffling — data order is preserved to prevent look-ahead bias.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = ModelMetrics()
        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr_s, y_tr)

            y_prob = m.predict_proba(X_val_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            if len(np.unique(y_val)) >= 2:
                auc = roc_auc_score(y_val, y_prob)
                fold_aucs.append(auc)

        # Final metrics on last fold
        if fold_aucs:
            metrics.wf_scores = fold_aucs
            metrics.wf_mean   = float(np.mean(fold_aucs))
            metrics.wf_std    = float(np.std(fold_aucs))
            metrics.roc_auc   = metrics.wf_mean
            metrics.accuracy  = metrics.wf_mean   # proxy

        return metrics

    def _fit_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
    ) -> None:
        """Fit the final model on all data."""
        scaler  = StandardScaler()
        X_s     = scaler.fit_transform(X)

        model.fit(X_s, y)

        # Calibrate probabilities via Platt scaling
        try:
            cal_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            cal_model.fit(X_s, y)
            self._models[self._best_algo]  = cal_model
        except Exception:
            self._models[self._best_algo]  = model

        self._scalers[self._best_algo] = scaler

    def _extract_feature_importance(self) -> None:
        """Extract feature importance from tree-based models."""
        model = self._models.get(self._best_algo)
        if model is None:
            return

        # Unwrap calibrated model
        inner = getattr(model, "estimator", model)

        importance = None

        if hasattr(inner, "feature_importances_"):
            importance = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            importance = np.abs(inner.coef_[0])

        if importance is not None and len(self._feature_names) == len(importance):
            pairs = sorted(
                zip(self._feature_names, importance.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            self._feature_importance = {k: float(v) for k, v in pairs}

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """Ensure feature columns match training order."""
        if not self._feature_names:
            return X.values

        for col in self._feature_names:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0

        return X[self._feature_names].values