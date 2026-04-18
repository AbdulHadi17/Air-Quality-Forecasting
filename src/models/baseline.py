"""
Baseline models for AQI forecasting benchmarking.

Provides simple but important baselines to measure whether complex models
(LSTM, ConvLSTM) actually add value:

    1. Naive Persistence: Predict PM2.5(t+6) = PM2.5(t)
    2. Historical Mean:   Predict the overall training set mean
    3. Random Forest:     Tabular ML on flattened sequences
    4. XGBoost:           Gradient-boosted trees on flattened sequences

All baselines use the same train/test split as the deep learning models
to ensure fair comparison.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.logger import logging
from src.config_loader import get_config


class BaselineModels:
    """Collection of baseline forecasting models.

    Args:
        config: Project configuration dict. If None, loads from default path.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.models_dir = self.config["paths"]["models_dir"]
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        logging.info("BaselineModels initialized")

    # ══════════════════════════════════════════════════════════
    # NAIVE PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def naive_persistence(
        self,
        X_test: np.ndarray,
        primary_param_idx: int = 0,
    ) -> np.ndarray:
        """Predict PM2.5(t+h) = PM2.5(t) — the simplest possible forecast.

        Uses the last timestep's primary parameter value as the prediction.

        Args:
            X_test: Test sequences of shape (n_samples, seq_length, n_features).
            primary_param_idx: Column index of PM2.5 in the feature array.

        Returns:
            Predictions array of shape (n_samples,).
        """
        # Last timestep, primary parameter column
        y_pred = X_test[:, -1, primary_param_idx]

        logging.info(f"Naive Persistence: {len(y_pred)} predictions generated")
        return y_pred

    # ══════════════════════════════════════════════════════════
    # HISTORICAL MEAN
    # ══════════════════════════════════════════════════════════

    def historical_mean(
        self,
        y_train: np.ndarray,
        n_test: int,
    ) -> np.ndarray:
        """Predict the training set mean for all test samples.

        Args:
            y_train: Training targets.
            n_test: Number of test samples to generate predictions for.

        Returns:
            Predictions array of shape (n_test,).
        """
        mean_val = np.nanmean(y_train)
        y_pred = np.full(n_test, mean_val, dtype=np.float32)

        logging.info(
            f"Historical Mean: predicting {mean_val:.4f} for all "
            f"{n_test} test samples"
        )
        return y_pred

    # ══════════════════════════════════════════════════════════
    # RIDGE REGRESSION
    # ══════════════════════════════════════════════════════════

    def train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alpha: float = 1.0,
    ) -> Ridge:
        """Train a Ridge regression on flattened sequences.

        Flattens (n_samples, seq_length, n_features) → (n_samples, seq_length*n_features).

        Args:
            X_train: Training sequences.
            y_train: Training targets.
            alpha: Regularization strength.

        Returns:
            Fitted Ridge model.
        """
        X_flat = X_train.reshape(X_train.shape[0], -1)

        model = Ridge(alpha=alpha)
        model.fit(X_flat, y_train)

        # Save
        path = os.path.join(self.models_dir, "ridge_baseline.pkl")
        joblib.dump(model, path)
        logging.info(f"Ridge trained on {X_flat.shape}, saved to {path}")

        return model

    def predict_ridge(
        self,
        model: Ridge,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Generate predictions with the Ridge model.

        Args:
            model: Fitted Ridge model.
            X_test: Test sequences.

        Returns:
            Predictions array.
        """
        X_flat = X_test.reshape(X_test.shape[0], -1)
        return model.predict(X_flat).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    # RANDOM FOREST
    # ══════════════════════════════════════════════════════════

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 200,
        max_depth: int = 20,
        max_samples: Optional[float] = 0.5,
        n_jobs: int = -1,
    ) -> RandomForestRegressor:
        """Train a Random Forest on flattened sequences.

        Args:
            X_train: Training sequences (n_samples, seq_length, n_features).
            y_train: Training targets.
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            max_samples: Fraction of samples per tree (for speed).
            n_jobs: Parallel jobs (-1 = all cores).

        Returns:
            Fitted RandomForestRegressor.
        """
        X_flat = X_train.reshape(X_train.shape[0], -1)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=42,
        )

        logging.info(
            f"Training Random Forest: {n_estimators} trees, "
            f"max_depth={max_depth}, shape={X_flat.shape}..."
        )
        model.fit(X_flat, y_train)

        # Save
        path = os.path.join(self.models_dir, "rf_baseline.pkl")
        joblib.dump(model, path)
        logging.info(f"Random Forest trained, saved to {path}")

        return model

    def predict_random_forest(
        self,
        model: RandomForestRegressor,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Generate predictions with the Random Forest.

        Args:
            model: Fitted model.
            X_test: Test sequences.

        Returns:
            Predictions array.
        """
        X_flat = X_test.reshape(X_test.shape[0], -1)
        return model.predict(X_flat).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    # XGBOOST
    # ══════════════════════════════════════════════════════════

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        """Train an XGBoost model on flattened sequences.

        Args:
            X_train: Training sequences.
            y_train: Training targets.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Step size shrinkage.
            subsample: Row sampling ratio.

        Returns:
            Fitted XGBRegressor (or None if xgboost not installed).
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logging.warning(
                "xgboost not installed — skipping XGBoost baseline. "
                "Install with: pip install xgboost"
            )
            return None

        X_flat = X_train.reshape(X_train.shape[0], -1)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        logging.info(
            f"Training XGBoost: {n_estimators} rounds, "
            f"max_depth={max_depth}, shape={X_flat.shape}..."
        )
        model.fit(X_flat, y_train)

        # Save
        path = os.path.join(self.models_dir, "xgb_baseline.pkl")
        joblib.dump(model, path)
        logging.info(f"XGBoost trained, saved to {path}")

        return model

    def predict_xgboost(self, model, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions with the XGBoost model.

        Args:
            model: Fitted XGBRegressor.
            X_test: Test sequences.

        Returns:
            Predictions array.
        """
        X_flat = X_test.reshape(X_test.shape[0], -1)
        return model.predict(X_flat).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    # RUN ALL BASELINES
    # ══════════════════════════════════════════════════════════

    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        evaluator=None,
    ) -> dict[str, np.ndarray]:
        """Train and evaluate all baseline models.

        Args:
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            evaluator: Optional ModelEvaluator instance for metrics.

        Returns:
            Dictionary of model_name → predictions on test set.
        """
        logging.info("=" * 60)
        logging.info("  BASELINE MODELS")
        logging.info("=" * 60)

        predictions = {}

        # 1. Naive Persistence
        pred_naive = self.naive_persistence(X_test)
        predictions["Naive Persistence"] = pred_naive
        if evaluator:
            evaluator.evaluate(y_test, pred_naive, "Naive Persistence")

        # 2. Historical Mean
        pred_mean = self.historical_mean(y_train, len(y_test))
        predictions["Historical Mean"] = pred_mean
        if evaluator:
            evaluator.evaluate(y_test, pred_mean, "Historical Mean")

        # 3. Ridge Regression
        ridge = self.train_ridge(X_train, y_train)
        pred_ridge = self.predict_ridge(ridge, X_test)
        predictions["Ridge Regression"] = pred_ridge
        if evaluator:
            evaluator.evaluate(y_test, pred_ridge, "Ridge Regression")

        # 4. Random Forest
        rf = self.train_random_forest(X_train, y_train)
        pred_rf = self.predict_random_forest(rf, X_test)
        predictions["Random Forest"] = pred_rf
        if evaluator:
            evaluator.evaluate(y_test, pred_rf, "Random Forest")

        # 5. XGBoost (optional)
        xgb = self.train_xgboost(X_train, y_train)
        if xgb is not None:
            pred_xgb = self.predict_xgboost(xgb, X_test)
            predictions["XGBoost"] = pred_xgb
            if evaluator:
                evaluator.evaluate(y_test, pred_xgb, "XGBoost")

        logging.info(f"All baselines complete: {list(predictions.keys())}")
        return predictions
