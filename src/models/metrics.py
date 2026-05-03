"""
Model evaluation and metrics module.

Provides standardized evaluation across all model types (baseline, LSTM,
ConvLSTM) with consistent metric computation, comparison tables, and
visualization utilities.

ConvLSTM models are evaluated at actual station grid cells only, not on
the full interpolated grid, to avoid artificially inflated metrics from
IDW-smoothed background cells.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from src.logger import logging
from src.config_loader import get_config


class ModelEvaluator:
    """Standardized evaluation framework for all forecasting models.

    Args:
        config: Project configuration dict. If None, loads from default path.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.outputs_dir = self.config["paths"]["outputs_dir"]
        Path(self.outputs_dir).mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

        logging.info("ModelEvaluator initialized")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        target_scaler=None,
    ) -> dict:
        """Compute all regression metrics for a model's predictions.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            model_name: Name string for logging and comparison tables.
            target_scaler: Optional scaler to inverse-transform values before evaluating.

        Returns:
            Dictionary of metric name → value.
        """
        # Flatten if multi-dimensional (e.g., ConvLSTM grid output)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Remove NaN / inf pairs
        valid_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_t = y_true_flat[valid_mask]
        y_p = y_pred_flat[valid_mask]

        if target_scaler is not None:
            # Inverse transform requires 2D array
            y_t = target_scaler.inverse_transform(y_t.reshape(-1, 1)).flatten()
            y_p = target_scaler.inverse_transform(y_p.reshape(-1, 1)).flatten()

        if len(y_t) == 0:
            logging.error(f"[{model_name}] No valid predictions to evaluate")
            return {}

        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2 = r2_score(y_t, y_p)

        # MAPE — only where y_true != 0
        nonzero_mask = y_t != 0
        if nonzero_mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_t[nonzero_mask], y_p[nonzero_mask]) * 100
        else:
            mape = float("nan")

        # Within-10 accuracy: % of predictions within ±10 µg/m³
        within_10 = (np.abs(y_t - y_p) <= 10).mean() * 100

        # Within-25% accuracy: % of predictions within ±25% of true value
        pct_errors = np.abs(y_t - y_p) / np.maximum(y_t, 1e-8)
        within_25pct = (pct_errors <= 0.25).mean() * 100

        metrics = {
            "model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE_%": round(mape, 2),
            "Within_10": round(within_10, 2),
            "Within_25pct": round(within_25pct, 2),
            "n_samples": len(y_t),
        }

        self.results.append(metrics)

        logging.info(
            f"[{model_name}] MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"R²={r2:.4f}, MAPE={mape:.2f}%, Within±10={within_10:.1f}%"
        )

        return metrics

    def evaluate_convlstm_at_stations(
        self,
        y_true_grid: np.ndarray,
        y_pred_grid: np.ndarray,
        station_grid_map: dict,
        model_name: str = "ConvLSTM",
        target_scaler=None,
    ) -> dict:
        """Evaluate ConvLSTM predictions only at actual station grid cells.

        Instead of flattening the entire (n, H, W) prediction grid —
        which is dominated by IDW-interpolated background cells — this
        method extracts predictions at the exact (row, col) grid indices
        where real monitoring stations exist, then computes metrics on
        those points only.

        Args:
            y_true_grid: Ground truth grids, shape (n_samples, H, W).
            y_pred_grid: Predicted grids, shape (n_samples, H, W).
            station_grid_map: Dict mapping location_id → (row, col).
            model_name: Name for the comparison table.
            target_scaler: Optional scaler for inverse transform.

        Returns:
            Dictionary of metric name → value.
        """
        if not station_grid_map:
            logging.warning(
                f"[{model_name}] No station_grid_map provided — "
                f"falling back to full-grid evaluation"
            )
            return self.evaluate(y_true_grid, y_pred_grid, model_name, target_scaler)

        # Extract values only at station grid cells
        station_cells = list(station_grid_map.values())
        rows = [cell[0] for cell in station_cells]
        cols = [cell[1] for cell in station_cells]

        # y_true_grid, y_pred_grid: (n_samples, H, W)
        y_true_station = y_true_grid[:, rows, cols]  # (n_samples, n_stations)
        y_pred_station = y_pred_grid[:, rows, cols]

        logging.info(
            f"[{model_name}] Evaluating at {len(station_cells)} station cells "
            f"out of {y_true_grid.shape[1]}×{y_true_grid.shape[2]} grid "
            f"({len(station_cells) * y_true_grid.shape[0]} total points)"
        )

        return self.evaluate(
            y_true_station, y_pred_station, model_name, target_scaler
        )

    def comparison_table(self) -> pd.DataFrame:
        """Build a comparison table of all evaluated models.

        Returns:
            DataFrame sorted by RMSE (ascending = best first).
        """
        if not self.results:
            logging.warning("No evaluation results to compare")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)

        # Save to CSV
        path = os.path.join(self.outputs_dir, "model_comparison.csv")
        df.to_csv(path, index=False)
        logging.info(f"Saved comparison table to {path}")

        return df

    def print_comparison(self) -> None:
        """Print a formatted comparison table."""
        df = self.comparison_table()
        if df.empty:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print("  MODEL COMPARISON (sorted by RMSE)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
