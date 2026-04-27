"""
Sequence builder for the AQI Forecasting project.

Converts the feature-engineered DataFrame into sliding-window sequences
suitable for LSTM and ConvLSTM input.

Outputs:
    - LSTM:     (samples, timesteps, features) per station
    - ConvLSTM: (samples, timesteps, grid_H, grid_W, features) for spatial grid

Sequences are saved as NumPy arrays in data/sequences/ for fast reloading.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.logger import logging
from src.config_loader import get_config


class SequenceBuilder:
    """Builds sliding-window sequences for LSTM and ConvLSTM models.

    Args:
        config: Project configuration dict. If None, loads from default path.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.seq_length = self.config["forecasting"].get("sequence_length", 24)
        self.horizon = self.config["forecasting"].get("horizon_hours", 6)
        self.grid_resolution = self.config["forecasting"].get("grid_resolution", 0.005)
        self.primary_param = self.config["openaq"].get("primary_parameter", "pm25")
        self.bbox = self.config["openaq"]["bbox"]

        self.sequences_dir = self.config["paths"]["sequences_dir"]
        self.models_dir = self.config["paths"]["models_dir"]
        Path(self.sequences_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.scaler = None  # Will be fitted during build

        logging.info(
            f"SequenceBuilder initialized: "
            f"seq_length={self.seq_length}, horizon={self.horizon}h"
        )

    # ══════════════════════════════════════════════════════════
    # LSTM SEQUENCES (per-station)
    # ══════════════════════════════════════════════════════════

    def build_lstm_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        test_split: float = 0.2,
    ) -> dict[str, np.ndarray]:
        """Build sliding-window sequences for per-station LSTM.

        For each station, creates overlapping windows of shape
        (seq_length, n_features) with corresponding target values.

        Args:
            df: Feature-engineered DataFrame with location_id, timestamp,
                features, and target columns.
            feature_cols: List of feature column names to include.
                          If None, auto-detects numeric columns.
            test_split: Fraction of data to hold out for testing (time-based).

        Returns:
            Dictionary with keys:
                X_train, y_train: Training sequences and targets
                X_test, y_test: Test sequences and targets
                feature_names: List of feature column names
                scaler: Fitted MinMaxScaler
        """
        logging.info("Building LSTM sequences (per-station)...")

        if df.empty:
            logging.error("Empty DataFrame — cannot build sequences")
            return {}

        # Auto-detect feature columns
        if feature_cols is None:
            exclude = {"timestamp", "location_id", "target"}
            feature_cols = [c for c in df.columns
                           if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.float16]]

        logging.info(f"  Using {len(feature_cols)} features: {feature_cols[:10]}...")

        # Drop rows where target is NaN
        df_clean = df.dropna(subset=["target"]).copy()

        # Drop rows where ALL features are NaN
        df_clean = df_clean.dropna(subset=feature_cols, how="all")

        # Fill remaining NaN in features with forward-fill then 0
        df_clean[feature_cols] = df_clean[feature_cols].ffill().fillna(0)

        logging.info(f"  Clean rows: {len(df_clean)} (dropped {len(df) - len(df_clean)} with NaN target)")

        # Scale features
        self.scaler = MinMaxScaler()
        df_clean[feature_cols] = self.scaler.fit_transform(df_clean[feature_cols])

        # Build sequences per station
        all_X, all_y = [], []

        for loc_id in sorted(df_clean["location_id"].unique()):
            loc_data = df_clean[df_clean["location_id"] == loc_id].sort_values("timestamp")
            features = loc_data[feature_cols].values
            targets = loc_data["target"].values

            X_loc, y_loc = self._create_sliding_windows(features, targets)
            if len(X_loc) > 0:
                all_X.append(X_loc)
                all_y.append(y_loc)

        if not all_X:
            logging.error("No valid sequences created")
            return {}

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        # Time-based train/test split (no shuffle — preserves temporal order)
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Save
        self._save_sequences(X_train, y_train, X_test, y_test, prefix="lstm")
        self._save_scaler()

        logging.info(
            f"LSTM sequences built:\n"
            f"  X_train: {X_train.shape}, y_train: {y_train.shape}\n"
            f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}\n"
            f"  Features: {len(feature_cols)}"
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_cols,
            "scaler": self.scaler,
        }

    def _create_sliding_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from a continuous array.

        Args:
            features: Array of shape (n_timesteps, n_features).
            targets: Array of shape (n_timesteps,).

        Returns:
            X: Array of shape (n_samples, seq_length, n_features).
            y: Array of shape (n_samples,).
        """
        n = len(features)
        if n <= self.seq_length:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(n - self.seq_length):
            X.append(features[i : i + self.seq_length])
            y.append(targets[i + self.seq_length - 1])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ══════════════════════════════════════════════════════════
    # ConvLSTM GRID SEQUENCES (spatio-temporal)
    # ══════════════════════════════════════════════════════════

    def build_convlstm_sequences(
        self,
        df: pd.DataFrame,
        stations_meta: pd.DataFrame,
        test_split: float = 0.2,
    ) -> dict[str, np.ndarray]:
        """Build grid-based sequences for ConvLSTM.

        Maps station data onto a spatial grid and creates sequences of
        shape (samples, timesteps, grid_H, grid_W, n_features).

        Each grid cell gets the value from the nearest station, or
        inverse-distance-weighted average if multiple stations are close.

        Args:
            df: Feature-engineered DataFrame.
            stations_meta: Station metadata with lat/lon.
            test_split: Fraction for test set.

        Returns:
            Dictionary with X_train, y_train, X_test, y_test,
            grid_shape, station_to_grid mapping.
        """
        logging.info("Building ConvLSTM grid sequences...")

        # Build the spatial grid
        grid_lats, grid_lons, grid_shape = self._build_grid()
        logging.info(f"  Grid shape: {grid_shape} ({grid_shape[0]}x{grid_shape[1]})")

        # Map stations to nearest grid cells
        station_grid_map = self._map_stations_to_grid(
            stations_meta, grid_lats, grid_lons
        )
        logging.info(f"  Mapped {len(station_grid_map)} stations to grid cells")

        # Select feature columns (only primary param + weather for grid)
        grid_features = [self.primary_param]
        weather_cols = [c for c in df.columns if c in [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "pressure_msl", "precipitation", "cloud_cover"
        ]]
        grid_features.extend(weather_cols)
        grid_features = [c for c in grid_features if c in df.columns]

        logging.info(f"  Grid features: {grid_features}")

        # Drop NaN targets
        df_clean = df.dropna(subset=["target"]).copy()
        df_clean[grid_features] = df_clean[grid_features].ffill().fillna(0)

        # Scale
        if not hasattr(self, "grid_scaler") or self.grid_scaler is None:
            self.grid_scaler = MinMaxScaler()
            df_clean[grid_features] = self.grid_scaler.fit_transform(df_clean[grid_features])
        else:
            df_clean[grid_features] = self.grid_scaler.transform(df_clean[grid_features])

        # Get unique sorted timestamps
        timestamps = sorted(df_clean["timestamp"].unique())
        n_features = len(grid_features)

        # Build grid tensor for each timestep
        grid_series = np.zeros(
            (len(timestamps), grid_shape[0], grid_shape[1], n_features),
            dtype=np.float32,
        )
        target_series = np.zeros(
            (len(timestamps), grid_shape[0], grid_shape[1]),
            dtype=np.float32,
        )

        for t_idx, ts in enumerate(timestamps):
            ts_data = df_clean[df_clean["timestamp"] == ts]

            for _, row in ts_data.iterrows():
                loc_id = row["location_id"]
                if loc_id in station_grid_map:
                    gi, gj = station_grid_map[loc_id]
                    for f_idx, feat in enumerate(grid_features):
                        grid_series[t_idx, gi, gj, f_idx] = row[feat]
                    target_series[t_idx, gi, gj] = row["target"]

        # Create sliding window sequences over grid
        X, y = [], []
        for i in range(len(timestamps) - self.seq_length):
            X.append(grid_series[i : i + self.seq_length])
            # Target: primary param grid at the last timestep
            y.append(target_series[i + self.seq_length - 1])

        if not X:
            logging.error("No valid ConvLSTM sequences created")
            return {}

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Time-based split
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Save
        self._save_sequences(X_train, y_train, X_test, y_test, prefix="convlstm")

        logging.info(
            f"ConvLSTM sequences built:\n"
            f"  X_train: {X_train.shape}, y_train: {y_train.shape}\n"
            f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}\n"
            f"  Grid: {grid_shape}, Features: {n_features}"
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "grid_shape": grid_shape,
            "station_grid_map": station_grid_map,
            "grid_lats": grid_lats,
            "grid_lons": grid_lons,
            "feature_names": grid_features,
        }

    def _build_grid(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Build a regular lat/lon grid over the Lahore bounding box.

        Returns:
            grid_lats: 1D array of latitude values.
            grid_lons: 1D array of longitude values.
            grid_shape: (n_rows, n_cols) tuple.
        """
        min_lon, min_lat, max_lon, max_lat = self.bbox
        res = self.grid_resolution

        grid_lats = np.arange(min_lat, max_lat, res)
        grid_lons = np.arange(min_lon, max_lon, res)

        return grid_lats, grid_lons, (len(grid_lats), len(grid_lons))

    def _map_stations_to_grid(
        self,
        stations_meta: pd.DataFrame,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[int, tuple[int, int]]:
        """Map each station to its nearest grid cell.

        Args:
            stations_meta: Station metadata with lat/lon.
            grid_lats: Grid latitude values.
            grid_lons: Grid longitude values.

        Returns:
            Dictionary mapping location_id → (grid_row, grid_col).
        """
        station_grid_map = {}
        for _, row in stations_meta.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            # Find nearest grid indices
            lat_idx = np.argmin(np.abs(grid_lats - lat))
            lon_idx = np.argmin(np.abs(grid_lons - lon))
            station_grid_map[row["location_id"]] = (lat_idx, lon_idx)

        return station_grid_map

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def _save_sequences(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prefix: str,
    ) -> None:
        """Save sequence arrays to data/sequences/.

        Args:
            X_train, y_train, X_test, y_test: NumPy arrays.
            prefix: File prefix (e.g., 'lstm' or 'convlstm').
        """
        for name, arr in [
            (f"{prefix}_X_train", X_train),
            (f"{prefix}_y_train", y_train),
            (f"{prefix}_X_test", X_test),
            (f"{prefix}_y_test", y_test),
        ]:
            path = os.path.join(self.sequences_dir, f"{name}.npy")
            np.save(path, arr)
            logging.info(f"  Saved {name}: {arr.shape} -> {path}")

    def _save_scaler(self) -> None:
        """Save the fitted scaler for inverse-transforming predictions."""
        if self.scaler is not None:
            path = os.path.join(self.models_dir, "scaler.pkl")
            joblib.dump(self.scaler, path)
            logging.info(f"  Saved scaler to {path}")

    @staticmethod
    def load_sequences(sequences_dir: str, prefix: str) -> dict[str, np.ndarray]:
        """Load previously saved sequence arrays.

        Args:
            sequences_dir: Path to sequences directory.
            prefix: File prefix ('lstm' or 'convlstm').

        Returns:
            Dictionary with X_train, y_train, X_test, y_test arrays.
        """
        result = {}
        for key in ["X_train", "y_train", "X_test", "y_test"]:
            path = os.path.join(sequences_dir, f"{prefix}_{key}.npy")
            if Path(path).exists():
                result[key] = np.load(path)
                logging.info(f"  Loaded {prefix}_{key}: {result[key].shape}")
            else:
                logging.warning(f"  {path} not found")

        return result
