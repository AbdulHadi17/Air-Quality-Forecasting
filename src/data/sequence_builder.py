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
        self.target_scaler = None

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
        Windows that span temporal gaps (> 1 hour between consecutive
        rows) are skipped to preserve temporal continuity.

        Scalers are fit on the training split only, then applied to the
        test split to prevent data leakage.

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

        # ── TIME-BASED SPLIT BEFORE SCALING (prevents data leakage) ──
        # Sort globally by timestamp to find the split point
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
        global_timestamps = np.sort(df_clean["timestamp"].unique())
        split_ts = global_timestamps[int(len(global_timestamps) * (1 - test_split))]

        df_train = df_clean[df_clean["timestamp"] < split_ts].copy()
        df_test = df_clean[df_clean["timestamp"] >= split_ts].copy()

        logging.info(
            f"  Time-based split: train={len(df_train)} rows "
            f"(before {split_ts}), test={len(df_test)} rows"
        )

        # ── FIT SCALERS ON TRAIN ONLY ──
        self.scaler = MinMaxScaler()
        df_train[feature_cols] = self.scaler.fit_transform(df_train[feature_cols])
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols])

        self.target_scaler = MinMaxScaler()
        df_train["target"] = self.target_scaler.fit_transform(df_train[["target"]])
        df_test["target"] = self.target_scaler.transform(df_test[["target"]])

        # ── BUILD SEQUENCES PER STATION ──
        train_X, train_y = [], []
        test_X, test_y = [], []

        for loc_id in sorted(df_clean["location_id"].unique()):
            # Train sequences
            loc_train = df_train[df_train["location_id"] == loc_id].sort_values("timestamp")
            if len(loc_train) > self.seq_length:
                X_loc, y_loc = self._create_sliding_windows(
                    loc_train[feature_cols].values,
                    loc_train["target"].values,
                    loc_train["timestamp"].values,
                )
                if len(X_loc) > 0:
                    train_X.append(X_loc)
                    train_y.append(y_loc)

            # Test sequences
            loc_test = df_test[df_test["location_id"] == loc_id].sort_values("timestamp")
            if len(loc_test) > self.seq_length:
                X_loc, y_loc = self._create_sliding_windows(
                    loc_test[feature_cols].values,
                    loc_test["target"].values,
                    loc_test["timestamp"].values,
                )
                if len(X_loc) > 0:
                    test_X.append(X_loc)
                    test_y.append(y_loc)

        if not train_X:
            logging.error("No valid training sequences created")
            return {}

        X_train = np.concatenate(train_X, axis=0)
        y_train = np.concatenate(train_y, axis=0)
        X_test = np.concatenate(test_X, axis=0) if test_X else np.array([])
        y_test = np.concatenate(test_y, axis=0) if test_y else np.array([])

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
            "target_scaler": self.target_scaler,
        }

    def _create_sliding_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences, skipping windows with time gaps.

        A window is only valid if every pair of consecutive timestamps
        within it differs by exactly 1 hour. This prevents the model
        from learning on temporally discontinuous sequences.

        Args:
            features: Array of shape (n_timesteps, n_features).
            targets: Array of shape (n_timesteps,).
            timestamps: Array of datetime64 timestamps. If provided,
                        windows spanning temporal gaps are skipped.

        Returns:
            X: Array of shape (n_samples, seq_length, n_features).
            y: Array of shape (n_samples,).
        """
        n = len(features)
        if n <= self.seq_length:
            return np.array([]), np.array([])

        # Pre-compute per-row temporal continuity flags
        if timestamps is not None:
            ts = pd.Series(pd.to_datetime(timestamps))
            diffs_hours = ts.diff().dt.total_seconds() / 3600.0
            # True if this row is exactly 1 hour after the previous
            is_contiguous = np.array(diffs_hours == 1.0, dtype=bool)
            if len(is_contiguous) > 0:
                is_contiguous[0] = True  # First row has no predecessor
        else:
            is_contiguous = None

        X, y = [], []
        skipped = 0
        for i in range(n - self.seq_length):
            window_end = i + self.seq_length

            # Check temporal continuity within this window
            if is_contiguous is not None:
                # All rows from i+1 to window_end-1 must be contiguous
                # (i.e., each is exactly 1h after its predecessor)
                if not is_contiguous[i + 1 : window_end].all():
                    skipped += 1
                    continue

            X.append(features[i : window_end])
            y.append(targets[window_end - 1])

        if skipped > 0:
            logging.info(f"    Skipped {skipped} windows with temporal gaps")

        if not X:
            return np.array([]), np.array([])

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

        Maps station data onto a spatial grid using Haversine-based
        Inverse Distance Weighting (IDW) and creates sequences of
        shape (samples, timesteps, grid_H, grid_W, n_features).

        Scalers are fit on the training split only, then applied to
        the test split to prevent data leakage.

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

        # Save station_grid_map for evaluation (station-level metric extraction)
        map_path = os.path.join(self.models_dir, "station_grid_map.pkl")
        joblib.dump(station_grid_map, map_path)
        logging.info(f"  Saved station_grid_map to {map_path}")

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

        # ── TIME-BASED SPLIT BEFORE SCALING (prevents data leakage) ──
        df_clean = df_clean.sort_values("timestamp")
        global_timestamps = np.sort(df_clean["timestamp"].unique())
        split_ts = global_timestamps[int(len(global_timestamps) * (1 - test_split))]

        df_train = df_clean[df_clean["timestamp"] < split_ts].copy()
        df_test = df_clean[df_clean["timestamp"] >= split_ts].copy()

        logging.info(
            f"  Time-based split: train={len(df_train)} rows, "
            f"test={len(df_test)} rows (split at {split_ts})"
        )

        # ── FIT SCALERS ON TRAIN ONLY ──
        self.grid_scaler = MinMaxScaler()
        df_train[grid_features] = self.grid_scaler.fit_transform(df_train[grid_features])
        df_test[grid_features] = self.grid_scaler.transform(df_test[grid_features])

        if self.target_scaler is None:
            self.target_scaler = MinMaxScaler()
            df_train["target"] = self.target_scaler.fit_transform(df_train[["target"]])
        else:
            df_train["target"] = self.target_scaler.transform(df_train[["target"]])
        df_test["target"] = self.target_scaler.transform(df_test[["target"]])

        # ── BUILD GRIDS USING HAVERSINE IDW ──
        n_features = len(grid_features)

        # Precompute grid points and Haversine distances
        Lon, Lat = np.meshgrid(grid_lons, grid_lats)
        grid_points = np.column_stack([Lat.ravel(), Lon.ravel()])

        def _build_grid_tensor(df_split, split_name):
            """Build grid tensor and target tensor for a split."""
            timestamps = sorted(df_split["timestamp"].unique())
            grid_tensor = np.zeros(
                (len(timestamps), grid_shape[0], grid_shape[1], n_features),
                dtype=np.float32,
            )
            target_tensor = np.zeros(
                (len(timestamps), grid_shape[0], grid_shape[1]),
                dtype=np.float32,
            )

            for t_idx, ts in enumerate(timestamps):
                ts_data = df_split[df_split["timestamp"] == ts]
                ts_data = ts_data.merge(
                    stations_meta[["location_id", "latitude", "longitude"]],
                    on="location_id",
                )

                if len(ts_data) == 0:
                    continue

                station_points = ts_data[["latitude", "longitude"]].values

                # Haversine distance matrix (grid_points × station_points)
                dists_km = self._haversine_matrix(grid_points, station_points)
                dists_km[dists_km == 0] = 1e-6  # avoid div-by-zero

                # IDW weights (inverse square of distance in km)
                weights = 1.0 / (dists_km ** 2)
                weights /= weights.sum(axis=1, keepdims=True)

                for f_idx, feat in enumerate(grid_features):
                    station_vals = ts_data[feat].values
                    interp_vals = np.sum(weights * station_vals, axis=1)
                    grid_tensor[t_idx, :, :, f_idx] = interp_vals.reshape(grid_shape)

                target_vals = ts_data["target"].values
                interp_target = np.sum(weights * target_vals, axis=1)
                target_tensor[t_idx, :, :] = interp_target.reshape(grid_shape)

            logging.info(
                f"  {split_name} grid tensor: {grid_tensor.shape}, "
                f"{len(timestamps)} timesteps"
            )
            return grid_tensor, target_tensor, timestamps

        train_grid, train_target, train_timestamps = _build_grid_tensor(
            df_train, "Train"
        )
        test_grid, test_target, test_timestamps = _build_grid_tensor(
            df_test, "Test"
        )

        # ── CREATE SLIDING WINDOWS (with continuity check) ──
        def _grid_sliding_windows(grid_tensor, target_tensor, timestamps):
            ts = pd.Series(pd.to_datetime(timestamps))
            diffs_hours = ts.diff().dt.total_seconds() / 3600.0
            is_contiguous = np.array(diffs_hours == 1.0, dtype=bool)
            if len(is_contiguous) > 0:
                is_contiguous[0] = True

            X, y = [], []
            skipped = 0
            for i in range(len(timestamps) - self.seq_length):
                window_end = i + self.seq_length
                # Check temporal continuity (each step must be ~1 hour apart)
                if not is_contiguous[i + 1 : window_end].all():
                    skipped += 1
                    continue
                X.append(grid_tensor[i:window_end])
                y.append(target_tensor[window_end - 1])
            if skipped > 0:
                logging.info(f"    Skipped {skipped} grid windows with temporal gaps")
            return X, y

        train_X_list, train_y_list = _grid_sliding_windows(
            train_grid, train_target, train_timestamps
        )
        test_X_list, test_y_list = _grid_sliding_windows(
            test_grid, test_target, test_timestamps
        )

        if not train_X_list:
            logging.error("No valid ConvLSTM training sequences created")
            return {}

        X_train = np.array(train_X_list, dtype=np.float32)
        y_train = np.array(train_y_list, dtype=np.float32)
        X_test = np.array(test_X_list, dtype=np.float32) if test_X_list else np.array([])
        y_test = np.array(test_y_list, dtype=np.float32) if test_y_list else np.array([])

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

    @staticmethod
    def _haversine_matrix(
        points_a: np.ndarray,
        points_b: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise Haversine distances between two sets of points.

        Args:
            points_a: Array of shape (N, 2) with [latitude, longitude] in degrees.
            points_b: Array of shape (M, 2) with [latitude, longitude] in degrees.

        Returns:
            Distance matrix of shape (N, M) in kilometres.
        """
        R = 6371.0  # Earth radius in km

        lat1 = np.radians(points_a[:, 0]).reshape(-1, 1)
        lon1 = np.radians(points_a[:, 1]).reshape(-1, 1)
        lat2 = np.radians(points_b[:, 0]).reshape(1, -1)
        lon2 = np.radians(points_b[:, 1]).reshape(1, -1)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return R * c

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
            
        if self.target_scaler is not None:
            path = os.path.join(self.models_dir, "target_scaler.pkl")
            joblib.dump(self.target_scaler, path)
            logging.info(f"  Saved target scaler to {path}")

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
