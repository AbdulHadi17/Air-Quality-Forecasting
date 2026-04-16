"""
Feature engineering module for the AQI Forecasting project.

Transforms the preprocessed merged DataFrame into ML-ready features:
    - Lag features (past N hours of PM2.5)
    - Rolling statistics (mean, std, min, max over windows)
    - Cyclical encoding for time features (hour, day_of_week, month)
    - Rate of change features
    - Spatial distance weights between stations
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.logger import logging
from src.config_loader import get_config


class FeatureEngineer:
    """Creates ML features from preprocessed AQ + weather data.

    Args:
        config: Project configuration dict. If None, loads from default path.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.primary_param = self.config["openaq"].get("primary_parameter", "pm25")
        self.horizon = self.config["forecasting"].get("horizon_hours", 6)
        self.sequence_length = self.config["forecasting"].get("sequence_length", 24)

        logging.info("FeatureEngineer initialized")

    # ══════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════

    def run(
        self,
        merged_df: pd.DataFrame,
        stations_meta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Execute the full feature engineering pipeline.

        Args:
            merged_df: Merged AQ + weather DataFrame from preprocessing.
            stations_meta: Station metadata with lat/lon for spatial features.

        Returns:
            Feature-enriched DataFrame ready for sequence building.
        """
        logging.info("=" * 60)
        logging.info("  FEATURE ENGINEERING PIPELINE")
        logging.info("=" * 60)

        df = merged_df.copy()

        # Step 1: Lag features
        logging.info("Step 1/5: Creating lag features...")
        df = self.add_lag_features(df)

        # Step 2: Rolling statistics
        logging.info("Step 2/5: Creating rolling statistics...")
        df = self.add_rolling_features(df)

        # Step 3: Rate of change
        logging.info("Step 3/5: Creating rate-of-change features...")
        df = self.add_rate_of_change(df)

        # Step 4: Cyclical encoding
        logging.info("Step 4/5: Encoding cyclical time features...")
        df = self.add_cyclical_encoding(df)

        # Step 5: Target variable
        logging.info("Step 5/5: Creating target variable...")
        df = self.add_target(df)

        # Summary
        feature_cols = [c for c in df.columns
                        if c not in ["timestamp", "location_id", "target"]]
        logging.info(
            f"Feature engineering complete: {len(feature_cols)} features, "
            f"{len(df)} rows"
        )
        logging.info(f"Feature list: {feature_cols}")

        return df

    # ══════════════════════════════════════════════════════════
    # LAG FEATURES
    # ══════════════════════════════════════════════════════════

    def add_lag_features(
        self,
        df: pd.DataFrame,
        lags: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Add lagged values of the primary parameter.

        Creates columns like pm25_lag_1h, pm25_lag_3h, pm25_lag_6h, etc.

        Args:
            df: Input DataFrame with timestamp and location_id.
            lags: List of lag hours. Defaults to [1, 2, 3, 6, 12, 24].

        Returns:
            DataFrame with lag feature columns added.
        """
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]

        if self.primary_param not in df.columns:
            logging.warning(f"{self.primary_param} not in DataFrame — skipping lags")
            return df

        df = df.copy()
        df = df.sort_values(["location_id", "timestamp"])

        for lag in lags:
            col_name = f"{self.primary_param}_lag_{lag}h"
            df[col_name] = df.groupby("location_id")[self.primary_param].shift(lag)

        logging.info(f"  Added {len(lags)} lag features: {lags}")
        return df

    # ══════════════════════════════════════════════════════════
    # ROLLING STATISTICS
    # ══════════════════════════════════════════════════════════

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Add rolling mean, std, min, max over sliding windows.

        Creates columns like pm25_rolling_6h_mean, pm25_rolling_24h_std, etc.

        Args:
            df: Input DataFrame.
            windows: Window sizes in hours. Defaults to [6, 12, 24].

        Returns:
            DataFrame with rolling feature columns added.
        """
        if windows is None:
            windows = [6, 12, 24]

        if self.primary_param not in df.columns:
            logging.warning(f"{self.primary_param} not in DataFrame — skipping rolling")
            return df

        df = df.copy()
        df = df.sort_values(["location_id", "timestamp"])

        for window in windows:
            grouped = df.groupby("location_id")[self.primary_param]

            df[f"{self.primary_param}_roll_{window}h_mean"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{self.primary_param}_roll_{window}h_std"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f"{self.primary_param}_roll_{window}h_min"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f"{self.primary_param}_roll_{window}h_max"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

        n_features = len(windows) * 4
        logging.info(f"  Added {n_features} rolling features for windows: {windows}")
        return df

    # ══════════════════════════════════════════════════════════
    # RATE OF CHANGE
    # ══════════════════════════════════════════════════════════

    def add_rate_of_change(
        self,
        df: pd.DataFrame,
        periods: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Add rate-of-change (difference) features.

        Creates columns like pm25_diff_1h (absolute change) and
        pm25_pct_change_1h (percentage change).

        Args:
            df: Input DataFrame.
            periods: Differencing periods in hours. Defaults to [1, 3, 6].

        Returns:
            DataFrame with rate-of-change columns added.
        """
        if periods is None:
            periods = [1, 3, 6]

        if self.primary_param not in df.columns:
            return df

        df = df.copy()
        df = df.sort_values(["location_id", "timestamp"])

        for period in periods:
            grouped = df.groupby("location_id")[self.primary_param]

            df[f"{self.primary_param}_diff_{period}h"] = grouped.diff(period)
            df[f"{self.primary_param}_pct_change_{period}h"] = grouped.pct_change(period)

        # Replace inf values from pct_change (div by zero)
        pct_cols = [c for c in df.columns if "pct_change" in c]
        df[pct_cols] = df[pct_cols].replace([np.inf, -np.inf], np.nan)

        n_features = len(periods) * 2
        logging.info(f"  Added {n_features} rate-of-change features for periods: {periods}")
        return df

    # ══════════════════════════════════════════════════════════
    # CYCLICAL ENCODING
    # ══════════════════════════════════════════════════════════

    def add_cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode cyclical time features using sin/cos transformation.

        Converts hour, day_of_week, and month into continuous
        sin/cos pairs so that, e.g., hour 23 is close to hour 0.

        Args:
            df: Input DataFrame with hour, day_of_week, month columns.

        Returns:
            DataFrame with sin/cos columns, original columns removed.
        """
        df = df.copy()
        cyclical_maps = {
            "hour": 24,
            "day_of_week": 7,
            "month": 12,
        }

        for col, period in cyclical_maps.items():
            if col in df.columns:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
                df = df.drop(columns=[col])

        # Also drop is_weekend (now encoded via day_of_week sin/cos)
        if "is_weekend" in df.columns:
            df = df.drop(columns=["is_weekend"])

        logging.info("  Added cyclical encoding for hour, day_of_week, month")
        return df

    # ══════════════════════════════════════════════════════════
    # TARGET VARIABLE
    # ══════════════════════════════════════════════════════════

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the target column: PM2.5 value N hours ahead.

        target = pm25 shifted backward by horizon_hours (default: 6).
        This means for each row at time t, target = pm25(t + 6h).

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with 'target' column added.
        """
        if self.primary_param not in df.columns:
            logging.warning(f"{self.primary_param} not in DataFrame — cannot create target")
            return df

        df = df.copy()
        df = df.sort_values(["location_id", "timestamp"])

        # Shift PM2.5 backward by horizon hours to get future value
        df["target"] = df.groupby("location_id")[self.primary_param].shift(
            -self.horizon
        )

        valid_targets = df["target"].notna().sum()
        total = len(df)
        logging.info(
            f"  Target created: {self.primary_param} +{self.horizon}h ahead, "
            f"{valid_targets}/{total} valid ({valid_targets/total*100:.1f}%)"
        )

        return df

    # ══════════════════════════════════════════════════════════
    # SPATIAL FEATURES
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def compute_distance_matrix(stations_meta: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise distance matrix between stations.

        Uses Haversine formula for geographic distance in kilometers.

        Args:
            stations_meta: DataFrame with location_id, latitude, longitude.

        Returns:
            Square DataFrame with station distances in km.
        """
        coords = stations_meta[["latitude", "longitude"]].values
        # Convert to radians for Haversine
        coords_rad = np.radians(coords)

        # Haversine distance matrix
        n = len(coords_rad)
        R = 6371  # Earth radius in km

        lat1 = coords_rad[:, 0].reshape(-1, 1)
        lat2 = coords_rad[:, 0].reshape(1, -1)
        lon1 = coords_rad[:, 1].reshape(-1, 1)
        lon2 = coords_rad[:, 1].reshape(1, -1)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = R * c

        loc_ids = stations_meta["location_id"].values
        dist_df = pd.DataFrame(distances, index=loc_ids, columns=loc_ids)

        logging.info(
            f"Distance matrix: {n}x{n} stations, "
            f"min={distances[distances > 0].min():.2f} km, "
            f"max={distances.max():.2f} km"
        )

        return dist_df

    @staticmethod
    def compute_spatial_weights(
        distance_matrix: pd.DataFrame,
        bandwidth: float = 5.0,
    ) -> pd.DataFrame:
        """Compute distance-weighted spatial influence matrix.

        Uses Gaussian kernel: w(d) = exp(-d² / (2 * bandwidth²))
        Self-weight (diagonal) is set to 0.

        Args:
            distance_matrix: Station pairwise distances in km.
            bandwidth: Gaussian kernel bandwidth in km. Controls
                       how quickly influence decays with distance.

        Returns:
            Square DataFrame with spatial weights (0 to 1).
        """
        distances = distance_matrix.values
        weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))

        # Zero out self-influence
        np.fill_diagonal(weights, 0)

        # Row-normalize so weights sum to 1
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid div by zero
        weights = weights / row_sums

        weight_df = pd.DataFrame(
            weights,
            index=distance_matrix.index,
            columns=distance_matrix.columns,
        )

        logging.info(
            f"Spatial weights computed: bandwidth={bandwidth}km, "
            f"avg non-zero weight={weights[weights > 0].mean():.4f}"
        )

        return weight_df
