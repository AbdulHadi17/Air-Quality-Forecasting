"""
Data preprocessing module for the AQI Forecasting project.

Takes raw Parquet files (stations, measurements, weather) from the ingestion
layer and produces a clean, merged, analysis-ready dataset.

Pipeline:
    1. Load raw data from Parquet
    2. Pivot measurements: (sensor_id, timestamp, value) → wide-format per location
    3. Remove physical-range outliers
    4. Interpolate missing hourly values (gaps ≤ 6 hours)
    5. Merge AQ data with weather data on timestamp
    6. Add time-based metadata columns
    7. Save to data/processed/
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.logger import logging
from src.config_loader import get_config


class DataPreprocessor:
    """Cleans, interpolates, and merges raw AQ + weather data.

    Args:
        config: Project configuration dict. If None, loads from default path.
    """

    # Maximum gap (in hours) to interpolate across
    MAX_INTERPOLATION_GAP = 6

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.paths = self.config["paths"]
        self.value_ranges = self.config["stations"].get("value_range", {})
        self.primary_param = self.config["openaq"].get("primary_parameter", "pm25")

        # File paths
        self.raw_dir = self.paths["raw_dir"]
        self.processed_dir = self.paths["processed_dir"]
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        logging.info("DataPreprocessor initialized")

    # ══════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════

    def run(self) -> dict[str, pd.DataFrame]:
        """Execute the full preprocessing pipeline.

        Returns:
            Dictionary with keys:
                - "station_ts": Per-station time-series DataFrame
                - "merged": Merged AQ + weather DataFrame (analysis-ready)
                - "stations_meta": Station metadata DataFrame
        """
        logging.info("=" * 60)
        logging.info("  PREPROCESSING PIPELINE")
        logging.info("=" * 60)

        # Step 1: Load raw data
        stations_df, measurements_df, weather_df = self._load_raw_data()

        # Step 2: Build per-station time-series (pivot from long to wide)
        logging.info("Step 1/5: Building per-station time-series...")
        station_ts = self._build_station_timeseries(measurements_df, stations_df)

        # Step 3: Clean outliers
        logging.info("Step 2/5: Removing physical-range outliers...")
        station_ts = self._remove_outliers(station_ts)

        # Step 4: Interpolate gaps
        logging.info("Step 3/5: Interpolating missing values...")
        station_ts = self._interpolate_gaps(station_ts)

        # Step 5: Merge with weather
        logging.info("Step 4/5: Merging with weather data...")
        merged = self._merge_with_weather(station_ts, weather_df)

        # Step 6: Add time features
        logging.info("Step 5/5: Adding temporal metadata...")
        merged = self._add_time_features(merged)

        # Build station metadata
        stations_meta = self._build_station_metadata(stations_df, station_ts)

        # Save
        self._save_processed(station_ts, merged, stations_meta)

        # Summary
        self._log_summary(station_ts, merged, stations_meta)

        return {
            "station_ts": station_ts,
            "merged": merged,
            "stations_meta": stations_meta,
        }

    # ══════════════════════════════════════════════════════════
    # STEP 1: LOAD RAW DATA
    # ══════════════════════════════════════════════════════════

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw Parquet files from data/raw/.

        Returns:
            Tuple of (stations_df, measurements_df, weather_df).

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        stations_path = os.path.join(self.raw_dir, "stations.parquet")
        measurements_path = os.path.join(self.raw_dir, "measurements.parquet")
        weather_path = os.path.join(self.raw_dir, "weather.parquet")

        for path, name in [
            (stations_path, "stations"),
            (measurements_path, "measurements"),
            (weather_path, "weather"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{name}.parquet not found at {path}. "
                    f"Run the ingestion pipeline first."
                )

        stations_df = pd.read_parquet(stations_path)
        measurements_df = pd.read_parquet(measurements_path)
        weather_df = pd.read_parquet(weather_path)

        logging.info(
            f"Loaded raw data: {len(stations_df)} stations, "
            f"{len(measurements_df)} measurements, {len(weather_df)} weather records"
        )
        return stations_df, measurements_df, weather_df

    # ══════════════════════════════════════════════════════════
    # STEP 2: BUILD PER-STATION TIME-SERIES
    # ══════════════════════════════════════════════════════════

    def _build_station_timeseries(
        self,
        measurements_df: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pivot measurements from long-format to station-indexed time-series.

        Input format (long):
            location_id | sensor_id | parameter | timestamp | value

        Output format (wide, multi-indexed):
            timestamp | location_id | pm25 | pm10 | temperature | ...
            With columns for each parameter available at that station.

        Also resamples to regular hourly frequency per station to expose gaps.

        Args:
            measurements_df: Raw measurements DataFrame.
            stations_df: Station metadata for sensor→location mapping.

        Returns:
            Wide-format DataFrame indexed by (timestamp, location_id).
        """
        if measurements_df.empty:
            logging.warning("Empty measurements DataFrame")
            return pd.DataFrame()

        # Ensure timestamp is proper datetime
        measurements_df = measurements_df.copy()
        measurements_df["timestamp"] = pd.to_datetime(
            measurements_df["timestamp"], utc=True
        )

        # Floor timestamps to exact hours to align readings
        measurements_df["timestamp"] = measurements_df["timestamp"].dt.floor("h")

        # Drop duplicate (location_id, parameter, timestamp) keeping last
        measurements_df = measurements_df.drop_duplicates(
            subset=["location_id", "parameter", "timestamp"],
            keep="last",
        )

        # Pivot: one column per parameter
        pivoted = measurements_df.pivot_table(
            index=["timestamp", "location_id"],
            columns="parameter",
            values="value",
            aggfunc="mean",  # Average if multiple readings in same hour
        )

        # Flatten column names
        pivoted.columns = [str(c) for c in pivoted.columns]
        pivoted = pivoted.reset_index()

        # Resample each station to regular hourly frequency
        # This inserts NaN rows for missing hours, making gaps explicit
        date_from = pd.Timestamp(
            self.config["openaq"]["date_from"], tz="UTC"
        )
        date_to = pd.Timestamp(
            self.config["openaq"]["date_to"], tz="UTC"
        )
        full_hourly_index = pd.date_range(
            start=date_from, end=date_to, freq="h", tz="UTC"
        )

        resampled_dfs = []
        for loc_id in pivoted["location_id"].unique():
            loc_data = pivoted[pivoted["location_id"] == loc_id].copy()
            loc_data = loc_data.set_index("timestamp")
            loc_data = loc_data.drop(columns=["location_id"], errors="ignore")

            # Reindex to full hourly range
            loc_data = loc_data.reindex(full_hourly_index)
            loc_data.index.name = "timestamp"
            loc_data["location_id"] = loc_id

            resampled_dfs.append(loc_data.reset_index())

        result = pd.concat(resampled_dfs, ignore_index=True)

        n_stations = result["location_id"].nunique()
        n_records = len(result)
        params = [c for c in result.columns if c not in ["timestamp", "location_id"]]
        logging.info(
            f"Built time-series: {n_stations} stations, {n_records} rows, "
            f"parameters: {params}"
        )

        return result

    # ══════════════════════════════════════════════════════════
    # STEP 3: REMOVE OUTLIERS
    # ══════════════════════════════════════════════════════════

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace values outside physical range with NaN.

        Uses the value_range config (e.g., pm25: [0, 1000]) to cap values.
        Also replaces negative values with NaN for all AQ parameters.

        Args:
            df: Station time-series DataFrame.

        Returns:
            DataFrame with outliers replaced by NaN.
        """
        df = df.copy()
        total_replaced = 0

        for param, (lo, hi) in self.value_ranges.items():
            if param in df.columns:
                mask = (df[param] < lo) | (df[param] > hi)
                count = mask.sum()
                if count > 0:
                    df.loc[mask, param] = np.nan
                    total_replaced += count
                    logging.info(f"  {param}: replaced {count} out-of-range values with NaN")

        # Also replace negative values for any remaining numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aq_cols = [c for c in numeric_cols if c not in ["location_id"]]
        for col in aq_cols:
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                df.loc[neg_mask, col] = np.nan
                total_replaced += neg_count

        logging.info(f"Outlier removal: {total_replaced} total values replaced with NaN")
        return df

    # ══════════════════════════════════════════════════════════
    # STEP 4: INTERPOLATE GAPS
    # ══════════════════════════════════════════════════════════

    def _interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values for gaps of ≤ MAX_INTERPOLATION_GAP hours.

        Uses linear interpolation within each station. Larger gaps are left
        as NaN — the sequence builder will handle them by splitting sequences.

        Args:
            df: Station time-series DataFrame with NaN gaps.

        Returns:
            DataFrame with small gaps interpolated.
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aq_cols = [c for c in numeric_cols if c not in ["location_id"]]

        total_filled = 0

        for loc_id in df["location_id"].unique():
            loc_mask = df["location_id"] == loc_id

            for col in aq_cols:
                before_nulls = df.loc[loc_mask, col].isnull().sum()

                # Interpolate with limit to avoid bridging large gaps
                df.loc[loc_mask, col] = (
                    df.loc[loc_mask, col]
                    .interpolate(method="linear", limit=self.MAX_INTERPOLATION_GAP)
                )

                after_nulls = df.loc[loc_mask, col].isnull().sum()
                filled = before_nulls - after_nulls
                total_filled += filled

        logging.info(
            f"Interpolation: filled {total_filled} values "
            f"(max gap: {self.MAX_INTERPOLATION_GAP} hours)"
        )

        # Report remaining nulls
        null_report = {}
        for col in aq_cols:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                pct = (nulls / len(df)) * 100
                null_report[col] = f"{nulls} ({pct:.1f}%)"
        if null_report:
            logging.info(f"Remaining NaN after interpolation: {null_report}")

        return df

    # ══════════════════════════════════════════════════════════
    # STEP 5: MERGE WITH WEATHER
    # ══════════════════════════════════════════════════════════

    def _merge_with_weather(
        self,
        station_ts: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge station time-series with weather data on timestamp.

        Weather is a single time-series for Lahore's centroid, broadcast
        to all stations (same weather for all stations in the 30km bbox).

        Args:
            station_ts: Per-station time-series DataFrame.
            weather_df: Weather DataFrame with timestamp index.

        Returns:
            Merged DataFrame with both AQ and weather columns.
        """
        if weather_df.empty:
            logging.warning("Empty weather DataFrame — skipping merge")
            return station_ts

        # Prepare weather: ensure timestamp column (not index)
        weather = weather_df.copy()
        if isinstance(weather.index, pd.DatetimeIndex):
            weather = weather.reset_index()
            # Rename 'timestamp' or first column to 'timestamp' for join
            if weather.columns[0] != "timestamp":
                weather = weather.rename(columns={weather.columns[0]: "timestamp"})

        # Ensure UTC timezone consistency
        weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
        weather["timestamp"] = weather["timestamp"].dt.floor("h")

        # Drop duplicate timestamps in weather (keep first)
        weather = weather.drop_duplicates(subset=["timestamp"], keep="first")

        # Merge: left join to keep all station rows
        merged = station_ts.merge(weather, on="timestamp", how="left")

        weather_cols = [c for c in weather.columns if c != "timestamp"]
        weather_coverage = merged[weather_cols[0]].notna().mean() * 100 if weather_cols else 0
        logging.info(
            f"Merged AQ + weather: {len(merged)} rows, "
            f"weather coverage: {weather_coverage:.1f}%"
        )

        return merged

    # ══════════════════════════════════════════════════════════
    # STEP 6: ADD TIME FEATURES
    # ══════════════════════════════════════════════════════════

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal metadata columns for downstream feature engineering.

        Adds:
            - hour (0-23)
            - day_of_week (0=Mon, 6=Sun)
            - month (1-12)
            - is_weekend (bool)

        Args:
            df: Merged DataFrame with timestamp column.

        Returns:
            DataFrame with added time columns.
        """
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"], utc=True)

        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

        logging.info("Added temporal features: hour, day_of_week, month, is_weekend")
        return df

    # ══════════════════════════════════════════════════════════
    # STATION METADATA
    # ══════════════════════════════════════════════════════════

    def _build_station_metadata(
        self,
        stations_df: pd.DataFrame,
        station_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build enriched station metadata with data quality stats.

        Args:
            stations_df: Raw station metadata from ingestion.
            station_ts: Processed per-station time-series.

        Returns:
            DataFrame with one row per location containing:
                location_id, name, latitude, longitude,
                total_hours, valid_pm25_hours, pm25_completeness_pct
        """
        if stations_df.empty or station_ts.empty:
            return pd.DataFrame()

        # Base metadata (one row per location)
        meta = stations_df.drop_duplicates(subset=["location_id"])[
            ["location_id", "name", "latitude", "longitude"]
        ].copy()

        # Compute per-station data quality stats
        quality_stats = []
        for loc_id in meta["location_id"]:
            loc_data = station_ts[station_ts["location_id"] == loc_id]
            total_hours = len(loc_data)

            pm25_valid = 0
            pm25_pct = 0.0
            if self.primary_param in loc_data.columns:
                pm25_valid = loc_data[self.primary_param].notna().sum()
                pm25_pct = (pm25_valid / total_hours * 100) if total_hours > 0 else 0.0

            quality_stats.append({
                "location_id": loc_id,
                "total_hours": total_hours,
                f"valid_{self.primary_param}_hours": pm25_valid,
                f"{self.primary_param}_completeness_pct": round(pm25_pct, 1),
            })

        quality_df = pd.DataFrame(quality_stats)
        meta = meta.merge(quality_df, on="location_id", how="left")

        logging.info(f"Built station metadata: {len(meta)} locations")
        return meta

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def _save_processed(
        self,
        station_ts: pd.DataFrame,
        merged: pd.DataFrame,
        stations_meta: pd.DataFrame,
    ) -> None:
        """Save processed DataFrames to Parquet.

        Outputs:
            data/processed/station_timeseries.parquet
            data/processed/merged_aq_weather.parquet
            data/processed/stations_metadata.parquet
        """
        if not station_ts.empty:
            path = os.path.join(self.processed_dir, "station_timeseries.parquet")
            station_ts.to_parquet(path, index=False, engine="pyarrow")
            logging.info(f"Saved station time-series to {path}")

        if not merged.empty:
            path = os.path.join(self.processed_dir, "merged_aq_weather.parquet")
            merged.to_parquet(path, index=False, engine="pyarrow")
            logging.info(f"Saved merged AQ+weather to {path}")

        if not stations_meta.empty:
            path = os.path.join(self.processed_dir, "stations_metadata.parquet")
            stations_meta.to_parquet(path, index=False, engine="pyarrow")
            logging.info(f"Saved station metadata to {path}")

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════

    def _log_summary(
        self,
        station_ts: pd.DataFrame,
        merged: pd.DataFrame,
        stations_meta: pd.DataFrame,
    ) -> None:
        """Log a human-readable summary."""
        lines = [
            "\n" + "=" * 60,
            "  PREPROCESSING COMPLETE",
            "=" * 60,
        ]

        if not station_ts.empty:
            n_stations = station_ts["location_id"].nunique()
            params = [c for c in station_ts.columns
                      if c not in ["timestamp", "location_id"]]
            lines.append(f"  Stations: {n_stations}")
            lines.append(f"  AQ parameters: {params}")
            lines.append(f"  Time-series rows: {len(station_ts):,}")

        if not merged.empty:
            lines.append(f"  Merged rows: {len(merged):,}")
            lines.append(f"  Total columns: {len(merged.columns)}")
            ts = merged["timestamp"]
            lines.append(f"  Date range: {ts.min()} to {ts.max()}")

        if not stations_meta.empty and f"{self.primary_param}_completeness_pct" in stations_meta.columns:
            avg_completeness = stations_meta[f"{self.primary_param}_completeness_pct"].mean()
            lines.append(f"  Avg PM2.5 completeness: {avg_completeness:.1f}%")

        # Null summary for primary parameter
        if not merged.empty and self.primary_param in merged.columns:
            null_pct = merged[self.primary_param].isnull().mean() * 100
            lines.append(f"  PM2.5 null rate (after interpolation): {null_pct:.1f}%")

        # File sizes
        for fname in ["station_timeseries.parquet", "merged_aq_weather.parquet",
                       "stations_metadata.parquet"]:
            fpath = os.path.join(self.processed_dir, fname)
            if Path(fpath).exists():
                size_mb = Path(fpath).stat().st_size / (1024 * 1024)
                lines.append(f"  {fname}: {size_mb:.2f} MB")

        lines.append("=" * 60)
        summary = "\n".join(lines)
        logging.info(summary)
        print(summary)
