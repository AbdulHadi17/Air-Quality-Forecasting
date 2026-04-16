"""
Data validation framework for the AQI Forecasting project.

Provides post-fetch validation checks for stations, measurements, and
weather data before persisting to Parquet. Flags issues without
silently dropping data — the pipeline decides what to do with the report.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.logger import logging


@dataclass
class ValidationReport:
    """Container for validation results.

    Attributes:
        dataset_name: Human-readable name of the dataset being validated.
        is_valid: True if no critical errors were found.
        total_rows: Number of rows in the dataset.
        null_counts: Per-column null/NaN counts.
        outlier_counts: Per-column outlier counts (values outside physical range).
        warnings: List of human-readable warning strings.
        errors: List of critical error strings.
    """
    dataset_name: str
    is_valid: bool = True
    total_rows: int = 0
    null_counts: dict = field(default_factory=dict)
    outlier_counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def log_summary(self) -> None:
        """Log a human-readable summary of the validation report."""
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        logging.info(f"Validation [{self.dataset_name}]: {status} — {self.total_rows} rows")

        for w in self.warnings:
            logging.warning(f"  ⚠ {w}")
        for e in self.errors:
            logging.error(f"  ✗ {e}")

        if self.null_counts:
            null_summary = {k: v for k, v in self.null_counts.items() if v > 0}
            if null_summary:
                logging.info(f"  Null counts: {null_summary}")

        if self.outlier_counts:
            outlier_summary = {k: v for k, v in self.outlier_counts.items() if v > 0}
            if outlier_summary:
                logging.info(f"  Outliers flagged: {outlier_summary}")

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"ValidationReport({self.dataset_name}): {status}, {self.total_rows} rows"]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


class DataValidator:
    """Validates fetched datasets against physical bounds and completeness rules.

    Args:
        config: Project configuration dictionary (from get_config()).
    """

    def __init__(self, config: dict):
        self.config = config
        self.bbox = config["openaq"]["bbox"]
        self.value_ranges = config["stations"].get("value_range", {})

    def validate_stations(self, df: pd.DataFrame) -> ValidationReport:
        """Validate the stations DataFrame.

        Checks:
            - Non-empty DataFrame
            - Required columns present
            - Valid lat/lon within Lahore bounding box
            - No duplicate location IDs

        Args:
            df: Stations DataFrame with columns:
                location_id, name, latitude, longitude, sensor_id, parameter, etc.

        Returns:
            ValidationReport with findings.
        """
        report = ValidationReport(dataset_name="stations", total_rows=len(df))

        # Critical: empty dataset
        if df.empty:
            report.is_valid = False
            report.errors.append("Stations DataFrame is empty — no stations found in bbox")
            report.log_summary()
            return report

        # Required columns
        required_cols = ["location_id", "latitude", "longitude"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            report.is_valid = False
            report.errors.append(f"Missing required columns: {missing}")
            report.log_summary()
            return report

        # Null checks
        report.null_counts = df.isnull().sum().to_dict()

        # Geographic bounds check
        min_lon, min_lat, max_lon, max_lat = self.bbox
        out_of_bounds = df[
            (df["latitude"] < min_lat) | (df["latitude"] > max_lat) |
            (df["longitude"] < min_lon) | (df["longitude"] > max_lon)
        ]
        if len(out_of_bounds) > 0:
            report.warnings.append(
                f"{len(out_of_bounds)} stations have coordinates outside Lahore bbox"
            )

        # Duplicate location IDs
        if "location_id" in df.columns:
            dup_locs = df["location_id"].duplicated().sum()
            if dup_locs > 0:
                report.warnings.append(
                    f"{dup_locs} duplicate location_id entries (expected if multi-sensor)"
                )

        report.log_summary()
        return report

    def validate_measurements(self, df: pd.DataFrame) -> ValidationReport:
        """Validate the measurements DataFrame.

        Checks:
            - Non-empty DataFrame
            - No future timestamps
            - Values within physical range per parameter
            - Flags outliers (>3σ) without removing

        Args:
            df: Measurements DataFrame with columns:
                location_id, sensor_id, parameter, timestamp, value.

        Returns:
            ValidationReport with findings.
        """
        report = ValidationReport(dataset_name="measurements", total_rows=len(df))

        if df.empty:
            report.is_valid = False
            report.errors.append("Measurements DataFrame is empty")
            report.log_summary()
            return report

        # Null checks
        report.null_counts = df.isnull().sum().to_dict()
        null_values = df["value"].isnull().sum() if "value" in df.columns else 0
        if null_values > 0:
            pct = (null_values / len(df)) * 100
            report.warnings.append(f"{null_values} null values ({pct:.1f}%)")

        # Future timestamp check
        if "timestamp" in df.columns:
            future = df[df["timestamp"] > pd.Timestamp.now(tz="UTC")]
            if len(future) > 0:
                report.warnings.append(f"{len(future)} measurements have future timestamps")

        # Physical range checks per parameter
        if "value" in df.columns and "parameter" in df.columns:
            for param, (lo, hi) in self.value_ranges.items():
                mask = df["parameter"] == param
                if mask.sum() == 0:
                    continue

                param_values = df.loc[mask, "value"].dropna()
                out_of_range = ((param_values < lo) | (param_values > hi)).sum()
                report.outlier_counts[param] = int(out_of_range)

                if out_of_range > 0:
                    pct = (out_of_range / len(param_values)) * 100
                    report.warnings.append(
                        f"{param}: {out_of_range} values outside [{lo}, {hi}] ({pct:.1f}%)"
                    )

            # Statistical outlier flagging (>3σ) per parameter
            for param in df["parameter"].unique():
                param_values = df.loc[df["parameter"] == param, "value"].dropna()
                if len(param_values) < 10:
                    continue
                mean = param_values.mean()
                std = param_values.std()
                if std > 0:
                    outliers_3sigma = ((param_values - mean).abs() > 3 * std).sum()
                    if outliers_3sigma > 0:
                        report.warnings.append(
                            f"{param}: {outliers_3sigma} values >3σ from mean "
                            f"(mean={mean:.1f}, std={std:.1f})"
                        )

        report.log_summary()
        return report

    def validate_weather(self, df: pd.DataFrame) -> ValidationReport:
        """Validate the weather DataFrame.

        Checks:
            - Non-empty DataFrame
            - Expected hourly frequency (no large gaps)
            - Reasonable physical ranges for weather variables

        Args:
            df: Weather DataFrame with timestamp index and weather columns.

        Returns:
            ValidationReport with findings.
        """
        report = ValidationReport(dataset_name="weather", total_rows=len(df))

        if df.empty:
            report.is_valid = False
            report.errors.append("Weather DataFrame is empty")
            report.log_summary()
            return report

        # Null checks
        report.null_counts = df.isnull().sum().to_dict()

        # Physical range checks for weather variables
        weather_ranges = {
            "temperature_2m": (-10, 55),          # °C — Lahore range
            "relative_humidity_2m": (0, 100),      # %
            "wind_speed_10m": (0, 50),             # m/s
            "wind_direction_10m": (0, 360),        # degrees
            "pressure_msl": (950, 1060),           # hPa
            "precipitation": (0, 200),             # mm
            "cloud_cover": (0, 100),               # %
        }

        for col, (lo, hi) in weather_ranges.items():
            if col not in df.columns:
                continue
            values = df[col].dropna()
            out_of_range = ((values < lo) | (values > hi)).sum()
            report.outlier_counts[col] = int(out_of_range)
            if out_of_range > 0:
                report.warnings.append(
                    f"{col}: {out_of_range} values outside [{lo}, {hi}]"
                )

        # Hourly completeness check
        if isinstance(df.index, pd.DatetimeIndex):
            expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
            actual_hours = len(df)
            completeness = (actual_hours / expected_hours) * 100 if expected_hours > 0 else 0
            if completeness < 95:
                report.warnings.append(
                    f"Hourly completeness: {completeness:.1f}% "
                    f"({actual_hours}/{int(expected_hours)} hours)"
                )

        report.log_summary()
        return report
