"""
Data Ingestion Pipeline — orchestrates the full data collection workflow.

Ties together OpenAQ station discovery, measurement fetching, weather data
retrieval, quality filtering, validation, and Parquet persistence into a
single run() call.

Supports both full-refresh and incremental modes.

Usage:
    from src.data.ingestion_pipeline import DataIngestionPipeline

    pipeline = DataIngestionPipeline()
    results = pipeline.run()
    # results = {"stations": df, "measurements": df, "weather": df}
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.logger import logging
from src.config_loader import get_config
from src.data.fetch_openaq import OpenAQClient
from src.data.fetch_weather import OpenMeteoClient
from src.data.validators import DataValidator


class DataIngestionPipeline:
    """Orchestrates the full data ingestion workflow.

    Steps:
        1. Load config & secrets
        2. Discover & filter stations → saves stations.parquet
        3. Fetch hourly measurements → saves measurements.parquet
           (incremental by default)
        4. Post-fetch station re-filter (actual completeness check)
        5. Fetch weather data → saves weather.parquet
        6. Validate all datasets
        7. Log summary statistics

    Args:
        config_path: Path to config YAML. Defaults to configs/config.yaml.
        secrets_path: Path to secrets JSON. Defaults to secrets.json.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        secrets_path: Optional[str] = None,
    ):
        self.config = get_config(
            config_path=config_path,
            secrets_path=secrets_path,
        )
        self.paths = self.config["paths"]

        # Output file paths
        self.stations_path = os.path.join(self.paths["raw_dir"], "stations.parquet")
        self.measurements_path = os.path.join(self.paths["raw_dir"], "measurements.parquet")
        self.weather_path = os.path.join(self.paths["raw_dir"], "weather.parquet")

        # Initialize clients
        self.openaq_client = OpenAQClient(self.config)
        self.weather_client = OpenMeteoClient(self.config)
        self.validator = DataValidator(self.config)

        logging.info("DataIngestionPipeline initialized")

    def run(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Execute the full ingestion pipeline.

        Args:
            force_refresh: If True, re-download everything from scratch.
                           If False (default), use incremental mode for
                           measurements (only fetch new data since last pull).

        Returns:
            Dictionary with keys "stations", "measurements", "weather",
            each mapping to the corresponding DataFrame.
        """
        logging.info(
            f"{'=' * 60}\n"
            f"  DATA INGESTION PIPELINE — {'FULL REFRESH' if force_refresh else 'INCREMENTAL'}\n"
            f"  Date range: {self.config['openaq']['date_from']} to {self.config['openaq']['date_to']}\n"
            f"{'=' * 60}"
        )

        try:
            # ── Step 1: Discover stations ────────────────────
            logging.info("Step 1/6: Discovering stations...")
            stations_df = self.openaq_client.discover_stations()
            self.validator.validate_stations(stations_df)

            # ── Step 2: Quality filter stations ──────────────
            logging.info("Step 2/6: Filtering stations (6-tier quality gate)...")
            stations_filtered = self.openaq_client.filter_stations(stations_df)

            if stations_filtered.empty:
                logging.error(
                    "No stations passed quality filters. "
                    "Consider relaxing filter thresholds in config.yaml."
                )
                return {
                    "stations": stations_filtered,
                    "measurements": pd.DataFrame(),
                    "weather": pd.DataFrame(),
                }

            self.openaq_client.save_stations(stations_filtered, self.stations_path)

            # ── Step 3: Fetch measurements ───────────────────
            logging.info("Step 3/6: Fetching hourly measurements...")
            existing_measurements = None
            if not force_refresh:
                existing_measurements = self.openaq_client.load_existing(
                    self.measurements_path
                )

            measurements_df = self.openaq_client.fetch_all_measurements(
                stations_filtered,
                existing_df=existing_measurements,
            )

            if measurements_df.empty:
                logging.error("No measurements fetched — check API connectivity and sensor IDs")
            else:
                # ── Step 4: Post-fetch re-filter ─────────────
                logging.info("Step 4/6: Post-fetch quality re-filtering...")
                stations_filtered, measurements_df = self.openaq_client.post_fetch_filter(
                    stations_filtered, measurements_df
                )

                # Re-save after post-fetch filtering
                self.openaq_client.save_stations(stations_filtered, self.stations_path)
                self.openaq_client.save_measurements(measurements_df, self.measurements_path)

                # Validate measurements
                self.validator.validate_measurements(measurements_df)

            # ── Step 5: Fetch weather data ───────────────────
            logging.info("Step 5/6: Fetching weather data...")
            weather_df = self._fetch_weather(force_refresh)

            # ── Step 6: Summary ──────────────────────────────
            logging.info("Step 6/6: Pipeline summary")
            self._log_summary(stations_filtered, measurements_df, weather_df)

            return {
                "stations": stations_filtered,
                "measurements": measurements_df,
                "weather": weather_df,
            }

        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            self.openaq_client.close()
            self.weather_client.close()

    def _fetch_weather(self, force_refresh: bool) -> pd.DataFrame:
        """Fetch weather data with incremental support.

        Args:
            force_refresh: If True, re-download all weather data.

        Returns:
            Weather DataFrame.
        """
        existing_weather = None
        if not force_refresh:
            existing_weather = self.weather_client.load_existing(self.weather_path)

        if existing_weather is not None and not existing_weather.empty:
            # Check if we have all the data we need
            last_date = existing_weather.index.max()
            target_end = pd.Timestamp(self.config["weather"]["date_to"])

            if last_date >= target_end:
                logging.info("Weather data already up to date")
                return existing_weather

            # Fetch only the missing range
            new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_weather = self.weather_client.fetch_historical(
                date_from=new_start,
                date_to=self.config["weather"]["date_to"],
            )

            if not new_weather.empty:
                weather_df = pd.concat([existing_weather, new_weather])
                weather_df = weather_df[~weather_df.index.duplicated(keep="first")]
                weather_df = weather_df.sort_index()
            else:
                weather_df = existing_weather
        else:
            weather_df = self.weather_client.fetch_historical()

        if not weather_df.empty:
            self.weather_client.save_weather(weather_df, self.weather_path)
            self.validator.validate_weather(weather_df)

        return weather_df

    def _log_summary(
        self,
        stations_df: pd.DataFrame,
        measurements_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> None:
        """Log a human-readable summary of the pipeline results.

        Args:
            stations_df: Final filtered stations.
            measurements_df: Final measurements.
            weather_df: Final weather data.
        """
        summary_lines = [
            "\n" + "=" * 60,
            "  INGESTION PIPELINE COMPLETE",
            "=" * 60,
        ]

        # Stations summary
        if not stations_df.empty:
            n_locations = stations_df["location_id"].nunique()
            n_sensors = len(stations_df)
            params = stations_df["parameter"].value_counts().to_dict()
            summary_lines.append(f"  Stations: {n_locations} locations, {n_sensors} sensors")
            summary_lines.append(f"  Parameters: {params}")
        else:
            summary_lines.append("  Stations: NONE")

        # Measurements summary
        if not measurements_df.empty:
            n_records = len(measurements_df)
            date_range = (
                f"{measurements_df['timestamp'].min()} to "
                f"{measurements_df['timestamp'].max()}"
            )
            n_sensors_with_data = measurements_df["sensor_id"].nunique()
            summary_lines.append(f"  Measurements: {n_records:,} records")
            summary_lines.append(f"  Date range: {date_range}")
            summary_lines.append(f"  Sensors with data: {n_sensors_with_data}")
        else:
            summary_lines.append("  Measurements: NONE")

        # Weather summary
        if not weather_df.empty:
            summary_lines.append(
                f"  Weather: {len(weather_df):,} hourly records, "
                f"{weather_df.index.min()} to {weather_df.index.max()}"
            )
            summary_lines.append(f"  Weather columns: {list(weather_df.columns)}")
        else:
            summary_lines.append("  Weather: NONE")

        # File sizes
        for label, path in [
            ("stations", self.stations_path),
            ("measurements", self.measurements_path),
            ("weather", self.weather_path),
        ]:
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                summary_lines.append(f"  File: {path} ({size_mb:.2f} MB)")

        summary_lines.append("=" * 60)
        summary = "\n".join(summary_lines)
        logging.info(summary)
        print(summary)
