"""
OpenAQ v3 API client for the Lahore AQI Forecasting project.

Handles:
    - Station discovery within the Lahore bounding box
    - Multi-tier quality filtering to keep only reliable stations
    - Hourly measurement fetching per sensor (paginated, rate-limited)
    - Incremental data updates (only fetch new data since last pull)
    - Parquet persistence

Uses the official openaq Python SDK (v0.7) for station discovery and
direct httpx calls for bulk measurement fetching.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from openaq import OpenAQ

from src.logger import logging
from src.utils import retry_with_backoff


class OpenAQClient:
    """Production-grade OpenAQ v3 API client for Lahore station data.

    Args:
        config: Project configuration dict (from get_config()).
    """

    BASE_URL = "https://api.openaq.org/v3"

    def __init__(self, config: dict):
        self.config = config
        self.openaq_config = config["openaq"]
        self.stations_config = config["stations"]
        self.api_key = self.openaq_config.get("api_key", "")
        self.bbox = self.openaq_config["bbox"]
        self.rate_limit_delay = self.openaq_config.get("rate_limit_delay", 0.25)
        self.max_retries = self.openaq_config.get("max_retries", 3)
        self.chunk_days = self.openaq_config.get("chunk_days", 90)

        # HTTP client for direct API calls (measurements)
        self._http = httpx.Client(
            base_url=self.BASE_URL,
            headers={"X-API-Key": self.api_key} if self.api_key else {},
            timeout=30.0,
        )

        logging.info(
            f"OpenAQClient initialized — bbox: {self.bbox}, "
            f"date range: {self.openaq_config['date_from']} to {self.openaq_config['date_to']}"
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    # ══════════════════════════════════════════════════════════
    # STATION DISCOVERY
    # ══════════════════════════════════════════════════════════

    def discover_stations(self) -> pd.DataFrame:
        """Discover all air quality stations within the Lahore bounding box.

        Uses the OpenAQ Python SDK (v0.7) to query locations. The SDK returns
        Location objects with attributes: id, name, coordinates, sensors,
        datetime_first, datetime_last, is_mobile, etc.

        Returns:
            DataFrame with columns:
                location_id, name, latitude, longitude, sensor_id,
                parameter, datetime_first, datetime_last, is_mobile
        """
        logging.info(f"Discovering stations in bbox: {self.bbox}")

        try:
            sdk_client = OpenAQ(api_key=self.api_key)
            # SDK requires bbox as tuple, not list
            bbox_tuple = tuple(self.bbox)
            response = sdk_client.locations.list(
                bbox=bbox_tuple,
                limit=self.openaq_config.get("limit", 1000),
            )
            sdk_client.close()
        except Exception as e:
            logging.error(f"SDK station discovery failed: {e}. Falling back to direct API call.")
            return self._discover_stations_direct()

        records = []
        # Access results via .results attribute or treat response as iterable
        locations = response.results if hasattr(response, 'results') else response

        for loc in locations:
            # Extract location-level attributes
            loc_id = loc.id
            loc_name = loc.name
            lat = loc.coordinates.latitude
            lon = loc.coordinates.longitude
            is_mobile = loc.is_mobile

            # Extract datetime_first/last from location level
            dt_first = loc.datetime_first.utc if loc.datetime_first else None
            dt_last = loc.datetime_last.utc if loc.datetime_last else None

            # Extract sensors
            sensors = loc.sensors if loc.sensors else []
            if not sensors:
                records.append({
                    "location_id": loc_id,
                    "name": loc_name,
                    "latitude": lat,
                    "longitude": lon,
                    "sensor_id": None,
                    "parameter": None,
                    "datetime_first": dt_first,
                    "datetime_last": dt_last,
                    "is_mobile": is_mobile,
                })
                continue

            for sensor in sensors:
                param_name = sensor.parameter.name if sensor.parameter else "unknown"

                records.append({
                    "location_id": loc_id,
                    "name": loc_name,
                    "latitude": lat,
                    "longitude": lon,
                    "sensor_id": sensor.id,
                    "parameter": param_name,
                    "datetime_first": dt_first,
                    "datetime_last": dt_last,
                    "is_mobile": is_mobile,
                })

        df = pd.DataFrame(records)

        if df.empty:
            logging.warning("No stations found in the bounding box")
            return df

        # Parse timestamps
        for col in ["datetime_first", "datetime_last"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        logging.info(
            f"Discovered {df['location_id'].nunique()} locations "
            f"with {len(df)} sensors total"
        )
        return df

    def _discover_stations_direct(self) -> pd.DataFrame:
        """Fallback: discover stations via direct HTTP instead of SDK."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        all_records = []
        page = 1
        limit = self.openaq_config.get("limit", 1000)

        while True:
            try:
                response = self._http.get(
                    "/locations",
                    params={"bbox": bbox_str, "limit": limit, "page": page},
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logging.error(f"Direct API station discovery failed (page {page}): {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for loc in results:
                loc_id = loc.get("id")
                loc_name = loc.get("name", "Unknown")
                coords = loc.get("coordinates", {})
                lat = coords.get("latitude", 0)
                lon = coords.get("longitude", 0)
                is_mobile = loc.get("isMobile", False)

                dt_first_raw = loc.get("datetimeFirst", {})
                dt_last_raw = loc.get("datetimeLast", {})
                dt_first = dt_first_raw.get("utc") if isinstance(dt_first_raw, dict) else dt_first_raw
                dt_last = dt_last_raw.get("utc") if isinstance(dt_last_raw, dict) else dt_last_raw

                for sensor in loc.get("sensors", []):
                    param = sensor.get("parameter", {})
                    param_name = param.get("name", "unknown") if isinstance(param, dict) else str(param)

                    all_records.append({
                        "location_id": loc_id,
                        "name": loc_name,
                        "latitude": lat,
                        "longitude": lon,
                        "sensor_id": sensor.get("id"),
                        "parameter": param_name,
                        "datetime_first": dt_first,
                        "datetime_last": dt_last,
                        "is_mobile": is_mobile,
                    })

            if len(results) < limit:
                break
            page += 1
            time.sleep(self.rate_limit_delay)

        df = pd.DataFrame(all_records)
        if not df.empty:
            for col in ["datetime_first", "datetime_last"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    # ══════════════════════════════════════════════════════════
    # STATION QUALITY FILTERING (Multi-Tier Gate)
    # ══════════════════════════════════════════════════════════

    def filter_stations(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-tier quality gate to filter stations.

        Pre-fetch tiers (using metadata only):
            1. Must have PM2.5 sensor (required_parameters)
            2. Not a mobile station
            3. Geographic sanity (coordinates within Lahore bbox)
            4. Non-null sensor ID
            5. Data recency (datetime_last within reasonable range)
            6. Data history (datetime_first early enough to have useful data)

        Post-fetch tiers (completeness, continuity) are applied separately
        via post_fetch_filter() after measurements are downloaded.

        Args:
            stations_df: Raw stations DataFrame from discover_stations().

        Returns:
            Filtered DataFrame containing only stations passing all pre-fetch tiers.
        """
        if stations_df.empty:
            logging.warning("No stations to filter (empty input)")
            return stations_df

        initial_count = stations_df["location_id"].nunique()
        initial_sensors = len(stations_df)
        df = stations_df.copy()

        # ── Tier 1: Required parameters ──────────────────────
        required = self.stations_config.get("required_parameters", ["pm25"])
        tier1_mask = df["parameter"].isin(required)
        locations_with_required = df.loc[tier1_mask, "location_id"].unique()
        dropped_t1 = df[~df["location_id"].isin(locations_with_required)]["location_id"].nunique()
        df = df[df["location_id"].isin(locations_with_required)]
        logging.info(f"  Tier 1 (required params {required}): dropped {dropped_t1} locations")

        # ── Tier 2: Not mobile ───────────────────────────────
        if "is_mobile" in df.columns:
            pre_t2 = len(df)
            df = df[~df["is_mobile"].fillna(False).astype(bool)]
            dropped_t2 = pre_t2 - len(df)
            logging.info(f"  Tier 2 (not mobile): dropped {dropped_t2} sensors")

        # ── Tier 3: Geographic bounds ────────────────────────
        min_lon, min_lat, max_lon, max_lat = self.bbox
        pre_t3 = len(df)
        df = df[
            (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat) &
            (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
        ]
        dropped_t3 = pre_t3 - len(df)
        logging.info(f"  Tier 3 (geographic bounds): dropped {dropped_t3} sensors")

        # ── Tier 4: Non-null sensor ID ───────────────────────
        pre_t4 = len(df)
        df = df[df["sensor_id"].notna()]
        dropped_t4 = pre_t4 - len(df)
        logging.info(f"  Tier 4 (non-null sensor_id): dropped {dropped_t4} sensors")

        # ── Tier 5: Data recency ─────────────────────────────
        # Station must have data at least within the configured date range
        if "datetime_last" in df.columns and df["datetime_last"].notna().any():
            config_from = pd.Timestamp(self.openaq_config["date_from"], tz="UTC")
            pre_t5 = len(df)
            # Station's last data must be AFTER the start of our target range
            df = df[df["datetime_last"] >= config_from]
            dropped_t5 = pre_t5 - len(df)
            logging.info(f"  Tier 5 (data extends into target range): dropped {dropped_t5} sensors")
        else:
            logging.info("  Tier 5 (data recency): skipped — no datetime_last data")

        # ── Tier 6: Data history ─────────────────────────────
        # Station must have started collecting before the end of our range
        if "datetime_first" in df.columns and df["datetime_first"].notna().any():
            config_to = pd.Timestamp(self.openaq_config["date_to"], tz="UTC")
            pre_t6 = len(df)
            df = df[df["datetime_first"] <= config_to]
            dropped_t6 = pre_t6 - len(df)
            logging.info(f"  Tier 6 (data started before range end): dropped {dropped_t6} sensors")
        else:
            logging.info("  Tier 6 (data history): skipped — no datetime_first data")

        final_locations = df["location_id"].nunique()
        final_sensors = len(df)
        logging.info(
            f"Pre-fetch filtering complete: {initial_count} locations ({initial_sensors} sensors) "
            f"→ {final_locations} locations ({final_sensors} sensors)"
        )

        return df.reset_index(drop=True)

    def post_fetch_filter(
        self,
        stations_df: pd.DataFrame,
        measurements_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Post-fetch re-filtering based on actual data completeness.

        After downloading measurements, compute real hourly coverage per
        station and drop those below the min_completeness_pct threshold.
        Also checks for temporal continuity (min_continuous_days).

        Args:
            stations_df: Filtered stations DataFrame.
            measurements_df: Fetched measurements DataFrame.

        Returns:
            Tuple of (filtered_stations, filtered_measurements).
        """
        if measurements_df.empty:
            return stations_df, measurements_df

        min_completeness = self.stations_config.get("min_completeness_pct", 40)
        min_continuous = self.stations_config.get("min_continuous_days", 30)

        date_from = pd.Timestamp(self.openaq_config["date_from"], tz="UTC")
        date_to = pd.Timestamp(self.openaq_config["date_to"], tz="UTC")
        total_expected_hours = (date_to - date_from).total_seconds() / 3600

        good_sensors = []
        all_sensors = measurements_df["sensor_id"].unique()

        for sensor_id in all_sensors:
            sensor_data = measurements_df[measurements_df["sensor_id"] == sensor_id]

            # ── Completeness check ─────────────────────────
            actual_hours = len(sensor_data)
            completeness = (actual_hours / total_expected_hours) * 100 if total_expected_hours > 0 else 0

            if completeness < min_completeness:
                logging.info(
                    f"  Post-filter: dropping sensor {sensor_id} — "
                    f"completeness {completeness:.1f}% < {min_completeness}%"
                )
                continue

            # ── Temporal continuity check ──────────────────
            if "timestamp" in sensor_data.columns and len(sensor_data) > 1:
                timestamps = sensor_data["timestamp"].sort_values()
                dates = sorted(timestamps.dt.date.unique())
                if len(dates) >= min_continuous:
                    max_streak = 1
                    current_streak = 1
                    for i in range(1, len(dates)):
                        if (dates[i] - dates[i - 1]).days == 1:
                            current_streak += 1
                            max_streak = max(max_streak, current_streak)
                        else:
                            current_streak = 1

                    if max_streak < min_continuous:
                        logging.info(
                            f"  Post-filter: dropping sensor {sensor_id} — "
                            f"max continuous streak {max_streak} days < {min_continuous}"
                        )
                        continue
                else:
                    logging.info(
                        f"  Post-filter: dropping sensor {sensor_id} — "
                        f"only {len(dates)} unique days < {min_continuous}"
                    )
                    continue

            good_sensors.append(sensor_id)

        dropped_count = len(all_sensors) - len(good_sensors)
        logging.info(
            f"Post-fetch filtering: {len(all_sensors)} sensors → {len(good_sensors)} "
            f"(dropped {dropped_count} for low completeness/continuity)"
        )

        # Filter both DataFrames
        measurements_filtered = measurements_df[
            measurements_df["sensor_id"].isin(good_sensors)
        ].reset_index(drop=True)

        stations_filtered = stations_df[
            stations_df["sensor_id"].isin(good_sensors)
        ].reset_index(drop=True)

        return stations_filtered, measurements_filtered

    # ══════════════════════════════════════════════════════════
    # MEASUREMENT FETCHING
    # ══════════════════════════════════════════════════════════

    def fetch_sensor_hours(
        self,
        sensor_id: int,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        """Fetch hourly aggregated measurements for a single sensor.

        Calls the /v3/sensors/{id}/hours endpoint with pagination,
        rate limiting, and retry logic.

        Args:
            sensor_id: OpenAQ sensor ID.
            date_from: Start date string (YYYY-MM-DD).
            date_to: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with columns: sensor_id, timestamp, value.
        """
        all_records = []

        # Chunk date range to avoid API timeouts
        chunks = self._chunk_date_range(date_from, date_to, self.chunk_days)

        for chunk_from, chunk_to in chunks:
            page = 1
            limit = 1000

            while True:
                try:
                    data = self._fetch_hours_page(
                        sensor_id, chunk_from, chunk_to, page, limit
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to fetch sensor {sensor_id} "
                        f"({chunk_from} to {chunk_to}, page {page}): {e}"
                    )
                    break

                results = data.get("results", [])
                if not results:
                    break

                for record in results:
                    period = record.get("period", {})
                    dt_from = period.get("datetimeFrom", {})
                    timestamp = dt_from.get("utc") if isinstance(dt_from, dict) else dt_from

                    all_records.append({
                        "sensor_id": sensor_id,
                        "timestamp": timestamp,
                        "value": record.get("value"),
                    })

                if len(results) < limit:
                    break
                page += 1
                time.sleep(self.rate_limit_delay)

            time.sleep(self.rate_limit_delay)

        df = pd.DataFrame(all_records)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException))
    def _fetch_hours_page(
        self,
        sensor_id: int,
        date_from: str,
        date_to: str,
        page: int,
        limit: int,
    ) -> dict:
        """Fetch a single page of hourly data from the API.

        Args:
            sensor_id: Sensor ID to fetch for.
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).
            page: Page number (1-indexed).
            limit: Results per page (max 1000).

        Returns:
            Raw JSON response as dict.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        response = self._http.get(
            f"/sensors/{sensor_id}/hours",
            params={
                "datetime_from": date_from,
                "datetime_to": date_to,
                "page": page,
                "limit": limit,
            },
        )
        response.raise_for_status()
        return response.json()

    def fetch_all_measurements(
        self,
        stations_df: pd.DataFrame,
        existing_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fetch hourly measurements for all sensors in filtered stations.

        Supports incremental mode: if existing_df is provided, only fetches
        data newer than the last timestamp per sensor.

        Args:
            stations_df: Filtered stations DataFrame with sensor_id and parameter columns.
            existing_df: Optional existing measurements DataFrame for incremental update.

        Returns:
            Combined DataFrame with columns:
                location_id, sensor_id, parameter, timestamp, value
        """
        if stations_df.empty:
            logging.warning("No stations to fetch measurements for")
            return pd.DataFrame()

        # Build sensor → metadata mapping
        sensors = stations_df[["location_id", "sensor_id", "parameter"]].drop_duplicates()
        total_sensors = len(sensors)

        logging.info(f"Fetching measurements for {total_sensors} sensors...")

        # Determine per-sensor start dates (for incremental)
        sensor_start_dates = {}
        default_from = self.openaq_config["date_from"]
        default_to = self.openaq_config["date_to"]

        if existing_df is not None and not existing_df.empty:
            for sid in sensors["sensor_id"].unique():
                existing_sensor = existing_df[existing_df["sensor_id"] == sid]
                if not existing_sensor.empty:
                    last_ts = existing_sensor["timestamp"].max()
                    # Start from the day after the last measurement
                    next_day = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    sensor_start_dates[sid] = next_day

        all_dfs = []
        for idx, (_, row) in enumerate(sensors.iterrows()):
            sensor_id = row["sensor_id"]
            location_id = row["location_id"]
            parameter = row["parameter"]

            date_from = sensor_start_dates.get(sensor_id, default_from)
            if date_from > default_to:
                logging.info(
                    f"  [{idx + 1}/{total_sensors}] Sensor {sensor_id} ({parameter}): "
                    f"already up to date"
                )
                continue

            logging.info(
                f"  [{idx + 1}/{total_sensors}] Sensor {sensor_id} ({parameter}): "
                f"{date_from} → {default_to}"
            )

            df = self.fetch_sensor_hours(sensor_id, date_from, default_to)

            if not df.empty:
                df["location_id"] = location_id
                df["parameter"] = parameter
                all_dfs.append(df)

            # Rate limiting between sensors
            time.sleep(self.rate_limit_delay)

        if not all_dfs:
            logging.warning("No measurements fetched for any sensor")
            return pd.DataFrame()

        new_data = pd.concat(all_dfs, ignore_index=True)

        # Merge with existing data if incremental
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_data], ignore_index=True)
            # Drop duplicates (same sensor + timestamp)
            combined = combined.drop_duplicates(
                subset=["sensor_id", "timestamp"], keep="last"
            )
            combined = combined.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
            logging.info(
                f"Incremental merge: {len(existing_df)} existing + {len(new_data)} new "
                f"→ {len(combined)} total (after dedup)"
            )
            return combined

        new_data = new_data.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
        logging.info(
            f"Fetched {len(new_data)} total measurements "
            f"across {new_data['sensor_id'].nunique()} sensors"
        )
        return new_data

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def save_stations(df: pd.DataFrame, path: str) -> None:
        """Save stations DataFrame to Parquet.

        Args:
            df: Stations DataFrame.
            path: Output file path (e.g., data/raw/stations.parquet).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        logging.info(f"Saved {len(df)} station records to {path}")

    @staticmethod
    def save_measurements(df: pd.DataFrame, path: str) -> None:
        """Save measurements DataFrame to Parquet.

        Args:
            df: Measurements DataFrame.
            path: Output file path (e.g., data/raw/measurements.parquet).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        logging.info(f"Saved {len(df)} measurement records to {path}")

    @staticmethod
    def load_existing(path: str) -> Optional[pd.DataFrame]:
        """Load existing Parquet file if it exists.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame if file exists, None otherwise.
        """
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p, engine="pyarrow")
            logging.info(f"Loaded {len(df)} existing records from {path}")
            return df
        return None

    @staticmethod
    def get_last_timestamp(path: str) -> Optional[pd.Timestamp]:
        """Get the maximum timestamp from an existing measurements Parquet.

        Args:
            path: Path to the measurements Parquet file.

        Returns:
            Maximum timestamp or None if file doesn't exist.
        """
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p, engine="pyarrow", columns=["timestamp"])
            if not df.empty:
                return df["timestamp"].max()
        return None

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _chunk_date_range(
        date_from: str,
        date_to: str,
        chunk_days: int,
    ) -> list[tuple[str, str]]:
        """Split a date range into smaller chunks.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).
            chunk_days: Maximum days per chunk.

        Returns:
            List of (chunk_from, chunk_to) string tuples.
        """
        start = datetime.strptime(date_from, "%Y-%m-%d")
        end = datetime.strptime(date_to, "%Y-%m-%d")
        chunks = []

        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            ))
            current = chunk_end

        return chunks
