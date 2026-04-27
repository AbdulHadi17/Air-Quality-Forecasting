"""
Open-Meteo historical weather data client for the AQI Forecasting project.

Fetches hourly meteorological data for Lahore from the Open-Meteo Archive API.
Weather features (temperature, humidity, wind, pressure, precipitation, cloud cover)
serve as covariates for the ConvLSTM air quality model.

No API key required (free tier).
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

from src.logger import logging
from src.utils import retry_with_backoff


class OpenMeteoClient:
    """Fetches historical hourly weather data for Lahore from Open-Meteo.

    Uses the Archive API (archive-api.open-meteo.com/v1/archive) which provides
    reanalysis data dating back to 1940 at 9km resolution.

    Args:
        config: Project configuration dict (from get_config()).
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config: dict):
        self.config = config
        self.weather_config = config["weather"]
        self.latitude = self.weather_config["latitude"]
        self.longitude = self.weather_config["longitude"]
        self.hourly_variables = self.weather_config["hourly_variables"]
        self.timezone = self.weather_config.get("timezone", "Asia/Karachi")

        self._http = httpx.Client(timeout=60.0)

        logging.info(
            f"OpenMeteoClient initialized — "
            f"lat: {self.latitude}, lon: {self.longitude}, "
            f"date range: {self.weather_config['date_from']} to {self.weather_config['date_to']}"
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def fetch_historical(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical hourly weather data for Lahore.

        Chunks the request into 1-year windows (API best practice) and
        concatenates the results.

        Args:
            date_from: Start date (YYYY-MM-DD). Defaults to config value.
            date_to: End date (YYYY-MM-DD). Defaults to config value.

        Returns:
            DataFrame with timestamp index and weather variable columns:
                temperature_2m, relative_humidity_2m, wind_speed_10m,
                wind_direction_10m, pressure_msl, precipitation, cloud_cover
        """
        date_from = date_from or self.weather_config["date_from"]
        date_to = date_to or self.weather_config["date_to"]

        logging.info(f"Fetching weather data: {date_from} → {date_to}")

        # Chunk into 1-year windows
        chunks = self._chunk_by_year(date_from, date_to)
        all_dfs = []

        for chunk_from, chunk_to in chunks:
            logging.info(f"  Fetching weather chunk: {chunk_from} → {chunk_to}")
            df = self._fetch_chunk(chunk_from, chunk_to)
            if df is not None and not df.empty:
                all_dfs.append(df)
            time.sleep(0.5)  # Be polite to the free API

        if not all_dfs:
            logging.error("No weather data fetched")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=False)
        # Remove any duplicates from overlapping chunks
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        logging.info(
            f"Weather data fetched: {len(combined)} hourly records, "
            f"{combined.index.min()} to {combined.index.max()}"
        )
        return combined

    @retry_with_backoff(max_retries=3, base_delay=2.0, retryable_exceptions=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException))
    def _fetch_chunk(self, date_from: str, date_to: str) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of weather data from Open-Meteo.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).

        Returns:
            DataFrame with timestamp index and weather columns, or None on error.

        Raises:
            httpx.HTTPStatusError: On API error responses.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": date_from,
            "end_date": date_to,
            "hourly": ",".join(self.hourly_variables),
            "timezone": self.timezone,
        }

        response = self._http.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # Check for API-level errors
        if data.get("error"):
            reason = data.get("reason", "Unknown error")
            logging.error(f"Open-Meteo API error: {reason}")
            return None

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            logging.warning(f"No hourly data in response for {date_from} to {date_to}")
            return None

        # Build DataFrame
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df.index.name = "timestamp"

        return df

    def save_weather(self, df: pd.DataFrame, path: str) -> None:
        """Save weather DataFrame to Parquet.

        Args:
            df: Weather DataFrame with timestamp index.
            path: Output file path (e.g., data/raw/weather.parquet).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")
        logging.info(f"Saved {len(df)} weather records to {path}")

    @staticmethod
    def load_existing(path: str) -> Optional[pd.DataFrame]:
        """Load existing weather Parquet file if it exists.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame if file exists, None otherwise.
        """
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p, engine="pyarrow")
            logging.info(f"Loaded {len(df)} existing weather records from {path}")
            return df
        return None

    @staticmethod
    def _chunk_by_year(date_from: str, date_to: str) -> list[tuple[str, str]]:
        """Split a date range into 1-year chunks.

        Args:
            date_from: Start date (YYYY-MM-DD).
            date_to: End date (YYYY-MM-DD).

        Returns:
            List of (chunk_from, chunk_to) string tuples.
        """
        start = datetime.strptime(date_from, "%Y-%m-%d")
        end = datetime.strptime(date_to, "%Y-%m-%d")
        chunks = []

        current = start
        while current < end:
            year_end = datetime(current.year, 12, 31)
            chunk_end = min(year_end, end)
            chunks.append((
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            ))
            current = chunk_end + timedelta(days=1)

        return chunks
