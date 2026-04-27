"""
Shared utility functions for the AQI Forecasting project.

Provides YAML/JSON loading, directory management, and a retry decorator
for resilient API interactions.
"""

import json
import os
import time
import functools
from pathlib import Path
from typing import Any, Callable

import yaml

from src.logger import logging


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file and return as dict.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded YAML config from {path}")
    return config


def load_secrets(path: str = "secrets.json") -> dict:
    """Load secrets from a JSON file.

    Args:
        path: Path to the secrets JSON file. Defaults to 'secrets.json'.

    Returns:
        Parsed secrets as a Python dictionary.

    Raises:
        FileNotFoundError: If the secrets file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Secrets file not found: {path}. "
            f"Copy secrets-example.json to secrets.json and fill in your API key."
        )

    with open(path, "r", encoding="utf-8") as f:
        secrets = json.load(f)

    logging.info("Loaded secrets file (keys hidden)")
    return secrets


def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist.

    Args:
        *paths: One or more directory paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {p}")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def call_api():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logging.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logging.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            raise last_exception  # Should never reach here

        return wrapper

    return decorator


def get_project_root() -> Path:
    """Get the project root directory (parent of src/).

    Returns:
        Path object pointing to the project root.
    """
    # Walk up from this file: src/utils.py -> src/ -> project_root/
    return Path(__file__).resolve().parent.parent
