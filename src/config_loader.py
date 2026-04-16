"""
Centralized configuration loader for the AQI Forecasting project.

Merges configs/config.yaml with secrets.json and provides a singleton
accessor used by all modules.
"""

from pathlib import Path
from typing import Optional

from src.logger import logging
from src.utils import load_yaml, load_secrets, get_project_root, ensure_dirs


# Module-level cache for the singleton config
_config_cache: Optional[dict] = None


def get_config(
    config_path: Optional[str] = None,
    secrets_path: Optional[str] = None,
    force_reload: bool = False,
) -> dict:
    """Load and return the merged project configuration (singleton).

    On first call, loads config.yaml and merges the API key from secrets.json.
    Subsequent calls return the cached config unless force_reload=True.

    Args:
        config_path: Path to config YAML. Defaults to configs/config.yaml
                     relative to project root.
        secrets_path: Path to secrets JSON. Defaults to secrets.json
                      relative to project root.
        force_reload: If True, bypass cache and reload from disk.

    Returns:
        Merged configuration dictionary with structure:
        {
            "project": {...},
            "openaq": {..., "api_key": "..."},
            "stations": {...},
            "weather": {...},
            "paths": {...},
            "forecasting": {...},
        }
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    root = get_project_root()

    # Resolve paths
    if config_path is None:
        config_path = str(root / "configs" / "config.yaml")
    if secrets_path is None:
        secrets_path = str(root / "secrets.json")

    # Load YAML config
    config = load_yaml(config_path)

    # Load and merge secrets
    try:
        secrets = load_secrets(secrets_path)
        config["openaq"]["api_key"] = secrets.get("openaq-api-key", "")
        if not config["openaq"]["api_key"]:
            logging.warning("OpenAQ API key is empty in secrets.json")
    except FileNotFoundError as e:
        logging.warning(f"Secrets file not found: {e}. OpenAQ API key will be empty.")
        config["openaq"]["api_key"] = ""

    # Resolve relative paths to absolute
    config["paths"] = _resolve_paths(config.get("paths", {}), root)

    # Ensure output directories exist
    ensure_dirs(
        config["paths"]["raw_dir"],
        config["paths"]["processed_dir"],
        config["paths"]["sequences_dir"],
        config["paths"]["models_dir"],
        config["paths"]["outputs_dir"],
    )

    _config_cache = config
    logging.info(
        f"Configuration loaded — "
        f"OpenAQ date range: {config['openaq']['date_from']} to {config['openaq']['date_to']}, "
        f"bbox: {config['openaq']['bbox']}"
    )

    return config


def _resolve_paths(paths_config: dict, root: Path) -> dict:
    """Resolve relative paths in the config to absolute paths.

    Args:
        paths_config: Dictionary of path key-value pairs from config.yaml.
        root: Project root directory.

    Returns:
        Dictionary with all paths resolved to absolute strings.
    """
    resolved = {}
    for key, value in paths_config.items():
        abs_path = root / value
        resolved[key] = str(abs_path)
    return resolved


def reset_config_cache() -> None:
    """Reset the config cache. Useful for testing."""
    global _config_cache
    _config_cache = None
    logging.info("Config cache reset")
