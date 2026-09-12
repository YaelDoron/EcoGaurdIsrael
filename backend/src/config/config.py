"""YAML configuration loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_NEWS_CONFIG_PATH = Path(__file__).with_name("news_config.yaml")


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load a YAML config file, defaulting to the bundled news config."""
    path = Path(config_path) if config_path else DEFAULT_NEWS_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data
