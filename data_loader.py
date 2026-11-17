"""Helpers for loading JSON datasets and normalising user input."""

import json
from pathlib import Path
from typing import Any

from src.config import CITY_ALIASES


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON dataset from disk.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file contains invalid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def normalize_city(name: str) -> str:
    """Convert user input such as 'bengaluru ' or 'goa' to the dataset's city name."""
    cleaned = name.strip().lower()
    if cleaned in CITY_ALIASES:
        return CITY_ALIASES[cleaned]
    return name.strip().title()
