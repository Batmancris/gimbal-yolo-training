from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in YAML file: {yaml_path}")
    return data


def merge_not_none(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def normalize_boolish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return text


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
