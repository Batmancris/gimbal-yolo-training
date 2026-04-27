from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ((base_dir or REPO_ROOT) / path).resolve()


def runtime_dir() -> Path:
    root = Path(tempfile.gettempdir()).resolve() / "gimbal_yolo_training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_yolo_config_dir() -> Path:
    config_dir = runtime_dir() / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def resolve_dataset_yaml(data_yaml: Path) -> Path:
    source = data_yaml.resolve()
    with source.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in dataset YAML: {source}")

    yaml_dir = source.parent
    root_value = data.get("path")
    if root_value:
        dataset_root = resolve_path(root_value, base_dir=yaml_dir)
    else:
        dataset_root = yaml_dir
    if dataset_root is None:
        dataset_root = yaml_dir

    data["path"] = dataset_root.as_posix()

    tmp_dir = runtime_dir() / "datasets"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resolved = tmp_dir / f"{source.stem}_resolved_{os.getpid()}_{uuid4().hex[:8]}.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved


def output_project_and_name(output: Path | None, default_project: Path, default_name: str) -> tuple[Path, str]:
    if output is None:
        return default_project, default_name
    output = output.resolve()
    return output.parent, output.name
