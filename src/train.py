from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.config import load_yaml, merge_not_none, normalize_boolish, parse_bool
from utils.logging import get_logger
from utils.paths import ensure_yolo_config_dir, resolve_dataset_yaml, resolve_path


LOG = get_logger("train")


def import_yolo() -> type:
    ensure_yolo_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for training. Install it with: pip install ultralytics") from exc
    return YOLO


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector.")
    parser.add_argument("--config", type=Path, default=None, help="optional training YAML")
    parser.add_argument("--data", type=Path, default=None, help="YOLO dataset YAML")
    parser.add_argument("--model", type=str, default=None, help="model yaml or weights")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    parser.add_argument("--cache", nargs="?", const="true", default=None, help="true, false, ram, or disk")
    return parser


def resolve_device(device: str | int | None) -> str | int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for device selection. Install it with: pip install torch") from exc

    if device is None:
        return 0 if torch.cuda.is_available() else "cpu"
    text = str(device).strip()
    if text.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return text
    raise RuntimeError("CUDA device was requested, but torch.cuda is not available. Use --device cpu.")


def resolve_cli_or_config_path(
    cli_value: Path | None,
    cfg: dict[str, Any],
    key: str,
    *,
    config_dir: Path | None,
) -> Path | None:
    if cli_value is not None:
        return resolve_path(cli_value, base_dir=Path.cwd())
    value = cfg.get(key)
    if value is None:
        return None
    return resolve_path(value, base_dir=config_dir)


def main() -> None:
    args = build_argparser().parse_args()
    config_dir = args.config.resolve().parent if args.config else None
    cfg = load_yaml(args.config)
    cfg = merge_not_none(
        cfg,
        {
            "data": args.data,
            "model": args.model,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "cache": args.cache,
        },
    )

    data_yaml = resolve_cli_or_config_path(args.data, cfg, "data", config_dir=config_dir)
    if data_yaml is None:
        raise ValueError("dataset YAML is required via --data or config data")
    resolved_data = resolve_dataset_yaml(data_yaml)

    if args.project is not None:
        project = resolve_path(args.project, base_dir=Path.cwd()) or Path("runs").resolve()
    else:
        project = resolve_path(cfg.get("project", "runs"), base_dir=config_dir) or Path("runs").resolve()
    model_name = str(cfg.get("model", "yolov8n.pt"))
    resume = args.resume or parse_bool(cfg.get("resume", False))
    if resume:
        resume_model = cfg.get("resume_model")
        if resume_model:
            resume_path = resolve_path(resume_model, base_dir=config_dir)
        else:
            resume_path = project / str(cfg.get("name", "train")) / "weights" / "last.pt"
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        model_name = str(resume_path)

    device = resolve_device(cfg.get("device"))
    cache = normalize_boolish(cfg.get("cache", False))
    YOLO = import_yolo()

    train_args: dict[str, Any] = {
        "data": str(resolved_data),
        "imgsz": int(cfg.get("imgsz", 640)),
        "epochs": int(cfg.get("epochs", 100)),
        "batch": int(cfg.get("batch", 16)),
        "device": device,
        "workers": int(cfg.get("workers", 8)),
        "project": str(project),
        "name": str(cfg.get("name", "train")),
        "pretrained": cfg.get("pretrained", True),
        "optimizer": cfg.get("optimizer", "auto"),
        "lr0": cfg.get("lr0", 0.01),
        "lrf": cfg.get("lrf", 0.01),
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "warmup_epochs": cfg.get("warmup_epochs", 3.0),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.7),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "degrees": cfg.get("degrees", 0.0),
        "translate": cfg.get("translate", 0.1),
        "scale": cfg.get("scale", 0.5),
        "fliplr": cfg.get("fliplr", 0.5),
        "mosaic": cfg.get("mosaic", 1.0),
        "close_mosaic": cfg.get("close_mosaic", 0),
        "mixup": cfg.get("mixup", 0.0),
        "cache": cache,
        "patience": cfg.get("patience", 50),
        "cos_lr": cfg.get("cos_lr", False),
        "amp": cfg.get("amp", True),
        "exist_ok": cfg.get("exist_ok", True),
        "resume": resume,
    }

    LOG.info("model=%s data=%s project=%s name=%s device=%s", model_name, data_yaml, project, train_args["name"], device)
    YOLO(model_name).train(**train_args)


if __name__ == "__main__":
    main()
