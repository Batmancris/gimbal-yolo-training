from __future__ import annotations

import argparse
from pathlib import Path

from utils.logging import get_logger
from utils.paths import ensure_yolo_config_dir, output_project_and_name, resolve_dataset_yaml


LOG = get_logger("validate")


def import_yolo() -> type:
    ensure_yolo_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for validation. Install it with: pip install ultralytics") from exc
    return YOLO


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 detector.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--weights", type=Path, required=True, help="model weights")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--save-predictions", action="store_true", help="save validation predictions when supported")
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data_yaml = resolve_dataset_yaml(args.data)
    project, name = output_project_and_name(args.output, Path("runs").resolve(), "val")
    YOLO = import_yolo()

    LOG.info("weights=%s data=%s split=%s output=%s/%s", args.weights, args.data, args.split, project, name)
    YOLO(str(args.weights)).val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(project),
        name=name,
        exist_ok=True,
        save_json=args.save_predictions,
    )


if __name__ == "__main__":
    main()
