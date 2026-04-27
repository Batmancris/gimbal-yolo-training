from __future__ import annotations

import argparse
from pathlib import Path

from utils.logging import get_logger
from utils.paths import ensure_yolo_config_dir, output_project_and_name


LOG = get_logger("detect")


def import_yolo() -> type:
    ensure_yolo_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detection. Install it with: pip install ultralytics") from exc
    return YOLO


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on images or video.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    project, name = output_project_and_name(args.output, Path("runs").resolve(), "detect")
    YOLO = import_yolo()
    LOG.info("weights=%s source=%s output=%s/%s", args.weights, args.source, project, name)
    YOLO(str(args.weights)).predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(project),
        name=name,
        exist_ok=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
