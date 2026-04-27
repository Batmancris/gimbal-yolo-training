from __future__ import annotations

import argparse
from pathlib import Path

from camera.hik_camera import HikCameraConfig
from utils.logging import get_logger


LOG = get_logger("realtime-ui")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime detection UI skeleton for Hikvision and YOLO.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "pt", "onnx"], default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--camera-serial", type=str, default=None)
    parser.add_argument("--source", choices=["hikvision", "image", "video"], default="hikvision")
    parser.add_argument("--input", type=Path, default=None, help="image or video path for non-camera sources")
    parser.add_argument("--dry-run", action="store_true", help="print planned runtime and source config without opening hardware")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    camera_config = HikCameraConfig(serial_number=args.camera_serial) if args.source == "hikvision" else None
    if args.dry_run:
        LOG.info(
            "planned weights=%s backend=%s source=%s input=%s camera=%s imgsz=%s conf=%s iou=%s",
            args.weights,
            args.backend,
            args.source,
            args.input,
            camera_config,
            args.imgsz,
            args.conf,
            args.iou,
        )
        return

    from yolo.runtime import create_runtime

    runtime = create_runtime(args.weights, backend=args.backend, device=args.device)
    if args.source != "hikvision":
        if args.input is None:
            raise ValueError("--input is required when --source is image or video")
        LOG.info(
            "runtime=%s source=%s input=%s imgsz=%s conf=%s iou=%s",
            runtime.__class__.__name__,
            args.source,
            args.input,
            args.imgsz,
            args.conf,
            args.iou,
        )
        raise NotImplementedError("Image/video realtime loop is planned for a later round.")

    LOG.info(
        "runtime=%s camera=%s imgsz=%s conf=%s iou=%s",
        runtime.__class__.__name__,
        camera_config,
        args.imgsz,
        args.conf,
        args.iou,
    )
    raise NotImplementedError("Realtime UI loop is planned for a later round.")


if __name__ == "__main__":
    main()
