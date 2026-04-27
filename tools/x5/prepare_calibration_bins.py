from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for calibration image conversion. Install it with: pip install opencv-python") from exc
    return cv2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (repo_root() / path).resolve()


def letterbox(image: np.ndarray, new_shape: int, cv2_module: Any) -> np.ndarray:
    shape = image.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        image = cv2_module.resize(image, new_unpad, interpolation=cv2_module.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2_module.copyMakeBorder(
        image, top, bottom, left, right, cv2_module.BORDER_CONSTANT, value=(114, 114, 114)
    )


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def convert_image(image_path: Path, output_path: Path, imgsz: int) -> None:
    cv2 = import_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    image = letterbox(image, imgsz, cv2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.uint8)
    image.tofile(output_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate X5 calibration bins for YOLOv8 NV12 deployment.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/vehicle_detection_scene/images/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_rgb_uint8_nchw"))
    parser.add_argument("--imgsz", type=int, default=640, help="square model input size")
    parser.add_argument("--limit", type=int, default=100, help="maximum number of images to convert")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    images = iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"no calibration images found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = images[: args.limit] if args.limit > 0 else images
    print(f"[CAL] input_dir={input_dir}")
    print(f"[CAL] output_dir={output_dir}")
    print(f"[CAL] imgsz={args.imgsz}, limit={len(selected)}")

    for index, image_path in enumerate(selected, start=1):
        output_path = output_dir / f"{image_path.stem}.bin"
        convert_image(image_path, output_path, args.imgsz)
        if index == 1 or index == len(selected) or index % 20 == 0:
            print(f"[CAL] {index}/{len(selected)} -> {output_path.name}", flush=True)

    print("[CAL] done", flush=True)


if __name__ == "__main__":
    main()
