from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad: tuple[float, float]
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for image preprocessing. Install it with: pip install opencv-python") from exc
    return cv2


def letterbox(image: np.ndarray, new_shape: int | tuple[int, int] = 640) -> tuple[np.ndarray, LetterboxInfo]:
    cv2 = import_cv2()
    if isinstance(new_shape, int):
        target_h, target_w = new_shape, new_shape
    else:
        target_h, target_w = new_shape

    original_h, original_w = image.shape[:2]
    ratio = min(target_h / original_h, target_w / original_w)
    new_unpad = (int(round(original_w * ratio)), int(round(original_h * ratio)))
    dw = (target_w - new_unpad[0]) / 2
    dh = (target_h - new_unpad[1]) / 2

    if (original_w, original_h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    info = LetterboxInfo(
        ratio=ratio,
        pad=(dw, dh),
        original_shape=(original_h, original_w),
        input_shape=(target_h, target_w),
    )
    return padded, info


def preprocess_bgr(image: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxInfo]:
    cv2 = import_cv2()
    padded, info = letterbox(image, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(np.ascontiguousarray(tensor), axis=0), info


def preprocess_path(image_path: str, imgsz: int = 640) -> tuple[np.ndarray, LetterboxInfo]:
    cv2 = import_cv2()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return preprocess_bgr(image, imgsz)
