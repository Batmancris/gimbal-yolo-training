from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from yolo.preprocess import LetterboxInfo


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for NMS postprocessing. Install it with: pip install opencv-python") from exc
    return cv2


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= info.pad[0]
    scaled[:, [1, 3]] -= info.pad[1]
    scaled[:, :4] /= info.ratio
    h, w = info.original_shape
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, w)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, h)
    return scaled


def nms(detections: list[Detection], iou_threshold: float = 0.7) -> list[Detection]:
    if not detections:
        return []
    cv2 = import_cv2()
    boxes = [
        [det.xyxy[0], det.xyxy[1], det.xyxy[2] - det.xyxy[0], det.xyxy[3] - det.xyxy[1]]
        for det in detections
    ]
    scores = [det.confidence for det in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    flat_indices = np.array(indices).reshape(-1).tolist()
    return [detections[index] for index in flat_indices]


def decode_yolov8_output(output: np.ndarray, info: LetterboxInfo, conf_threshold: float = 0.25, iou_threshold: float = 0.7) -> list[Detection]:
    tensor = np.asarray(output, dtype=np.float32)
    tensor = np.squeeze(tensor)
    if tensor.ndim != 2:
        raise ValueError(f"expected 2D YOLO output after squeeze, got {tensor.shape}")
    if tensor.shape[0] < tensor.shape[1]:
        tensor = tensor.T

    boxes = tensor[:, :4]
    class_scores = tensor[:, 4:]
    if class_scores.size == 0:
        return []

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    boxes_xyxy = scale_boxes(xywh_to_xyxy(boxes[keep]), info)
    kept_class_ids = class_ids[keep]
    kept_confidences = confidences[keep]
    detections = [
        Detection(
            class_id=int(class_id),
            confidence=float(confidence),
            xyxy=tuple(float(value) for value in box),
        )
        for class_id, confidence, box in zip(kept_class_ids, kept_confidences, boxes_xyxy, strict=True)
    ]
    # TODO: add class-aware NMS if multi-class models are introduced.
    return nms(detections, iou_threshold=iou_threshold)
