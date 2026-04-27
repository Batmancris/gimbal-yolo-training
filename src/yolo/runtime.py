from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class DetectionRuntime(ABC):
    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> Any:
        raise NotImplementedError


class UltralyticsRuntime(DetectionRuntime):
    def __init__(self, weights: Path | str, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def predict(self, input_tensor: np.ndarray) -> Any:
        return self.model.predict(input_tensor, device=self.device, verbose=False)


class OnnxRuntime(DetectionRuntime):
    def __init__(self, weights: Path | str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(weights), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: input_tensor})


def infer_backend(weights: Path | str) -> str:
    suffix = Path(weights).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth"}:
        return "pt"
    raise ValueError(f"cannot infer runtime backend from suffix: {weights}")


def create_runtime(weights: Path | str, backend: str = "auto", device: str | None = None) -> DetectionRuntime:
    selected = infer_backend(weights) if backend == "auto" else backend
    if selected == "pt":
        return UltralyticsRuntime(weights, device=device)
    if selected == "onnx":
        return OnnxRuntime(weights)
    raise ValueError(f"unsupported backend: {backend}")
