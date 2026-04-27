from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HikCameraConfig:
    serial_number: str | None = None
    pixel_format: str = "BGR8"
    width: int | None = None
    height: int | None = None
    exposure_time: float | None = None
    gain: float | None = None


def import_mvs_sdk() -> Any:
    try:
        from MvCameraControl_class import MvCamera  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Hikvision MVS SDK is not available. "
            "Please install MVS SDK and configure Python wrapper path."
        ) from exc
    return MvCamera


class HikCamera:
    def __init__(self, config: HikCameraConfig | None = None) -> None:
        self.config = config or HikCameraConfig()
        self._camera: Any | None = None

    def open(self) -> None:
        camera_cls = import_mvs_sdk()
        self._camera = camera_cls()
        # TODO: enumerate devices, select by serial number when provided, and create the camera handle.
        # TODO: apply width, height, exposure, gain, trigger, and pixel format settings.
        raise NotImplementedError("Hikvision camera initialization is a skeleton in this round.")

    def read(self) -> Any:
        if self._camera is None:
            raise RuntimeError("camera is not open")
        # TODO: grab one frame and return a BGR numpy array.
        raise NotImplementedError("frame grabbing is not implemented yet.")

    def close(self) -> None:
        if self._camera is None:
            return
        # TODO: stop grabbing, close device, and destroy handle.
        self._camera = None

    def __enter__(self) -> "HikCamera":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
