# Inference

Single-image or batch detection is handled by:

```bash
python src/detect.py --weights weights/model.pt --source datasets/vehicle_detection_scene/images/test --output runs/detect
```

Real-time detection is intentionally only a clean skeleton in this round:

```bash
python src/realtime_ui.py --source image --weights weights/model.pt --dry-run
```

Hikvision camera runtime requires physical camera and MVS SDK. Training, export, and no-camera dry-run paths do not require camera hardware.

The planned module split is:

```text
src/camera/hik_camera.py     Hikvision camera adapter
src/yolo/runtime.py          PT and ONNX runtime adapters
src/yolo/preprocess.py       Letterbox and tensor preparation
src/yolo/postprocess.py      Decode, threshold, and NMS helpers
src/realtime_ui.py           CLI and UI orchestration
```
