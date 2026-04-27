# Gimbal YOLO Training

Clean training and deployment workspace for gimbal vision object detection. It covers YOLOv8 training, validation, ONNX export, Hikvision real-time detection boundaries, and RDK X5 quantization preparation.

## Directory Layout

```text
configs/                 Dataset, training, and X5 example configs
src/                     Main Python entry points and reusable modules
tools/debug/             PT/ONNX comparison and ONNX inspection tools
tools/x5/                X5 calibration and quantization helpers
datasets/                Dataset placeholders and data.yaml templates
docs/                    Task-focused operation notes
assets/results/          Small result images for reports
weights/                 Local weights only, ignored by Git
runs/                    Local training outputs only, ignored by Git
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Hikvision MVS SDK is not a pip dependency. Hikvision camera runtime requires physical camera and MVS SDK; this repository can still run training/export without camera.

## Dataset Preparation

Place YOLO-format data under `datasets/bear_detection_scene` or `datasets/vehicle_detection_scene`.

Bear LabelMe conversion:

```bash
python src/datasets/prepare_bear.py --source-root beardata --output-root datasets/bear_detection_scene
```

The converter expects `bear/` LabelMe images with matching JSON files, optional `background/` negative samples, and optional `test/` images.

## Training

Bear:

```bash
python src/train.py --config configs/train_bear.yaml
```

Vehicle:

```bash
python src/train.py --config configs/train_vehicle.yaml
```

All paths should come from CLI arguments or config files. Do not hard-code machine-local paths in scripts or configs.

## Validation

```bash
python src/validate.py --data datasets/vehicle_detection_scene/data.yaml --weights runs/vehicle_yolov8n_x5_640/weights/best.pt --split val --output runs/vehicle_val
```

## ONNX Export

```bash
python src/export.py --weights runs/vehicle_yolov8n_x5_640/weights/best.pt --format onnx --imgsz 640 --opset 11 --fix-opset11 --output weights/vehicle_yolov8n_x5_640.onnx
```

## Detection

```bash
python src/detect.py --weights weights/model.pt --source datasets/vehicle_detection_scene/images/test --output runs/detect
```

## Hikvision Real-Time Detection

`src/realtime_ui.py` is a clean boundary for future real-time detection work. The current round provides argparse, camera/runtime interfaces, and preprocessing/postprocessing helpers. Install Hikvision MVS SDK separately before enabling real camera capture.

No-camera dry-run:

```bash
python src/realtime_ui.py --source image --weights weights/best.pt --dry-run
```

## RDK X5 Deployment

Prepare calibration bins:

```bash
python tools/x5/prepare_calibration_bins.py --input-dir datasets/vehicle_detection_scene/images/val --output-dir calibration_rgb_uint8_nchw
```

Run the X5 quantization helper inside a Horizon/OpenExplorer environment:

```bash
MODEL_PATH=weights/vehicle_yolov8n_x5_640.onnx bash tools/x5/run_x5_quant_pipeline.sh
```

RDK X5 quantization requires OpenExplorer / hb_mapper environment on the target deployment machine. Horizon/OpenExplorer and `hb_mapper` are not pip dependencies.

## Large Files

Datasets, weights, calibration bins, and `runs/` outputs are local artifacts and should not be committed. Keep only templates, small docs, and selected result images in Git. Put report images under `assets/results/` instead of uploading full training runs.
