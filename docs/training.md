# Training

Install dependencies first:

```bash
pip install -r requirements.txt
```

Bear training:

```bash
python src/train.py --config configs/train_bear.yaml
```

Vehicle training:

```bash
python src/train.py --config configs/train_vehicle.yaml
```

Override common settings from the CLI:

```bash
python src/train.py --data datasets/vehicle_detection_scene/data.yaml --model yolov8n.pt --imgsz 640 --epochs 100 --batch 16 --device 0 --workers 2 --project runs --name vehicle_yolov8n_x5_640 --cache disk
```

Use `--resume` only when the configured or inferred `last.pt` checkpoint exists.
