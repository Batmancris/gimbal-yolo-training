# ONNX Export

Export from a trained YOLOv8 checkpoint:

```bash
python src/export.py --weights runs/vehicle_yolov8n_x5_640/weights/best.pt --format onnx --imgsz 640 --opset 11 --output weights/vehicle_yolov8n_x5_640.onnx
```

For Horizon toolchain compatibility, use the opset 11 repair pass:

```bash
python src/export.py --weights runs/vehicle_yolov8n_x5_640/weights/best.pt --format onnx --imgsz 640 --opset 11 --fix-opset11 --output weights/vehicle_yolov8n_x5_640_opset11.onnx
```

The repair pass rewrites selected `Split`, `Resize`, and `Reshape` patterns that can break older opset 11 consumers.
