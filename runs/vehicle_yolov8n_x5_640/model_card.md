# vehicle_yolov8n_x5_640

- 实验名称: vehicle_yolov8n_x5_640
- 任务类型: object detection
- 类别 names: 0: vehicle
- 模型结构: YOLOv8n
- 输入尺寸: 640
- 训练数据来源: yolo_datasets\formal\vehicle_detection_scene\data.yaml
- best.pt: yes
- last.pt: yes
- 普通 ONNX weights/best.onnx: yes
- results.csv: yes
- 关键结果图:
  - results.png: yes
  - confusion_matrix.png: yes
  - confusion_matrix_normalized.png: yes
  - PR_curve.png: yes
  - P_curve.png: yes
  - R_curve.png: yes
  - F1_curve.png: yes

## 当前训练资产保留内容
- weights/best.pt
- weights/last.pt
- weights/best.onnx（普通训练导出 ONNX）
- args.yaml
- results.csv
- 训练结果图和评估图

## X5/RDK 量化内容状态
- 当前实验目录不包含 bin、hbm、x5_quant、nv12、hb_mapper、model_output 等 X5/RDK 量化内容。
- best_fixed.onnx 已移出，不属于当前训练资产包。
- 已移出的量化残留位于: E:\research\1\yolo\quantization_excluded_from_training_assets\vehicle_quant_residuals
- 后续 X5/RDK 量化资产将单独整理。

## 已知问题
- args.yaml 记录的是原始训练时路径，迁移后仅作为历史元数据使用。

- 最后整理时间: 2026-04-27 15:02:46 +08:00
