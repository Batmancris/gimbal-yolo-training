# ARTIFACT_INDEX

- 当前 yolo_artifacts 不包含 X5/RDK 量化产物。
- 量化残留已移出到: E:\research\1\yolo\quantization_excluded_from_training_assets\vehicle_quant_residuals
- 当前 training 保留内容只包括训练权重、普通 ONNX、训练参数、训练结果和评估图。
- base_models/yolov8n 为基础模型留档，不属于 X5/RDK 量化产物。

## 实验摘要
| experiment | best.pt | last.pt | weights/best.onnx | results.csv | model_card | artifact_manifest |
|---|---|---|---|---|---|---|
| training/vehicle_yolov8n_x5_640 | True | True | True | True | True | True |
| training/bear_yolov8n_x5_640 | True | True | True | True | True | True |

## ONNX 分类
| path | class | 结论 |
|---|---|---|
| training\bear_yolov8n_x5_640\weights\best.onnx | A | 普通训练导出 ONNX，可以保留 |
| training\vehicle_yolov8n_x5_640\weights\best.onnx | A | 普通训练导出 ONNX，可以保留 |

## 量化残留复查
- 路径/文件名复查未发现 .bin、.hbm、x5_quant、quant、calibration、hb_mapper、hb_check、checker、nv12、model_output、best_fixed.onnx。

- 最后整理时间: 2026-04-27 15:02:46 +08:00
