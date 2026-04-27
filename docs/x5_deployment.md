# X5 Deployment

RDK X5 deployment uses Horizon/OpenExplorer tooling outside pip. RDK X5 quantization requires OpenExplorer / hb_mapper environment on the target deployment machine. Install the toolchain separately and verify that `hb_mapper` is available in the deployment shell.

Prepare RGB uint8 NCHW calibration bins:

```bash
python tools/x5/prepare_calibration_bins.py --input-dir datasets/vehicle_detection_scene/images/val --output-dir calibration_rgb_uint8_nchw --imgsz 640 --limit 100
```

The example quantization config is `configs/x5_quant.example.yaml`. Copy it to a local config and update:

```text
model_parameters.onnx_model
model_parameters.working_dir
calibration_parameters.cal_data_dir
```

Run checker and makertbin:

```bash
MODEL_PATH=weights/model.onnx CFG_PATH=configs/x5_quant.example.yaml bash tools/x5/run_x5_quant_pipeline.sh
```

The runtime input type is expected to be NV12, while calibration data is generated from RGB uint8 NCHW tensors.
