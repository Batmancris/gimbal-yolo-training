from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils.logging import get_logger
from utils.paths import ensure_yolo_config_dir


LOG = get_logger("export")


def import_yolo() -> type:
    ensure_yolo_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for export. Install it with: pip install ultralytics") from exc
    return YOLO


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights.")
    parser.add_argument("--weights", type=Path, required=True, help="input weights")
    parser.add_argument("--format", choices=["onnx"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="output file or directory")
    parser.add_argument("--fix-opset11", action="store_true", help="rewrite selected opset 11 compatibility patterns")
    return parser


def fix_opset11(input_path: Path, output_path: Path) -> dict[str, int]:
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("ONNX post-processing requires numpy and onnx. Install them with: pip install numpy onnx") from exc

    model = onnx.load(input_path)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    counts = {"split": 0, "resize": 0, "reshape": 0}

    for node in model.graph.node:
        if node.op_type != "Split" or len(node.input) != 2:
            continue
        split_input = node.input[1]
        if split_input not in initializers:
            continue
        split_values = [int(value) for value in np.asarray(initializers[split_input]).tolist()]
        kept_attrs = [attr for attr in node.attribute if attr.name != "split"]
        kept_attrs.append(helper.make_attribute("split", split_values))
        data_input = node.input[0]
        del node.input[:]
        node.input.extend([data_input])
        del node.attribute[:]
        node.attribute.extend(kept_attrs)
        counts["split"] += 1

    roi_name = "_hb_empty_roi"
    if not any(item.name == roi_name for item in model.graph.initializer):
        model.graph.initializer.append(numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name))

    for node in model.graph.node:
        if node.op_type == "Resize" and len(node.input) >= 2 and node.input[1] == "":
            node.input[1] = roi_name
            counts["resize"] += 1
        if node.op_type == "Reshape":
            kept_attrs = [attr for attr in node.attribute if attr.name != "allowzero"]
            if len(kept_attrs) != len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(kept_attrs)
                counts["reshape"] += 1

    used_inputs = {name for node in model.graph.node for name in node.input}
    kept_initializers = [item for item in model.graph.initializer if item.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return counts


def resolve_output_path(exported_path: Path, output: Path | None) -> Path:
    if output is None:
        return exported_path
    output = output.resolve()
    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / exported_path.name


def main() -> None:
    args = build_argparser().parse_args()
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
        "opset": args.opset,
        "simplify": args.simplify,
        "dynamic": args.dynamic,
        "half": args.half,
    }

    YOLO = import_yolo()
    LOG.info("exporting weights=%s format=%s imgsz=%s opset=%s", args.weights, args.format, args.imgsz, args.opset)
    exported = YOLO(str(args.weights)).export(**export_kwargs)
    exported_path = Path(exported).resolve()
    output_path = resolve_output_path(exported_path, args.output)

    if args.fix_opset11:
        counts = fix_opset11(exported_path, output_path)
        LOG.info("fixed opset11 output=%s split=%s resize=%s reshape=%s", output_path, counts["split"], counts["resize"], counts["reshape"])
        return

    if output_path != exported_path:
        shutil.copy2(exported_path, output_path)
    LOG.info("exported output=%s", output_path)


if __name__ == "__main__":
    main()
