#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
CFG_PATH="${CFG_PATH:-${2:-$ROOT_DIR/configs/x5_quant.example.yaml}}"
HB_MAPPER_BIN="${HB_MAPPER_BIN:-hb_mapper}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required, or pass the ONNX path as the first argument."
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ONNX model not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Quant config not found: $CFG_PATH"
  exit 1
fi

if ! command -v "$HB_MAPPER_BIN" >/dev/null 2>&1; then
  echo "hb_mapper not found: $HB_MAPPER_BIN"
  exit 1
fi

export HOME="${HOME:-$ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-x5}"
export HORIZON_LIB_PATH="${HORIZON_LIB_PATH:-$ROOT_DIR/.horizon}"
export DDK_LIB_PATH="${DDK_LIB_PATH:-$ROOT_DIR/.horizon/ddk}"
export X5_X86_GCC1140_PATH="${X5_X86_GCC1140_PATH:-$ROOT_DIR/.horizon/ddk/x5_x86_64_gcc_11.4.0}"
export HB_DNN_SIM_PLATFORM="${HB_DNN_SIM_PLATFORM:-BAYESE}"
export LD_LIBRARY_PATH="${X5_X86_GCC1140_PATH}/dnn_x86/lib:${LD_LIBRARY_PATH:-}"

echo "[X5] model=$MODEL_PATH"
echo "[X5] config=$CFG_PATH"
echo "[X5] hb_mapper=$HB_MAPPER_BIN"

echo "[X5] step 1/2: checker"
"$HB_MAPPER_BIN" checker --model-type onnx --model "$MODEL_PATH" --march bayes-e

echo "[X5] step 2/2: makertbin"
"$HB_MAPPER_BIN" makertbin --config "$CFG_PATH" --model-type onnx

echo "[X5] done"
