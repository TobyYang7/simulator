#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  bash humanlm/train_bdi_aux_pipeline.sh <gpu_list> <rl_data_dir> <work_dir> [teacher_model] [model_path] [extra hydra overrides...]

This script:
  1. Generates pseudo-BDI targets for train/val parquet splits
  2. Builds a slot-level BDI auxiliary SFT dataset
  3. Launches auxiliary SFT training

Example:
  bash humanlm/train_bdi_aux_pipeline.sh "0,1,2,3" /path/to/rl_data /path/to/work
  bash humanlm/train_bdi_aux_pipeline.sh "0,1,2,3" /path/to/rl_data /path/to/work openai/gpt-5-mini /path/to/model trainer.total_epochs=1
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

GPU_LIST=${1:?"Error: GPU list required"}
RL_DATA_DIR=${2:?"Error: RL parquet directory required"}
WORK_DIR=${3:?"Error: working directory required"}
shift 3

TEACHER_MODEL="openai/gpt-5-mini"
if [[ $# -ge 1 && "$1" != *"="* ]]; then
  TEACHER_MODEL="$1"
  shift 1
fi

MODEL_PATH="Qwen/Qwen3-8B"
if [[ $# -ge 1 && "$1" != *"="* ]]; then
  MODEL_PATH="$1"
  shift 1
fi

if [[ ! -d "$RL_DATA_DIR" ]]; then
  echo "Error: RL data directory not found: $RL_DATA_DIR" >&2
  exit 1
fi

for split in train val; do
  if [[ ! -f "$RL_DATA_DIR/${split}.parquet" ]]; then
    echo "Error: missing required split parquet: $RL_DATA_DIR/${split}.parquet" >&2
    exit 1
  fi
done

TARGETS_DIR="${WORK_DIR}/pseudo_bdi_targets"
DATASET_DIR="${WORK_DIR}/bdi_aux_dataset"
mkdir -p "$TARGETS_DIR" "$DATASET_DIR"

BOOTSTRAP_MAX_SAMPLES="${BOOTSTRAP_MAX_SAMPLES:-}"
BUILD_MAX_SAMPLES="${BUILD_MAX_SAMPLES:-}"
BOOTSTRAP_EXTRA_ARGS=()
BUILD_EXTRA_ARGS=()

if [[ -n "$BOOTSTRAP_MAX_SAMPLES" ]]; then
  BOOTSTRAP_EXTRA_ARGS+=(--max-samples "$BOOTSTRAP_MAX_SAMPLES")
fi
if [[ -n "$BUILD_MAX_SAMPLES" ]]; then
  BUILD_EXTRA_ARGS+=(--max-samples "$BUILD_MAX_SAMPLES")
fi

for split in train val; do
  python3 "${PROJECT_DIR}/humanlm/bootstrap_bdi_targets.py" \
    --input "$RL_DATA_DIR/${split}.parquet" \
    --output "$TARGETS_DIR/${split}.jsonl" \
    --model "$TEACHER_MODEL" \
    "${BOOTSTRAP_EXTRA_ARGS[@]}"
done

python3 "${PROJECT_DIR}/humanlm/build_bdi_aux_dataset.py" \
  --input-dir "$RL_DATA_DIR" \
  --targets-dir "$TARGETS_DIR" \
  --output-dir "$DATASET_DIR" \
  --splits train val \
  "${BUILD_EXTRA_ARGS[@]}"

bash "${PROJECT_DIR}/humanlm/train_sft_bdi_aux.sh" \
  "$GPU_LIST" \
  "$DATASET_DIR" \
  "$MODEL_PATH" \
  "$@"
