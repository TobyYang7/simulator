#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  bash humanlm/train_bdi_rl_from_aux.sh <gpu_list> <dataset_name> <aux_model_path> <aux_targets_path> [extra rl hydra overrides...]

This script launches BDI RL with:
  - the auxiliary SFT checkpoint as the model warm start
  - split-matched pseudo-BDI sidecars injected into StateDataset

Examples:
  bash humanlm/train_bdi_rl_from_aux.sh "0,1,2,3" amazon /path/to/aux_ckpt /path/to/pseudo_bdi_targets
  bash humanlm/train_bdi_rl_from_aux.sh "0,1,2,3" reddit /path/to/aux_ckpt /path/to/train.jsonl trainer.total_epochs=1
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

GPU_LIST=${1:?"Error: GPU list required"}
DATASET_NAME=${2:?"Error: dataset name required"}
AUX_MODEL_PATH=${3:?"Error: auxiliary model path required"}
AUX_TARGETS_PATH=${4:?"Error: auxiliary targets path required"}
shift 4

if [[ ! -e "$AUX_MODEL_PATH" ]]; then
  echo "Error: auxiliary model path not found: $AUX_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -e "$AUX_TARGETS_PATH" ]]; then
  echo "Error: auxiliary targets path not found: $AUX_TARGETS_PATH" >&2
  exit 1
fi

MODEL_PATH_OVERRIDE="$AUX_MODEL_PATH" \
AUX_TARGETS_PATH="$AUX_TARGETS_PATH" \
bash "${PROJECT_DIR}/humanlm/train_rl_humanlm.sh" \
  "$GPU_LIST" \
  "$DATASET_NAME" \
  train_bdi_humanlm \
  "" \
  base \
  "$@"
