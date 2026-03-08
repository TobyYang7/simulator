#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  bash humanlm/train_sft_bdi_aux.sh <gpu_list> <data_path> [model_path] [extra hydra overrides...]

Example:
  bash humanlm/train_sft_bdi_aux.sh "0,1,2,3" /path/to/bdi_aux_dataset
  bash humanlm/train_sft_bdi_aux.sh "0,1,2,3" /path/to/bdi_aux_dataset /path/to/model trainer.total_epochs=1
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

GPU_LIST=${1:?"Error: GPU list required (e.g., '0,1,2,3')"}
DATA_PATH=${2:?"Error: built BDI auxiliary SFT dataset path required"}
shift 2

MODEL_PATH="Qwen/Qwen3-8B"
if [[ $# -ge 1 && "$1" != *"="* ]]; then
  MODEL_PATH="$1"
  shift 1
fi

if [[ ! -f "$DATA_PATH/train.parquet" ]]; then
  echo "Error: missing train parquet at $DATA_PATH/train.parquet" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH/val.parquet" ]]; then
  echo "Error: missing val parquet at $DATA_PATH/val.parquet" >&2
  exit 1
fi

CHAT_TEMPLATE="${PROJECT_DIR}/humanlm/chat_templates/qwen3_multi_role_template_think.jinja"
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "Error: chat template not found at $CHAT_TEMPLATE" >&2
  exit 1
fi

cluster_root="//llm_twin/${USER}"
local_root="${PROJECT_DIR}/outputs"
cache_local_root="${PROJECT_DIR}/.cache/${USER}"
if mkdir -p "$cluster_root" >/dev/null 2>&1; then
  OUTPUT_ROOT_DEFAULT="${cluster_root}/outputs"
  CACHE_ROOT_DEFAULT="//llm_twin/hf-cache/${USER}"
else
  OUTPUT_ROOT_DEFAULT="$local_root"
  CACHE_ROOT_DEFAULT="${cache_local_root}/hf-cache"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
CACHE_ROOT="${CACHE_ROOT:-$CACHE_ROOT_DEFAULT}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
NUM_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')

EXP_NAME="sft_bdi_aux_$(basename "$DATA_PATH")"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

export WANDB_ENTITY="${WANDB_ENTITY:-dsp-team}"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export XDG_CACHE_HOME="$CACHE_ROOT"
export VLLM_DOWNLOAD_DIR="$CACHE_ROOT/hub"
export VERL_CACHE_DIR="$CACHE_ROOT/verl-cache"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$VERL_CACHE_DIR"

cat <<EOF
================================================================================
                    BDI Auxiliary SFT Configuration Summary
================================================================================
GPUs:                 $GPU_LIST ($NUM_GPUS GPUs)
Data Path:            $DATA_PATH
Chat Template:        $CHAT_TEMPLATE
Model Path:           $MODEL_PATH
Experiment Name:      $EXP_NAME
Output Dir:           $OUTPUT_DIR
Cache Root:           $CACHE_ROOT
================================================================================
EOF

python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_PATH/train.parquet" \
  data.val_files="$DATA_PATH/val.parquet" \
  +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.multiturn.enable=false \
  data.max_length=6144 \
  data.truncation=right \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=2 \
  data.prompt_key=prompt \
  data.response_key=generation \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-6 \
  optim.warmup_steps_ratio=0.1 \
  optim.lr_scheduler=cosine \
  +trainer.val_before_train=true \
  trainer.total_epochs=2 \
  trainer.project_name=humanlm_bdi_aux \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq=300 \
  trainer.test_freq=20 \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  "$@"

echo "BDI auxiliary SFT completed"
echo "Model saved to: $OUTPUT_DIR"
