#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH to a dedicated local base-model copy}"
: "${DATA_ROOT:?Set DATA_ROOT to the modified LIBERO RLDS root}"
OFT_SOURCE="$VLA_WORKSPACE/openvla-oft"
VLA_TORCHRUN="${VLA_TORCHRUN:-$OFT_SOURCE/.venv/bin/torchrun}"
RUN_ROOT="${RUN_ROOT:-$VLA_WORKSPACE/oft_runs}"
require_file "$VLA_TORCHRUN"
require_file "$MODEL_PATH/config.json"
require_file "$OFT_SOURCE/vla-scripts/finetune.py"
if [[ ! -d "$DATA_ROOT/libero_spatial_no_noops" ]]; then
    echo "Expected dataset directory: $DATA_ROOT/libero_spatial_no_noops" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
cd "$OFT_SOURCE"
export OFT_8BIT=1
export WANDB_MODE=offline
# The upstream trainer updates model config/source files in MODEL_PATH.
# Always supply an OFT-only base model copy.
"$VLA_TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "$MODEL_PATH" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 1 --use_proprio False \
    --batch_size 1 --grad_accumulation_steps 16 --learning_rate 5e-4 \
    --num_steps_before_decay 8000 --max_steps 10000 --save_freq 2500 \
    --save_latest_checkpoint_only False --image_aug True --lora_rank 32 \
    --wandb_entity local --wandb_project oft_full \
    --run_id_note oft_8bit_lora_r32 \
    2>&1 | tee "$RUN_ROOT/train-$(date +%Y%m%d-%H%M%S).log"
