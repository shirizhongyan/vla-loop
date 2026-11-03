#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH to a dedicated base-model copy for this paradigm}"
: "${DATA_ROOT:?Set DATA_ROOT to the modified LIBERO RLDS root}"
PARADIGM="${PARADIGM:-fm}"
case "$PARADIGM" in
    fm) flags=(--use_l1_regression False --use_diffusion False --use_flow_matching True) ;;
    diff) flags=(--use_l1_regression False --use_diffusion True --use_flow_matching False) ;;
    l1) flags=(--use_l1_regression True --use_diffusion False --use_flow_matching False) ;;
    *) echo "PARADIGM must be fm, diff or l1" >&2; exit 1 ;;
esac
OFT_SOURCE="$VLA_WORKSPACE/openvla-oft"
VLA_TORCHRUN="${VLA_TORCHRUN:-$OFT_SOURCE/.venv/bin/torchrun}"
RUN_ROOT="${RUN_ROOT:-$VLA_WORKSPACE/paradigm_runs}"
require_file "$VLA_TORCHRUN"
require_file "$MODEL_PATH/config.json"
require_file "$OFT_SOURCE/vla-scripts/finetune.py"
[[ -d "$DATA_ROOT/libero_spatial_no_noops" ]]
mkdir -p "$RUN_ROOT"
cd "$OFT_SOURCE"
export OFT_8BIT=1 WANDB_MODE=offline
"$VLA_TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vla_path "$MODEL_PATH" --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_spatial_no_noops --run_root_dir "$RUN_ROOT" \
    "${flags[@]}" --num_images_in_input 1 --use_proprio False \
    --batch_size 1 --grad_accumulation_steps 16 --learning_rate 5e-4 \
    --num_steps_before_decay 8000 --max_steps 10000 --save_freq 2500 \
    --save_latest_checkpoint_only False --image_aug True --lora_rank 32 \
    --grad_clip_max_norm 1.0 --diffusion_sample_freq 50 \
    --wandb_entity local --wandb_project "paradigm_$PARADIGM" \
    --run_id_note "paradigm_$PARADIGM" \
    2>&1 | tee "$RUN_ROOT/train-$PARADIGM-$(date +%Y%m%d-%H%M%S).log"
