#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to a complete OFT checkpoint directory}"
OFT_SOURCE="$VLA_WORKSPACE/openvla-oft"
VLA_PYTHON="${VLA_PYTHON:-$OFT_SOURCE/.venv/bin/python}"
require_file "$VLA_PYTHON"
require_file "$CHECKPOINT/config.json"
require_file "$CHECKPOINT/dataset_statistics.json"
if ! compgen -G "$CHECKPOINT/action_head--*checkpoint.pt" >/dev/null; then
    echo "Missing continuous action head in CHECKPOINT" >&2
    exit 1
fi
cd "$OFT_SOURCE"
export PYTHONPATH="$OFT_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
"$VLA_PYTHON" experiments/robot/libero/run_libero_eval.py \
    --model_family openvla --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name libero_spatial --center_crop True \
    --use_l1_regression True --load_in_8bit True \
    --use_diffusion False --use_film False \
    --num_images_in_input 1 --use_proprio False \
    --num_open_loop_steps 8 --num_steps_wait 10 \
    --seed 7 --num_trials_per_task "${TRIALS:-50}" --use_wandb False
