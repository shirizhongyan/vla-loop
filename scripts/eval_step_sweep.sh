#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to the complete flow-matching 10000-step checkpoint}"
OFT_SOURCE="$VLA_WORKSPACE/openvla-oft"
VLA_PYTHON="${VLA_PYTHON:-$OFT_SOURCE/.venv/bin/python}"
LOG_ROOT="${LOG_ROOT:-$VLA_WORKSPACE/step-sweep}"
TRIALS="${TRIALS:-50}"
SEED="${SEED:-7}"
require_file "$VLA_PYTHON"
require_file "$CHECKPOINT/config.json"
require_file "$CHECKPOINT/dataset_statistics.json"
require_file "$OFT_SOURCE/experiments/robot/libero/run_libero_eval.py"
compgen -G "$CHECKPOINT/action_head--*checkpoint.pt" >/dev/null
compgen -G "$CHECKPOINT/noisy_action_projector--*checkpoint.pt" >/dev/null
if [[ ! "$TRIALS" =~ ^[1-9][0-9]*$ ]] || (( TRIALS > 50 )); then
    echo "TRIALS must be an integer between 1 and 50" >&2
    exit 1
fi
read -r -a sweep_steps <<< "${STEPS:-1 2}"
[[ ${#sweep_steps[@]} -gt 0 ]]
cd "$OFT_SOURCE"
export PYTHONPATH="$OFT_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
for inference_steps in "${sweep_steps[@]}"; do
    [[ "$inference_steps" =~ ^[1-9][0-9]*$ ]]
    destination="$LOG_ROOT/steps_${inference_steps}_trials_${TRIALS}_seed_${SEED}_$(date +%Y%m%d-%H%M%S-%N)"
    command=(
        "$VLA_PYTHON" experiments/robot/libero/run_libero_eval.py
        --model_family openvla --pretrained_checkpoint "$CHECKPOINT"
        --task_suite_name libero_spatial --center_crop True
        --use_l1_regression False --use_diffusion False --use_flow_matching True
        --num_flow_matching_steps_inference "$inference_steps"
        --load_in_8bit True --num_images_in_input 1 --use_proprio False
        --num_open_loop_steps 8 --num_steps_wait 10
        --seed "$SEED" --num_trials_per_task "$TRIALS" --use_wandb False
        --local_log_dir "$destination"
    )
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    mkdir -p "$destination"
    printf '%q ' "${command[@]}" > "$destination/command.sh"
    printf '\n' >> "$destination/command.sh"
    "${command[@]}" 2>&1 | tee "$destination/stdout.log"
done
