#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to the matching official LIBERO-finetuned model}"
SUITE="${SUITE:-spatial}"
case "$SUITE" in
    spatial|object|goal) ;;
    *) echo "Supported suites: spatial, object, goal" >&2; exit 1 ;;
esac
OPENVLA_SOURCE="$VLA_WORKSPACE/openvla"
VLA_PYTHON="${VLA_PYTHON:-$OPENVLA_SOURCE/.venv/bin/python}"
require_file "$VLA_PYTHON"
require_file "$CHECKPOINT/config.json"
cd "$OPENVLA_SOURCE"
export PYTHONPATH="$OPENVLA_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
"$VLA_PYTHON" experiments/robot/libero/run_libero_eval.py \
    --model_family openvla --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name "libero_$SUITE" --center_crop True \
    --num_trials_per_task "${TRIALS:-50}" --use_wandb False
