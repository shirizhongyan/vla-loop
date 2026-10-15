#!/usr/bin/env bash
# Sourced by explicit train/eval commands. Nothing starts by sourcing this file.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PUBLISH_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VLA_WORKSPACE="${VLA_WORKSPACE:-$HOME/vla-work}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}
