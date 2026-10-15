#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

command -v git >/dev/null
mkdir -p "$VLA_WORKSPACE"
for name in openvla openvla-oft LIBERO; do
    if [[ -e "$VLA_WORKSPACE/$name" ]]; then
        echo "Destination already exists; inspect it before continuing: $VLA_WORKSPACE/$name" >&2
        exit 1
    fi
done

# This only fetches source code and applies patches. No dependencies, data,
# checkpoints, training jobs or evaluation jobs are started.
python3 - "$PUBLISH_ROOT" "$VLA_WORKSPACE" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root, workspace = map(Path, sys.argv[1:])
versions = json.loads((root / "configs/versions.json").read_text())
for name in ("openvla", "openvla-oft", "LIBERO"):
    source = versions["sources"][name]
    target = workspace / name
    subprocess.run(["git", "clone", "--no-checkout", source["url"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", source["commit"]], check=True)
for name, patch in (("openvla", "openvla-eval.patch"), ("openvla-oft", "oft-16gb.patch")):
    subprocess.run(["git", "-C", str(workspace / name),
                    "apply", "--check", str(root / "patches" / patch)], check=True)
    subprocess.run(["git", "-C", str(workspace / name),
                    "apply", str(root / "patches" / patch)], check=True)
print("Source preparation complete. Install dependencies and provide model/data separately.")
PY
