"""
Read-only collector for the paradigm comparison runs.

Extracts the W&B offline history of every run into a tidy CSV, samples the live
training logs for step counts, and records GPU memory and utilisation samples.

This script never starts, stops or restarts a training job.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

WANDB_ROOT = Path("/home/ps/openvla/openvla-oft/wandb")
OUT_ROOT = Path("/home/ps/openvla/analysis/monitoring")
LOG_ROOT = Path("/home/ps/openvla")

# W&B metric key -> column name
METRIC_COLUMNS = {
    "VLA Train/Loss": "train_loss",
    "VLA Train/Curr Action L1 Loss": "curr_action_l1_loss",
    "VLA Train/Next Actions L1 Loss": "next_actions_l1_loss",
    "VLA Train/Learning Rate": "learning_rate",
    "VLA Train/Grad Norm": "grad_norm",
    "VLA Train/Grad Norm Max": "grad_norm_max",
}


def read_history(run_dir):
    """Return the history rows of one W&B offline run."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2

    files = sorted(run_dir.glob("*.wandb"))
    if not files:
        return []
    store = DataStore()
    store.open_for_scan(str(files[-1]))
    rows = []
    while True:
        try:
            data = store.scan_data()
        except AssertionError:
            # The run file is still being written, so the last record may be
            # incomplete. Keep the records read so far.
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                continue
        if row:
            rows.append(row)
    return rows


def identify_runs():
    """
    Map each paradigm label to its W&B offline run directory.

    The run directory name does not appear in the training log, so the mapping is
    read from the run metadata: the trainer writes the run id into the W&B metadata
    and the paradigm label into the run id note.
    """
    runs = {}
    for run_dir in sorted(WANDB_ROOT.glob("offline-run-*")):
        metadata = run_dir / "files" / "wandb-metadata.json"
        if not metadata.is_file():
            continue
        try:
            data = json.loads(metadata.read_text())
        except json.JSONDecodeError:
            continue
        args = data.get("args") or []
        joined = " ".join(str(item) for item in args)
        for label in ("fm", "diff"):
            if f"openvla-7b-{label}" in joined:
                runs[label] = run_dir
    return runs


def sample_gpu():
    """One sample of GPU memory, utilisation, power and temperature."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        rows.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu_index": int(parts[0]),
            "memory_used_mib": int(parts[1]),
            "memory_total_mib": int(parts[2]),
            "utilization_percent": int(parts[3]),
            "power_watts": float(parts[4]),
            "temperature_c": int(parts[5]),
        })
    return rows


def sample_progress(label):
    """Extract the current step and loss from the live training log."""
    log = LOG_ROOT / f"paradigm_{label}.log"
    if not log.is_file():
        return {}
    text = log.read_text(errors="replace")
    steps = re.findall(r"(\d+)/10000 \[", text)
    step = int(steps[-1]) if steps else None
    elapsed = None
    rate = None
    match = re.findall(r"(\d+)/10000 \[(\d+):(\d+):(\d+)<([^,]+),\s*([\d.]+)s/it\]", text)
    if match:
        hours, minutes, seconds = (int(match[-1][1]), int(match[-1][2]), int(match[-1][3]))
        elapsed = hours * 3600 + minutes * 60 + seconds
        try:
            rate = float(match[-1][4])
        except ValueError:
            # tqdm sometimes prints the remaining time in the rate field
            rate = None
    return {"paradigm": label, "step": step, "max_steps": 10000,
            "elapsed_seconds": elapsed, "seconds_per_step": rate}


def write_history_csv(label, run_dir):
    rows = read_history(run_dir)
    if not rows:
        return 0
    columns = ["step", "runtime_seconds", "timestamp"] + list(METRIC_COLUMNS.values())
    target = OUT_ROOT / "training" / f"{label}_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = {"step": row.get("_step"), "runtime_seconds": row.get("_runtime"),
                      "timestamp": row.get("_timestamp")}
            for source, name in METRIC_COLUMNS.items():
                record[name] = row.get(source)
            writer.writerow(record)
    return len(rows)


def append_gpu_samples(rows):
    target = OUT_ROOT / "gpu_samples.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.is_file()
    with target.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-interval", type=int, default=0,
                        help="Seconds between GPU samples; 0 collects only once")
    parser.add_argument("--duration", type=int, default=0,
                        help="Total seconds to keep sampling; 0 runs a single pass")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.duration if args.duration else 0
    while True:
        runs = identify_runs()
        status = {"collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}}
        for label, run_dir in runs.items():
            count = write_history_csv(label, run_dir)
            status["runs"][label] = {
                "run_dir": run_dir.name,
                "history_points": count,
                **sample_progress(label),
            }
        append_gpu_samples(sample_gpu())
        (OUT_ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
        if not args.duration or time.time() >= deadline:
            break
        time.sleep(min(args.sample_interval, max(1, deadline - time.time())))


if __name__ == "__main__":
    main()
