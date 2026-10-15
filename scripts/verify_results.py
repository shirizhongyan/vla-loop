#!/usr/bin/env python3
"""Cross-check published episode CSV, task CSV, log counters, hashes and summary."""
import csv
import hashlib
import json
from pathlib import Path

from summarize_eval import parse_log, summarize


def main():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "results/summary.json").read_text())
    with (root / "results/episodes.csv").open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    with (root / "results/tasks.csv").open(newline="") as stream:
        tasks = list(csv.DictReader(stream))
    for record in records:
        rows, errors = parse_log((root / record["evidence"]).read_text())
        assert not errors, f'{record["run_id"]}: runtime errors in selected log'
        actual = summarize(rows)
        for key in ("episodes", "successes", "success_rate"):
            assert actual[key] == record[key], (record["run_id"], key)
        selected = [row for row in episodes if row["run_id"] == record["run_id"]]
        assert len(selected) == len(rows)
        for left, right in zip(selected, rows):
            assert left["task"] == right["task"]
            assert int(left["episode"]) == right["episode"]
            assert int(left["success"]) == int(right["success"])
        selected_tasks = [row for row in tasks if row["run_id"] == record["run_id"]]
        assert len(selected_tasks) == len(actual["tasks"])
        for left, right in zip(selected_tasks, actual["tasks"]):
            assert left["task"] == right["task"]
            assert int(left["episodes"]) == right["episodes"]
            assert int(left["successes"]) == right["successes"]
            assert abs(float(left["success_rate"]) - right["success_rate"]) < 1e-12
    manifest = json.loads((root / "results/source_manifest.json").read_text())
    for entry in manifest["published_files"]:
        actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        assert actual == entry["sha256"], entry["path"]
    with (root / "results/training/oft.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 1001
    assert int(history[0]["step"]) == 0 and int(history[-1]["step"]) == 10000
    assert abs(float(history[-1]["loss"]) - 0.09079742431640625) < 1e-12
    print(f"Verified {len(records)} runs, {len(episodes)} outcomes, "
          f"{len(history)} training records and published evidence hashes.")


if __name__ == "__main__":
    main()
