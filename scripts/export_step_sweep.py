#!/usr/bin/env python3
"""Rebuild or verify step-sweep tables from the published, separate run logs."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from summarize_eval import parse_log, summarize


def wilson_interval(successes, episodes):
    rate = successes / episodes
    critical = 1.959963984540054
    denominator = 1 + critical**2 / episodes
    center = (rate + critical**2 / (2 * episodes)) / denominator
    half = critical * math.sqrt(
        rate * (1 - rate) / episodes + critical**2 / (4 * episodes**2)
    ) / denominator
    return [center - half, center + half]


def csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_outputs(root):
    protocol = json.loads((root / "configs/flow_matching_sweep.json").read_text())
    summaries, episodes, tasks, full_runs = [], [], [], {}
    for run in protocol["runs"]:
        run_id = f'flow_matching_steps_{run["steps"]}_{run["scope"]}'
        relative = f"results/paradigm/evaluations/{run_id}.txt"
        content = (root / relative).read_bytes()
        rows, errors = parse_log(content.decode())
        if errors:
            raise ValueError(f"{run_id}: episode errors: {errors}")
        actual = summarize(rows)
        for key in ("episodes", "successes"):
            if actual[key] != run[key]:
                raise ValueError(f"{run_id}: {key}: {actual[key]} != {run[key]}")
        expected_trials = run["episodes"] // 10
        if len(actual["tasks"]) != 10 or any(
            task["episodes"] != expected_trials for task in actual["tasks"]
        ):
            raise ValueError(f"{run_id}: incomplete task coverage")
        if any(
            sorted(row["episode"] for row in rows if row["task"] == task["task"])
            != list(range(1, expected_trials + 1))
            for task in actual["tasks"]
        ):
            raise ValueError(f"{run_id}: unexpected episode identifiers")
        summaries.append({
            "run_id": run_id, **run, "success_rate": actual["success_rate"],
            "wilson_95_interval": wilson_interval(run["successes"], run["episodes"]),
            "evidence": relative, "sha256": hashlib.sha256(content).hexdigest(),
        })
        episodes.extend(
            {"run_id": run_id, **row, "success": int(row["success"])} for row in rows
        )
        tasks.extend({"run_id": run_id, **task} for task in actual["tasks"])
        if run["scope"] == "full":
            full_runs[run["steps"]] = {
                (row["task"], row["episode"]): row["success"] for row in rows
            }
    first, second = full_runs[1], full_runs[2]
    if first.keys() != second.keys():
        raise ValueError("Full runs do not cover the same task/initial-state IDs")
    first_only = sum(first[key] and not second[key] for key in first)
    second_only = sum(second[key] and not first[key] for key in first)
    discordant = first_only + second_only
    probability = min(1.0, 2 * sum(
        math.comb(discordant, count)
        for count in range(min(first_only, second_only) + 1)
    ) / 2**discordant)
    document = {
        "protocol": {key: value for key, value in protocol.items() if key != "runs"},
        "runs": summaries,
        "paired_full_steps_1_vs_2": {
            "pairing": "Task description and LIBERO default initial-state index",
            "warning": "Same initial states do not imply identical sampled action noise.",
            "both_success": sum(first[key] and second[key] for key in first),
            "steps_1_only_success": first_only,
            "steps_2_only_success": second_only,
            "both_failure": sum(not first[key] and not second[key] for key in first),
            "mcnemar_exact_two_sided_p": probability,
        },
    }
    return {
        "results/paradigm/steps_sweep_verified.json":
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        "results/paradigm/steps_sweep_verified.csv": csv_text(
            ["run_id", "steps", "scope", "episodes", "successes", "success_rate"],
            [{key: row[key] for key in (
                "run_id", "steps", "scope", "episodes", "successes", "success_rate"
            )} for row in summaries],
        ),
        "results/paradigm/steps_sweep_episodes.csv": csv_text(
            ["run_id", "task", "episode", "success"], episodes,
        ),
        "results/paradigm/steps_sweep_tasks.csv": csv_text(
            ["run_id", "task", "episodes", "successes", "success_rate"], tasks,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    outputs = build_outputs(root)
    for relative, content in outputs.items():
        target = root / relative
        if args.check:
            if not target.is_file() or target.read_text() != content:
                raise ValueError(f"Stale or missing generated evidence: {relative}")
        else:
            target.write_text(content)
    print(f'{"Verified" if args.check else "Exported"} 8 runs / 1600 outcomes.')


if __name__ == "__main__":
    main()
