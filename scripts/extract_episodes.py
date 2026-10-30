"""
Extract per-episode outcomes from a captured LIBERO evaluation stdout log.

The Rich console wraps every message with a fixed prefix, so the plain-text
patterns are matched loosely and the task/success pairs are recovered in order.
"""
import argparse
import csv
import re
from pathlib import Path


def extract(path):
    text = Path(path).read_text(errors="replace")
    tasks = []
    successes = []
    for line in text.splitlines():
        if "| >>" not in line:
            continue
        message = line.split("| >>", 1)[1]
        message = message.split("run_libero_eval.py")[0].strip()
        if message.startswith("Task: "):
            tasks.append(message[len("Task: "):].strip())
        elif message.startswith("Success: "):
            successes.append(message[len("Success: "):].strip() == "True")
    return tasks, successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=500)
    args = parser.parse_args()

    tasks, successes = extract(args.log)
    if len(successes) < args.expected:
        raise SystemExit(f"only {len(successes)} outcomes found, expected {args.expected}")
    if len(tasks) != len(successes):
        raise SystemExit(f"task/outcome count mismatch: {len(tasks)} vs {len(successes)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["run_id", "task", "episode", "success"])
        for index, (task, success) in enumerate(zip(tasks, successes), start=1):
            writer.writerow([args.out.stem, task, index, int(success)])
    print(f"{args.out.name}: {len(successes)} episodes, {sum(successes)} successes")


if __name__ == "__main__":
    main()
