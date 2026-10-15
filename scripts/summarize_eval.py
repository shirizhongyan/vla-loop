#!/usr/bin/env python3
"""Parse the task/episode/result lines of one OpenVLA or OFT evaluation log."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import re


def parse_log(text):
    rows = []
    task = None
    episode = None
    errors = []
    for line in text.splitlines():
        if line.startswith("Task: "):
            task = line.removeprefix("Task: ").strip()
            episode = None
        elif match := re.fullmatch(r"Starting episode (\d+)\.\.\.", line):
            episode = int(match[1])
        elif match := re.fullmatch(r"Success: (True|False)", line):
            if task is None or episode is None:
                raise ValueError("Outcome without a preceding task and episode")
            rows.append({"task": task, "episode": episode, "success": match[1] == "True"})
            episode = None
        elif line.startswith("Episode error:"):
            errors.append(line)
    if not rows:
        raise ValueError("No completed episodes found")
    seen = set()
    for row in rows:
        key = row["task"], row["episode"]
        if key in seen:
            raise ValueError(f"Duplicate task/episode: {key}. Do not concatenate runs.")
        seen.add(key)
    counters = re.findall(r"# episodes completed so far: (\d+)", text)
    successes = re.findall(r"# successes: (\d+)", text)
    if counters and int(counters[-1]) != len(rows):
        raise ValueError("Episode count disagrees with the final logged counter")
    if successes and int(successes[-1]) != sum(row["success"] for row in rows):
        raise ValueError("Success count disagrees with the final logged counter")
    return rows, errors


def summarize(rows):
    tasks = OrderedDict()
    for row in rows:
        entry = tasks.setdefault(row["task"], {"episodes": 0, "successes": 0})
        entry["episodes"] += 1
        entry["successes"] += row["success"]
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "tasks": [
            {"task": task, **counts,
             "success_rate": counts["successes"] / counts["episodes"]}
            for task, counts in tasks.items()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows, errors = parse_log(args.log.read_text())
    print(json.dumps({**summarize(rows), "episode_errors": len(errors)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
