"""
Rebuild the per-episode outcomes of two concurrent LIBERO evaluations.

The evaluation writes one MP4 per episode and encodes the success flag in the
file name. When two runs agree on an episode the names are identical and one
file replaces the other; when they disagree both files survive. Sorting the
surviving files by modification time therefore separates the two runs, because
the faster run finishes each episode first.

The reconstruction is accepted only when it reproduces the totals that each
evaluation process reported on its own output.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(r"episode=(\d+)--success=(True|False)--task=(.*)\.mp4$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--known", required=True,
                        help='JSON mapping label to reported success count, ordered fastest run first')
    args = parser.parse_args()

    known = json.loads(args.known)
    labels = list(known)
    if len(labels) != 2:
        raise SystemExit("exactly two runs are expected")

    by_episode = defaultdict(list)
    for path in args.rollouts.glob("*.mp4"):
        match = PATTERN.search(path.name)
        if match:
            by_episode[int(match[1])].append(
                dict(success=match[2] == "True", task=match[3],
                     mtime=path.stat().st_mtime, file=path.name))

    records = {label: {} for label in labels}
    task_of = {}
    disagreements = 0
    for episode, items in sorted(by_episode.items()):
        items.sort(key=lambda item: item["mtime"])
        task_of[episode] = items[0]["task"]
        if len(items) == 1:
            for label in labels:
                records[label][episode] = items[0]["success"]
        else:
            disagreements += 1
            # The earlier file belongs to the faster run.
            records[labels[0]][episode] = items[0]["success"]
            records[labels[1]][episode] = items[1]["success"]

    summary = {}
    for label in labels:
        data = records[label]
        ok = sum(data.values())
        if ok != known[label]:
            raise SystemExit(f"{label}: reconstructed {ok}, expected {known[label]}")
        summary[label] = dict(episodes=len(data), successes=ok, success_rate=ok / len(data))
        out = args.out_dir / f"{label}_episodes.csv"
        with out.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["run_id", "task", "episode", "success"])
            for episode in sorted(data):
                writer.writerow([label, task_of[episode], episode, int(data[episode])])
        print(f"{label}: {ok}/{len(data)} = {ok/len(data):.1%} -> {out.name}")

    summary["episodes_with_disagreeing_outcomes"] = disagreements
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "episodes_reconstruction.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"episodes where the two runs disagreed: {disagreements}")


if __name__ == "__main__":
    main()
