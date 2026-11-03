"""
Plot the flow-matching inference step sweep.

Reads the per-step-count evaluation logs and the shared observation set.
"""
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SWEEP = Path("/home/ps/openvla/steps_sweep")
RESULTS = Path("/home/ps/openvla/analysis/results/paradigm")
OUTPUT = Path("/home/ps/openvla/analysis/results/paradigm")
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(FONT).is_file():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def load():
    rows = []
    for step_dir in sorted(SWEEP.glob("steps_*"), key=lambda p: int(p.name.split("_")[1])):
        step = int(step_dir.name.split("_")[1])
        log = next(step_dir.glob("EVAL-*.txt"))
        text = log.read_text(errors="replace")
        tasks = re.findall(r"^Task: (.+)$", text, re.M)
        successes = re.findall(r"^Success: (True|False)$", text, re.M)
        aggregate = OrderedDict()
        for task, success in zip(tasks, successes):
            entry = aggregate.setdefault(task.strip(), [0, 0])
            entry[0] += 1
            entry[1] += success == "True"
        rows.append(dict(steps=step, episodes=sum(v[0] for v in aggregate.values()),
                         successes=sum(v[1] for v in aggregate.values()),
                         per_task={k: dict(episodes=v[0], successes=v[1]) for k, v in aggregate.items()}))
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "axes.unicode_minus": False})
    steps = [row["steps"] for row in rows]
    rates = [row["successes"] / row["episodes"] * 100 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    axes[0].plot(steps, rates, marker="o", color="#2f6f9f", linewidth=2)
    axes[0].axhline(95.6, color="#b8763c", linestyle="--", linewidth=1.4)
    axes[0].text(1.4, 96.4, "L1 回归基线 95.6%（500 次）", color="#b8763c", fontsize=10)
    axes[0].set_xlabel("流匹配推理积分步数")
    axes[0].set_ylabel("闭环成功率 (%)")
    axes[0].set_ylim(60, 102)
    axes[0].set_title("步数越少，成功率越高", loc="left")
    axes[0].grid(alpha=.16)

    tasks = list(rows[0]["per_task"].keys())
    matrix = np.array([[row["per_task"][task]["successes"] / row["per_task"][task]["episodes"] * 100
                        if task in row["per_task"] else np.nan for task in tasks] for row in rows])
    image = axes[1].imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    axes[1].set_yticks(range(len(tasks)))
    axes[1].set_yticklabels([t.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", "")
                              .replace("_and_place_it_on_the_pla", "").replace("pick_up_the_black_bowl_", "")
                              for t in tasks], fontsize=8)
    axes[1].set_xticks(range(len(steps)))
    axes[1].set_xticklabels([str(s) for s in steps])
    axes[1].set_xlabel("推理积分步数")
    axes[1].set_title("逐任务成功率 (%)", loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                axes[1].text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.03)
    fig.suptitle("OpenVLA-OFT 流匹配 · 推理步数扫描（每档 100 次闭环试验）",
                 x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUTPUT / "steps_sweep.png")

    with (OUTPUT / "steps_sweep.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["steps", "episodes", "successes", "success_rate"])
        for row in rows:
            writer.writerow([row["steps"], row["episodes"], row["successes"],
                             row["successes"] / row["episodes"]])
    (OUTPUT / "steps_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Figures and tables written to {OUTPUT}")


if __name__ == "__main__":
    main()
