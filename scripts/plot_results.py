#!/usr/bin/env python3
"""Regenerate figures using only the published CSV/JSON records."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    with (root / "results/training/oft.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    fig, (ax, lr_ax) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    loss = [float(row["loss"]) for row in rows]
    # The moving average is only a visual aid; CSV values remain unchanged.
    smooth = [sum(loss[max(0, i - 19):i + 1]) / min(i + 1, 20)
              for i in range(len(loss))]
    ax.plot(steps, loss, color="#9ab8cf", alpha=.5, linewidth=.7,
            label="Logged training L1")
    ax.plot(steps, smooth, color="#245b80", linewidth=1.8,
            label="20-point moving mean")
    ax.set_ylabel("Normalized action L1")
    ax.set_title("OpenVLA-OFT / Spatial / 8-bit + LoRA")
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    lr_ax.plot(steps, [float(row["learning_rate"]) for row in rows],
               color="#ae7032")
    lr_ax.set_xlabel("Logged training step")
    lr_ax.set_ylabel("LR")
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lr_ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output / "training.png")
    plt.close(fig)

    summary = {row["run_id"]: row
               for row in json.loads((root / "results/summary.json").read_text())}
    runs = ["openvla_spatial", "openvla_object", "openvla_goal", "oft_spatial"]
    labels = ["OpenVLA\nSpatial", "OpenVLA\nObject", "OpenVLA\nGoal",
              "Local OFT\nSpatial"]
    values = [summary[name]["success_rate"] * 100 for name in runs]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    bars = ax.bar(labels, values, width=.58,
                  color=["#99b1c2"] * 3 + ["#245b80"])
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=5)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("LIBERO closed-loop evaluation / 500 trials per suite")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.15)
    fig.text(.5, .015, "Results aggregated from archived episode logs.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / "evaluation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
