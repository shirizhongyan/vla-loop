#!/usr/bin/env python3
"""Plot verified subset and full-suite results without accessing the GPU server."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (root / "results/paradigm/steps_sweep_verified.json").read_text()
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, scope, title in zip(
        axes, ("subset", "full"), ("Exploratory: 100 episodes", "Full suite: 500 episodes")
    ):
        rows = [row for row in document["runs"] if row["scope"] == scope]
        steps = [row["steps"] for row in rows]
        rates = [row["success_rate"] * 100 for row in rows]
        errors = [
            [row["success_rate"] * 100 - row["wilson_95_interval"][0] * 100 for row in rows],
            [row["wilson_95_interval"][1] * 100 - row["success_rate"] * 100 for row in rows],
        ]
        axis.errorbar(steps, rates, yerr=errors, marker="o", capsize=4, color="#286b96")
        axis.axhline(95.6, linestyle="--", color="#b06e30", label="Historical L1: 95.6%")
        axis.set(title=title, xlabel="Euler integration steps", ylabel="Success rate (%)")
        axis.set_ylim(60, 100)
        axis.set_xticks(steps)
        axis.grid(alpha=0.2)
        for step, rate in zip(steps, rates):
            axis.annotate(f"{rate:.1f}%", (step, rate), xytext=(4, 7),
                          textcoords="offset points", fontsize=9)
        axis.legend(fontsize=8)
    figure.suptitle("Flow matching: fixed 10k checkpoint; Wilson 95% intervals")
    figure.savefig(root / "figures/paradigm_steps_sweep_full.png", dpi=160)


if __name__ == "__main__":
    main()
