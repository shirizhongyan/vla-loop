"""
Plot the paradigm comparison: training metrics and GPU load.

Reads only the exported CSV files under /home/ps/openvla/analysis/monitoring.
"""
import csv
from pathlib import Path

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
for candidate in FONT_CANDIDATES:
    if Path(candidate).is_file():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break

MONITORING = Path("/home/ps/openvla/analysis/monitoring")
OUTPUT = Path("/home/ps/openvla/analysis/monitoring/figures")
LABELS = {"fm": ("Flow Matching", "#2f6f9f"), "diff": ("Diffusion", "#b8763c")}
LABELS_ZH = {"fm": "流匹配", "diff": "扩散"}


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, column):
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return values


def smooth(values, width=21):
    array = np.asarray(values, dtype=float)
    output = []
    for index in range(len(array)):
        window = array[max(0, index - width + 1):index + 1]
        finite = window[np.isfinite(window)]
        output.append(finite.mean() if len(finite) else np.nan)
    return output


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "axes.unicode_minus": False})
    panels = [
        ("train_loss", "训练损失 ↓"),
        ("curr_action_l1_loss", "当前动作 L1 ↓"),
        ("next_actions_l1_loss", "后续动作 L1 ↓"),
        ("grad_norm", "梯度范数（裁剪前）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for axis, (column, title) in zip(axes.flatten(), panels):
        for key, (name, color) in LABELS.items():
            rows = read_csv(MONITORING / "training" / f"{key}_metrics.csv")
            if not rows:
                continue
            x = numeric(rows, "step")
            y = numeric(rows, column)
            axis.plot(x, y, color=color, alpha=.25, linewidth=.6)
            axis.plot(x, smooth(y), color=color, linewidth=1.7, label=LABELS_ZH[key])
        axis.set_title(title, loc="left")
        axis.set_xlabel("训练步号")
        axis.grid(alpha=.16)
        axis.legend(frameon=False)
    fig.suptitle("OpenVLA-OFT · 动作生成范式对比（流匹配 / 扩散）", x=.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUTPUT / "training_paradigms.png")
    plt.close(fig)

    gpu = read_csv(MONITORING / "gpu_samples.csv")
    if gpu:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for index in (0, 1):
            subset = [row for row in gpu if int(row["gpu_index"]) == index]
            if not subset:
                continue
            def parse_time(value):
                return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").timestamp()

            start = parse_time(subset[0]["timestamp"])
            x = [(parse_time(row["timestamp"]) - start) / 3600 for row in subset]
            axes[0].plot(x, [int(row["memory_used_mib"]) / 1024 for row in subset],
                         label=f"GPU {index}", linewidth=1.4)
            axes[1].plot(x, [int(row["utilization_percent"]) for row in subset],
                         label=f"GPU {index}", linewidth=1.4)
        axes[0].set_ylabel("显存占用 (GiB)")
        axes[1].set_ylabel("GPU 利用率 (%)")
        for axis in axes:
            axis.set_xlabel("自采集起的小时数")
            axis.grid(alpha=.16)
            axis.legend(frameon=False)
        fig.suptitle("训练期 GPU 资源占用", x=.02, ha="left", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(OUTPUT / "gpu_usage.png")
        plt.close(fig)
    print(f"Figures written to {OUTPUT}")


if __name__ == "__main__":
    main()
