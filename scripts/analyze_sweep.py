"""
Explain the inverse relation between integration steps and success rate.

Two candidate causes are checked:
1. The integration drifts away from the trained time distribution.
2. The number of forward passes changes the number of noise draws, which changes
   the episode outcomes for reasons unrelated to the step count.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/ps/openvla/analysis/results/paradigm")
action_stats = json.loads((ROOT / "action_statistics.json").read_text())
sweep = json.loads((ROOT / "steps_sweep.json").read_text())

print("=== 流匹配：推理步数与首步动作误差 ===")
per_step = action_stats["fm"].get("per_step_mean_l1", {})
for step in sorted(per_step, key=int):
    print(f"  steps={step:>3}: mean_l1 {per_step[step]:.4f}")

print("\n=== 闭环成功率 ===")
for row in sweep:
    print(f"  steps={row['steps']:>3}: {row['successes']}/{row['episodes']} = "
          f"{row['successes']/row['episodes']:.0%}")

steps = np.array([row["steps"] for row in sweep], dtype=float)
rates = np.array([row["successes"] / row["episodes"] for row in sweep])
order = np.argsort(steps)
correlation = np.corrcoef(steps[order], rates[order])[0, 1]
print(f"\n步数与成功率的相关系数: {correlation:.3f}")

# Binomial uncertainty at n = 100 per point.
p = rates.mean()
sigma = float(np.sqrt(p * (1 - p) / 100))
print(f"每档 100 次试验的二项标准误约 {sigma*100:.1f} 个百分点")
print("相邻档之间的差距是否超过 2 倍标准误：")
for i in range(len(sweep) - 1):
    gap = abs(rates[i + 1] - rates[i]) * 100
    print(f"  {sweep[i]['steps']:>3} vs {sweep[i+1]['steps']:>3}: {gap:.1f} 点 -> "
          f"{'显著' if gap > 2 * sigma * 100 else '不显著'}")
