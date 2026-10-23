"""
Print the current state of the paradigm comparison runs.

Read-only. Reports progress, latest training metrics, GPU load and the
checkpoint files that exist so far.
"""
import json
from pathlib import Path
import re
import subprocess

MONITORING = Path("/home/ps/openvla/analysis/monitoring")
RUNS = Path("/home/ps/openvla/paradigm_runs")
LOGS = Path("/home/ps/openvla")


def read_csv(path):
    import csv
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def tail_progress(label):
    log = LOGS / f"paradigm_{label}.log"
    if not log.is_file():
        return None, None
    text = log.read_text(errors="replace")
    steps = re.findall(r"(\d+)/10000", text)
    step = int(steps[-1]) if steps else 0
    return step, text


def main():
    status = {}
    status_path = MONITORING / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())

    print("=" * 78)
    print("动作生成范式对比 · 运行状态")
    print("=" * 78)
    for label, name in (("fm", "流匹配 FlowMatching"), ("diff", "扩散 Diffusion")):
        step, _ = tail_progress(label)
        info = status.get("runs", {}).get(label, {})
        rows = read_csv(MONITORING / "training" / f"{label}_metrics.csv")
        print(f"\n[{name}]  {info.get('run_dir', '-')}")
        print(f"  进度      : {step or info.get('step', 0)} / 10000 步")
        if info.get("elapsed_seconds"):
            elapsed = int(info["elapsed_seconds"])
            print(f"  已运行    : {elapsed // 3600} 小时 {(elapsed % 3600) // 60} 分钟")
        if rows:
            last = rows[-1]
            print(f"  最新 loss : {last.get('train_loss')}  (step {last.get('step')})")
            print(f"  动作 L1   : 当前 {last.get('curr_action_l1_loss')} / 后续 {last.get('next_actions_l1_loss')}")
            print(f"  梯度范数 : 均值 {last.get('grad_norm')} / 峰值 {last.get('grad_norm_max')}")
            print(f"  学习率   : {last.get('learning_rate')}")
            print(f"  已记录点  : {len(rows)}")
        checkpoints = sorted(RUNS.glob(f"*{label}*--*_chkpt")) + sorted(RUNS.glob("*--*_chkpt"))
        found = sorted({p.name for p in RUNS.glob("*_chkpt")})
        if found:
            print(f"  已保存权重: {len(found)} 个")

    gpu = read_csv(MONITORING / "gpu_samples.csv")
    if gpu:
        print("\n" + "-" * 78)
        print("GPU 采样（最近一次）")
        for row in gpu[-2:]:
            print(f"  GPU {row['gpu_index']}  显存 {row['memory_used_mib']}/{row['memory_total_mib']} MiB"
                  f"  利用率 {row['utilization_percent']}%  功耗 {row['power_watts']}W  温度 {row['temperature_c']}C")
        print(f"  累计采样 {len(gpu)} 条")
    print("=" * 78)


if __name__ == "__main__":
    main()
