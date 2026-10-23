# 动作生成范式对比 · 训练监控

## 目录

```text
/home/ps/openvla/analysis/monitoring/
├── collect_metrics.py          # 只读采集器：导出训练曲线、GPU 采样与进度
├── report.py                   # 打印当前状态
├── plot_paradigm.py            # 绘制训练曲线与 GPU 占用图
├── run_collector.sh            # 后台采集循环（tmux 会话 collector）
├── check_paradigm.sh           # 定时检查入口（cron 每 30 分钟）
├── status.json                 # 最近一次采集的状态快照
├── gpu_samples.csv             # GPU 显存/利用率/功耗/温度采样
├── training/
│   ├── fm_metrics.csv          # 流匹配训练指标
│   └── diff_metrics.csv        # 扩散训练指标
├── snapshots/                  # 每 30 分钟一份状态文本快照
├── collector.log               # 采集器运行日志
└── figures/                    # 生成的图表
```

## 记录的指标

训练指标来自 W&B 离线事件，每 10 步一个点：

| 列 | 含义 |
| --- | --- |
| `train_loss` | 训练损失。流匹配是速度场均方误差，扩散是噪声预测均方误差，两者量纲不同 |
| `curr_action_l1_loss` | 采样后当前动作的 L1，衡量生成动作与示教的接近程度 |
| `next_actions_l1_loss` | 采样后后续动作的 L1 |
| `learning_rate` | 学习率 |
| `grad_norm` | **裁剪前**的梯度总范数均值 |
| `grad_norm_max` | 裁剪前梯度总范数的窗口峰值 |

系统采样每 5 分钟一条，记录两张卡的显存占用、GPU 利用率、功耗和温度。

## 查看方式

```bash
# 当前状态
cd /home/ps/openvla/openvla-oft && .venv/bin/python /home/ps/openvla/analysis/monitoring/report.py

# 重新绘图
.venv/bin/python /home/ps/openvla/analysis/monitoring/plot_paradigm.py

# 手动触发一次定时检查
bash /home/ps/openvla/analysis/monitoring/check_paradigm.sh

# 查看历史快照
ls -lt /home/ps/openvla/analysis/monitoring/snapshots/ | head
```

## 运行中的会话

| tmux 会话 | 作用 |
| --- | --- |
| `paradigm_fm` | 流匹配训练（GPU 0） |
| `paradigm_diff` | 扩散训练（GPU 1） |
| `collector` | 后台指标采集循环 |

## 边界

监控脚本全部只读，不会启动、停止或重启训练进程。定时检查只生成快照和刷新 CSV。
