#!/usr/bin/env bash
# 只读采集器: 定期导出训练曲线、GPU 采样与进度状态。
# 不会启动、停止或重启任何训练进程。
cd /home/ps/openvla/openvla-oft
while true; do
  .venv/bin/python /home/ps/openvla/analysis/monitoring/collect_metrics.py >> /home/ps/openvla/analysis/monitoring/collector.log 2>&1
  sleep 300
done
