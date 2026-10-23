#!/usr/bin/env bash
# 定时检查: 只读快照, 不做任何自动重启或干预。
STAMP=$(date +%Y%m%d-%H%M%S)
cd /home/ps/openvla/openvla-oft
.venv/bin/python /home/ps/openvla/analysis/monitoring/report.py > /home/ps/openvla/analysis/monitoring/snapshots/report-$STAMP.txt 2>&1
.venv/bin/python /home/ps/openvla/analysis/monitoring/collect_metrics.py >> /home/ps/openvla/analysis/monitoring/collector.log 2>&1
# 只保留最近 120 份快照
ls -1t /home/ps/openvla/analysis/monitoring/snapshots/report-*.txt 2>/dev/null | tail -n +121 | xargs -r rm -f
