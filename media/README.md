# 操作视频

两个片段来自已完成的 LIBERO-Spatial 评测，左侧为官方套件微调权重的 OpenVLA，右侧为本项目训练的 OpenVLA-OFT。

| 片段 | 任务顺序 | 任务内试验号 | 全局试验号 | OpenVLA | OFT |
| --- | ---: | ---: | ---: | --- | --- |
| 桌面中央取碗 | 3 | 1 | 101 | 未完成 | 成功 |
| 木柜顶层抽屉取碗 | 5 | 3 | 203 | 未完成 | 成功 |

## 配对方式

按完整任务指令与任务内试验号配对，并与 `results/episodes.csv` 中的成功状态核对。两套评测均按同一固定版本的 LIBERO 任务顺序、默认初始状态列表运行，环境种子为 0。

原版评测使用 `initial_states[episode_idx]`，OFT 的 `initial_states_path=DEFAULT` 路径同样使用该索引。两组视频在统一到 224 × 224 后，首帧 RGB 平均绝对像素差分别约 1.88 与 1.82，来源包括原视频分辨率及编码差异。

片段按 OpenVLA 未完成、OFT 完成的对应结果选取，用于展示具体执行过程。完整套件表现保留在评测统计中。

## 视频制作

- 原始视频均为每秒 30 帧。
- 两侧均从首帧开始，等比显示完整方形画面。
- MP4 保留原帧率，GIF 以每秒 12 帧预览，播放时长保持一致。
- 任务较早结束的一侧保留末帧，直至另一侧结束。
- 仅添加模型名称与日志记录的结果，未剪除试验中间片段。

原始片段文件名、哈希与渲染后文件信息见 [sources.json](sources.json)。

## 重建示例

```bash
python3 scripts/render_comparison.py \
  /path/to/openvla-rollout.mp4 \
  /path/to/oft-rollout.mp4 \
  media/table-center \
  --font /path/to/DejaVuSans.ttf
```

脚本依赖 FFmpeg / ffprobe。标题中的结果标签对应本次选定的 OpenVLA 未完成、OFT 成功案例。
