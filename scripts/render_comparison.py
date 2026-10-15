#!/usr/bin/env python3
"""Create synchronized side-by-side MP4/GIF previews from two rollout videos."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height",
        "-of", "json", str(path),
    ], text=True))["streams"][0]


def render(left, right, output, font):
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required executable: {command}")
    streams = [probe(path) for path in (left, right)]
    if any(stream["r_frame_rate"] != "30/1" for stream in streams):
        raise ValueError("This renderer expects the archived 30 fps rollouts.")
    frame_count = max(int(stream["nb_frames"]) for stream in streams)
    width, image_height, header = 320, 320, 44
    panels = []
    for i, label, color in (
        (0, "OpenVLA | Not completed", "0xd68372"),
        (1, "OFT | Success", "0x8bc8a7"),
    ):
        hold = frame_count - int(streams[i]["nb_frames"])
        panels.append(
            f"[{i}:v]setpts=PTS-STARTPTS,"
            f"scale={width}:{image_height}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop={hold},"
            f"pad={width}:{image_height + header}:0:{header}:color=0x14212b,"
            f"drawtext=fontfile={font}:text='{label}':"
            f"fontsize=19:fontcolor={color}:x=12:y=12[p{i}]"
        )
    filters = ";".join(panels) + ";[p0][p1]hstack=inputs=2[out]"
    output.parent.mkdir(parents=True, exist_ok=True)
    mp4 = output.with_suffix(".mp4")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(left), "-i", str(right),
        "-filter_complex", filters, "-map", "[out]", "-frames:v", str(frame_count),
        "-an", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4),
    ], check=True)
    # Change preview sampling only; playback duration and MP4 frame timing remain.
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(mp4),
        "-filter_complex",
        "[0:v]fps=12,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0", str(output.with_suffix(".gif")),
    ], check=True)
    return frame_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openvla", type=Path)
    parser.add_argument("oft", type=Path)
    parser.add_argument("output", type=Path, help="Output path without extension")
    parser.add_argument(
        "--font", type=Path,
        default=Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    args = parser.parse_args()
    if not args.font.is_file():
        raise FileNotFoundError("Supply a readable TrueType font via --font")
    render(args.openvla, args.oft, args.output, args.font)


if __name__ == "__main__":
    main()
