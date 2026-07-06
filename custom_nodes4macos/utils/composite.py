"""Multi-character compositing engine — ffmpeg overlay for multiple avatars on one scene."""
from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger("custom_nodes4macos.utils.composite")


def composite_scene(
    background: str,
    avatar_layers: list[dict],
    output_path: str,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Composite multiple avatar videos onto a background scene.

    Args:
        background: Path to background image/video.
        avatar_layers: List of dicts with keys:
            - video_path: str — avatar video clip path
            - x: int — horizontal position (0=left)
            - y: int — vertical position (0=top)
            - scale: float — size multiplier (0.0-1.0)
            - z_index: int — layer order (higher = on top)
        output_path: Where to save the composited clip.
        width, height: Output video dimensions.

    Returns:
        output_path on success.
    """
    if not avatar_layers:
        logger.warning("composite_scene: no avatar layers, copying background")
        import shutil
        shutil.copy2(background, output_path)
        return output_path

    # Build ffmpeg filter_complex for multi-layer compositing
    filter_parts = []
    input_count = 1  # background is input 0

    # Scale and position each avatar
    for i, layer in enumerate(avatar_layers):
        idx = input_count
        input_count += 1
        s = layer.get("scale", 0.3)
        x = layer.get("x", 0)
        y = layer.get("y", 0)
        # Scale to fit and position
        filter_parts.append(
            f"[{idx}:v]scale=iw*{s}:ih*{s}:force_original_aspect_ratio=decrease[av{i}]"
        )

    # Start with background, overlay each avatar
    overlay_chain = "[0:v]"
    for i in range(len(avatar_layers)):
        overlay_chain += f"[av{i}]overlay={avatar_layers[i].get('x', 0)}:{avatar_layers[i].get('y', 0)}[ov{i}]"
        if i < len(avatar_layers) - 1:
            overlay_chain = f"[ov{i}]"

    filter_complex = ";".join(filter_parts)
    if overlay_chain:
        filter_complex += ";" + overlay_chain

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    cmd += ["-loop", "1", "-i", background]  # background
    for layer in avatar_layers:
        cmd += ["-i", layer["video_path"]]
    cmd += ["-filter_complex", filter_complex, "-map", f"[ov{len(avatar_layers)-1}]"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            logger.error("composite failed: %s", proc.stderr[-500:])
            raise RuntimeError(f"Composite failed: {proc.stderr[-200:]}")
        logger.info("composite_scene -> %s (%d layers)", output_path, len(avatar_layers))
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("Composite timed out")


def multi_layer_composite(
    clips: list[str],
    layout: str = "side_by_side",
    output_path: str = "",
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Simple multi-clip layout: side_by_side, pip (picture-in-picture).

    Args:
        clips: List of video paths.
        layout: 'side_by_side' | 'pip' (right-bottom overlay).
        output_path: Output path.
    """
    if not clips:
        raise ValueError("No clips provided")
    if len(clips) == 1:
        import shutil
        shutil.copy2(clips[0], output_path)
        return output_path

    n = len(clips)

    if layout == "side_by_side":
        # Split screen equally
        seg_w = width // n
        filter_parts = []
        for i, clip in enumerate(clips):
            filter_parts.append(f"[{i}:v]scale={seg_w}:{height}:force_original_aspect_ratio=decrease,pad={seg_w}:{height}:(ow-iw)/2:(oh-ih)/2[vid{i}]")
        hstack_in = "".join(f"[vid{i}]" for i in range(n))
        filter_parts.append(f"{hstack_in}hstack=inputs={n}[out]")

        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for clip in clips:
            cmd += ["-i", clip]
        cmd += ["-filter_complex", ";".join(filter_parts), "-map", "[out]"]
    else:
        # PIP: main video + small overlay
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", clips[0], "-i", clips[1],
               "-filter_complex",
               "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2[bg];"
               "[1:v]scale=iw*0.25:ih*0.25[fg];"
               "[bg][fg]overlay=W-w-20:H-h-20[out]",
               "-map", "[out]"]

    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"Multi-composite failed: {proc.stderr[-200:]}")
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("Multi-composite timed out")
