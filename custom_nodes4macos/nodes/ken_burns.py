import logging
import os
import random
import time
import uuid

from .. import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.nodes.ken_burns")

_PRESETS = ["zoom-in", "zoom-out", "pan-right", "pan-left", "random"]


def _output_directory() -> str:
    try:
        from folder_paths import get_output_directory

        out = get_output_directory()
        if out:
            os.makedirs(out, exist_ok=True)
            return out
    except Exception as exc:
        logger.debug("folder_paths unavailable, fallback to tempfile: %s", exc)
    import tempfile

    return tempfile.gettempdir()


def _image_tensor_to_png(tensor, png_path: str) -> None:
    import numpy
    from PIL import Image

    arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else numpy.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        pass
    else:
        raise ValueError(f"不支持的图像张量形状: {arr.shape}")
    if arr.dtype != numpy.uint8:
        arr = numpy.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(numpy.uint8)
    Image.fromarray(arr).save(png_path)
    logger.info("ken_burns 写入源图 %s shape=%s", png_path, arr.shape)


def _build_zoompan(preset: str, out_w: int, out_h: int, fps: int, total_frames: int) -> str:
    chosen = preset
    if preset == "random":
        chosen = random.choice([p for p in _PRESETS if p != "random"])
    D = max(1, total_frames)
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"
    if chosen == "zoom-in":
        z = "1+0.25*on/" + str(D)
        x, y = cx, cy
    elif chosen == "zoom-out":
        z = "1.25-0.25*on/" + str(D)
        x, y = cx, cy
    elif chosen == "pan-right":
        z = "1.2"
        x = "(iw-iw/zoom)*(on/" + str(D) + ")"
        y = cy
    elif chosen == "pan-left":
        z = "1.2"
        x = "(iw-iw/zoom)*(1-on/" + str(D) + ")"
        y = cy
    else:
        z = "1+0.2*on/" + str(D)
        x, y = cx, cy
    return (f"scale={out_w * 2}:{out_h * 2}:force_original_aspect_ratio=increase,"
            f"crop={out_w * 2}:{out_h * 2},setsar=1,"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={D}:s={out_w}x{out_h}:fps={fps},"
            f"format=yuv420p")


class FusionMLXKenBurns:
    """单图 Ken Burns 推镜 → 9:16 mp4。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 120.0, "step": 0.1}),
                "motion_preset": (_PRESETS, {"default": "zoom-in"}),
                "width": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 2}),
                "height": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 2}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "filename_prefix": ("STRING", {"default": "ken_burns"}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render"
    CATEGORY = "FusionMLX/Horror"

    def render(self, image, duration_seconds, motion_preset, width, height, fps,
               filename_prefix, audio_path=""):
        import tempfile

        out_dir = _output_directory()
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        work_dir = tempfile.mkdtemp(prefix="ken_burns_")
        png_path = os.path.join(work_dir, "src.png")
        out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}.mp4")

        _image_tensor_to_png(image, png_path)

        total_frames = max(1, round(duration_seconds * fps))
        filter_chain = _build_zoompan(motion_preset, width, height, fps, total_frames)

        has_audio = bool(audio_path) and os.path.exists(audio_path) and ffmpeg_util.probe_has_audio(audio_path)
        if has_audio:
            args = ["-loop", "1", "-i", png_path, "-i", audio_path,
                    "-filter_complex", f"[0:v]{filter_chain}[v]",
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", out_path]
            label = f"ken_burns+audio {width}x{height} {total_frames}f"
        else:
            args = ["-loop", "1", "-i", png_path,
                    "-filter_complex", f"[0:v]{filter_chain}[v]",
                    "-map", "[v]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-t", f"{total_frames / fps:.3f}",
                    "-movflags", "+faststart", out_path]
            label = f"ken_burns {width}x{height} {total_frames}f"

        logger.info("ken_burns 渲染 preset=%s dur=%.2f audio=%s out=%s",
                    motion_preset, duration_seconds, has_audio, out_path)
        ffmpeg_util.run_ffmpeg(args, timeout=600, label=label)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ken_burns 输出为空: {out_path}")
        dur = ffmpeg_util.probe_duration(out_path)
        logger.info("ken_burns 完成 out=%s dur=%.2fs size=%d", out_path, dur, os.path.getsize(out_path))
        return (out_path,)
