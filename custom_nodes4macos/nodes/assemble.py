import logging
import os
import time
import uuid

from .. import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.nodes.assemble")

_TRANSITIONS = ["none", "fade"]
_BGM_VOLUME = 0.3


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


def _parse_clips(text: str) -> list[str]:
    clips = []
    for raw in (text or "").splitlines():
        p = raw.strip().strip('"').strip("'")
        if not p:
            continue
        if not os.path.exists(p):
            logger.warning("assemble 跳过不存在的片段: %s", p)
            continue
        clips.append(p)
    return clips


class FusionMLXAssemble:
    """多段 mp4 → 9:16 单片（concat + 可选 BGM + 可选淡入淡出）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("STRING", {"multiline": True, "default": ""}),
                "transition": (_TRANSITIONS, {"default": "none"}),
                "width": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 2}),
                "height": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 2}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "filename_prefix": ("STRING", {"default": "assemble"}),
            },
            "optional": {
                "bgm_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render"
    CATEGORY = "FusionMLX/Horror"

    def render(self, clips, transition, width, height, fps, filename_prefix, bgm_path=""):
        valid = _parse_clips(clips)
        if len(valid) < 1:
            raise RuntimeError("assemble 没有可用片段（clips 为空或全部不存在）")

        out_dir = _output_directory()
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        out_path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}.mp4")

        audio_flags = [ffmpeg_util.probe_has_audio(p) for p in valid]
        all_audio = all(audio_flags)
        any_audio = any(audio_flags)
        use_clip_audio = all_audio
        if any_audio and not all_audio:
            logger.warning("assemble 片段音频不一致（部分无音轨），本次丢弃片段音频，仅用 BGM")
            use_clip_audio = False

        bgm_ok = bool(bgm_path) and os.path.exists(bgm_path) and ffmpeg_util.probe_has_audio(bgm_path)
        if bgm_path and not bgm_ok:
            logger.warning("assemble BGM 不可用（不存在或无音轨）: %s", bgm_path)

        n = len(valid)
        filters = []
        if use_clip_audio:
            seg_labels = []
            for i, _ in enumerate(valid):
                v = f"v{i}"
                a = f"a{i}"
                filters.append(
                    f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"fps={fps},setsar=1,format=yuv420p[{v}]"
                )
                filters.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[{a}]")
                seg_labels.append(f"[{v}][{a}]")
            concat_in = "".join(seg_labels)
            filters.append(f"{concat_in}concat=n={n}:v=1:a=1[vcat][acat]")
        else:
            v_labels = []
            for i, _ in enumerate(valid):
                v = f"v{i}"
                v_labels.append(f"[{v}]")
                filters.append(
                    f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"fps={fps},setsar=1,format=yuv420p[{v}]"
                )
            concat_in = "".join(v_labels)
            filters.append(f"{concat_in}concat=n={n}:v=1:a=0[vcat]")

        video_label = "[vcat]"
        if transition == "fade":
            total = sum(ffmpeg_util.probe_duration(p) for p in valid)
            fade_out_start = max(0.0, total - 0.4)
            filters.append(
                f"[vcat]fade=t=in:st=0:d=0.4,fade=t=out:st={fade_out_start:.3f}:d=0.4[vfade]"
            )
            video_label = "[vfade]"

        bgm_index = n
        if use_clip_audio and bgm_ok:
            filters.append(
                f"[{bgm_index}:a]volume={_BGM_VOLUME},aresample=44100,"
                f"aformat=channel_layouts=stereo[bgm]"
            )
            filters.append("[acat][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]")
            audio_map = "[aout]"
        elif use_clip_audio:
            audio_map = "[acat]"
        elif bgm_ok:
            filters.append(
                f"[{bgm_index}:a]aresample=44100,aformat=channel_layouts=stereo[aout]"
            )
            audio_map = "[aout]"
        else:
            audio_map = None

        filter_complex = ";".join(filters)
        args = []
        for p in valid:
            args += ["-i", p]
        if bgm_ok:
            args += ["-i", bgm_path]
        args += ["-filter_complex", filter_complex]
        args += ["-map", video_label]
        if audio_map:
            args += ["-map", audio_map]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
        if bgm_ok or use_clip_audio:
            args += ["-shortest"]
        args.append(out_path)

        logger.info("assemble 渲染 n=%d audio=clip:%s bgm:%s transition=%s out=%s",
                    n, use_clip_audio, bgm_ok, transition, out_path)
        ffmpeg_util.run_ffmpeg(args, timeout=900, label=f"assemble n={n}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"assemble 输出为空: {out_path}")
        dur = ffmpeg_util.probe_duration(out_path)
        logger.info("assemble 完成 out=%s dur=%.2fs size=%d", out_path, dur, os.path.getsize(out_path))
        return (out_path,)
