from __future__ import annotations

import logging
import os

from ..stage import Stage, StageInfo
from ... import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.assemble")

_TRANSITIONS = ["none", "fade"]
_BGM_VOLUME = 0.3


class AssembleStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="assemble",
            description="多段 clip → 最终 mp4 (concat + 可选 BGM + 可选淡入淡出)",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["clip"],
            output_kinds=["final"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        width = ctx.config.get("assemble_width", ctx.config.get("ken_burns_width", 1080))
        height = ctx.config.get("assemble_height", ctx.config.get("ken_burns_height", 1920))
        fps = ctx.config.get("assemble_fps", ctx.config.get("ken_burns_fps", 30))
        transition = ctx.config.get("assemble_transition", "none")
        bgm_path = ctx.config.get("assemble_bgm_path", "")

        clips = self._collect_clips(ctx)
        if not clips:
            raise RuntimeError("assemble: no clips available")

        out_path = ctx.artifact_path(0, "final")

        self._render_final(
            clips, width, height, fps, transition, bgm_path, out_path,
        )
        ctx.set_artifact(0, "final", out_path)
        ctx.update_progress("assemble", 1, 1)

    @staticmethod
    def _collect_clips(ctx) -> list[str]:
        clips = []
        for i, scene in enumerate(ctx.scenes):
            scene_id = scene.get("scene_id", i + 1)
            clip_path = ctx.get_artifact(scene_id, "clip")
            if clip_path and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clips.append(clip_path)
            else:
                logger.warning("assemble scene %d: clip missing, skip", scene_id)
        return clips

    @staticmethod
    def _render_final(
        clips: list[str],
        width: int,
        height: int,
        fps: int,
        transition: str,
        bgm_path: str,
        out_path: str,
    ) -> None:
        n = len(clips)

        audio_flags = [ffmpeg_util.probe_has_audio(p) for p in clips]
        all_audio = all(audio_flags)
        any_audio = any(audio_flags)
        use_clip_audio = all_audio
        if any_audio and not all_audio:
            logger.warning("assemble clips audio inconsistent, dropping clip audio")
            use_clip_audio = False

        bgm_ok = (
            bgm_path
            and os.path.exists(bgm_path)
            and ffmpeg_util.probe_has_audio(bgm_path)
        )
        if bgm_path and not bgm_ok:
            logger.warning("assemble BGM unusable: %s", bgm_path)

        filters = []
        if use_clip_audio:
            seg_labels = []
            for i in range(n):
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
            for i in range(n):
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
            total = sum(ffmpeg_util.probe_duration(p) for p in clips)
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
        for p in clips:
            args += ["-i", p]
        if bgm_ok:
            args += ["-i", bgm_path]
        args += ["-filter_complex", filter_complex]
        args += ["-map", video_label]
        if audio_map:
            args += ["-map", audio_map]
        args += [
            "-c:v", _video_encoder(), "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        ]
        if bgm_ok or use_clip_audio:
            args += ["-shortest"]
        args.append(out_path)

        logger.info(
            "assemble n=%d audio=clip:%s bgm:%s transition=%s",
            n, use_clip_audio, bgm_ok, transition,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=900, label=f"assemble n={n}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"assemble output empty: {out_path}")

        dur = ffmpeg_util.probe_duration(out_path)
        logger.info("assemble done dur=%.2fs size=%d", dur, os.path.getsize(out_path))


_ASSEMBLE_VT_CACHE: bool | None = None


def _video_encoder() -> str:
    global _ASSEMBLE_VT_CACHE
    if _ASSEMBLE_VT_CACHE is not None:
        return "h264_videotoolbox" if _ASSEMBLE_VT_CACHE else "libx264"
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _ASSEMBLE_VT_CACHE = "h264_videotoolbox" in result.stdout
        if _ASSEMBLE_VT_CACHE:
            logger.info("assemble: VideoToolbox h264 encoder available")
        return "h264_videotoolbox" if _ASSEMBLE_VT_CACHE else "libx264"
    except Exception:
        _ASSEMBLE_VT_CACHE = False
        return "libx264"
