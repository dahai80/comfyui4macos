from __future__ import annotations

import logging
import os
import time

from ..stage import Stage, StageInfo
from ... import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.wan_video")

_DEFAULT_SCENE_DUR = 8.0
_MOTION_FALLBACK = (
    "subtle cinematic motion, gentle camera push-in, "
    "character blinking and breathing, soft wind in hair"
)


class WanVideoStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="wan_video",
            description="场景图 → Wan2.2 i2v 动作片段 → 拉伸到旁白时长（替代 ken_burns 静态推镜）",
            model_requirements=[],
            memory_estimate_gb=18.0,
            input_kinds=["image", "audio"],
            output_kinds=["clip"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return
        if not ctx.config.get("wan_enabled", False):
            logger.info("wan_video: disabled (wan_enabled=False), ken_burns will fill")
            return

        frames = int(ctx.config.get("wan_frames", 41))
        if frames % 4 != 1:
            raise ValueError(f"wan_frames must be 4n+1, got {frames}")
        steps = int(ctx.config.get("wan_steps", 20))
        wan_fps = int(ctx.config.get("wan_fps", 24))
        size = tuple(ctx.config.get("wan_size", (480, 832)))
        max_area = int(ctx.config.get("wan_max_area", 399360))
        base_seed = int(ctx.config.get("wan_seed", 1001))
        vary_seed = bool(ctx.config.get("wan_vary_seed", True))
        out_w = int(ctx.config.get("ken_burns_width", 1080))
        out_h = int(ctx.config.get("ken_burns_height", 1920))
        out_fps = int(ctx.config.get("ken_burns_fps", 30))

        from ..checkpoint import CheckpointManager
        from ..wan_utils import load_wan_pipe, wan_i2v
        checkpoint = CheckpointManager(ctx.job_dir)

        try:
            pipe = load_wan_pipe(ctx.config.get("wan_checkpoint_dir", ""))
        except Exception as exc:
            logger.error(
                "wan_video: pipe load failed (%s); ALL scenes fall back to ken_burns",
                exc,
                exc_info=True,
            )
            return

        from PIL import Image
        t_start = time.time()
        produced = 0
        for i, scene in enumerate(ctx.scenes):
            scene_id = scene.get("scene_id", i + 1)
            if ctx.has_artifact_on_disk(scene_id, "clip"):
                logger.info("wan_video scene %d skipped (clip exists)", scene_id)
                continue

            img_path = ctx.get_artifact(scene_id, "image")
            if not img_path or not os.path.exists(img_path):
                logger.warning(
                    "wan_video scene %d: no image, skip (ken_burns fallback)", scene_id,
                )
                continue

            audio_path = ctx.get_artifact(scene_id, "audio")
            duration = float(
                scene.get("duration_seconds", scene.get("duration", 0)) or 0
            )
            if duration <= 0 and audio_path and os.path.exists(audio_path):
                duration = ffmpeg_util.probe_duration(audio_path)
            if duration <= 0:
                duration = _DEFAULT_SCENE_DUR

            out_path = ctx.artifact_path(scene_id, "clip")
            motion_path = os.path.join(ctx.job_dir, f"scene_{scene_id:03d}_wan_motion.mp4")
            scene_seed = (base_seed + scene_id) if vary_seed else base_seed
            motion_prompt = self._build_motion_prompt(scene)

            try:
                img = Image.open(img_path).convert("RGB")
                wan_i2v(
                    pipe, motion_prompt, img, motion_path,
                    frames=frames, size=size, max_area=max_area,
                    steps=steps, fps=wan_fps, seed=scene_seed,
                )
            except Exception as exc:
                logger.error(
                    "wan_video scene %d i2v failed: %s (ken_burns fallback)",
                    scene_id, exc,
                    exc_info=True,
                )
                self._safe_remove(motion_path)
                continue

            try:
                self._extend_to_duration(
                    motion_path, audio_path, duration,
                    out_w, out_h, out_fps, out_path,
                )
            except Exception as exc:
                logger.error(
                    "wan_video scene %d extend failed: %s (ken_burns fallback)",
                    scene_id, exc,
                    exc_info=True,
                )
                continue

            self._safe_remove(motion_path)
            ctx.set_artifact(scene_id, "clip", out_path)
            produced += 1
            ctx.update_progress("wan_video", i + 1, len(ctx.scenes))
            if ctx.should_checkpoint_scene(i + 1):
                checkpoint.save(ctx)
                logger.info("wan_video scene-level checkpoint at scene %d", scene_id)

        if produced:
            logger.info(
                "wan_video total: %d/%d clips in %.1fs",
                produced, len(ctx.scenes), time.time() - t_start,
            )

    @staticmethod
    def _build_motion_prompt(scene: dict) -> str:
        hint = (scene.get("motion_hint") or "").strip()
        visual = (scene.get("visual_prompt") or "").strip()
        sfx = (scene.get("sound_effect") or "").strip()
        parts = []
        if visual:
            parts.append(visual)
        if hint:
            parts.append(hint)
        elif sfx:
            parts.append(f"motion cue: {sfx}")
        else:
            parts.append(_MOTION_FALLBACK)
        return ", ".join(parts)

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("wan_video: cannot remove %s: %s", path, exc)

    @staticmethod
    def _extend_to_duration(
        motion_path: str,
        audio_path: str | None,
        duration: float,
        out_w: int,
        out_h: int,
        out_fps: int,
        out_path: str,
    ) -> None:
        motion_dur = ffmpeg_util.probe_duration(motion_path)
        if motion_dur <= 0:
            motion_dur = 0.1
        has_audio = (
            audio_path
            and os.path.exists(audio_path)
            and ffmpeg_util.probe_has_audio(audio_path)
        )

        extend = max(0.0, duration - motion_dur + 0.5)
        vf_parts = [
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase",
            f"crop={out_w}:{out_h}",
            "setsar=1",
            f"fps={out_fps}",
        ]
        if extend > 0.05:
            vf_parts.append(f"tpad=stop_mode=clone:stop_duration={extend:.3f}")
        vf_parts.append("format=yuv420p")
        vf = ",".join(vf_parts)

        args = ["-i", motion_path]
        if has_audio:
            args += ["-i", audio_path]
        args += ["-vf", vf]
        args += ffmpeg_util.video_encoder_args()
        args += ["-pix_fmt", "yuv420p"]
        if has_audio:
            args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            args += ["-t", f"{duration:.3f}"]
        args += ["-movflags", "+faststart", out_path]

        logger.info(
            "wan_video extend motion=%.2fs target=%.2fs audio=%s -> %s",
            motion_dur, duration, has_audio, out_path,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=900, label="wan_video extend")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"wan_video output empty: {out_path}")
