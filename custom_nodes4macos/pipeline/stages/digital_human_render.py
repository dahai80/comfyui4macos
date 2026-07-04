from __future__ import annotations

import logging
import os

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.digital_human_render")


class DigitalHumanRenderStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="digital_human_render",
            description="数字人渲染（旧版）: TTS → 静态头像+音频合成（推荐使用 avatar_create+avatar_animate 流水线）",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["audio"],
            output_kinds=["clip"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        avatar_package = ctx.artifacts.get("avatar_package") or ctx.config.get("avatar_package", "")
        if avatar_package and os.path.isdir(avatar_package):
            logger.info(
                "digital_human_render: avatar_package detected, delegating to avatar_animate logic"
            )
            self._render_with_avatar(ctx, avatar_package)
            return

        avatar_reference = ctx.config.get("avatar_reference", "")
        logger.warning(
            "digital_human_render: no avatar_package, generating static avatar + audio composite"
        )
        self._render_fallback(ctx, avatar_reference)

    def _render_with_avatar(self, ctx, avatar_dir: str) -> None:
        import json
        import wave

        import cv2
        import numpy as np

        from ..checkpoint import CheckpointManager

        ref_path = ctx.artifacts.get("avatar_reference") or ctx.config.get("avatar_reference", "")
        if not ref_path or not os.path.isfile(ref_path):
            ref_path = os.path.join(avatar_dir, "reference.png")
        if not os.path.isfile(ref_path):
            logger.error("digital_human_render: reference.png not found in avatar_package")
            return

        avatar_meta_path = os.path.join(avatar_dir, "avatar_meta.json")
        avatar_meta = {}
        if os.path.isfile(avatar_meta_path):
            with open(avatar_meta_path, "r", encoding="utf-8") as f:
                avatar_meta = json.load(f)

        motion_dir = os.path.join(avatar_dir, "motion_frames")
        motion_frames = []
        if os.path.isdir(motion_dir):
            for fn in sorted(os.listdir(motion_dir)):
                if fn.endswith(".png") or fn.endswith(".jpg"):
                    motion_frames.append(os.path.join(motion_dir, fn))

        checkpoint = CheckpointManager(ctx.job_dir)
        scenes = ctx.scenes
        if not scenes:
            logger.warning("digital_human_render: no scenes to process")
            return

        for i, scene in enumerate(scenes):
            sid = scene.get("scene_id", i + 1)
            if ctx.has_artifact_on_disk(sid, "clip"):
                logger.info("digital_human_render scene %d: clip exists, skipping", sid)
                continue

            audio_path = ctx.get_artifact(sid, "audio")
            if not audio_path or not os.path.isfile(audio_path):
                logger.warning("digital_human_render scene %d: no audio, skipping", sid)
                continue

            clip_path = ctx.artifact_path(sid, "clip")
            duration = scene.get("duration_seconds", 10)

            from .avatar_animate import AvatarAnimateStage
            animator = AvatarAnimateStage()
            animator._animate_scene(
                ref_path=ref_path,
                audio_path=audio_path,
                output_path=clip_path,
                duration=duration,
                motion_frames=motion_frames,
                avatar_meta=avatar_meta,
                scene=scene,
            )
            ctx.set_artifact(sid, "clip", clip_path)
            logger.info("digital_human_render scene %d: avatar-animated clip rendered", sid)

            ctx.update_progress("digital_human_render", i + 1, len(scenes))
            if ctx.should_checkpoint_scene(i + 1):
                checkpoint.save(ctx)

    def _render_fallback(self, ctx, avatar_reference: str) -> None:
        scenes = ctx.scenes
        if not scenes:
            logger.warning("digital_human_render: no scenes to process")
            return

        avatar_path = avatar_reference
        if not avatar_path or not os.path.isfile(avatar_path):
            avatar_path = self._generate_placeholder_avatar(ctx)

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        for i, scene in enumerate(scenes):
            sid = scene.get("scene_id", 0)
            if ctx.has_artifact_on_disk(sid, "clip"):
                logger.info("scene %d clip exists, skipping", sid)
                continue

            audio_path = ctx.get_artifact(sid, "audio")
            if not audio_path or not os.path.isfile(audio_path):
                logger.warning("scene %d: no audio, skipping", sid)
                continue

            clip_path = ctx.artifact_path(sid, "clip")
            duration = scene.get("duration_seconds", 10)

            from ...ffmpeg_util import run_ffmpeg, video_encoder_args

            cmd = [
                "-loop", "1", "-i", avatar_path,
                "-i", audio_path,
            ]
            cmd += video_encoder_args()
            cmd += [
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-t", str(duration),
                "-shortest",
                "-y", clip_path,
            ]
            try:
                run_ffmpeg(cmd)
                ctx.set_artifact(sid, "clip", clip_path)
                logger.info("scene %d: fallback avatar clip rendered", sid)
            except Exception as exc:
                logger.error("scene %d: fallback render failed: %s", sid, exc)

            ctx.update_progress("digital_human_render", i + 1, len(scenes))
            if ctx.should_checkpoint_scene(i + 1):
                checkpoint.save(ctx)
                logger.info("digital_human_render scene-level checkpoint at scene %d", sid)

    @staticmethod
    def _generate_placeholder_avatar(ctx) -> str:
        avatar_path = os.path.join(ctx.job_dir, "_avatar_placeholder.png")
        if os.path.exists(avatar_path):
            return avatar_path
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new("RGB", (1080, 1920), (40, 40, 60))
            draw = ImageDraw.Draw(img)
            draw.ellipse([390, 300, 690, 600], fill=(180, 180, 200))
            draw.rectangle([440, 620, 640, 1200], fill=(100, 100, 140))
            draw.text((440, 1400), "数字人", fill=(200, 200, 220))
            img.save(avatar_path)
            logger.info("placeholder avatar generated: %s", avatar_path)
        except ImportError:
            import struct
            import zlib
            w, h = 1080, 1920
            def _png_chunk(typ, data):
                c = typ + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            raw = b""
            for _y in range(h):
                raw += b"\x00"
                for _x in range(w):
                    raw += b"\x28\x28\x3c"
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            png = b"\x89PNG\r\n\x1a\n"
            png += _png_chunk(b"IHDR", ihdr)
            png += _png_chunk(b"IDAT", zlib.compress(raw))
            png += _png_chunk(b"IEND", b"")
            with open(avatar_path, "wb") as f:
                f.write(png)
            logger.warning("PIL not available, created minimal placeholder avatar (%d bytes)", len(png))
        return avatar_path
