from __future__ import annotations

import logging
import os
import random

from ..stage import Stage, StageInfo
from ... import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.ken_burns")


class KenBurnsStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="ken_burns",
            description="Ken Burns 推镜: PNG + 音频 → 9:16 mp4",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["image", "audio"],
            output_kinds=["clip"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        preset = ctx.config.get("ken_burns_preset", "random")
        width = ctx.config.get("ken_burns_width", 1080)
        height = ctx.config.get("ken_burns_height", 1920)
        fps = ctx.config.get("ken_burns_fps", 30)
        workers = ctx.config.get("ken_burns_workers", 2)

        tasks = []
        for i, scene in enumerate(ctx.scenes):
            scene_id = scene.get("scene_id", i + 1)
            if ctx.has_artifact_on_disk(scene_id, "clip"):
                logger.info("ken_burns scene %d skipped (exists)", scene_id)
                continue

            img_path = ctx.get_artifact(scene_id, "image")
            if not img_path or not os.path.exists(img_path):
                logger.warning("ken_burns scene %d: no image, skip", scene_id)
                continue

            audio_path = ctx.get_artifact(scene_id, "audio")
            duration = scene.get("duration_seconds", scene.get("duration", 8))
            out_path = ctx.artifact_path(scene_id, "clip")

            tasks.append((img_path, audio_path, duration, preset, width, height, fps, scene_id, out_path))

        if not tasks:
            logger.info("ken_burns: all clips already exist")
            return

        if workers > 1 and len(tasks) > 1:
            self._render_parallel(ctx, tasks, workers)
        else:
            self._render_sequential(ctx, tasks)

    def _render_sequential(self, ctx, tasks):
        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)
        for idx, (img, audio, dur, preset, w, h, fps, sid, out) in enumerate(tasks):
            self._render_clip(img, audio, dur, preset, w, h, fps, sid, out)
            ctx.set_artifact(sid, "clip", out)
            ctx.update_progress("ken_burns", idx + 1, len(tasks))
            if ctx.should_checkpoint_scene(idx + 1):
                checkpoint.save(ctx)
                logger.info("ken_burns scene-level checkpoint at scene %d", sid)

    def _render_parallel(self, ctx, tasks, workers):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("ken_burns: parallel render workers=%d tasks=%d", workers, len(tasks))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for idx, (img, audio, dur, preset, w, h, fps, sid, out) in enumerate(tasks):
                fut = pool.submit(
                    self._render_clip, img, audio, dur, preset, w, h, fps, sid, out,
                )
                futures[fut] = (idx, sid, out)
            for fut in as_completed(futures):
                idx, sid, out = futures[fut]
                exc = fut.exception()
                if exc:
                    logger.error("ken_burns scene %d failed: %s", sid, exc)
                else:
                    ctx.set_artifact(sid, "clip", out)
                done += 1
                ctx.update_progress("ken_burns", done, len(tasks))

    @staticmethod
    def _render_clip(
        img_path: str,
        audio_path: str | None,
        duration: float,
        preset: str,
        width: int,
        height: int,
        fps: int,
        scene_id: int,
        out_path: str,
    ) -> None:
        total_frames = max(1, round(duration * fps))
        zoompan = _build_zoompan(preset, width, height, fps, total_frames)

        has_audio = (
            audio_path
            and os.path.exists(audio_path)
            and ffmpeg_util.probe_has_audio(audio_path)
        )

        args = ["-loop", "1", "-i", img_path]
        if has_audio:
            args += ["-i", audio_path]

        args += [
            "-vf", zoompan,
        ]

        hw_accel = ffmpeg_util.has_videotoolbox()
        if hw_accel:
            args += ["-c:v", "h264_videotoolbox", "-q:v", "65"]
        else:
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
        args += ["-pix_fmt", "yuv420p"]

        if has_audio:
            args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            args += ["-t", f"{total_frames / fps:.3f}"]

        args += ["-movflags", "+faststart", out_path]

        logger.info(
            "ken_burns scene=%d preset=%s dur=%.1f audio=%s",
            scene_id, preset, duration, has_audio,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=600, label=f"ken_burns scene={scene_id}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ken_burns output empty: {out_path}")


def _build_zoompan(preset: str, out_w: int, out_h: int, fps: int, total_frames: int) -> str:
    chosen = preset
    if preset == "random":
        options = ["zoom-in", "zoom-out", "pan-right", "pan-left"]
        chosen = random.choice(options)
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
    return (
        f"scale={out_w * 2}:{out_h * 2}:force_original_aspect_ratio=increase,"
        f"crop={out_w * 2}:{out_h * 2},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={D}:s={out_w}x{out_h}:fps={fps},"
        f"format=yuv420p"
    )
