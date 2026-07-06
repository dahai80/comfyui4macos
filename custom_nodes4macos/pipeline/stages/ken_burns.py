from __future__ import annotations

import logging
import os
import random
import re
import subprocess

from ..stage import Stage, StageInfo
from ... import ffmpeg_util

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.ken_burns")

_PRESET_OPTIONS = ["zoom-in", "zoom-out", "pan-right", "pan-left"]
_MULTISHOT_ROTATION = ["zoom-in", "pan-right", "zoom-out", "pan-left"]
_MIN_SCENE_FOR_CUT = 3.0
_EDGE_PAD = 0.3


class KenBurnsStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="ken_burns",
            description="Ken Burns 推镜: PNG + 音频 → 9:16 mp4（可选音频对齐细切）",
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

        audio_cut = bool(ctx.config.get("ken_burns_audio_cut", False))
        sdb = ctx.config.get("ken_burns_silence_noise_db", -30)
        smd = float(ctx.config.get("ken_burns_silence_min_dur", 0.3))
        max_shots = int(ctx.config.get("ken_burns_max_shots", 4))
        min_cut_dur = float(ctx.config.get("ken_burns_audio_cut_min_dur", _MIN_SCENE_FOR_CUT))

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

        cut_cfg = (audio_cut, sdb, smd, max_shots, min_cut_dur)

        if workers > 1 and len(tasks) > 1:
            self._render_parallel(ctx, tasks, workers, cut_cfg)
        else:
            self._render_sequential(ctx, tasks, cut_cfg)

    def _render_sequential(self, ctx, tasks, cut_cfg):
        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)
        for idx, (img, audio, dur, preset, w, h, fps, sid, out) in enumerate(tasks):
            self._render_scene(img, audio, dur, preset, w, h, fps, sid, out, cut_cfg)
            ctx.set_artifact(sid, "clip", out)
            ctx.update_progress("ken_burns", idx + 1, len(tasks))
            if ctx.should_checkpoint_scene(idx + 1):
                checkpoint.save(ctx)
                logger.info("ken_burns scene-level checkpoint at scene %d", sid)

    def _render_parallel(self, ctx, tasks, workers, cut_cfg):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("ken_burns: parallel render workers=%d tasks=%d", workers, len(tasks))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for idx, (img, audio, dur, preset, w, h, fps, sid, out) in enumerate(tasks):
                fut = pool.submit(
                    self._render_scene, img, audio, dur, preset, w, h, fps, sid, out, cut_cfg,
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

    def _render_scene(self, img, audio, dur, preset, w, h, fps, sid, out, cut_cfg):
        audio_cut, sdb, smd, max_shots, min_cut_dur = cut_cfg
        if audio_cut and audio and os.path.exists(audio) and dur >= min_cut_dur:
            silences = _detect_silence(audio, sdb, smd)
            total_frames = max(1, round(dur * fps))
            cut_frames = _silence_to_cut_frames(silences, dur, fps, max_shots)
            if cut_frames:
                seg_lengths = _cut_frames_to_seg_lengths(cut_frames, total_frames)
                presets = _pick_multishot_presets(len(seg_lengths), preset)
                zoompan = _build_zoompan_multishot(presets, seg_lengths, total_frames, w, h, fps)
                logger.info(
                    "ken_burns scene=%d audio-cut shots=%d cuts=%s segs=%s",
                    sid, len(seg_lengths), cut_frames, seg_lengths,
                )
                KenBurnsStage._run_zoompan_render(img, audio, dur, zoompan, fps, sid, out)
                return
        logger.info("ken_burns scene=%d preset=%s dur=%.1f", sid, preset, dur)
        self._render_clip(img, audio, dur, preset, w, h, fps, sid, out)

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
        KenBurnsStage._run_zoompan_render(img_path, audio_path, duration, zoompan, fps, scene_id, out_path)

    @staticmethod
    def _run_zoompan_render(
        img_path: str,
        audio_path: str | None,
        duration: float,
        zoompan: str,
        fps: int,
        scene_id: int,
        out_path: str,
    ) -> None:
        total_frames = max(1, round(duration * fps))

        has_audio = (
            audio_path
            and os.path.exists(audio_path)
            and ffmpeg_util.probe_has_audio(audio_path)
        )

        args = ["-loop", "1", "-i", img_path]
        if has_audio:
            args += ["-i", audio_path]

        args += ["-vf", zoompan]

        args += ffmpeg_util.video_encoder_args()
        args += ["-pix_fmt", "yuv420p"]

        if has_audio:
            args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            args += ["-t", f"{total_frames / fps:.3f}"]

        args += ["-movflags", "+faststart", out_path]

        logger.info(
            "ken_burns scene=%d dur=%.1f audio=%s",
            scene_id, duration, has_audio,
        )
        ffmpeg_util.run_ffmpeg(args, timeout=600, label=f"ken_burns scene={scene_id}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ken_burns output empty: {out_path}")


def _preset_motion(preset: str, D: int, on_var: str) -> tuple[str, str, str]:
    chosen = preset
    if preset == "random":
        chosen = random.choice(_PRESET_OPTIONS)
    d = max(1, D)
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"
    if chosen == "zoom-in":
        z = f"1+0.25*{on_var}/{d}"
        x, y = cx, cy
    elif chosen == "zoom-out":
        z = f"1.25-0.25*{on_var}/{d}"
        x, y = cx, cy
    elif chosen == "pan-right":
        z = "1.2"
        x = f"(iw-iw/zoom)*({on_var}/{d})"
        y = cy
    elif chosen == "pan-left":
        z = "1.2"
        x = f"(iw-iw/zoom)*(1-{on_var}/{d})"
        y = cy
    else:
        z = f"1+0.2*{on_var}/{d}"
        x, y = cx, cy
    return z, x, y


def _build_zoompan(preset: str, out_w: int, out_h: int, fps: int, total_frames: int) -> str:
    D = max(1, total_frames)
    z, x, y = _preset_motion(preset, D, "on")
    return (
        f"scale={out_w * 2}:{out_h * 2}:force_original_aspect_ratio=increase,"
        f"crop={out_w * 2}:{out_h * 2},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={D}:s={out_w}x{out_h}:fps={fps},"
        f"format=yuv420p"
    )


def _build_zoompan_multishot(
    presets: list[str], seg_lengths: list[int], total_frames: int,
    out_w: int, out_h: int, fps: int,
) -> str:
    n = len(seg_lengths)
    start_frames = [0]
    for i in range(n - 1):
        start_frames.append(start_frames[-1] + seg_lengths[i])

    z_parts: list[str] = []
    x_parts: list[str] = []
    y_parts: list[str] = []
    for i in range(n):
        on_var = "on" if i == 0 else f"(on-{start_frames[i]})"
        z, x, y = _preset_motion(presets[i], seg_lengths[i], on_var)
        z_parts.append(z)
        x_parts.append(x)
        y_parts.append(y)

    def nest(parts: list[str]) -> str:
        expr = parts[n - 1]
        for i in range(n - 2, -1, -1):
            cond = f"lt(on,{start_frames[i + 1]})"
            expr = f"if({cond},{parts[i]},{expr})"
        return expr

    z = nest(z_parts)
    x = nest(x_parts)
    y = nest(y_parts)
    D = max(1, total_frames)
    return (
        f"scale={out_w * 2}:{out_h * 2}:force_original_aspect_ratio=increase,"
        f"crop={out_w * 2}:{out_h * 2},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={D}:s={out_w}x{out_h}:fps={fps},"
        f"format=yuv420p"
    )


def _detect_silence(audio_path: str, noise_db: int, min_dur: float) -> list[tuple[float, float]]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
             "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        logger.warning("ken_burns: silence detect failed: %s", exc)
        return []
    stderr = proc.stderr or ""
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)]
    intervals: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        if i < len(ends):
            intervals.append((s, ends[i]))
    logger.info("ken_burns: detected %d silence intervals", len(intervals))
    return intervals


def _silence_to_cut_frames(
    silences: list[tuple[float, float]], duration: float, fps: int, max_shots: int,
) -> list[int]:
    if not silences or duration <= 0:
        return []
    total_frames = max(1, round(duration * fps))
    candidates: list[int] = []
    for (s, e) in silences:
        mid = (s + e) / 2.0
        if mid <= _EDGE_PAD or mid >= duration - _EDGE_PAD:
            continue
        candidates.append(round(mid * fps))
    candidates = sorted(set(candidates))
    candidates = [c for c in candidates if 0 < c < total_frames]

    max_cuts = max(0, max_shots - 1)
    if len(candidates) > max_cuts:
        candidates = _even_pick(candidates, max_cuts)
    return candidates


def _even_pick(items: list[int], k: int) -> list[int]:
    if k <= 0 or not items:
        return []
    if k == 1:
        return [items[len(items) // 2]]
    if len(items) <= k:
        return items
    return [items[int(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _cut_frames_to_seg_lengths(cut_frames: list[int], total_frames: int) -> list[int]:
    if not cut_frames:
        return [max(1, total_frames)]
    bounds = [0] + list(cut_frames) + [total_frames]
    segs = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    return [max(1, s) for s in segs]


def _pick_multishot_presets(n: int, base_preset: str) -> list[str]:
    if base_preset in _PRESET_OPTIONS:
        return [base_preset] * n
    return [_MULTISHOT_ROTATION[i % len(_MULTISHOT_ROTATION)] for i in range(n)]
