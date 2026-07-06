from __future__ import annotations

import logging
import os
import time

from ..stage import Stage, StageInfo
from ..checkpoint import CheckpointManager
from ... import ffmpeg_util
from .image_generate import ImageGenerateStage

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.multi_pose")

_DEFAULT_POSE_SUFFIXES = (
    "facing forward, neutral pose",
    "turning to the side, mid-action pose",
    "reaching outward, dynamic action pose",
    "looking back over shoulder, dramatic pose",
)
_MIN_SEG_DUR = 0.5
_REALISTIC_REF_STRENGTH = 0.6
_REALISTIC_REF_MODE = "redux"


class MultiPoseStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="multi_pose",
            description="多姿态关键帧: 同角色 N 姿态 → 定格剪辑 clip（卡通动作幻觉）",
            model_requirements=["flux"],
            memory_estimate_gb=7.0,
            input_kinds=["image", "audio"],
            output_kinds=["clip"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        n_poses = max(2, int(ctx.config.get("multi_pose_count", 3)))
        pose_suffixes = ctx.config.get("multi_pose_poses") or list(_DEFAULT_POSE_SUFFIXES)
        out_w = ctx.config.get("ken_burns_width", 1080)
        out_h = ctx.config.get("ken_burns_height", 1920)
        fps = ctx.config.get("ken_burns_fps", 30)
        base_seed = ctx.config.get("flux_seed", 0)
        vary_seed = ctx.config.get("flux_vary_seed", True)
        flux_w = ctx.config.get("flux_width", 1024)
        flux_h = ctx.config.get("flux_height", 1024)
        steps = ctx.config.get("flux_steps", 8)
        guidance = ctx.config.get("flux_guidance", 4.0)
        global_style = ctx.config.get("global_style", "")
        character_registry = ctx.config.get("character_registry", [])
        char_lookup = {c["name"]: c for c in character_registry if "name" in c}
        character_style = ctx.config.get("character_style", "none")
        ref_strength = float(ctx.config.get("realistic_reference_strength", _REALISTIC_REF_STRENGTH))
        ref_mode = ctx.config.get("realistic_conditioning_mode", _REALISTIC_REF_MODE)

        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("flux") as handle:
            t_start = time.time()
            produced = 0
            for i, scene in enumerate(ctx.scenes):
                scene_id = scene.get("scene_id", i + 1)
                if ctx.has_artifact_on_disk(scene_id, "clip"):
                    logger.info("multi_pose scene %d skipped (clip exists)", scene_id)
                    continue

                base_img = ctx.get_artifact(scene_id, "image")
                if not base_img or not os.path.exists(base_img):
                    logger.warning("multi_pose scene %d: no base image, skip", scene_id)
                    continue

                audio_path = ctx.get_artifact(scene_id, "audio")
                duration = float(scene.get("duration_seconds", scene.get("duration", 8)))

                visual_prompt = scene.get("visual_prompt", "")
                scene_chars = scene.get("characters", [])
                char_appearance = ImageGenerateStage._get_character_appearance(scene_chars, char_lookup)
                scene_seed = ImageGenerateStage._compute_scene_seed(base_seed, scene_id, vary_seed, scene_chars)

                pose_paths = self._generate_poses(
                    handle, visual_prompt, global_style, char_appearance, scene_seed,
                    n_poses, pose_suffixes, flux_w, flux_h, steps, guidance,
                    ctx.job_dir, scene_id, base_img,
                    character_style=character_style,
                    reference_strength=ref_strength,
                    conditioning_mode=ref_mode,
                )

                out_path = ctx.artifact_path(scene_id, "clip")
                self._stitch_clip(
                    pose_paths, audio_path, duration, out_w, out_h, fps, out_path, scene_id,
                )
                ctx.set_artifact(scene_id, "clip", out_path)
                produced += 1
                ctx.update_progress("multi_pose", i + 1, len(ctx.scenes))

                if ctx.should_checkpoint_scene(i + 1):
                    checkpoint.save(ctx)
                    logger.info("multi_pose scene-level checkpoint at scene %d", scene_id)

            if produced:
                logger.info(
                    "multi_pose: %d clips in %.1fs (avg %.1fs/clip)",
                    produced, time.time() - t_start, (time.time() - t_start) / produced,
                )

    @staticmethod
    def _generate_poses(
        handle,
        visual_prompt: str,
        global_style: str,
        char_appearance: str,
        scene_seed: int,
        n_poses: int,
        pose_suffixes: list[str],
        width: int,
        height: int,
        steps: int,
        guidance: float,
        job_dir: str,
        scene_id: int,
        base_img: str,
        character_style: str = "none",
        reference_strength: float = _REALISTIC_REF_STRENGTH,
        conditioning_mode: str = _REALISTIC_REF_MODE,
    ) -> list[str]:
        if not visual_prompt:
            logger.warning("multi_pose scene %d: empty visual_prompt, reuse base image", scene_id)
            return [base_img] * n_poses

        pose_dir = os.path.join(job_dir, "poses")
        os.makedirs(pose_dir, exist_ok=True)
        base_prompt = ImageGenerateStage._build_prompt(visual_prompt, global_style, char_appearance)

        ref_b64: str | None = None
        if character_style == "realistic" and os.path.exists(base_img):
            ref_b64 = ImageGenerateStage._image_to_b64(base_img)
            logger.info(
                "multi_pose scene %d: realistic reference conditioning mode=%s strength=%.2f ref_bytes=%d",
                scene_id, conditioning_mode, reference_strength, len(base_img),
            )

        paths: list[str] = []
        for k in range(n_poses):
            suffix = pose_suffixes[k % len(pose_suffixes)]
            prompt = f"{base_prompt}, {suffix}"
            out = os.path.join(pose_dir, f"scene_{scene_id:03d}_pose_{k + 1:03d}.png")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                logger.info("multi_pose scene %d pose %d reused (cached)", scene_id, k + 1)
                paths.append(out)
                continue
            try:
                ImageGenerateStage._generate_http(
                    handle, prompt, width, height, steps, guidance, scene_seed, out,
                    reference_image=ref_b64,
                    reference_strength=reference_strength if ref_b64 else None,
                    conditioning_mode=conditioning_mode if ref_b64 else None,
                )
                paths.append(out)
                logger.info("multi_pose scene %d pose %d/%d done", scene_id, k + 1, n_poses)
            except Exception as exc:
                logger.warning(
                    "multi_pose scene %d pose %d gen failed: %s (reuse base image)",
                    scene_id, k + 1, exc,
                )
                paths.append(base_img)
        return paths

    @staticmethod
    def _stitch_clip(
        pose_paths: list[str],
        audio_path: str | None,
        duration: float,
        out_w: int,
        out_h: int,
        fps: int,
        out_path: str,
        scene_id: int,
    ) -> None:
        n = len(pose_paths)
        seg_dur = max(_MIN_SEG_DUR, (duration / n) if duration > 0 else _MIN_SEG_DUR)
        seg_frames = max(1, round(seg_dur * fps))

        inputs: list[str] = []
        chains: list[str] = []
        for k, p in enumerate(pose_paths):
            inputs += ["-loop", "1", "-t", f"{seg_dur:.3f}", "-i", p]
            z = f"1+0.15*on/{seg_frames}"
            cx = "iw/2-(iw/zoom/2)"
            cy = "ih/2-(ih/zoom/2)"
            zoompan = (
                f"scale={out_w * 2}:{out_h * 2}:force_original_aspect_ratio=increase,"
                f"crop={out_w * 2}:{out_h * 2},setsar=1,"
                f"zoompan=z='{z}':x='{cx}':y='{cy}':d={seg_frames}:s={out_w}x{out_h}:fps={fps},"
                f"format=yuv420p"
            )
            chains.append(f"[{k}:v]{zoompan}[v{k}]")
        concat_in = "".join(f"[v{k}]" for k in range(n))
        chains.append(f"{concat_in}concat=n={n}:v=1:a=0[vout]")
        filter_complex = ";".join(chains)

        has_audio = (
            audio_path
            and os.path.exists(audio_path)
            and ffmpeg_util.probe_has_audio(audio_path)
        )

        args: list[str] = list(inputs)
        if has_audio:
            args += ["-i", audio_path]
        args += ["-filter_complex", filter_complex, "-map", "[vout]"]
        if has_audio:
            args += ["-map", f"{n}:a", "-c:a", "aac", "-b:a", "128k", "-shortest"]

        args += ffmpeg_util.video_encoder_args()
        args += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]

        logger.info(
            "multi_pose stitch scene=%d poses=%d seg_dur=%.2fs audio=%s",
            scene_id, n, seg_dur, bool(has_audio),
        )
        ffmpeg_util.run_ffmpeg(args, timeout=600, label=f"multi_pose scene={scene_id}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"multi_pose output empty: {out_path}")
