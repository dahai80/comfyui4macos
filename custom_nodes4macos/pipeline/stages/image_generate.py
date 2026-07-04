from __future__ import annotations

import logging
import os
import time

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.image_generate")


class ImageGenerateStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="image_generate",
            description="visual_prompt → PNG (FluxPipeline MLX native)",
            model_requirements=["flux"],
            memory_estimate_gb=7.0,
            input_kinds=["scenes"],
            output_kinds=["image"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        width = ctx.config.get("flux_width", 1024)
        height = ctx.config.get("flux_height", 1024)
        steps = ctx.config.get("flux_steps", 8)
        guidance = ctx.config.get("flux_guidance", 4.0)
        seed = ctx.config.get("flux_seed", 0)
        vary_seed = ctx.config.get("flux_vary_seed", True)
        consistency_check = ctx.config.get("consistency_check", False)
        char_ref_dir = ctx.config.get("character_reference_dir", "")
        enable_tiling = ctx.config.get("flux_tiling", width > 1024 or height > 1024)

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("flux") as handle:
            pipeline = handle.model
            if enable_tiling:
                self._apply_tiling(pipeline, width, height)

            t_start = time.time()
            generated = 0
            for i, scene in enumerate(ctx.scenes):
                scene_id = scene.get("scene_id", i + 1)
                if ctx.has_artifact_on_disk(scene_id, "image"):
                    logger.info("image_generate scene %d skipped (exists)", scene_id)
                    continue

                visual_prompt = scene.get("visual_prompt", "")
                global_style = ctx.config.get("global_style", "")
                if not visual_prompt:
                    logger.warning("image_generate scene %d: no visual_prompt, skip", scene_id)
                    continue

                prompt = self._build_prompt(visual_prompt, global_style)
                if consistency_check:
                    char_desc = scene.get("character_description", "")
                    if char_desc:
                        prompt = f"{prompt}, consistent character: {char_desc}"
                    episode_title = scene.get("episode_title", "")
                    if episode_title:
                        prompt = f"{prompt}, from episode: {episode_title}"
                out_path = ctx.artifact_path(scene_id, "image")

                scene_seed = (seed + scene_id) if (seed and vary_seed) else seed
                t_scene = time.time()
                self._generate_image(
                    pipeline, prompt, width, height, steps, guidance, scene_seed, out_path,
                )
                elapsed = time.time() - t_scene
                generated += 1
                logger.info(
                    "scene %d/%d done in %.1fs → %s",
                    i + 1, len(ctx.scenes), elapsed, out_path,
                )

                ctx.set_artifact(scene_id, "image", out_path)

                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except ImportError:
                    pass

                ctx.update_progress("image_generate", i + 1, len(ctx.scenes))

                if ctx.should_checkpoint_scene(i + 1):
                    checkpoint.save(ctx)
                    logger.info("scene-level checkpoint saved at scene %d", scene_id)

            total_elapsed = time.time() - t_start
            if generated > 0:
                logger.info(
                    "image_generate total: %d images in %.1fs (avg %.1fs/image)",
                    generated, total_elapsed, total_elapsed / generated,
                )

    @staticmethod
    def _apply_tiling(pipeline, width: int, height: int) -> None:
        try:
            from mflux.models.common.vae.tiling_config import TilingConfig
            tiles = 4 if max(width, height) > 1536 else 2
            pipeline.tiling_config = TilingConfig(
                vae_decode_tiles_per_dim=tiles,
                vae_decode_overlap=8,
            )
            logger.info("tiling enabled: %dx%d tiles_per_dim=%d", width, height, tiles)
        except ImportError:
            logger.warning("tiling_config not available in this mflux version, skipping")

    @staticmethod
    def _build_prompt(visual_prompt: str, global_style: str) -> str:
        vp = (visual_prompt or "").strip()
        if not vp:
            raise ValueError("visual_prompt is empty")
        style = (global_style or "").strip()
        if style:
            return f"{vp}, {style}"
        return vp

    @staticmethod
    def _generate_image(
        pipeline,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
    ) -> None:
        try:
            ImageGenerateStage._generate_mlx(
                pipeline, prompt, width, height, steps, guidance, seed, out_path,
            )
        except ImportError:
            logger.warning("mlx/FluxPipeline not available, falling back to HTTP")
            ImageGenerateStage._generate_http(
                prompt, width, height, steps, guidance, seed, out_path,
            )

    @staticmethod
    def _generate_mlx(
        pipeline,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
    ) -> None:
        seed_arg = seed if seed else 42
        logger.info(
            "image_generate MLX prompt_len=%d size=%dx%d steps=%d guidance=%.1f seed=%s",
            len(prompt), width, height, steps, guidance, seed_arg,
        )
        image = pipeline.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance=guidance,
            seed=seed_arg,
        )
        if image is None:
            raise RuntimeError("Flux1.generate_image returned None")

        image.save(path=out_path)
        logger.info("image_generate MLX saved: %s (%d bytes)", out_path, os.path.getsize(out_path))

    @staticmethod
    def _generate_http(
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
    ) -> None:
        from ...fusion_client import FusionMLXClient

        seed_arg = None if seed == 0 else seed
        logger.info("image_generate HTTP prompt_len=%d", len(prompt))
        with FusionMLXClient() as client:
            if not client.health():
                raise RuntimeError("fusion-mlx unreachable (HTTP fallback)")
            images = client.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed_arg,
                guidance=guidance,
                n=1,
                response_format="b64_json",
            )
        if not images:
            raise RuntimeError("generate_image returned empty")

        import base64
        from PIL import Image
        import io
        img_bytes = images[0] if isinstance(images[0], bytes) else base64.b64decode(images[0])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(out_path)
        logger.info("image_generate HTTP saved: %s (%d bytes)", out_path, os.path.getsize(out_path))
