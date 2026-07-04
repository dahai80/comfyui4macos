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
        scheduler = ctx.config.get("flux_scheduler", "linear")
        steps_auto = ctx.config.get("flux_steps_auto", False)
        consistency_check = ctx.config.get("consistency_check", False)

        if steps_auto and scheduler == "flow_match_euler_discrete" and steps > 6:
            logger.info("flux_steps_auto: reducing steps %d→6 for FlowMatchEuler", steps)
            steps = 6
        char_ref_dir = ctx.config.get("character_reference_dir", "")
        global_style = ctx.config.get("global_style", "")
        enable_tiling = ctx.config.get("flux_tiling", width > 1024 or height > 1024)
        character_registry = ctx.config.get("character_registry", [])
        char_lookup = {c["name"]: c for c in character_registry if "name" in c}

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("flux") as handle:
            pipeline = handle.model
            if enable_tiling:
                self._apply_tiling(pipeline, width, height)

            # Pre-encode all prompts before denoising, then evict text encoders
            evict_encoders = ctx.config.get("flux_evict_encoders", True)
            if evict_encoders:
                self._preencode_and_evict(pipeline, ctx, char_lookup, global_style,
                                          consistency_check)

            t_start = time.time()
            generated = 0
            for i, scene in enumerate(ctx.scenes):
                scene_id = scene.get("scene_id", i + 1)
                if ctx.has_artifact_on_disk(scene_id, "image"):
                    logger.info("image_generate scene %d skipped (exists)", scene_id)
                    continue

                visual_prompt = scene.get("visual_prompt", "")
                if not visual_prompt:
                    logger.warning("image_generate scene %d: no visual_prompt, skip", scene_id)
                    continue

                scene_chars = scene.get("characters", [])
                char_appearance = self._get_character_appearance(scene_chars, char_lookup)
                prompt = self._build_prompt(visual_prompt, global_style, char_appearance)
                if consistency_check:
                    char_desc = scene.get("character_description", "")
                    if char_desc:
                        prompt = f"{prompt}, consistent character: {char_desc}"
                    episode_title = scene.get("episode_title", "")
                    if episode_title:
                        prompt = f"{prompt}, from episode: {episode_title}"
                out_path = ctx.artifact_path(scene_id, "image")

                scene_seed = self._compute_scene_seed(seed, scene_id, vary_seed, scene_chars)
                t_scene = time.time()
                self._generate_image(
                    pipeline, prompt, width, height, steps, guidance, scene_seed, out_path,
                    scheduler=scheduler,
                )
                elapsed = time.time() - t_scene
                generated += 1
                logger.info(
                    "scene %d/%d done in %.1fs → %s",
                    i + 1, len(ctx.scenes), elapsed, out_path,
                )

                ctx.set_artifact(scene_id, "image", out_path)

                cache_every = ctx.config.get("clear_cache_every_n_scenes", 0)
                if cache_every > 0 and (i + 1) % cache_every == 0:
                    try:
                        import mlx.core as mx
                        mx.clear_cache()
                        logger.info("mx.clear_cache() at scene %d (every %d)", i + 1, cache_every)
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
    def _preencode_and_evict(pipeline, ctx, char_lookup: dict, global_style: str,
                             consistency_check: bool) -> None:
        try:
            from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        except ImportError:
            logger.warning("PromptEncoder not available, skipping pre-encode")
            return

        unique_prompts = set()
        for scene in ctx.scenes:
            vp = scene.get("visual_prompt", "")
            if not vp:
                continue
            chars = scene.get("characters", [])
            char_app = ImageGenerateStage._get_character_appearance(chars, char_lookup)
            prompt = ImageGenerateStage._build_prompt(vp, global_style, char_app)
            if consistency_check:
                char_desc = scene.get("character_description", "")
                if char_desc:
                    prompt = f"{prompt}, consistent character: {char_desc}"
                ep_title = scene.get("episode_title", "")
                if ep_title:
                    prompt = f"{prompt}, from episode: {ep_title}"
            unique_prompts.add(prompt)

        if not unique_prompts:
            return

        t0 = time.time()
        for prompt in unique_prompts:
            PromptEncoder.encode_prompt(
                prompt=prompt,
                prompt_cache=pipeline.prompt_cache,
                t5_tokenizer=pipeline.tokenizers["t5"],
                clip_tokenizer=pipeline.tokenizers["clip"],
                t5_text_encoder=pipeline.t5_text_encoder,
                clip_text_encoder=pipeline.clip_text_encoder,
            )
        encode_time = time.time() - t0
        logger.info(
            "pre-encoded %d unique prompts in %.1fs (prompt_cache=%d)",
            len(unique_prompts), encode_time, len(pipeline.prompt_cache),
        )

        del pipeline.t5_text_encoder
        del pipeline.clip_text_encoder
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
            logger.info("evicted text encoders, mx.clear_cache() done to free GPU memory")
        except ImportError:
            pass

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
    def _build_prompt(visual_prompt: str, global_style: str, char_appearance: str = "") -> str:
        vp = (visual_prompt or "").strip()
        if not vp:
            raise ValueError("visual_prompt is empty")
        style = (global_style or "").strip()
        parts = [vp]
        if char_appearance:
            parts.append(char_appearance)
        if style:
            parts.append(style)
        return ", ".join(parts)

    @staticmethod
    def _get_character_appearance(scene_chars: list, char_lookup: dict) -> str:
        if not scene_chars or not char_lookup:
            return ""
        appearances = []
        for name in scene_chars:
            c = char_lookup.get(name)
            if c and c.get("appearance"):
                appearances.append(c["appearance"])
        if not appearances:
            return ""
        return "character appearance: " + "; ".join(appearances)

    @staticmethod
    def _compute_scene_seed(base_seed: int, scene_id: int, vary_seed: bool, scene_chars: list) -> int:
        scene_seed = (base_seed + scene_id) if (base_seed and vary_seed) else base_seed
        if scene_chars:
            char_hash = 0
            for name in scene_chars:
                for ch in name:
                    char_hash = (char_hash * 31 + ord(ch)) & 0x7FFFFFFF
            scene_seed = (scene_seed or 0) + char_hash
        return scene_seed

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
        scheduler: str = "linear",
    ) -> None:
        try:
            ImageGenerateStage._generate_mlx(
                pipeline, prompt, width, height, steps, guidance, seed, out_path,
                scheduler=scheduler,
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
        scheduler: str = "linear",
    ) -> None:
        seed_arg = seed if seed else 42
        logger.info(
            "image_generate MLX prompt_len=%d size=%dx%d steps=%d guidance=%.1f seed=%s scheduler=%s",
            len(prompt), width, height, steps, guidance, seed_arg, scheduler,
        )

        use_fast = scheduler == "flow_match_euler_discrete"
        if use_fast:
            try:
                ImageGenerateStage._generate_mlx_fast(
                    pipeline, prompt, width, height, steps, guidance, seed_arg,
                    out_path, scheduler,
                )
                return
            except Exception as exc:
                logger.warning("fast path failed (%s), falling back to standard", exc)

        image = pipeline.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance=guidance,
            seed=seed_arg,
            scheduler=scheduler,
        )
        if image is None:
            raise RuntimeError("Flux1.generate_image returned None")

        image.save(path=out_path)
        logger.info("image_generate MLX saved: %s (%d bytes)", out_path, os.path.getsize(out_path))

    @staticmethod
    def _generate_mlx_fast(
        pipeline,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
        scheduler: str,
    ) -> None:
        import mlx.core as mx
        from mflux.models.common.config.config import Config
        from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        from mflux.models.flux.model.flux_vae.vae import VAE
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.utils.image_util import ImageUtil

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            model_config=pipeline.model_config,
            num_inference_steps=steps,
        )

        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=pipeline.vae,
                latent_creator=FluxLatentCreator,
                image_path=None,
                sigmas=config.scheduler.sigmas,
                init_time_step=0,
            ),
        )

        if prompt in pipeline.prompt_cache:
            prompt_embeds, pooled_embeds = pipeline.prompt_cache[prompt]
        else:
            has_encoders = (
                hasattr(pipeline, "t5_text_encoder") and hasattr(pipeline, "clip_text_encoder")
            )
            if not has_encoders:
                raise RuntimeError(
                    f"prompt not in cache and text encoders evicted: {prompt[:80]}..."
                )
            prompt_embeds, pooled_embeds = PromptEncoder.encode_prompt(
                prompt=prompt,
                prompt_cache=pipeline.prompt_cache,
                t5_tokenizer=pipeline.tokenizers["t5"],
                clip_tokenizer=pipeline.tokenizers["clip"],
                t5_text_encoder=pipeline.t5_text_encoder,
                clip_text_encoder=pipeline.clip_text_encoder,
            )

        # Pre-compute rotary embeddings (constant for same dimensions)
        rotary_emb = pipeline.transformer.compute_rotary_embeddings(
            prompt_embeds, pipeline.transformer.pos_embed, config,
        )

        # Pre-compute text embeddings for all timesteps
        num_steps = config.num_inference_steps
        sigmas = config.scheduler.sigmas
        all_text_emb = []
        for t in range(num_steps):
            time_step = sigmas[t] * config.num_train_steps
            time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
            guid_arr = mx.broadcast_to(
                config.guidance * config.num_train_steps, (1,),
            ).astype(config.precision)
            text_emb = pipeline.transformer.time_text_embed(
                time_step, pooled_embeds, guid_arr,
            )
            all_text_emb.append(text_emb)

        # Try mx.compile on the transformer forward
        compiled_forward = None
        try:
            compiled_forward = mx.compile(pipeline.transformer.__call__)
            # Warmup: trigger compilation with a dummy call
            _ = compiled_forward(
                t=0, config=config, hidden_states=latents,
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_embeds,
            )
            mx.eval(_)
            logger.info("mx.compile(transformer.__call__) succeeded")
        except Exception as exc:
            logger.warning("mx.compile(transformer) failed: %s, using uncompiled", exc)
            compiled_forward = None

        # Denoising loop
        t_loop = time.time()
        for t in range(num_steps):
            if compiled_forward is not None:
                noise = compiled_forward(
                    t=t, config=config, hidden_states=latents,
                    prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_embeds,
                )
            else:
                noise = pipeline.transformer(
                    t=t, config=config, hidden_states=latents,
                    prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_embeds,
                )
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            logger.debug("step %d/%d done", t + 1, num_steps)

        loop_elapsed = time.time() - t_loop
        logger.info("denoising loop: %d steps in %.1fs", num_steps, loop_elapsed)

        # VAE decode
        latents = FluxLatentCreator.unpack_latents(
            latents=latents, height=config.height, width=config.width,
        )
        decoded = VAEUtil.decode(
            vae=pipeline.vae, latent=latents, tiling_config=pipeline.tiling_config,
        )
        image = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=pipeline.bits,
            lora_paths=pipeline.lora_paths,
            lora_scales=pipeline.lora_scales,
            image_path=None,
            image_strength=None,
            generation_time=0,
        )
        image.save(path=out_path)
        logger.info("image_generate MLX fast saved: %s (%d bytes)", out_path, os.path.getsize(out_path))

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
