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
            description="visual_prompt → PNG (fusion-mlx HTTP)",
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

        self._warn_invalid_config(ctx)

        char_ref_dir = ctx.config.get("character_reference_dir", "")
        global_style = ctx.config.get("global_style", "")
        character_registry = ctx.config.get("character_registry", [])
        char_lookup = {c["name"]: c for c in character_registry if "name" in c}
        character_style = ctx.config.get("character_style", "none")
        ref_strength = float(ctx.config.get("realistic_reference_strength", 0.6))
        ref_mode = ctx.config.get("realistic_conditioning_mode", "redux")

        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        with model_manager.acquire("flux") as handle:
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
                ref_b64 = self._resolve_realistic_reference(
                    character_style, scene_chars, char_lookup, char_ref_dir,
                )
                t_scene = time.time()
                self._generate_image(
                    handle, prompt, width, height, steps, guidance, scene_seed, out_path,
                    reference_image=ref_b64,
                    reference_strength=ref_strength if ref_b64 else None,
                    conditioning_mode=ref_mode if ref_b64 else None,
                )
                elapsed = time.time() - t_scene
                generated += 1
                logger.info(
                    "scene %d/%d done in %.1fs → %s",
                    i + 1, len(ctx.scenes), elapsed, out_path,
                )

                ctx.set_artifact(scene_id, "image", out_path)

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
    def _warn_invalid_config(ctx) -> None:
        invalid = []
        if ctx.config.get("flux_scheduler", "linear") != "linear":
            invalid.append("flux_scheduler")
        if ctx.config.get("flux_tiling"):
            invalid.append("flux_tiling")
        if ctx.config.get("flux_evict_encoders"):
            invalid.append("flux_evict_encoders")
        if ctx.config.get("flux_steps_auto"):
            invalid.append("flux_steps_auto")
        if invalid:
            logger.warning(
                "config %s ignored by fusion-mlx (server manages memory/scheduler); "
                "image gen uses default flux pipeline",
                ", ".join(invalid),
            )

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
    def _resolve_realistic_reference(
        character_style: str,
        scene_chars: list,
        char_lookup: dict,
        char_ref_dir: str,
    ) -> str | None:
        if character_style != "realistic" or not scene_chars or not char_lookup:
            return None
        for name in scene_chars:
            c = char_lookup.get(name)
            if not c or not c.get("reference_image"):
                continue
            ref = c["reference_image"]
            path = ref if os.path.isabs(ref) else os.path.join(char_ref_dir or "", ref)
            if os.path.exists(path):
                logger.info(
                    "realistic reference for character %s: %s (mode=redux)",
                    name, path,
                )
                return ImageGenerateStage._image_to_b64(path)
            logger.warning(
                "realistic reference_image missing for character %s: %s",
                name, path,
            )
        return None

    @staticmethod
    def _image_to_b64(path: str) -> str:
        import base64
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")

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
        handle,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
        reference_image: str | None = None,
        reference_strength: float | None = None,
        conditioning_mode: str | None = None,
    ) -> None:
        ImageGenerateStage._generate_http(
            handle, prompt, width, height, steps, guidance, seed, out_path,
            reference_image=reference_image,
            reference_strength=reference_strength,
            conditioning_mode=conditioning_mode,
        )

    @staticmethod
    def _generate_http(
        handle,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: str,
        reference_image: str | None = None,
        reference_strength: float | None = None,
        conditioning_mode: str | None = None,
    ) -> None:
        seed_arg = None if seed == 0 else seed
        logger.info(
            "image_generate HTTP prompt_len=%d size=%dx%d steps=%d guidance=%.1f ref=%s",
            len(prompt), width, height, steps, guidance, conditioning_mode or "none",
        )
        images = handle.client.generate_image(
            prompt=prompt,
            model=handle.model_name or None,
            width=width,
            height=height,
            steps=steps,
            seed=seed_arg,
            guidance=guidance,
            n=1,
            response_format="b64_json",
            reference_image=reference_image,
            reference_strength=reference_strength,
            conditioning_mode=conditioning_mode,
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
