from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.prompt_expand")

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PromptExpandStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="prompt_expand",
            description="故事种子 → 结构化分镜 JSON (LLM 驱动)",
            model_requirements=["llm"],
            memory_estimate_gb=5.6,
            input_kinds=["text"],
            output_kinds=["scenes"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        story_seed = ctx.config.get("story_seed", "")
        episodes = ctx.config.get("episodes", [])

        if episodes and isinstance(episodes, list) and len(episodes) > 0:
            self._process_episodes(ctx, model_manager, episodes)
            return

        if not story_seed or not story_seed.strip():
            raise ValueError("prompt_expand: story_seed is empty")

        episode_title = ctx.config.get("episode_title", "")
        scene_count = ctx.config.get("scene_count", 8)
        style_preset = ctx.config.get("style_preset", "")
        temperature = ctx.config.get("prompt_expand_temperature", 0.75)

        style_presets = ctx.config.get("style_presets", {})
        if style_preset and style_preset in style_presets:
            style_text = style_presets[style_preset]
        elif style_preset:
            style_text = style_preset
        else:
            first_key = next(iter(style_presets), "")
            style_text = style_presets.get(first_key, "")

        system_prompt_file = (
            ctx.config.get("system_prompt_file")
            or ctx.config.get("system_prompt")
            or "horror_director.md"
        )
        system_prompt = self._load_system_prompt(system_prompt_file)
        user_msg = self._build_user_message(
            story_seed, episode_title, scene_count, style_preset, style_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        with model_manager.acquire("llm") as handle:
            model, tokenizer = handle.model
            content = self._generate(model, tokenizer, messages, temperature)

        parsed = self._parse_and_validate_raw(content)
        scenes = parsed.get("scenes", [])

        if "global_style" in parsed and "global_style" not in ctx.config:
            ctx.config["global_style"] = parsed["global_style"]
            logger.info("global_style from LLM: %s", parsed["global_style"])

        ctx.scenes = scenes
        ctx.update_progress("prompt_expand", 1, 1)
        logger.info("prompt_expand done scenes=%d", len(scenes))

    def _process_episodes(self, ctx, model_manager, episodes) -> None:
        scene_count = ctx.config.get("scene_count", 8)
        style_preset = ctx.config.get("style_preset", "")
        temperature = ctx.config.get("prompt_expand_temperature", 0.75)

        style_presets = ctx.config.get("style_presets", {})
        if style_preset and style_preset in style_presets:
            style_text = style_presets[style_preset]
        elif style_preset:
            style_text = style_preset
        else:
            first_key = next(iter(style_presets), "")
            style_text = style_presets.get(first_key, "")

        system_prompt_file = (
            ctx.config.get("system_prompt_file")
            or ctx.config.get("system_prompt")
            or "series_director.md"
        )
        system_prompt = self._load_system_prompt(system_prompt_file)

        all_scenes = []
        global_scene_offset = 0
        from ..checkpoint import CheckpointManager
        checkpoint = CheckpointManager(ctx.job_dir)

        for ep_idx, episode in enumerate(episodes):
            ep_title = episode.get("title", f"第{ep_idx + 1}集")
            ep_synopsis = episode.get("synopsis", "")
            ep_key_scenes = episode.get("key_scenes", [])
            ep_cliffhanger = episode.get("cliffhanger", "")

            ep_seed = (
                f"【{ep_title}】\n"
                f"剧情概要：{ep_synopsis}\n"
            )
            if ep_key_scenes:
                ep_seed += f"关键场景：{', '.join(str(s) for s in ep_key_scenes)}\n"
            if ep_cliffhanger:
                ep_seed += f"悬念结尾：{ep_cliffhanger}\n"

            user_msg = self._build_user_message(
                ep_seed, ep_title, scene_count, style_preset, style_text,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            with model_manager.acquire("llm") as handle:
                model, tokenizer = handle.model
                content = self._generate(model, tokenizer, messages, temperature)

            parsed = self._parse_and_validate_raw(content)
            scenes = parsed.get("scenes", [])
            for scene in scenes:
                scene["episode_id"] = episode.get("episode_id", ep_idx + 1)
                scene["episode_title"] = ep_title
            global_scene_offset = self._renumber_scenes(
                scenes, global_scene_offset,
            )

            if "global_style" in parsed and "global_style" not in ctx.config:
                ctx.config["global_style"] = parsed["global_style"]
                logger.info("global_style from LLM: %s", parsed["global_style"])

            all_scenes.extend(scenes)
            ctx.scenes = all_scenes
            ctx.update_progress("prompt_expand", ep_idx + 1, len(episodes))
            logger.info("prompt_expand episode %d/%d scenes=%d", ep_idx + 1, len(episodes), len(scenes))

            if ctx.should_checkpoint_scene(ep_idx + 1):
                checkpoint.save(ctx)

        logger.info("prompt_expand all episodes done total_scenes=%d", len(all_scenes))

    @staticmethod
    def _parse_and_validate(content: str) -> list[dict]:
        parsed = PromptExpandStage._parse_and_validate_raw(content)
        return parsed.get("scenes", [])

    @staticmethod
    def _parse_and_validate_raw(content: str) -> dict:
        parsed = PromptExpandStage._parse_json(content)
        if isinstance(parsed, list):
            logger.warning("model returned bare list, wrapping as {scenes: [...]}")
            parsed = {"scenes": parsed}
        if not isinstance(parsed, dict):
            raise RuntimeError(f"prompt_expand: output is not JSON object: {type(parsed).__name__}")

        scenes = parsed.get("scenes", [])
        if not scenes:
            raise RuntimeError("prompt_expand: no scenes in output")

        for i, scene in enumerate(scenes):
            if "scene_id" not in scene:
                scene["scene_id"] = i + 1

        return parsed

    @staticmethod
    def _renumber_scenes(scenes: list[dict], offset: int) -> int:
        for i, scene in enumerate(scenes):
            scene["scene_id"] = offset + i + 1
        return offset + len(scenes)

    @staticmethod
    def _load_system_prompt(filename: str) -> str:
        if os.path.isfile(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        path = _PROMPT_DIR / filename
        if not path.exists():
            logger.warning("system prompt missing: %s, using fallback", path)
            return "你是一位编剧，将故事种子扩展为分镜脚本，只输出 JSON。"
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _build_user_message(
        story_seed: str,
        episode_title: str,
        scene_count: int,
        style_preset: str,
        style_text: str,
    ) -> str:
        return (
            f"故事种子：{story_seed.strip()}\n"
            f"剧集标题：{episode_title.strip() or '（待定）'}\n"
            f"目标分镜数：{scene_count}\n"
            f"画风预设：{style_preset}（{style_text}）\n"
            f"请严格按 schema 输出 JSON，分镜数必须等于 {scene_count}。"
        )

    @staticmethod
    def _generate(model, tokenizer, messages: list[dict], temperature: float) -> str:
        try:
            from mlx_lm import generate as mlx_generate
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            output = mlx_generate(
                model, tokenizer,
                prompt=prompt_text,
                max_tokens=4096,
                temp=temperature,
                verbose=False,
            )
            return output.strip()
        except ImportError:
            logger.warning("mlx_lm not available, falling back to HTTP")
            return PromptExpandStage._generate_http(messages, temperature)

    @staticmethod
    def _generate_http(messages: list[dict], temperature: float) -> str:
        from ...fusion_client import FusionMLXClient
        with FusionMLXClient() as client:
            if not client.health():
                raise RuntimeError("fusion-mlx unreachable (HTTP fallback)")
            content, _ = client.chat(
                messages, temperature=temperature, json_mode=True,
            )
        return content

    @staticmethod
    def _parse_json(content: str) -> dict | list:
        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                inner = parts[1]
                if inner.lower().startswith("json"):
                    inner = inner[4:]
                text = inner.strip()
            else:
                text = text.strip("`")
        return json.loads(text)
