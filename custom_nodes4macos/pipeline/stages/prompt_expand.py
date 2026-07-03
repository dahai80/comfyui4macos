from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..stage import Stage, StageInfo

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.prompt_expand")

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_STYLE_PRESETS = {
    "水墨悬疑": "Chinese ink-wash dark fantasy, muted desaturated tones, misty mountains, cinematic, 8k",
    "纸扎阴森": "paper-crafted eerie folk horror, dim candlelight, cold palette, film grain, 8k",
    "暗黑道观": "dark Taoist temple interior, incense smoke, chiaroscuro, blood-red accents, 8k",
    "佛寺夜寂": "abandoned Buddhist shrine, moonlit, cold blue, silent dread, volumetric light, 8k",
}


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
        if not story_seed or not story_seed.strip():
            raise ValueError("prompt_expand: story_seed is empty")

        episode_title = ctx.config.get("episode_title", "")
        scene_count = ctx.config.get("scene_count", 8)
        style_preset = ctx.config.get("style_preset", "水墨悬疑")
        temperature = ctx.config.get("prompt_expand_temperature", 0.75)
        system_prompt_file = ctx.config.get("system_prompt_file", "horror_director.md")

        style_text = _STYLE_PRESETS.get(style_preset, _STYLE_PRESETS["水墨悬疑"])
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

        parsed = self._parse_json(content)
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

        ctx.scenes = scenes
        ctx.update_progress("prompt_expand", 1, 1)
        logger.info("prompt_expand done scenes=%d", len(scenes))

    @staticmethod
    def _load_system_prompt(filename: str) -> str:
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
