import json
import logging
from pathlib import Path

from ..fusion_client import FusionMLXClient, FusionMLXError, list_models_safe

logger = logging.getLogger("custom_nodes4macos.prompt_expand")

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

_STYLE_PRESETS = {
    "水墨悬疑": "Chinese ink-wash dark fantasy, muted desaturated tones, misty mountains, cinematic, 8k",
    "纸扎阴森": "paper-crafted eerie folk horror, dim candlelight, cold palette, film grain, 8k",
    "暗黑道观": "dark Taoist temple interior, incense smoke, chiaroscuro, blood-red accents, 8k",
    "佛寺夜寂": "abandoned Buddhist shrine, moonlit, cold blue, silent dread, volumetric light, 8k",
}


def _load_system_prompt() -> str:
    path = _PROMPT_DIR / "horror_director.md"
    if not path.exists():
        logger.warning("system prompt file missing: %s", path)
        return "你是中式恐怖短剧编剧，只输出 JSON。"
    return path.read_text(encoding="utf-8")


def _parse_json(content: str) -> dict:
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
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("json parse failed: %s head=%r", exc, text[:200])
        raise FusionMLXError(f"模型输出不是合法 JSON: {exc}") from exc


def _build_user_message(story_seed: str, episode_title: str, scene_count: int, style_preset: str, style_text: str) -> str:
    return (
        f"故事种子：{story_seed.strip()}\n"
        f"剧集标题：{episode_title.strip() or '（待定）'}\n"
        f"目标分镜数：{scene_count}\n"
        f"画风预设：{style_preset}（{style_text}）\n"
        f"请严格按 schema 输出 JSON，分镜数必须等于 {scene_count}。"
    )


class FusionMLXPromptExpand:
    system_prompt = _load_system_prompt()

    @classmethod
    def INPUT_TYPES(cls):
        models = list_models_safe()
        return {
            "required": {
                "story_seed": ("STRING", {"multiline": True, "default": ""}),
                "episode_title": ("STRING", {"default": ""}),
                "scene_count": ("INT", {"default": 8, "min": 1, "max": 40, "step": 1}),
                "model": (models, {"default": "(auto)"}),
                "style_preset": (list(_STYLE_PRESETS.keys()), {"default": "水墨悬疑"}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("scenes_json", "scene_count")
    FUNCTION = "expand"
    CATEGORY = "FusionMLX/Horror"

    def expand(
        self,
        story_seed: str,
        episode_title: str,
        scene_count: int,
        model: str,
        style_preset: str,
        temperature: float,
        base_url: str = "",
        api_key: str = "",
    ):
        if not story_seed or not story_seed.strip():
            raise ValueError("story_seed 不能为空")
        style_text = _STYLE_PRESETS.get(style_preset, _STYLE_PRESETS["水墨悬疑"])
        user_msg = _build_user_message(story_seed, episode_title, scene_count, style_preset, style_text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        model_name = None if model == "(auto)" else model
        logger.info(
            "expand title=%s scenes=%d style=%s model=%s temp=%.2f",
            episode_title or "(untitled)", scene_count, style_preset, model_name or "auto", temperature,
        )
        with FusionMLXClient(base_url=base_url or None, api_key=api_key or None) as client:
            if not client.health():
                raise FusionMLXError(f"fusion-mlx 不可达: {client.base_url}")
            content, usage = client.chat(
                messages,
                model=model_name,
                temperature=temperature,
                json_mode=True,
            )
        parsed = _parse_json(content)
        if isinstance(parsed, list):
            logger.warning("model returned bare list, wrapping as {scenes: [...]}; consider a stronger model")
            parsed = {"scenes": parsed}
        if not isinstance(parsed, dict):
            raise FusionMLXError(f"模型输出不是 JSON 对象: {type(parsed).__name__}")
        scenes = parsed.get("scenes", [])
        actual = len(scenes)
        logger.info("expand done scenes=%d tokens=%s", actual, usage)
        return (json.dumps(parsed, ensure_ascii=False, indent=2), actual)
