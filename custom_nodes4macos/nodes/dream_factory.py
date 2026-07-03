from __future__ import annotations

import json
import logging

logger = logging.getLogger("custom_nodes4macos.nodes.dream_factory")


class FusionMLXDreamFactory:
    """梦工厂 — 一键出片: content_type + story_seed → final video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_type": (
                    [
                        "short_drama", "ad_drama", "puppet_show",
                        "medium_video", "series",
                        "digital_human", "digital_human_live",
                    ],
                    {"default": "short_drama"},
                ),
                "story_seed": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "episode_title": ("STRING", {"default": ""}),
                "scene_count": ("INT", {"default": 0, "min": 0, "max": 200}),
                "style_preset": ("STRING", {"default": ""}),
                "story_file": ("STRING", {"default": ""}),
                "episode_count": ("INT", {"default": 30, "min": 1, "max": 200}),
                "avatar_reference": ("STRING", {"default": ""}),
                "resume_job_id": ("STRING", {"default": ""}),
                "config_overrides": ("STRING", {"multiline": True, "default": "{}"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "scenes_json")
    FUNCTION = "produce"
    CATEGORY = "FusionMLX/DreamFactory"
    OUTPUT_NODE = True

    def produce(
        self,
        content_type: str,
        story_seed: str,
        episode_title: str = "",
        scene_count: int = 0,
        style_preset: str = "",
        story_file: str = "",
        episode_count: int = 30,
        avatar_reference: str = "",
        resume_job_id: str = "",
        config_overrides: str = "{}",
    ):
        from ..pipeline.engine import PipelineEngine

        overrides = {}
        if story_seed and story_seed.strip():
            overrides["story_seed"] = story_seed.strip()
        if episode_title and episode_title.strip():
            overrides["episode_title"] = episode_title.strip()
        if scene_count > 0:
            overrides["scene_count"] = scene_count
        if style_preset and style_preset.strip():
            overrides["style_preset"] = style_preset.strip()
        if story_file and story_file.strip():
            overrides["story_file"] = story_file.strip()
        if episode_count > 0:
            overrides["episode_count"] = episode_count
        if avatar_reference and avatar_reference.strip():
            overrides["avatar_reference"] = avatar_reference.strip()
        try:
            extra = json.loads(config_overrides or "{}")
            if isinstance(extra, dict):
                overrides.update(extra)
        except json.JSONDecodeError:
            logger.warning("config_overrides JSON parse failed, ignoring")

        engine = PipelineEngine()

        if resume_job_id and resume_job_id.strip():
            result = engine.run(
                content_type=content_type,
                resume_from=resume_job_id.strip(),
                **overrides,
            )
        else:
            if not story_seed or not story_seed.strip():
                if not story_file or not story_file.strip():
                    raise ValueError("story_seed or story_file must be provided for new job")
            result = engine.run(
                content_type=content_type,
                **overrides,
            )

        scenes_data = {
            "job_id": result.job_id,
            "content_type": content_type,
            "scenes": [],
        }

        cp_path = f"{result.job_dir}/_checkpoint.json"
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            scenes_data["scenes"] = cp.get("scenes", [])
            scenes_data["completed_stages"] = cp.get("completed_stages", [])
        except Exception:
            pass

        scenes_json = json.dumps(scenes_data, ensure_ascii=False)
        return (result.final_video or "", scenes_json)
