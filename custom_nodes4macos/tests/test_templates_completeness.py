"""模板与配置完整性测试。

覆盖全部 7 模板的可解析性、stage 注册一致性、
content_type 与 DreamFactory 节点声明一致、prompts 文件存在性。

补充 REVIEW_REPORT：模板与节点声明的一致性无集中校验。
"""
from __future__ import annotations

import os
import unittest

import yaml

from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "pipeline", "templates",
)
_PROMPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "prompts",
)

# 所有合法 content_type（来自 DreamFactory 节点声明）
EXPECTED_CONTENT_TYPES = {
    "short_drama", "ad_drama", "puppet_show",
    "medium_video", "series", "digital_human", "digital_human_live",
}

# 已实现的 stage 名（来自 stages/__init__.py 注册）
IMPLEMENTED_STAGES = {
    "story_ingest", "prompt_expand", "image_generate",
    "tts_synthesize", "ken_burns", "multi_pose", "assemble", "sfx", "subtitle",
    "digital_human_render", "avatar_create", "avatar_animate", "voice_clone",
    "series_orchestrate", "publish",
}

# 产出 final 视频的收尾 stage（assemble 直出，subtitle 覆盖 final，series_orchestrate 逐集产出 final）
_FINAL_PRODUCERS = {"assemble", "subtitle", "series_orchestrate"}


def _load_all_templates():
    templates = {}
    for fname in sorted(os.listdir(_TEMPLATE_DIR)):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(_TEMPLATE_DIR, fname), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        templates[fname] = data
    return templates


class TestTemplateStructure(unittest.TestCase):

    def setUp(self):
        self.templates = _load_all_templates()

    def test_all_seven_templates_present(self):
        ct_set = {data["content_type"] for data in self.templates.values()}
        self.assertEqual(ct_set, EXPECTED_CONTENT_TYPES)

    def test_each_template_has_required_fields(self):
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                self.assertIsInstance(data, dict, f"{fname} not a dict")
                self.assertIn("content_type", data, f"{fname} missing content_type")
                self.assertIn("name", data, f"{fname} missing name")
                self.assertIn("stages", data, f"{fname} missing stages")
                self.assertIn("defaults", data, f"{fname} missing defaults")
                self.assertIsInstance(data["stages"], list, f"{fname} stages not a list")
                self.assertIsInstance(data["defaults"], dict, f"{fname} defaults not a dict")

    def test_each_template_stages_non_empty(self):
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                self.assertGreater(len(data["stages"]), 0, f"{fname} has no stages")

    def test_content_type_matches_filename(self):
        """content_type 应与文件名 stem 一致。"""
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                stem = fname.rsplit(".", 1)[0]
                self.assertEqual(data["content_type"], stem,
                                 f"{fname} content_type={data['content_type']} != {stem}")


class TestTemplateStagesConsistency(unittest.TestCase):

    def setUp(self):
        self.templates = _load_all_templates()

    def test_all_template_stages_are_implemented(self):
        """每个模板声明的 stage 都已在 stages/__init__.py 注册。"""
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                for stage_name in data["stages"]:
                    self.assertIn(stage_name, IMPLEMENTED_STAGES,
                                  f"{fname} references unimplemented stage: {stage_name}")

    def test_short_drama_uses_horror_pipeline(self):
        """short_drama 应含 prompt_expand→image→tts→ken_burns→assemble→sfx→subtitle。"""
        data = self.templates["short_drama.yaml"]
        expected = ["prompt_expand", "image_generate", "tts_synthesize", "ken_burns", "assemble", "sfx", "subtitle"]
        self.assertEqual(data["stages"], expected)

    def test_series_includes_story_ingest(self):
        """series 必须以 story_ingest 开头。"""
        data = self.templates["series.yaml"]
        self.assertEqual(data["stages"][0], "story_ingest")

    def test_digital_human_includes_avatar_stages(self):
        """digital_human 必须含 avatar_create 和 avatar_animate。"""
        data = self.templates["digital_human.yaml"]
        self.assertIn("avatar_create", data["stages"])
        self.assertIn("avatar_animate", data["stages"])

    def test_all_templates_end_with_final_producer(self):
        """所有内容类型最终以产出 final 的 stage 收尾（assemble 或 subtitle）。"""
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                self.assertIn(data["stages"][-1], _FINAL_PRODUCERS,
                              f"{fname} does not end with a final-producing stage")


class TestTemplateDefaults(unittest.TestCase):

    def setUp(self):
        self.templates = _load_all_templates()

    def test_short_drama_scene_count_sufficient(self):
        """短剧分镜数 ≥12 才能满足 2min 时长目标。"""
        sc = self.templates["short_drama.yaml"]["defaults"]["scene_count"]
        self.assertGreaterEqual(sc, 12)

    def test_series_has_episode_count(self):
        """series 模板必须声明 episode_count。"""
        defaults = self.templates["series.yaml"]["defaults"]
        self.assertIn("episode_count", defaults)
        self.assertGreaterEqual(defaults["episode_count"], 1)

    def test_medium_video_has_checkpoint_interval(self):
        """medium_video 应配置 checkpoint_every_n_scenes 以支持续跑。"""
        defaults = self.templates["medium_video.yaml"]["defaults"]
        self.assertIn("checkpoint_every_n_scenes", defaults)
        self.assertGreater(defaults["checkpoint_every_n_scenes"], 0)

    def test_series_has_checkpoint_interval(self):
        """series 应配置 checkpoint_every_n_scenes。"""
        defaults = self.templates["series.yaml"]["defaults"]
        self.assertIn("checkpoint_every_n_scenes", defaults)

    def test_all_templates_have_style_presets(self):
        """每个模板应有 style_presets 供选择。"""
        for fname, data in self.templates.items():
            with self.subTest(template=fname):
                if "style_presets" in data:
                    self.assertIsInstance(data["style_presets"], dict)
                    self.assertGreater(len(data["style_presets"]), 0)


class TestPromptFilesExist(unittest.TestCase):

    def setUp(self):
        self.templates = _load_all_templates()

    def test_referenced_prompt_files_exist(self):
        """模板通过 system_prompt_file 引用的 prompt 文件必须存在。"""
        for fname, data in self.templates.items():
            prompts = data.get("prompts", {})
            sp_file = prompts.get("system_prompt_file")
            if not sp_file:
                continue
            with self.subTest(template=fname, prompt=sp_file):
                path = os.path.join(_PROMPT_DIR, sp_file)
                self.assertTrue(os.path.exists(path),
                                f"{fname} references missing prompt: {sp_file}")

    def test_all_prompt_files_are_non_empty(self):
        """prompts 目录下所有 .md 文件非空。"""
        for fname in os.listdir(_PROMPT_DIR):
            if not fname.endswith(".md"):
                continue
            with self.subTest(prompt=fname):
                path = os.path.join(_PROMPT_DIR, fname)
                self.assertGreater(os.path.getsize(path), 0)

    def test_six_director_prompts_present(self):
        """应有 6 个 director prompt 文件。"""
        md_files = [f for f in os.listdir(_PROMPT_DIR) if f.endswith(".md")]
        self.assertEqual(len(md_files), 6)


class TestNodeTemplateAlignment(unittest.TestCase):

    def test_node_content_types_match_templates(self):
        """DreamFactory 节点声明的 content_type 与模板文件一一对应。"""
        inputs = FusionMLXDreamFactory.INPUT_TYPES()
        node_cts = set(inputs["required"]["content_type"][0])
        template_cts = {data["content_type"] for data in _load_all_templates().values()}
        self.assertEqual(node_cts, template_cts,
                         f"node={node_cts} != templates={template_cts}")


if __name__ == "__main__":
    unittest.main()
