from __future__ import annotations

import unittest

from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
from custom_nodes4macos.pipeline.stages.tts_synthesize import TTSSynthesizeStage
from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage


class TestCharacterAppearance(unittest.TestCase):

    def test_get_character_appearance_empty(self):
        result = ImageGenerateStage._get_character_appearance([], {})
        self.assertEqual(result, "")

    def test_get_character_appearance_no_lookup(self):
        result = ImageGenerateStage._get_character_appearance(["Alice"], {})
        self.assertEqual(result, "")

    def test_get_character_appearance_single(self):
        lookup = {"Alice": {"name": "Alice", "appearance": "young woman with black hair"}}
        result = ImageGenerateStage._get_character_appearance(["Alice"], lookup)
        self.assertIn("young woman with black hair", result)
        self.assertIn("character appearance:", result)

    def test_get_character_appearance_multiple(self):
        lookup = {
            "Alice": {"name": "Alice", "appearance": "young woman with black hair"},
            "Bob": {"name": "Bob", "appearance": "old man with white beard"},
        }
        result = ImageGenerateStage._get_character_appearance(["Alice", "Bob"], lookup)
        self.assertIn("young woman with black hair", result)
        self.assertIn("old man with white beard", result)

    def test_get_character_appearance_missing_appearance(self):
        lookup = {"Alice": {"name": "Alice"}}
        result = ImageGenerateStage._get_character_appearance(["Alice"], lookup)
        self.assertEqual(result, "")


class TestComputeSceneSeed(unittest.TestCase):

    def test_base_seed_zero_vary(self):
        result = ImageGenerateStage._compute_scene_seed(0, 3, True, [])
        self.assertEqual(result, 0)

    def test_base_seed_with_vary(self):
        result = ImageGenerateStage._compute_scene_seed(42, 5, True, [])
        self.assertEqual(result, 47)

    def test_base_seed_no_vary(self):
        result = ImageGenerateStage._compute_scene_seed(42, 5, False, [])
        self.assertEqual(result, 42)

    def test_seed_with_characters(self):
        result = ImageGenerateStage._compute_scene_seed(42, 1, True, ["Alice"])
        self.assertNotEqual(result, 43)

    def test_same_characters_same_seed(self):
        r1 = ImageGenerateStage._compute_scene_seed(0, 1, False, ["Alice"])
        r2 = ImageGenerateStage._compute_scene_seed(0, 2, False, ["Alice"])
        self.assertEqual(r1, r2)

    def test_different_characters_different_seed(self):
        r1 = ImageGenerateStage._compute_scene_seed(0, 1, False, ["Alice"])
        r2 = ImageGenerateStage._compute_scene_seed(0, 1, False, ["Bob"])
        self.assertNotEqual(r1, r2)


class TestBuildPromptWithCharacter(unittest.TestCase):

    def test_no_character_no_style(self):
        result = ImageGenerateStage._build_prompt("a dark forest", "")
        self.assertEqual(result, "a dark forest")

    def test_with_style(self):
        result = ImageGenerateStage._build_prompt("a dark forest", "ink-wash style")
        self.assertEqual(result, "a dark forest, ink-wash style")

    def test_with_character_appearance(self):
        result = ImageGenerateStage._build_prompt(
            "a dark forest", "ink-wash style", "character appearance: young woman"
        )
        self.assertIn("character appearance: young woman", result)

    def test_with_empty_character_appearance(self):
        result = ImageGenerateStage._build_prompt("a dark forest", "ink-wash style", "")
        self.assertEqual(result, "a dark forest, ink-wash style")


class TestMergeCharacterRegistry(unittest.TestCase):

    def test_merge_new_registry(self):
        ctx = type("Ctx", (), {"config": {}})()
        parsed = {
            "character_registry": [
                {"name": "Alice", "appearance": "young woman", "voice": "清脆女声"},
            ]
        }
        PromptExpandStage._merge_character_registry(ctx, parsed)
        self.assertEqual(len(ctx.config["character_registry"]), 1)
        self.assertEqual(ctx.config["character_registry"][0]["name"], "Alice")

    def test_merge_no_duplicate(self):
        ctx = type("Ctx", (), {"config": {
            "character_registry": [{"name": "Alice", "appearance": "young woman"}]
        }})()
        parsed = {
            "character_registry": [
                {"name": "Alice", "appearance": "different look"},
                {"name": "Bob", "appearance": "old man"},
            ]
        }
        PromptExpandStage._merge_character_registry(ctx, parsed)
        self.assertEqual(len(ctx.config["character_registry"]), 2)
        alice = next(c for c in ctx.config["character_registry"] if c["name"] == "Alice")
        self.assertEqual(alice["appearance"], "young woman")

    def test_merge_fills_missing_fields(self):
        ctx = type("Ctx", (), {"config": {
            "character_registry": [{"name": "Alice", "appearance": "young woman"}]
        }})()
        parsed = {
            "character_registry": [
                {"name": "Alice", "voice": "清脆女声"},
            ]
        }
        PromptExpandStage._merge_character_registry(ctx, parsed)
        alice = next(c for c in ctx.config["character_registry"] if c["name"] == "Alice")
        self.assertEqual(alice["voice"], "清脆女声")
        self.assertEqual(alice["appearance"], "young woman")

    def test_merge_from_character_descriptions_key(self):
        ctx = type("Ctx", (), {"config": {}})()
        parsed = {
            "character_descriptions": [
                {"name": "Bob", "appearance": "old man"},
            ]
        }
        PromptExpandStage._merge_character_registry(ctx, parsed)
        self.assertEqual(len(ctx.config["character_registry"]), 1)

    def test_merge_empty_registry(self):
        ctx = type("Ctx", (), {"config": {}})()
        parsed = {}
        PromptExpandStage._merge_character_registry(ctx, parsed)
        self.assertNotIn("character_registry", ctx.config)


class TestGetSceneInstructions(unittest.TestCase):

    def test_no_characters(self):
        result = TTSSynthesizeStage._get_scene_instructions("沉稳旁白", [], {})
        self.assertEqual(result, "沉稳旁白")

    def test_with_character_voice(self):
        lookup = {
            "Alice": {"name": "Alice", "voice": "清脆女声，语速快"},
            "Bob": {"name": "Bob", "voice": "低沉男声，语速慢"},
        }
        result = TTSSynthesizeStage._get_scene_instructions(
            "沉稳旁白", ["Alice"], lookup
        )
        self.assertIn("清脆女声", result)
        self.assertIn("角色配音", result)

    def test_multiple_characters(self):
        lookup = {
            "Alice": {"name": "Alice", "voice": "清脆女声"},
            "Bob": {"name": "Bob", "voice": "低沉男声"},
        }
        result = TTSSynthesizeStage._get_scene_instructions(
            "沉稳旁白", ["Alice", "Bob"], lookup
        )
        self.assertIn("清脆女声", result)
        self.assertIn("低沉男声", result)

    def test_no_voice_field(self):
        lookup = {"Alice": {"name": "Alice"}}
        result = TTSSynthesizeStage._get_scene_instructions(
            "沉稳旁白", ["Alice"], lookup
        )
        self.assertEqual(result, "沉稳旁白")


class TestBuildUserMessageWithRegistry(unittest.TestCase):

    def test_no_registry(self):
        msg = PromptExpandStage._build_user_message(
            "故事种子", "标题", 8, "水墨", "ink-wash",
        )
        self.assertNotIn("角色注册表", msg)
        self.assertIn("故事种子", msg)

    def test_with_registry(self):
        registry = [
            {"name": "Alice", "appearance": "young woman", "voice": "清脆"},
        ]
        msg = PromptExpandStage._build_user_message(
            "故事种子", "标题", 8, "水墨", "ink-wash",
            character_registry=registry,
        )
        self.assertIn("角色注册表", msg)
        self.assertIn("Alice", msg)
        self.assertIn("young woman", msg)


if __name__ == "__main__":
    unittest.main()
