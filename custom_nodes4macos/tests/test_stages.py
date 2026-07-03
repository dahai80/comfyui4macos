from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.ken_burns import KenBurnsStage, _build_zoompan
from custom_nodes4macos.pipeline.stages.assemble import AssembleStage
from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
from custom_nodes4macos.pipeline.stages.tts_synthesize import TTSSynthesizeStage


class TestBuildZoompan(unittest.TestCase):

    def test_zoom_in(self):
        result = _build_zoompan("zoom-in", 1080, 1920, 30, 240)
        self.assertIn("zoompan", result)
        self.assertIn("1+0.25*on/240", result)

    def test_zoom_out(self):
        result = _build_zoompan("zoom-out", 1080, 1920, 30, 240)
        self.assertIn("1.25-0.25*on/240", result)

    def test_pan_right(self):
        result = _build_zoompan("pan-right", 1080, 1920, 30, 240)
        self.assertIn("on/240", result)

    def test_pan_left(self):
        result = _build_zoompan("pan-left", 1080, 1920, 30, 240)
        self.assertIn("1-on/240", result)

    def test_unknown_defaults_to_zoom_in(self):
        result = _build_zoompan("unknown_preset", 1080, 1920, 30, 240)
        self.assertIn("1+0.2*on/240", result)

    def test_random_picks_one(self):
        result = _build_zoompan("random", 1080, 1920, 30, 240)
        self.assertIn("zoompan", result)


class TestKenBurnsStageInfo(unittest.TestCase):

    def test_info(self):
        info = KenBurnsStage.info()
        self.assertEqual(info.name, "ken_burns")
        self.assertEqual(info.model_requirements, [])
        self.assertEqual(info.memory_estimate_gb, 0.0)

    def test_skip_if_completed(self):
        stage = KenBurnsStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["ken_burns"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestAssembleStageInfo(unittest.TestCase):

    def test_info(self):
        info = AssembleStage.info()
        self.assertEqual(info.name, "assemble")
        self.assertEqual(info.model_requirements, [])

    def test_collect_clips(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = [
            {"scene_id": 1},
            {"scene_id": 2},
        ]
        clip1 = os.path.join(tmpdir, "scene_001_clip.mp4")
        clip2 = os.path.join(tmpdir, "scene_002_clip.mp4")
        with open(clip1, "wb") as f:
            f.write(b"fake_clip_1")
        with open(clip2, "wb") as f:
            f.write(b"fake_clip_2")
        ctx.artifacts = {"1_clip": clip1, "2_clip": clip2}

        clips = AssembleStage._collect_clips(ctx)
        self.assertEqual(len(clips), 2)


class TestPromptExpandStageInfo(unittest.TestCase):

    def test_info(self):
        info = PromptExpandStage.info()
        self.assertEqual(info.name, "prompt_expand")
        self.assertEqual(info.model_requirements, ["llm"])
        self.assertEqual(info.memory_estimate_gb, 5.6)

    def test_build_user_message(self):
        msg = PromptExpandStage._build_user_message(
            "深夜赶路遇白衣", "画皮", 8, "水墨悬疑",
            "Chinese ink-wash dark fantasy, cinematic, 8k",
        )
        self.assertIn("深夜赶路遇白衣", msg)
        self.assertIn("画皮", msg)
        self.assertIn("8", msg)


class TestImageGenerateStageInfo(unittest.TestCase):

    def test_info(self):
        info = ImageGenerateStage.info()
        self.assertEqual(info.name, "image_generate")
        self.assertEqual(info.model_requirements, ["flux"])
        self.assertEqual(info.memory_estimate_gb, 7.0)

    def test_build_prompt_with_style(self):
        result = ImageGenerateStage._build_prompt(
            "dark temple", "ink-wash style, 8k",
        )
        self.assertEqual(result, "dark temple, ink-wash style, 8k")

    def test_build_prompt_no_style(self):
        result = ImageGenerateStage._build_prompt("dark temple", "")
        self.assertEqual(result, "dark temple")

    def test_build_prompt_empty_raises(self):
        with self.assertRaises(ValueError):
            ImageGenerateStage._build_prompt("", "")


class TestTTSSynthesizeStageInfo(unittest.TestCase):

    def test_info(self):
        info = TTSSynthesizeStage.info()
        self.assertEqual(info.name, "tts_synthesize")
        self.assertEqual(info.model_requirements, ["tts"])
        self.assertEqual(info.memory_estimate_gb, 2.9)


class TestKenBurnsStageProcess(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_process_skips_completed(self, mock_ffmpeg):
        stage = KenBurnsStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["ken_burns"]
        stage.process(ctx, MagicMock())
        mock_ffmpeg.run_ffmpeg.assert_not_called()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_process_skips_existing_clip(self, mock_ffmpeg):
        tmpdir = tempfile.mkdtemp()
        stage = KenBurnsStage()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1}]

        img_path = os.path.join(tmpdir, "scene_001_image.png")
        with open(img_path, "wb") as f:
            f.write(b"fake_img")

        clip_path = os.path.join(tmpdir, "scene_001_clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake_clip")

        ctx.artifacts = {"1_image": img_path, "1_clip": clip_path}
        stage.process(ctx, MagicMock())
        mock_ffmpeg.run_ffmpeg.assert_not_called()


class TestPromptExpandParseJson(unittest.TestCase):

    def test_plain_json(self):
        result = PromptExpandStage._parse_json('{"scenes": []}')
        self.assertEqual(result, {"scenes": []})

    def test_json_in_code_block(self):
        result = PromptExpandStage._parse_json('```json\n{"scenes": []}\n```')
        self.assertEqual(result, {"scenes": []})

    def test_bare_list(self):
        result = PromptExpandStage._parse_json('[1, 2, 3]')
        self.assertEqual(result, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
