from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage


class TestStoryIngestStageInfo(unittest.TestCase):

    def test_info(self):
        info = StoryIngestStage.info()
        self.assertEqual(info.name, "story_ingest")
        self.assertEqual(info.model_requirements, ["llm"])
        self.assertEqual(info.input_kinds, ["text"])
        self.assertEqual(info.output_kinds, ["episodes"])

    def test_skip_if_completed(self):
        stage = StoryIngestStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["story_ingest"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestStoryIngestReadTxt(unittest.TestCase):

    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world\nSecond line\n")
            f.flush()
            path = f.name
        try:
            text = StoryIngestStage._read_file(path)
            self.assertIn("Hello world", text)
            self.assertIn("Second line", text)
        finally:
            os.unlink(path)


class TestStoryIngestReadPdf(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.story_ingest.StoryIngestStage._read_pdf")
    def test_read_pdf_dispatches(self, mock_pdf):
        mock_pdf.return_value = "pdf content"
        text = StoryIngestStage._read_file("/some/book.pdf")
        mock_pdf.assert_called_once_with("/some/book.pdf")
        self.assertEqual(text, "pdf content")


class TestStoryIngestSplitChapters(unittest.TestCase):

    def test_chinese_chapters(self):
        text = "前言内容\n第一章 开端\n第一段\n第二章 发展\n第二段"
        chapters = StoryIngestStage._split_chapters(text)
        self.assertGreaterEqual(len(chapters), 2)
        self.assertIn("开端", chapters[0]["title"])

    def test_english_chapters(self):
        text = "Intro\nChapter 1: The Beginning\nSome text\nChapter 2: The Middle\nMore text"
        chapters = StoryIngestStage._split_chapters(text)
        self.assertGreaterEqual(len(chapters), 2)

    def test_no_chapters_fallback(self):
        text = "A" * 20000
        chapters = StoryIngestStage._split_chapters(text)
        self.assertGreaterEqual(len(chapters), 2)
        self.assertTrue(all("chapter_id" in ch for ch in chapters))

    def test_short_text_single_chapter(self):
        text = "Short text"
        chapters = StoryIngestStage._split_chapters(text)
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["text"], "Short text")


class TestStoryIngestParseEpisodes(unittest.TestCase):

    def test_parse_json_dict(self):
        raw = '{"episodes": [{"title": "Ep1", "scenes": [], "episode_id": 1}]}'
        result = StoryIngestStage._parse_episodes(raw)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Ep1")

    def test_parse_json_list(self):
        raw = '[{"title": "Ep1", "scenes": [], "episode_id": 1}]'
        result = StoryIngestStage._parse_episodes(raw)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_parse_json_in_code_block(self):
        raw = '```json\n{"episodes": []}\n```'
        result = StoryIngestStage._parse_episodes(raw)
        self.assertIsInstance(result, list)

    def test_parse_invalid_returns_empty(self):
        raw = "not json at all"
        result = StoryIngestStage._parse_episodes(raw)
        self.assertEqual(result, [])

    def test_parse_adds_episode_id(self):
        raw = '[{"title": "Ep1"}, {"title": "Ep2"}]'
        result = StoryIngestStage._parse_episodes(raw)
        self.assertEqual(result[0]["episode_id"], 1)
        self.assertEqual(result[1]["episode_id"], 2)


class TestStoryIngestProcess(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.story_ingest.StoryIngestStage._generate_outline")
    def test_process_creates_scenes(self, mock_outline):
        mock_outline.return_value = json.dumps({
            "episodes": [
                {
                    "episode_id": 1,
                    "title": "Episode 1",
                    "chapters": "第1-2章",
                    "synopsis": "beginning",
                    "key_scenes": ["road", "temple"],
                    "cliffhanger": "something lurks",
                },
            ],
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Chapter 1: Start\nSome content\nChapter 2: End\nMore content")
            f.flush()
            story_path = f.name

        try:
            tmpdir = tempfile.mkdtemp()
            ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
                "story_file": story_path,
                "episode_count": 1,
            })
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            mock_mgr.acquire.return_value = mock_handle

            stage = StoryIngestStage()
            stage.process(ctx, mock_mgr)

            self.assertGreater(len(ctx.scenes), 0)
        finally:
            os.unlink(story_path)

    @patch("custom_nodes4macos.pipeline.stages.story_ingest.StoryIngestStage._generate_outline")
    def test_process_with_story_seed(self, mock_outline):
        mock_outline.return_value = json.dumps({
            "episodes": [
                {"episode_id": 1, "title": "Ep1", "chapters": "1", "synopsis": "test", "key_scenes": [], "cliffhanger": "end"},
            ],
        })

        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "story_seed": "A long story text that provides the seed content",
            "episode_count": 1,
        })
        mock_mgr = MagicMock()
        mock_handle = MagicMock()
        mock_handle.model = (MagicMock(), MagicMock())
        mock_handle.__enter__ = MagicMock(return_value=mock_handle)
        mock_handle.__exit__ = MagicMock(return_value=False)
        mock_mgr.acquire.return_value = mock_handle

        stage = StoryIngestStage()
        stage.process(ctx, mock_mgr)
        self.assertGreater(len(ctx.scenes), 0)

    def test_process_raises_without_story(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        stage = StoryIngestStage()
        with self.assertRaises(ValueError):
            stage.process(ctx, MagicMock())


class TestDigitalHumanRenderStageInfo(unittest.TestCase):

    def test_info(self):
        info = DigitalHumanRenderStage.info()
        self.assertEqual(info.name, "digital_human_render")
        self.assertEqual(info.model_requirements, [])
        self.assertEqual(info.output_kinds, ["clip"])

    def test_skip_if_completed(self):
        stage = DigitalHumanRenderStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["digital_human_render"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestDigitalHumanRenderProcess(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.digital_human_render.DigitalHumanRenderStage._render_fallback")
    def test_process_uses_fallback_when_no_lip_sync(self, mock_fallback):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = [
            {"scene_id": 1, "audio_script": "test narration", "emotion": "neutral"},
        ]

        stage = DigitalHumanRenderStage()
        stage.process(ctx, MagicMock())
        mock_fallback.assert_called_once()

    def test_process_raises_with_lip_sync_model(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "lip_sync_model": "wav2lip",
        })
        ctx.scenes = [{"scene_id": 1}]

        stage = DigitalHumanRenderStage()
        with self.assertRaises(NotImplementedError):
            stage.process(ctx, MagicMock())

    @patch("custom_nodes4macos.ffmpeg_util.run_ffmpeg")
    @patch("custom_nodes4macos.pipeline.stages.digital_human_render.DigitalHumanRenderStage._generate_placeholder_avatar")
    def test_render_fallback_skips_no_audio(self, mock_avatar, mock_ffmpeg):
        mock_avatar.return_value = "/tmp/avatar.png"
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1}]

        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")
        mock_ffmpeg.assert_not_called()

    @patch("custom_nodes4macos.ffmpeg_util.run_ffmpeg")
    @patch("custom_nodes4macos.pipeline.stages.digital_human_render.DigitalHumanRenderStage._generate_placeholder_avatar")
    def test_render_fallback_with_audio(self, mock_avatar, mock_ffmpeg):
        mock_avatar.return_value = "/tmp/avatar.png"
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, "scene_001_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(b"fake_audio")

        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 10}]
        ctx.artifacts = {"1_audio": audio_path}

        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")
        mock_ffmpeg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
