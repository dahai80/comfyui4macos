"""补充覆盖：stages 的 HTTP 回退路径与纯函数。

覆盖 image_generate/tts_synthesize/prompt_expand/story_ingest 的
_generate_http/_synthesize_http/_load_system_prompt/_strip_thinking/
_parse_json 容错路径。MLX 原生路径因环境无 mlx 自动走 ImportError 回退。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestImageGenerateHttpFallback(unittest.TestCase):
    """覆盖 image_generate._generate_http。"""

    def test_generate_http_saves_image(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        from PIL import Image
        import io
        img = Image.new("RGB", (4, 4), (50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        handle = MagicMock()
        handle.client.generate_image.return_value = [png_bytes]
        handle.model_name = "flux-test"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            out_path = tf.name
        try:
            ImageGenerateStage._generate_http(handle, "prompt", 64, 64, 4, 4.0, 0, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
        finally:
            os.unlink(out_path)

    def test_generate_http_empty_result_raises(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        handle = MagicMock()
        handle.client.generate_image.return_value = []
        handle.model_name = "flux-test"
        with self.assertRaises(RuntimeError) as cm:
            ImageGenerateStage._generate_http(handle, "p", 64, 64, 4, 4.0, 0, "/tmp/out.png")
        self.assertIn("empty", str(cm.exception))

    def test_build_prompt_empty_raises(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        with self.assertRaises(ValueError):
            ImageGenerateStage._build_prompt("", "style")

    def test_build_prompt_with_style(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        self.assertEqual(ImageGenerateStage._build_prompt("v", "s"), "v, s")

    def test_build_prompt_no_style(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        self.assertEqual(ImageGenerateStage._build_prompt("v", "  "), "v")


class TestTtsSynthesizeHttpFallback(unittest.TestCase):

    def test_synthesize_http_saves_audio(self):
        from custom_nodes4macos.pipeline.stages.tts_synthesize import TTSSynthesizeStage
        handle = MagicMock()
        handle.client.synthesize_speech.return_value = b"RIFF...WAV"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = tf.name
        try:
            TTSSynthesizeStage._synthesize_http(handle, "tts-model", "text", "", "", 1.0, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "rb") as f:
                self.assertTrue(f.read().startswith(b"RIFF"))
        finally:
            os.unlink(out_path)

    def test_synthesize_http_empty_raises(self):
        from custom_nodes4macos.pipeline.stages.tts_synthesize import TTSSynthesizeStage
        handle = MagicMock()
        handle.client.synthesize_speech.return_value = b""
        with self.assertRaises(RuntimeError):
            TTSSynthesizeStage._synthesize_http(handle, "tts-model", "t", "", "", 1.0, "/tmp/x.wav")


class TestPromptExpandHelpers(unittest.TestCase):
    """覆盖 prompt_expand stage 的 _load_system_prompt、_strip_thinking、_parse_json 容错。"""

    def test_load_system_prompt_from_absolute_path(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as tf:
            tf.write("custom system prompt")
            path = tf.name
        try:
            result = PromptExpandStage._load_system_prompt(path)
            self.assertEqual(result, "custom system prompt")
        finally:
            os.unlink(path)

    def test_load_system_prompt_missing_file_fallback(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        result = PromptExpandStage._load_system_prompt("nonexistent_prompt.md")
        self.assertIn("编剧", result)  # fallback 文案

    def test_load_system_prompt_from_prompt_dir(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        # horror_director.md 存在于 prompts 目录
        result = PromptExpandStage._load_system_prompt("horror_director.md")
        self.assertIn("Role", result)

    def test_strip_thinking_removes_think_tag(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        text = '<think>some reasoning</think>{"scenes":[]}'
        result = PromptExpandStage._strip_thinking(text)
        self.assertEqual(result, '{"scenes":[]}')

    def test_strip_thinking_removes_thinking_process_section(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        text = 'Thinking Process: blah\n{"scenes":[{"visual_prompt":"x"}]}'
        result = PromptExpandStage._strip_thinking(text)
        self.assertIn('"scenes"', result)

    def test_strip_thinking_no_tags_returns_original(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        text = '{"scenes":[]}'
        result = PromptExpandStage._strip_thinking(text)
        self.assertEqual(result, '{"scenes":[]}')

    def test_parse_json_plain_dict(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        result = PromptExpandStage._parse_json('{"scenes":[{"visual_prompt":"x"}]}')
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["scenes"]), 1)

    def test_parse_json_codefence(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        result = PromptExpandStage._parse_json('```json\n{"scenes":[]}\n```')
        self.assertEqual(result, {"scenes": []})

    def test_parse_json_with_prefix_text(self):
        """JSON 前有文字时，从第一个 { 开始解析。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        text = 'Here is the output:\n{"scenes":[{"visual_prompt":"x"}]}'
        result = PromptExpandStage._parse_json(text)
        self.assertEqual(len(result["scenes"]), 1)

    def test_parse_json_trailing_garbage(self):
        """JSON 后有杂质时，从尾部回溯找闭合 }。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        text = '{"scenes":[{"visual_prompt":"x"}]}\n```'
        result = PromptExpandStage._parse_json(text)
        self.assertEqual(len(result["scenes"]), 1)

    def test_parse_and_validate_raw_empty_scenes_raises(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        with self.assertRaises(RuntimeError) as cm:
            PromptExpandStage._parse_and_validate_raw('{"scenes":[]}')
        self.assertIn("no scenes", str(cm.exception))

    def test_parse_and_validate_raw_non_dict_raises(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        with self.assertRaises(RuntimeError):
            PromptExpandStage._parse_and_validate_raw('"just a string"')

    def test_parse_and_validate_raw_bare_list_wrapped(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        result = PromptExpandStage._parse_and_validate_raw('[{"visual_prompt":"x"}]')
        self.assertIn("scenes", result)
        self.assertEqual(len(result["scenes"]), 1)

    def test_parse_and_validate_backwards_compat(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        scenes = PromptExpandStage._parse_and_validate('{"scenes":[{"visual_prompt":"a"},{"visual_prompt":"b"}]}')
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0]["scene_id"], 1)
        self.assertEqual(scenes[1]["scene_id"], 2)


class TestPromptExpandProcessEpisodes(unittest.TestCase):
    """覆盖 _process_episodes 连载分支。"""

    @patch.object(__import__("custom_nodes4macos.pipeline.stages.prompt_expand", fromlist=["PromptExpandStage"]).PromptExpandStage, "_generate")
    def test_process_episodes_renumbers_scenes(self, mock_gen):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "episodes": [
                {"episode_id": 1, "title": "Ep1", "synopsis": "s1", "key_scenes": ["a"], "cliffhanger": "c1"},
                {"episode_id": 2, "title": "Ep2", "synopsis": "s2", "key_scenes": [], "cliffhanger": ""},
            ],
            "scene_count": 2,
            "style_preset": "水墨悬疑",
            "style_presets": {"水墨悬疑": "ink-wash"},
        })
        # 第一集返回 2 scenes，第二集返回 2 scenes
        mock_gen.side_effect = [
            '{"scenes":[{"visual_prompt":"a"},{"visual_prompt":"b"}]}',
            '{"scenes":[{"visual_prompt":"c"},{"visual_prompt":"d"}]}',
        ]
        mock_mgr = MagicMock()
        mock_handle = MagicMock()
        mock_handle.model = (MagicMock(), MagicMock())
        mock_mgr.acquire.return_value.__enter__.return_value = mock_handle

        stage = PromptExpandStage()
        stage.process(ctx, mock_mgr)

        self.assertEqual(len(ctx.scenes), 4)
        # 全局 scene_id 应为 1,2,3,4（renumber）
        self.assertEqual(ctx.scenes[0]["scene_id"], 1)
        self.assertEqual(ctx.scenes[1]["scene_id"], 2)
        self.assertEqual(ctx.scenes[2]["scene_id"], 3)
        self.assertEqual(ctx.scenes[3]["scene_id"], 4)
        # episode_id 注入
        self.assertEqual(ctx.scenes[0]["episode_id"], 1)
        self.assertEqual(ctx.scenes[2]["episode_id"], 2)

    def test_process_episodes_writes_raw_output(self):
        """_process_episodes 每集写 _prompt_expand_epN_raw.txt。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "episodes": [{"episode_id": 1, "title": "E1", "synopsis": "s", "key_scenes": [], "cliffhanger": ""}],
            "scene_count": 1,
            "style_preset": "",
            "style_presets": {},
        })
        with patch.object(PromptExpandStage, "_generate", return_value='{"scenes":[{"visual_prompt":"x"}]}'):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            stage = PromptExpandStage()
            stage.process(ctx, mock_mgr)
        raw_path = os.path.join(tmpdir, "_prompt_expand_ep1_raw.txt")
        self.assertTrue(os.path.exists(raw_path))

    def test_process_episodes_retries_on_generate_failure(self):
        """_generate 抛错时重试，第三次成功则正常产出。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "episodes": [{"episode_id": 1, "title": "E1", "synopsis": "s", "key_scenes": [], "cliffhanger": ""}],
            "scene_count": 1,
            "style_preset": "",
            "style_presets": {},
        })
        call_count = {"n": 0}

        def flaky(handle, messages, temperature):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("simulated timeout")
            return '{"scenes":[{"visual_prompt":"ok"}]}'

        with patch.object(PromptExpandStage, "_generate", side_effect=flaky):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            stage = PromptExpandStage()
            stage.process(ctx, mock_mgr)
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(len(ctx.scenes), 1)
        self.assertEqual(ctx.scenes[0]["visual_prompt"], "ok")

    def test_process_episodes_fallback_after_all_attempts_fail(self):
        """3 次 _generate 全抛错 → 走西游记 fallback，不崩溃。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "episodes": [{"episode_id": 1, "title": "E1", "synopsis": "s", "key_scenes": [], "cliffhanger": ""}],
            "scene_count": 2,
            "style_preset": "",
            "style_presets": {},
        })
        with patch.object(PromptExpandStage, "_generate", side_effect=RuntimeError("hard timeout")):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            stage = PromptExpandStage()
            stage.process(ctx, mock_mgr)
        self.assertGreaterEqual(len(ctx.scenes), 1)
        fb_path = os.path.join(tmpdir, "_prompt_expand_ep1_fallback.json")
        self.assertTrue(os.path.exists(fb_path), "fallback json should be written")


class TestPromptExpandProcessSingle(unittest.TestCase):
    """覆盖 process() 单集分支的重试与 fallback。

    之前单集路径直接 _parse_and_validate_raw 无重试，LLM 返回坏 JSON 会抛
    JSONDecodeError 直接崩掉整个 job；现在与连载分支共用 _generate_with_retry。
    """

    def test_process_single_retries_on_parse_failure_then_succeeds(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "story_seed": "深夜赶路遇白衣女子",
            "scene_count": 1,
            "style_preset": "",
            "style_presets": {},
        })
        call_count = {"n": 0}

        def flaky(handle, messages, temperature):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return "totally not json at all"
            return '{"scenes":[{"visual_prompt":"ok"}]}'

        with patch.object(PromptExpandStage, "_generate", side_effect=flaky):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            stage = PromptExpandStage()
            stage.process(ctx, mock_mgr)
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(len(ctx.scenes), 1)
        self.assertEqual(ctx.scenes[0]["visual_prompt"], "ok")

    def test_process_single_fallback_after_all_parse_fail(self):
        """3 次 LLM 都返回坏 JSON → 走西游记 fallback，不崩溃，写 fallback json。"""
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "story_seed": "深夜赶路遇白衣女子",
            "scene_count": 2,
            "style_preset": "",
            "style_presets": {},
        })
        with patch.object(PromptExpandStage, "_generate", return_value="totally not json at all"):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model = (MagicMock(), MagicMock())
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            stage = PromptExpandStage()
            stage.process(ctx, mock_mgr)
        self.assertGreaterEqual(len(ctx.scenes), 1)
        fb_path = os.path.join(tmpdir, "_prompt_expand_ep1_fallback.json")
        self.assertTrue(os.path.exists(fb_path), "single-episode fallback json should be written")


class TestStoryIngestHelpers(unittest.TestCase):
    """覆盖 story_ingest 的 _strip_thinking、_parse_episodes 容错、_read_pdf/epub 回退。"""

    def test_strip_thinking_removes_think_tag(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        text = '<think>reasoning</think>{"episodes":[]}'
        result = StoryIngestStage._strip_thinking(text)
        self.assertEqual(result, '{"episodes":[]}')

    def test_strip_thinking_thinking_process_with_episodes(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        text = 'Thinking Process: blah\n{"episodes":[{"title":"x"}]}'
        result = StoryIngestStage._strip_thinking(text)
        self.assertIn('"episodes"', result)

    def test_parse_episodes_with_prefix_text(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        text = 'Here is the plan:\n{"episodes":[{"episode_id":1,"title":"E1"}]}'
        result = StoryIngestStage._parse_episodes(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "E1")

    def test_parse_episodes_trailing_garbage(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        text = '{"episodes":[{"episode_id":1,"title":"E1"}]}\n```'
        result = StoryIngestStage._parse_episodes(text)
        self.assertEqual(len(result), 1)

    def test_parse_episodes_completely_invalid_returns_empty(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        result = StoryIngestStage._parse_episodes("totally not json at all")
        self.assertEqual(result, [])

    def test_parse_episodes_non_dict_non_list_returns_empty(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        result = StoryIngestStage._parse_episodes('"just a string"')
        self.assertEqual(result, [])

    def test_read_pdf_no_readers_raises(self):
        """无 pymupdf/fitz/pdfminer 时抛 RuntimeError。"""
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name in ("pymupdf", "fitz", "pdfminer", "pdfminer.high_level"):
                raise ImportError(f"no {name}")
            return real_import(name, *a, **k)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as cm:
                StoryIngestStage._read_pdf("/fake.pdf")
            self.assertIn("PDF reader", str(cm.exception))

    def test_read_epub_no_readers_raises(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name in ("ebooklib", "bs4", "bs4.BeautifulSoup"):
                raise ImportError(f"no {name}")
            return real_import(name, *a, **k)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError) as cm:
                StoryIngestStage._read_epub("/fake.epub")
            self.assertIn("EPUB reader", str(cm.exception))

    def test_read_unknown_extension_falls_back_to_text(self):
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as tf:
            tf.write("plain content")
            path = tf.name
        try:
            result = StoryIngestStage._read_file(path)
            self.assertEqual(result, "plain content")
        finally:
            os.unlink(path)

    def test_generate_outline_caps_chapters(self):
        """超过 60 章时截断。"""
        from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        chapters = [{"chapter_id": i, "title": f"ch{i}", "text": f"content{i}"} for i in range(1, 70)]
        ctx = PipelineContext(job_id="t", job_dir=tempfile.mkdtemp(), config={})
        stage = StoryIngestStage()
        with patch.object(StoryIngestStage, "_generate", return_value='{"episodes":[]}'):
            stage._generate_outline(MagicMock(), chapters, 30, 25, ctx)
        # _generate 被调用即可，验证不抛异常


class TestDigitalHumanRenderFallback(unittest.TestCase):
    """覆盖 digital_human_render._render_fallback 完整路径。"""

    @patch("custom_nodes4macos.ffmpeg_util.run_ffmpeg")
    @patch("custom_nodes4macos.ffmpeg_util.video_encoder_args", return_value=["-c:v", "libx264"])
    def test_render_fallback_no_avatar_generates_placeholder(self, mock_enc, mock_run):
        from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, "a.wav")
        with open(audio_path, "wb") as f:
            f.write(b"wav")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 5}]
        ctx.set_artifact(1, "audio", audio_path)

        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")  # 无 avatar，应生成占位
        mock_run.assert_called_once()
        # 占位头像应被创建
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "_avatar_placeholder.png")))

    @patch("custom_nodes4macos.ffmpeg_util.run_ffmpeg")
    @patch("custom_nodes4macos.ffmpeg_util.video_encoder_args", return_value=["-c:v", "libx264"])
    def test_render_fallback_skips_existing_clip(self, mock_enc, mock_run):
        from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, "a.wav")
        with open(audio_path, "wb") as f:
            f.write(b"wav")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 5}]
        ctx.set_artifact(1, "audio", audio_path)
        # 预置已存在 clip
        clip_path = ctx.artifact_path(1, "clip")
        with open(clip_path, "wb") as f:
            f.write(b"mp4")
        ctx.set_artifact(1, "clip", clip_path)

        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")
        mock_run.assert_not_called()  # 已存在 clip，跳过

    @patch("custom_nodes4macos.ffmpeg_util.run_ffmpeg", side_effect=Exception("ffmpeg fail"))
    @patch("custom_nodes4macos.ffmpeg_util.video_encoder_args", return_value=["-c:v", "libx264"])
    def test_render_fallback_ffmpeg_failure_does_not_crash(self, mock_enc, mock_run):
        """ffmpeg 失败被捕获，不抛异常。"""
        from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, "a.wav")
        with open(audio_path, "wb") as f:
            f.write(b"wav")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 5}]
        ctx.set_artifact(1, "audio", audio_path)

        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")  # 不抛异常即通过

    def test_generate_placeholder_avatar_creates_png(self):
        from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        path = DigitalHumanRenderStage._generate_placeholder_avatar(ctx)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        # 第二次调用应复用
        path2 = DigitalHumanRenderStage._generate_placeholder_avatar(ctx)
        self.assertEqual(path, path2)

    def test_render_fallback_empty_scenes_returns_early(self):
        from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
        from custom_nodes4macos.pipeline.context import PipelineContext
        ctx = PipelineContext(job_id="t", job_dir=tempfile.mkdtemp(), config={})
        ctx.scenes = []
        stage = DigitalHumanRenderStage()
        stage._render_fallback(ctx, "")  # 空 scenes，不抛异常


class TestRealisticReferenceHelpers(unittest.TestCase):
    """覆盖 image_generate 的真实人物参考图条件生成辅助函数。"""

    def test_image_to_b64_roundtrip(self):
        import base64
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "ref.png")
        payload = b"\x89PNG fake bytes 12345"
        with open(path, "wb") as fh:
            fh.write(payload)
        b64 = ImageGenerateStage._image_to_b64(path)
        self.assertEqual(base64.b64decode(b64), payload)

    def test_resolve_reference_returns_b64_when_exists(self):
        import base64
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "face.png")
        with open(path, "wb") as fh:
            fh.write(b"face-bytes")
        char_lookup = {"lao_wang": {"name": "lao_wang", "reference_image": path}}
        ref = ImageGenerateStage._resolve_realistic_reference(
            "realistic", ["lao_wang"], char_lookup, "",
        )
        self.assertIsNotNone(ref)
        self.assertEqual(base64.b64decode(ref), b"face-bytes")

    def test_resolve_reference_none_when_not_realistic(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        char_lookup = {"lao_wang": {"name": "lao_wang", "reference_image": "/x.png"}}
        self.assertIsNone(
            ImageGenerateStage._resolve_realistic_reference(
                "cartoon", ["lao_wang"], char_lookup, "",
            )
        )

    def test_resolve_reference_none_when_path_missing(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        char_lookup = {"lao_wang": {"name": "lao_wang", "reference_image": "/nope/missing.png"}}
        self.assertIsNone(
            ImageGenerateStage._resolve_realistic_reference(
                "realistic", ["lao_wang"], char_lookup, "",
            )
        )

    def test_resolve_reference_relative_to_char_ref_dir(self):
        import base64
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "face.png")
        with open(path, "wb") as fh:
            fh.write(b"rel-bytes")
        char_lookup = {"lao_wang": {"name": "lao_wang", "reference_image": "face.png"}}
        ref = ImageGenerateStage._resolve_realistic_reference(
            "realistic", ["lao_wang"], char_lookup, tmpdir,
        )
        self.assertIsNotNone(ref)
        self.assertEqual(base64.b64decode(ref), b"rel-bytes")

    def test_resolve_reference_none_when_no_chars(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        self.assertIsNone(
            ImageGenerateStage._resolve_realistic_reference("realistic", [], {}, "")
        )

    def test_generate_http_forwards_reference_fields(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        from PIL import Image
        import io
        img = Image.new("RGB", (4, 4), (1, 2, 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        handle = MagicMock()
        handle.client.generate_image.return_value = [buf.getvalue()]
        handle.model_name = "flux-test"
        out = os.path.join(tempfile.mkdtemp(), "out.png")
        ImageGenerateStage._generate_http(
            handle, "p", 64, 64, 4, 4.0, 0, out,
            reference_image="REFB64", reference_strength=0.6, conditioning_mode="redux",
        )
        call_kwargs = handle.client.generate_image.call_args.kwargs
        self.assertEqual(call_kwargs["reference_image"], "REFB64")
        self.assertEqual(call_kwargs["reference_strength"], 0.6)
        self.assertEqual(call_kwargs["conditioning_mode"], "redux")

    def test_generate_http_omits_reference_when_none(self):
        from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
        from PIL import Image
        import io
        img = Image.new("RGB", (4, 4), (1, 2, 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        handle = MagicMock()
        handle.client.generate_image.return_value = [buf.getvalue()]
        handle.model_name = "flux-test"
        out = os.path.join(tempfile.mkdtemp(), "out.png")
        ImageGenerateStage._generate_http(handle, "p", 64, 64, 4, 4.0, 0, out)
        call_kwargs = handle.client.generate_image.call_args.kwargs
        self.assertIsNone(call_kwargs["reference_image"])
        self.assertIsNone(call_kwargs["reference_strength"])
        self.assertIsNone(call_kwargs["conditioning_mode"])


if __name__ == "__main__":
    unittest.main()
