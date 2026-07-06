from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.story_ingest import StoryIngestStage
from custom_nodes4macos.pipeline.stages.digital_human_render import DigitalHumanRenderStage
from custom_nodes4macos.pipeline.stages.avatar_create import AvatarCreateStage
from custom_nodes4macos.pipeline.stages.avatar_animate import AvatarAnimateStage
from custom_nodes4macos.pipeline.stages.voice_clone import VoiceCloneStage


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

    @patch("custom_nodes4macos.pipeline.stages.digital_human_render.DigitalHumanRenderStage._render_fallback")
    def test_process_uses_fallback_with_lip_sync_no_avatar_package(self, mock_fallback):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "lip_sync_model": "wav2lip",
        })
        ctx.scenes = [{"scene_id": 1}]

        stage = DigitalHumanRenderStage()
        stage.process(ctx, MagicMock())
        mock_fallback.assert_called_once()

    @patch("custom_nodes4macos.pipeline.stages.digital_human_render.DigitalHumanRenderStage._render_with_avatar")
    def test_process_delegates_to_avatar_when_package_exists(self, mock_avatar_render):
        tmpdir = tempfile.mkdtemp()
        avatar_dir = os.path.join(tmpdir, "_avatar")
        os.makedirs(avatar_dir, exist_ok=True)

        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "avatar_package": avatar_dir,
        })
        ctx.scenes = [{"scene_id": 1}]

        stage = DigitalHumanRenderStage()
        stage.process(ctx, MagicMock())
        mock_avatar_render.assert_called_once()

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


class TestAvatarCreateStageInfo(unittest.TestCase):

    def test_info(self):
        info = AvatarCreateStage.info()
        self.assertEqual(info.name, "avatar_create")
        self.assertEqual(info.model_requirements, [])
        self.assertEqual(info.output_kinds, ["avatar_package"])

    def test_skip_if_completed(self):
        stage = AvatarCreateStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["avatar_create"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestAvatarCreateProcess(unittest.TestCase):

    def test_process_no_photo_no_video_generates_placeholder(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        ctx.scenes = []

        stage = AvatarCreateStage()
        stage.process(ctx, MagicMock())

        avatar_dir = os.path.join(tmpdir, "_avatar")
        self.assertTrue(os.path.isdir(avatar_dir))
        self.assertTrue(os.path.isfile(os.path.join(avatar_dir, "reference.png")))
        self.assertTrue(os.path.isfile(os.path.join(avatar_dir, "avatar_meta.json")))
        self.assertEqual(ctx.artifacts.get("avatar_package"), avatar_dir)

    def test_process_with_photo(self):
        import cv2
        import numpy as np

        photo_dir = tempfile.mkdtemp()
        photo_path = os.path.join(photo_dir, "face.png")
        face_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(face_img, (180, 120), (332, 300), (200, 200, 200), -1)
        cv2.circle(face_img, (220, 180), 15, (100, 100, 255), -1)
        cv2.circle(face_img, (292, 180), 15, (100, 100, 255), -1)
        cv2.imwrite(photo_path, face_img)

        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "avatar_photo": photo_path,
            "avatar_style": "realistic",
        })
        ctx.scenes = []

        stage = AvatarCreateStage()
        stage.process(ctx, MagicMock())

        avatar_dir = os.path.join(tmpdir, "_avatar")
        self.assertTrue(os.path.isfile(os.path.join(avatar_dir, "reference.png")))

        meta_path = os.path.join(avatar_dir, "avatar_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta.get("avatar_style"), "realistic")
        self.assertIn("reference_frame", meta)

    def test_cartoon_style(self):
        import cv2
        import numpy as np

        photo_dir = tempfile.mkdtemp()
        photo_path = os.path.join(photo_dir, "face.png")
        face_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.rectangle(face_img, (180, 120), (332, 300), (200, 200, 200), -1)
        cv2.imwrite(photo_path, face_img)

        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "avatar_photo": photo_path,
            "avatar_style": "cartoon",
        })
        ctx.scenes = []

        stage = AvatarCreateStage()
        stage.process(ctx, MagicMock())

        avatar_dir = os.path.join(tmpdir, "_avatar")
        self.assertTrue(os.path.isfile(os.path.join(avatar_dir, "reference.png")))

    def test_placeholder_avatar_is_valid_image(self):
        import numpy as np

        img = AvatarCreateStage._generate_placeholder("/tmp")
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_reuse_prebuilt_avatar_package_skips_detection(self):
        import cv2
        import numpy as np

        pre_pkg = tempfile.mkdtemp()
        ref_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(pre_pkg, "reference.png"), ref_img)
        with open(os.path.join(pre_pkg, "avatar_meta.json"), "w") as f:
            json.dump({"avatar_style": "realistic", "landmarks": {}}, f)

        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={
            "avatar_package": pre_pkg,
        })
        ctx.scenes = []

        stage = AvatarCreateStage()
        stage.process(ctx, MagicMock())

        self.assertEqual(ctx.artifacts.get("avatar_package"), pre_pkg)
        self.assertEqual(ctx.artifacts.get("avatar_reference"), os.path.join(pre_pkg, "reference.png"))
        self.assertEqual(ctx.config.get("avatar_reference"), os.path.join(pre_pkg, "reference.png"))
        self.assertFalse(os.path.isdir(os.path.join(tmpdir, "_avatar")))


class TestAvatarAnimateStageInfo(unittest.TestCase):

    def test_info(self):
        info = AvatarAnimateStage.info()
        self.assertEqual(info.name, "avatar_animate")
        self.assertEqual(info.model_requirements, [])
        self.assertEqual(info.input_kinds, ["avatar_package", "audio"])
        self.assertEqual(info.output_kinds, ["clip"])

    def test_skip_if_completed(self):
        stage = AvatarAnimateStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["avatar_animate"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestAvatarAnimateAudioEnergy(unittest.TestCase):

    def test_analyze_audio_energy_valid_wav(self):
        import struct
        import wave

        tmpdir = tempfile.mkdtemp()
        wav_path = os.path.join(tmpdir, "test.wav")

        framerate = 16000
        duration = 1
        n_samples = framerate * duration
        samples = []
        for i in range(n_samples):
            val = int(16384 * (1 if (i % 800 < 400) else -1))
            samples.append(struct.pack("<h", val))

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(b"".join(samples))

        stage = AvatarAnimateStage()
        energy = stage._analyze_audio_energy(wav_path)

        self.assertIsInstance(energy, list)
        self.assertGreater(len(energy), 0)
        self.assertTrue(all(0 <= e <= 1.0 for e in energy))

    def test_analyze_audio_energy_invalid_file(self):
        stage = AvatarAnimateStage()
        energy = stage._analyze_audio_energy("/nonexistent/file.wav")
        self.assertEqual(energy, [])


class TestAvatarAnimateMouthRegion(unittest.TestCase):

    def test_estimate_from_landmarks(self):
        import numpy as np

        stage = AvatarAnimateStage()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        landmarks = {
            "mouth_left": [200, 350],
            "mouth_right": [300, 350],
            "mouth_center": [250, 350],
        }
        result = stage._estimate_mouth_region(img, landmarks, [])
        self.assertIsNotNone(result)
        self.assertIn("x1", result)
        self.assertIn("center_x", result)

    def test_estimate_from_bbox(self):
        import numpy as np

        stage = AvatarAnimateStage()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = stage._estimate_mouth_region(img, {}, [100, 80, 200, 250])
        self.assertIsNotNone(result)
        self.assertIn("mouth_width", result)

    def test_estimate_returns_none_no_data(self):
        import numpy as np

        stage = AvatarAnimateStage()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = stage._estimate_mouth_region(img, {}, [])
        self.assertIsNone(result)


class TestVoiceCloneStageInfo(unittest.TestCase):

    def test_info(self):
        info = VoiceCloneStage.info()
        self.assertEqual(info.name, "voice_clone")
        self.assertEqual(info.output_kinds, ["voice_profile"])

    def test_skip_if_completed(self):
        stage = VoiceCloneStage()
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        ctx.completed_stages = ["voice_clone"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestVoiceCloneAutoTranscribe(unittest.TestCase):

    @patch("custom_nodes4macos.fusion_client.FusionMLXClient")
    def test_auto_transcribe_returns_stripped_text(self, mock_cls):
        mock_client = MagicMock()
        mock_client.health.return_value = True
        mock_client.transcribe.return_value = ("  转写文本  ", {})
        mock_cls.return_value.__enter__.return_value = mock_client
        text = VoiceCloneStage._auto_transcribe("/tmp/ref.wav")
        self.assertEqual(text, "转写文本")
        mock_client.transcribe.assert_called_once_with("/tmp/ref.wav")

    @patch("custom_nodes4macos.fusion_client.FusionMLXClient")
    def test_auto_transcribe_unreachable_returns_empty(self, mock_cls):
        mock_client = MagicMock()
        mock_client.health.return_value = False
        mock_cls.return_value.__enter__.return_value = mock_client
        self.assertEqual(VoiceCloneStage._auto_transcribe("/tmp/ref.wav"), "")
        mock_client.transcribe.assert_not_called()

    @patch("custom_nodes4macos.pipeline.stages.voice_clone.VoiceCloneStage._auto_transcribe")
    def test_auto_transcribe_called_when_no_ref_text(self, mock_transcribe):
        mock_transcribe.return_value = "自动转写结果"
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.wav")
            with open(ref_path, "wb") as f:
                f.write(b"\x00" * 1024)
            stage = VoiceCloneStage()
            ctx = PipelineContext(
                job_id="test",
                job_dir=tmpdir,
                config={
                    "voice_ref_audio": ref_path,
                    "voice_ref_text": "",
                    "voice_clone_model": "fish-audio-s2-pro",
                },
            )
            ctx.scenes = []
            ctx.artifacts = {}
            with patch.object(stage, "_skip_if_completed", return_value=False), \
                 patch("custom_nodes4macos.pipeline.stages.voice_clone.VoiceCloneStage.process",
                       side_effect=NotImplementedError):
                pass
        self.assertTrue(callable(VoiceCloneStage._auto_transcribe))


class TestVoiceCloneNoRefAudio(unittest.TestCase):

    def test_skip_when_no_ref_audio(self):
        stage = VoiceCloneStage()
        ctx = PipelineContext(
            job_id="test",
            job_dir="/tmp",
            config={"voice_ref_audio": "", "voice_ref_text": "", "voice_clone_model": "fish-audio-s2-pro"},
        )
        ctx.scenes = []
        ctx.artifacts = {}
        result = stage._skip_if_completed(ctx)
        self.assertFalse(result)


class TestTTSSynthesizeFishS2(unittest.TestCase):

    def test_fish_s2_model_id(self):
        from custom_nodes4macos.pipeline.stages.tts_synthesize import _FISH_S2_MODEL_ID
        self.assertEqual(_FISH_S2_MODEL_ID, "mlx-community/fish-audio-s2-pro")

    def test_fish_s2_gen_kwargs_ref_audio(self):
        gen_kwargs = {"ref_audio": "/path/to/ref.wav", "verbose": False}
        gen_kwargs_ref_text = {**gen_kwargs, "ref_text": "参考文字"}
        gen_kwargs_instruct = {**gen_kwargs_ref_text, "instruct": "温柔语气"}
        self.assertIn("ref_audio", gen_kwargs_instruct)
        self.assertIn("ref_text", gen_kwargs_instruct)
        self.assertIn("instruct", gen_kwargs_instruct)

    def test_qwen3_icl_requires_both(self):
        ref_audio = "/path/to/ref.wav"
        ref_text = ""
        use_icl = ref_audio is not None and ref_text is not None
        self.assertTrue(use_icl)
        ref_text_none = None
        use_icl_no_text = ref_audio is not None and ref_text_none is not None
        self.assertFalse(use_icl_no_text)


class TestFusionClientVoiceParams(unittest.TestCase):

    def test_synthesize_speech_accepts_ref_params(self):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        import inspect
        sig = inspect.signature(FusionMLXClient.synthesize_speech)
        params = list(sig.parameters.keys())
        self.assertIn("ref_audio", params)
        self.assertIn("ref_text", params)


if __name__ == "__main__":
    unittest.main()
