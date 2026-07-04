"""跨 stage 数据流验证。

验证 stage A 的输出（scenes / artifacts）被 stage B 正确消费，
覆盖 prompt_expand → image_generate → tts → ken_burns → assemble
的关键衔接契约。

填补 REVIEW_REPORT P1-2：无跨 stage 数据流验证。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
from custom_nodes4macos.pipeline.stages.tts_synthesize import TTSSynthesizeStage
from custom_nodes4macos.pipeline.stages.ken_burns import KenBurnsStage
from custom_nodes4macos.pipeline.stages.assemble import AssembleStage


def _make_ctx(scenes: list[dict], tmpdir: str, config: dict | None = None) -> PipelineContext:
    ctx = PipelineContext(job_id="dataflow", job_dir=tmpdir, config=config or {})
    ctx.scenes = scenes
    return ctx


class TestPromptExpandToImageGenerate(unittest.TestCase):
    """prompt_expand 产出 scenes → image_generate 消费 scene.visual_prompt + global_style。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch.object(ImageGenerateStage, "_generate_image")
    def test_image_generate_consumes_visual_prompt_per_scene(self, mock_gen):
        """每个 scene 的 visual_prompt 被传入 _generate_image，并写入 artifact。"""
        scenes = [
            {"scene_id": 1, "visual_prompt": "temple at night"},
            {"scene_id": 2, "visual_prompt": "white figure"},
        ]
        ctx = _make_ctx(scenes, self._tmpdir, {"global_style": "ink-wash, 8k"})

        def fake_gen(pipeline, prompt, w, h, steps, g, seed, out_path, scheduler="linear"):
            with open(out_path, "wb") as f:
                f.write(b"fake-png")
            return None

        mock_gen.side_effect = fake_gen

        stage = ImageGenerateStage()
        stage.process(ctx, MagicMock())

        # 验证每个 scene 都生成了 image artifact
        self.assertTrue(ctx.has_artifact_on_disk(1, "image"))
        self.assertTrue(ctx.has_artifact_on_disk(2, "image"))
        # 验证 prompt 拼接了 global_style
        called_prompts = [call.args[1] for call in mock_gen.call_args_list]
        self.assertTrue(all("ink-wash, 8k" in p for p in called_prompts))
        self.assertIn("temple at night", called_prompts[0])
        self.assertIn("white figure", called_prompts[1])

    @patch.object(ImageGenerateStage, "_generate_image")
    def test_image_generate_skips_scene_without_visual_prompt(self, mock_gen):
        """无 visual_prompt 的 scene 被跳过，不调用生成。"""
        scenes = [
            {"scene_id": 1, "visual_prompt": "ok"},
            {"scene_id": 2, "audio_script": "no visual"},
        ]
        ctx = _make_ctx(scenes, self._tmpdir, {})
        stage = ImageGenerateStage()
        stage.process(ctx, MagicMock())
        self.assertEqual(mock_gen.call_count, 1)

    @patch.object(ImageGenerateStage, "_generate_image")
    def test_image_generate_skips_existing_artifact(self, mock_gen):
        """已存在的 image artifact 跳过，不重复生成。"""
        scenes = [{"scene_id": 1, "visual_prompt": "v"}]
        ctx = _make_ctx(scenes, self._tmpdir, {})
        # 预置已存在的 artifact
        img_path = ctx.artifact_path(1, "image")
        with open(img_path, "wb") as f:
            f.write(b"existing")
        stage = ImageGenerateStage()
        stage.process(ctx, MagicMock())
        mock_gen.assert_not_called()


class TestImageGenerateToKenBurns(unittest.TestCase):
    """image_generate 产出 image artifact → ken_burns 消费 image_path。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_ken_burns_consumes_image_artifact(self, mock_ffmpeg):
        """ken_burns 从 ctx.get_artifact(scene_id, 'image') 取路径。"""
        img_path = os.path.join(self._tmpdir, "scene_001_image.png")
        with open(img_path, "wb") as f:
            f.write(b"png")
        scenes = [{"scene_id": 1, "duration_seconds": 2}]
        ctx = _make_ctx(scenes, self._tmpdir, {"ken_burns_workers": 1})
        ctx.set_artifact(1, "image", img_path)

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            out = args[-1]
            with open(out, "wb") as f:
                f.write(b"fake-mp4")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = KenBurnsStage()
        stage.process(ctx, MagicMock())

        clip = ctx.get_artifact(1, "clip")
        self.assertIsNotNone(clip)
        # 验证 ffmpeg 被调用时输入图片是 img_path
        first_call_args = mock_ffmpeg.run_ffmpeg.call_args_list[0].args[0]
        self.assertIn(img_path, first_call_args)

    @patch("custom_nodes4macos.ffmpeg_util")
    def test_ken_burns_skips_scene_without_image(self, mock_ffmpeg):
        """无 image artifact 的 scene 被跳过。"""
        scenes = [{"scene_id": 1, "duration_seconds": 2}]
        ctx = _make_ctx(scenes, self._tmpdir, {"ken_burns_workers": 1})
        stage = KenBurnsStage()
        stage.process(ctx, MagicMock())
        mock_ffmpeg.run_ffmpeg.assert_not_called()


class TestTTSToKenBurnsDuration(unittest.TestCase):
    """tts 产出 audio + duration_seconds → ken_burns 使用 duration。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_ken_burns_uses_scene_duration(self, mock_ffmpeg):
        """ken_burns 用 scene['duration_seconds'] 计算总帧数。"""
        img_path = os.path.join(self._tmpdir, "s1.png")
        with open(img_path, "wb") as f:
            f.write(b"png")
        scenes = [{"scene_id": 1, "duration_seconds": 5.0}]
        ctx = _make_ctx(scenes, self._tmpdir, {"ken_burns_workers": 1, "ken_burns_fps": 30})
        ctx.set_artifact(1, "image", img_path)

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"mp4")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = KenBurnsStage()
        stage.process(ctx, MagicMock())

        # zoompan filter 应包含 d=150 (5.0 * 30)
        zoompan_arg = mock_ffmpeg.run_ffmpeg.call_args_list[0].args[0]
        vf_idx = zoompan_arg.index("-vf") + 1
        self.assertIn("d=150", zoompan_arg[vf_idx])


class TestKenBurnsToAssemble(unittest.TestCase):
    """ken_burns 产出 clip artifact → assemble 收集所有 clip 拼接。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_collects_all_clips(self, mock_ffmpeg):
        """assemble 从 ctx 收集所有 scene 的 clip artifact。"""
        clips = []
        for i in range(1, 4):
            p = os.path.join(self._tmpdir, f"scene_{i:03d}_clip.mp4")
            with open(p, "wb") as f:
                f.write(b"mp4")
            clips.append(p)

        scenes = [{"scene_id": i} for i in range(1, 4)]
        ctx = _make_ctx(scenes, self._tmpdir, {})
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.probe_duration.return_value = 1.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            # 最后一个参数是输出路径
            with open(args[-1], "wb") as f:
                f.write(b"final-mp4")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())

        final = ctx.get_artifact(0, "final")
        self.assertIsNotNone(final)
        # ffmpeg 被调用一次，且所有 clip 作为 -i 输入
        call_args = mock_ffmpeg.run_ffmpeg.call_args_list[0].args[0]
        for c in clips:
            self.assertIn(c, call_args)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_skips_missing_clips(self, mock_ffmpeg):
        """assemble 跳过缺失的 clip，只用存在的。"""
        scenes = [{"scene_id": 1}, {"scene_id": 2}, {"scene_id": 3}]
        ctx = _make_ctx(scenes, self._tmpdir, {})
        p1 = os.path.join(self._tmpdir, "c1.mp4")
        p3 = os.path.join(self._tmpdir, "c3.mp4")
        for p in (p1, p3):
            with open(p, "wb") as f:
                f.write(b"mp4")
        ctx.set_artifact(1, "clip", p1)
        ctx.set_artifact(3, "clip", p3)

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.probe_duration.return_value = 1.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"final")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())

        call_args = mock_ffmpeg.run_ffmpeg.call_args_list[0].args[0]
        self.assertIn(p1, call_args)
        self.assertIn(p3, call_args)
        # scene 2 的 clip 不存在，不应出现在输入
        self.assertNotIn(os.path.join(self._tmpdir, "c2.mp4"), call_args)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_no_clips_raises(self, mock_ffmpeg):
        """无任何可用 clip 时 assemble 抛 RuntimeError。"""
        scenes = [{"scene_id": 1}]
        ctx = _make_ctx(scenes, self._tmpdir, {})
        stage = AssembleStage()
        with self.assertRaises(RuntimeError):
            stage.process(ctx, MagicMock())


class TestArtifactPathConvention(unittest.TestCase):
    """验证 artifact key 命名约定在 stage 间一致。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_artifact_key_format(self):
        """key = f'{scene_id}_{kind}'，所有 stage 用相同格式读写。"""
        ctx = _make_ctx([], self._tmpdir, {})
        ctx.set_artifact(5, "image", "/tmp/x.png")
        ctx.set_artifact(5, "audio", "/tmp/x.wav")
        ctx.set_artifact(5, "clip", "/tmp/x.mp4")
        self.assertEqual(ctx.artifacts["5_image"], "/tmp/x.png")
        self.assertEqual(ctx.artifacts["5_audio"], "/tmp/x.wav")
        self.assertEqual(ctx.artifacts["5_clip"], "/tmp/x.mp4")
        self.assertEqual(ctx.get_artifact(5, "image"), "/tmp/x.png")
        self.assertEqual(ctx.get_artifact(5, "audio"), "/tmp/x.wav")
        self.assertEqual(ctx.get_artifact(5, "clip"), "/tmp/x.mp4")

    def test_artifact_path_zero_final_for_assemble(self):
        """assemble 产物用 scene_id=0, kind='final'。"""
        ctx = _make_ctx([], self._tmpdir, {})
        p = ctx.artifact_path(0, "final")
        self.assertTrue(p.endswith("scene_000_final.mp4"))

    def test_artifact_path_extensions(self):
        """各 kind 的扩展名约定：image→png, audio→wav, clip→mp4, final→mp4。"""
        ctx = _make_ctx([], self._tmpdir, {})
        self.assertTrue(ctx.artifact_path(1, "image").endswith(".png"))
        self.assertTrue(ctx.artifact_path(1, "audio").endswith(".wav"))
        self.assertTrue(ctx.artifact_path(1, "clip").endswith(".mp4"))
        self.assertTrue(ctx.artifact_path(1, "final").endswith(".mp4"))
        # 未知 kind 默认 .bin
        self.assertTrue(ctx.artifact_path(1, "unknown").endswith(".bin"))


if __name__ == "__main__":
    unittest.main()
