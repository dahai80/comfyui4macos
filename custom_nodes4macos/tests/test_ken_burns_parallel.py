"""KenBurnsStage 并行渲染路径测试。

覆盖 _render_parallel 路径、失败传播、空任务、单任务串行回退。

填补 REVIEW_REPORT P2：ken_burns _render_parallel 路径无测试。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.ken_burns import KenBurnsStage


def _make_image(ctx, scene_id):
    p = os.path.join(ctx.job_dir, f"src_{scene_id}.png")
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nfake")
    ctx.set_artifact(scene_id, "image", p)
    return p


class TestKenBurnsParallelDispatch(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_parallel_path_taken_when_workers_gt_1_and_multi_tasks(self):
        """workers>1 且 tasks>1 时走 _render_parallel。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 3})
        ctx.scenes = [{"scene_id": i, "duration_seconds": 1} for i in range(1, 4)]
        for i in range(1, 4):
            _make_image(ctx, i)

        stage = KenBurnsStage()
        with patch.object(stage, "_render_parallel") as mock_par:
            with patch.object(stage, "_render_sequential") as mock_seq:
                stage.process(ctx, MagicMock())
            mock_par.assert_called_once()
            mock_seq.assert_not_called()

    def test_sequential_path_when_workers_is_1(self):
        """workers=1 时走 _render_sequential。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 1})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 1}, {"scene_id": 2, "duration_seconds": 1}]
        for i in (1, 2):
            _make_image(ctx, i)

        stage = KenBurnsStage()
        with patch.object(stage, "_render_parallel") as mock_par:
            with patch.object(stage, "_render_sequential") as mock_seq:
                stage.process(ctx, MagicMock())
            mock_seq.assert_called_once()
            mock_par.assert_not_called()

    def test_sequential_path_when_single_task(self):
        """workers>1 但仅 1 个 task 时走 _render_sequential。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 4})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 1}]
        _make_image(ctx, 1)

        stage = KenBurnsStage()
        with patch.object(stage, "_render_parallel") as mock_par:
            with patch.object(stage, "_render_sequential") as mock_seq:
                stage.process(ctx, MagicMock())
            mock_seq.assert_called_once()
            mock_par.assert_not_called()


class TestKenBurnsParallelExecution(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_parallel_renders_all_clips(self, mock_ffmpeg):
        """并行模式渲染所有 clip，每个 scene 写入 artifact。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 2})
        ctx.scenes = [{"scene_id": i, "duration_seconds": 1} for i in range(1, 4)]
        for i in range(1, 4):
            _make_image(ctx, i)

        mock_ffmpeg.probe_has_audio.return_value = False

        def fake_render_clip(img, audio, dur, preset, w, h, fps, render_fps, sid, out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(b"mp4")

        with patch.object(KenBurnsStage, "_render_clip", side_effect=fake_render_clip):
            stage = KenBurnsStage()
            stage.process(ctx, MagicMock())

        for i in range(1, 4):
            self.assertIsNotNone(ctx.get_artifact(i, "clip"))

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_parallel_failure_does_not_crash_others(self, mock_ffmpeg):
        """一个 scene 渲染失败不影响其他 scene。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 3})
        ctx.scenes = [{"scene_id": i, "duration_seconds": 1} for i in range(1, 4)]
        for i in range(1, 4):
            _make_image(ctx, i)

        mock_ffmpeg.probe_has_audio.return_value = False

        def flaky_render(img, audio, dur, preset, w, h, fps, render_fps, sid, out):
            if sid == 2:
                raise RuntimeError("scene 2 fails")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(b"mp4")

        with patch.object(KenBurnsStage, "_render_clip", side_effect=flaky_render):
            stage = KenBurnsStage()
            stage.process(ctx, MagicMock())

        # scene 1 和 3 成功，scene 2 失败（无 artifact）
        self.assertIsNotNone(ctx.get_artifact(1, "clip"))
        self.assertIsNone(ctx.get_artifact(2, "clip"))
        self.assertIsNotNone(ctx.get_artifact(3, "clip"))


class TestKenBurnsTaskCollection(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_existing_clips_skipped(self, mock_ffmpeg):
        """已存在的 clip 不进 tasks 列表。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 2})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 1}, {"scene_id": 2, "duration_seconds": 1}]
        _make_image(ctx, 1)
        _make_image(ctx, 2)
        # scene 1 已有 clip
        existing_clip = ctx.artifact_path(1, "clip")
        with open(existing_clip, "wb") as f:
            f.write(b"existing")
        ctx.set_artifact(1, "clip", existing_clip)

        mock_ffmpeg.probe_has_audio.return_value = False
        with patch.object(KenBurnsStage, "_render_clip") as mock_render:
            stage = KenBurnsStage()
            stage.process(ctx, MagicMock())
            # 只渲染 scene 2
            self.assertEqual(mock_render.call_count, 1)
            rendered_sid = mock_render.call_args.args[8]
            self.assertEqual(rendered_sid, 2)

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_no_tasks_returns_early(self, mock_ffmpeg):
        """所有 clip 都已存在时，不调用任何渲染。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 2})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 1}]
        _make_image(ctx, 1)
        existing = ctx.artifact_path(1, "clip")
        with open(existing, "wb") as f:
            f.write(b"x")
        ctx.set_artifact(1, "clip", existing)

        with patch.object(KenBurnsStage, "_render_clip") as mock_render:
            stage = KenBurnsStage()
            stage.process(ctx, MagicMock())
            mock_render.assert_not_called()

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_scene_without_image_skipped(self, mock_ffmpeg):
        """无 image artifact 的 scene 跳过。"""
        ctx = PipelineContext(job_id="kb", job_dir=self._tmpdir, config={"ken_burns_workers": 2})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 1}]  # 无 image
        with patch.object(KenBurnsStage, "_render_clip") as mock_render:
            stage = KenBurnsStage()
            stage.process(ctx, MagicMock())
            mock_render.assert_not_called()


class TestRenderClipFfmpegArgs(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_render_clip_with_audio(self, mock_ffmpeg):
        """有音频时 ffmpeg args 含 -shortest 和 aac 编码。"""
        import tempfile as tf
        img = tf.mktemp(suffix=".png")
        with open(img, "wb") as f:
            f.write(b"png")
        audio = tf.mktemp(suffix=".wav")
        with open(audio, "wb") as f:
            f.write(b"wav")
        out = tf.mktemp(suffix=".mp4")

        mock_ffmpeg.probe_has_audio.return_value = True
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"mp4")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        KenBurnsStage._render_clip(img, audio, 2.0, "zoom-in", 1080, 1920, 30, 30, 1, out)

        call_args = mock_ffmpeg.run_ffmpeg.call_args.args[0]
        self.assertIn("-shortest", call_args)
        self.assertIn("aac", call_args)
        self.assertIn(audio, call_args)

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_render_clip_no_audio_uses_duration_flag(self, mock_ffmpeg):
        """无音频时用 -t 限定时长。"""
        import tempfile as tf
        img = tf.mktemp(suffix=".png")
        with open(img, "wb") as f:
            f.write(b"png")
        out = tf.mktemp(suffix=".mp4")

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"mp4")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        KenBurnsStage._render_clip(img, None, 2.0, "zoom-in", 1080, 1920, 30, 30, 1, out)

        call_args = mock_ffmpeg.run_ffmpeg.call_args.args[0]
        self.assertIn("-t", call_args)

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.ffmpeg_util")
    def test_render_clip_empty_output_raises(self, mock_ffmpeg):
        """ffmpeg 成功但输出文件为空/不存在时抛 RuntimeError。"""
        import tempfile as tf
        img = tf.mktemp(suffix=".png")
        with open(img, "wb") as f:
            f.write(b"png")
        out = tf.mktemp(suffix=".mp4")  # 不创建该文件

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        mock_ffmpeg.run_ffmpeg.return_value = None  # ffmpeg "成功"但未产出

        with self.assertRaises(RuntimeError):
            KenBurnsStage._render_clip(img, None, 1.0, "zoom-in", 64, 128, 10, 10, 1, out)


if __name__ == "__main__":
    unittest.main()
