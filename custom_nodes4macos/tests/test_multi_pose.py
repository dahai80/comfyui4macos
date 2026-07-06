"""multi_pose 阶段测试：多姿态关键帧 → 定格剪辑 clip。

离线测试（无需 fusion-mlx）：mock Flux HTTP，真实 ffmpeg 拼接。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos import ffmpeg_util
from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.image_generate import ImageGenerateStage
from custom_nodes4macos.pipeline.stages.multi_pose import MultiPoseStage, _DEFAULT_POSE_SUFFIXES

HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _tiny_png(path: str, color: tuple = (80, 120, 160)) -> str:
    from PIL import Image
    Image.new("RGB", (8, 8), color).save(path)
    return path


def _tiny_wav(path: str, dur: float = 1.2) -> str:
    ffmpeg_util.run_ffmpeg(
        ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", path],
        timeout=30, label="test_wav",
    )
    return path


class TestMultiPoseStageInfo(unittest.TestCase):

    def test_info(self):
        info = MultiPoseStage.info()
        self.assertEqual(info.name, "multi_pose")
        self.assertEqual(info.model_requirements, ["flux"])
        self.assertEqual(info.input_kinds, ["image", "audio"])
        self.assertEqual(info.output_kinds, ["clip"])

    def test_default_pose_suffixes_non_empty(self):
        self.assertGreaterEqual(len(_DEFAULT_POSE_SUFFIXES), 3)


class TestMultiPoseSkip(unittest.TestCase):

    def test_skip_if_completed(self):
        stage = MultiPoseStage()
        ctx = PipelineContext(job_id="t", job_dir="/tmp", config={})
        ctx.completed_stages = ["multi_pose"]
        self.assertTrue(stage._skip_if_completed(ctx))

    def test_skip_scene_without_base_image(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 10,
            "multi_pose_count": 3,
        })
        ctx.scenes = [{"scene_id": 1, "visual_prompt": "a temple", "characters": []}]
        mock_mgr = MagicMock()
        mock_handle = MagicMock()
        mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
        MultiPoseStage().process(ctx, mock_mgr)
        self.assertIsNone(ctx.get_artifact(1, "clip"))
        mock_handle.client.generate_image.assert_not_called()


class TestGeneratePoses(unittest.TestCase):

    def test_empty_prompt_reuses_base(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        paths = MultiPoseStage._generate_poses(
            handle, "", "style", "", 42, 3, list(_DEFAULT_POSE_SUFFIXES),
            64, 64, 4, 4.0, tmpdir, 1, base,
        )
        self.assertEqual(paths, [base, base, base])
        handle.client.generate_image.assert_not_called()

    def test_reuses_cached_pose_files(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        pose_dir = os.path.join(tmpdir, "poses")
        os.makedirs(pose_dir, exist_ok=True)
        _tiny_png(os.path.join(pose_dir, "scene_001_pose_001.png"), (10, 20, 30))
        _tiny_png(os.path.join(pose_dir, "scene_001_pose_002.png"), (40, 50, 60))
        handle = MagicMock()
        paths = MultiPoseStage._generate_poses(
            handle, "a temple", "", "", 42, 2, list(_DEFAULT_POSE_SUFFIXES),
            64, 64, 4, 4.0, tmpdir, 1, base,
        )
        self.assertTrue(all(os.path.exists(p) for p in paths))
        self.assertEqual(len(paths), 2)
        handle.client.generate_image.assert_not_called()

    def test_falls_back_to_base_on_gen_failure(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        with patch.object(ImageGenerateStage, "_generate_http", side_effect=RuntimeError("flux down")):
            paths = MultiPoseStage._generate_poses(
                handle, "a temple", "", "", 42, 3, list(_DEFAULT_POSE_SUFFIXES),
                64, 64, 4, 4.0, tmpdir, 1, base,
            )
        self.assertEqual(paths, [base, base, base])


class TestRealisticReferencePassthrough(unittest.TestCase):
    """realistic 模式把 base_img 作为参考图透传给 _generate_http。"""

    def test_realistic_passes_reference_to_each_pose(self):
        import base64
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        with patch.object(ImageGenerateStage, "_generate_http") as mock_gen:
            MultiPoseStage._generate_poses(
                handle, "a temple", "", "", 42, 3, list(_DEFAULT_POSE_SUFFIXES),
                64, 64, 4, 4.0, tmpdir, 1, base,
                character_style="realistic",
            )
        self.assertEqual(mock_gen.call_count, 3)
        expected_ref = base64.b64encode(open(base, "rb").read()).decode("ascii")
        for call in mock_gen.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["reference_image"], expected_ref)
            self.assertEqual(kwargs["conditioning_mode"], "redux")
            self.assertAlmostEqual(kwargs["reference_strength"], 0.6)

    def test_non_realistic_omits_reference(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        with patch.object(ImageGenerateStage, "_generate_http") as mock_gen:
            MultiPoseStage._generate_poses(
                handle, "a temple", "", "", 42, 2, list(_DEFAULT_POSE_SUFFIXES),
                64, 64, 4, 4.0, tmpdir, 1, base,
                character_style="cartoon",
            )
        self.assertEqual(mock_gen.call_count, 2)
        for call in mock_gen.call_args_list:
            self.assertIsNone(call.kwargs["reference_image"])
            self.assertIsNone(call.kwargs["conditioning_mode"])

    def test_realistic_empty_prompt_skips_reference(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        with patch.object(ImageGenerateStage, "_generate_http") as mock_gen:
            paths = MultiPoseStage._generate_poses(
                handle, "", "", "", 42, 3, list(_DEFAULT_POSE_SUFFIXES),
                64, 64, 4, 4.0, tmpdir, 1, base,
                character_style="realistic",
            )
        self.assertEqual(paths, [base, base, base])
        mock_gen.assert_not_called()

    def test_realistic_respects_custom_strength_and_mode(self):
        tmpdir = tempfile.mkdtemp()
        base = _tiny_png(os.path.join(tmpdir, "base.png"))
        handle = MagicMock()
        with patch.object(ImageGenerateStage, "_generate_http") as mock_gen:
            MultiPoseStage._generate_poses(
                handle, "a temple", "", "", 42, 2, list(_DEFAULT_POSE_SUFFIXES),
                64, 64, 4, 4.0, tmpdir, 1, base,
                character_style="realistic",
                reference_strength=0.42,
                conditioning_mode="in_context",
            )
        for call in mock_gen.call_args_list:
            self.assertAlmostEqual(call.kwargs["reference_strength"], 0.42)
            self.assertEqual(call.kwargs["conditioning_mode"], "in_context")


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestStitchClip(unittest.TestCase):

    def test_stitch_hard_cut_no_audio(self):
        tmpdir = tempfile.mkdtemp()
        poses = [
            _tiny_png(os.path.join(tmpdir, f"p{k}.png"), (k * 60, 100, 150))
            for k in range(3)
        ]
        out = os.path.join(tmpdir, "out.mp4")
        MultiPoseStage._stitch_clip(poses, "", 1.5, 64, 128, 10, out, 1)
        self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)
        self.assertFalse(ffmpeg_util.probe_has_audio(out))
        dur = ffmpeg_util.probe_duration(out)
        self.assertGreater(dur, 0.5)

    def test_stitch_with_audio_muxed(self):
        tmpdir = tempfile.mkdtemp()
        poses = [
            _tiny_png(os.path.join(tmpdir, f"p{k}.png"), (k * 60, 100, 150))
            for k in range(3)
        ]
        wav = _tiny_wav(os.path.join(tmpdir, "narr.wav"), dur=1.2)
        out = os.path.join(tmpdir, "out_audio.mp4")
        MultiPoseStage._stitch_clip(poses, wav, 1.2, 64, 128, 10, out, 1)
        self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)
        self.assertTrue(ffmpeg_util.probe_has_audio(out))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestMultiPoseProcessFull(unittest.TestCase):

    def test_process_produces_clip_with_mocked_flux(self):
        tmpdir = tempfile.mkdtemp()
        base_img = _tiny_png(os.path.join(tmpdir, "base.png"))
        wav = _tiny_wav(os.path.join(tmpdir, "narr.wav"), dur=1.2)

        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 10,
            "multi_pose_count": 3,
        })
        ctx.scenes = [{
            "scene_id": 1,
            "visual_prompt": "a dark temple at night",
            "characters": [],
            "duration_seconds": 1.2,
        }]
        ctx.set_artifact(1, "image", base_img)
        ctx.set_artifact(1, "audio", wav)

        def fake_gen(handle, prompt, width, height, steps, guidance, seed, out_path):
            color = (100, (len(prompt) * 7) % 255, 200)
            _tiny_png(out_path, color)

        with patch.object(ImageGenerateStage, "_generate_http", side_effect=fake_gen):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_handle.model_name = "flux-test"
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            MultiPoseStage().process(ctx, mock_mgr)

        clip = ctx.get_artifact(1, "clip")
        self.assertIsNotNone(clip)
        self.assertTrue(os.path.exists(clip) and os.path.getsize(clip) > 0)
        self.assertTrue(ffmpeg_util.probe_has_audio(clip))
        self.assertEqual(ctx.progress.get("stage"), "multi_pose")
        self.assertEqual(ctx.progress.get("scene"), 1)

    def test_process_idempotent_skips_existing_clip(self):
        tmpdir = tempfile.mkdtemp()
        base_img = _tiny_png(os.path.join(tmpdir, "base.png"))
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 10,
            "multi_pose_count": 2,
        })
        ctx.scenes = [{"scene_id": 1, "visual_prompt": "x", "characters": [], "duration_seconds": 1.0}]
        ctx.set_artifact(1, "image", base_img)
        existing_clip = ctx.artifact_path(1, "clip")
        with open(existing_clip, "wb") as f:
            f.write(b"already_done")
        ctx.set_artifact(1, "clip", existing_clip)

        with patch.object(ImageGenerateStage, "_generate_http", side_effect=AssertionError("should not call flux")):
            mock_mgr = MagicMock()
            mock_handle = MagicMock()
            mock_mgr.acquire.return_value.__enter__.return_value = mock_handle
            MultiPoseStage().process(ctx, mock_mgr)
        self.assertEqual(ctx.get_artifact(1, "clip"), existing_clip)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestMultiPoseEngineIntegration(unittest.TestCase):
    """multi_pose 通过真实 engine.run 调度（注册表解析 + 分发）。"""

    def test_engine_runs_multi_pose_via_registry(self):
        import json
        from custom_nodes4macos.pipeline.engine import (
            PipelineEngine, _STAGE_REGISTRY, register_stage, _auto_discover_stages,
        )
        from custom_nodes4macos.pipeline.stage import Stage, StageInfo

        _auto_discover_stages()
        self.assertIn("multi_pose", _STAGE_REGISTRY)
        saved = dict(_STAGE_REGISTRY)

        class _PrepStage(Stage):
            @classmethod
            def info(cls) -> StageInfo:
                return StageInfo(name="_mp_prep", description="prep scenes+image+audio", input_kinds=[], output_kinds=[])

            def process(self, ctx, model_manager) -> None:
                tmp = ctx.job_dir
                img = _tiny_png(os.path.join(tmp, "prep_img.png"))
                wav = _tiny_wav(os.path.join(tmp, "prep.wav"), dur=1.2)
                ctx.scenes = [{
                    "scene_id": 1,
                    "visual_prompt": "a dark temple at night",
                    "characters": [],
                    "duration_seconds": 1.2,
                }]
                ctx.set_artifact(1, "image", img)
                ctx.set_artifact(1, "audio", wav)

        register_stage(_PrepStage)
        try:
            engine = PipelineEngine(output_root=tempfile.mkdtemp())

            def fake_http(handle, prompt, w, h, steps, guidance, seed, out_path):
                _tiny_png(out_path, (100, (len(prompt) * 7) % 255, 200))

            with patch.object(ImageGenerateStage, "_generate_http", side_effect=fake_http), \
                 patch.object(PipelineEngine, "_warmup_mlx", lambda self: None), \
                 patch("custom_nodes4macos.pipeline.engine.ModelManager") as MockMM:
                MockMM.return_value = MagicMock()
                result = engine.run("puppet_show", stages=["_mp_prep", "multi_pose"])

            cp_path = os.path.join(result.job_dir, "_checkpoint.json")
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            self.assertIn("multi_pose", cp.get("completed_stages", []))
            clip_artifacts = [
                a for a in cp.get("artifacts", {}).values()
                if isinstance(a, str) and a.endswith(".mp4") and "_clip" in os.path.basename(a)
            ]
            self.assertTrue(any(os.path.isfile(c) and os.path.getsize(c) > 0 for c in clip_artifacts),
                            f"no real clip produced: {clip_artifacts}")
        finally:
            _STAGE_REGISTRY.clear()
            _STAGE_REGISTRY.update(saved)

    def test_motion_mode_swaps_ken_burns_to_multi_pose(self):
        from custom_nodes4macos.pipeline.engine import PipelineEngine, _STAGE_REGISTRY, _auto_discover_stages
        from custom_nodes4macos.pipeline.context import PipelineContext
        from custom_nodes4macos.pipeline.checkpoint import CheckpointManager

        _auto_discover_stages()
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "content_type": "puppet_show",
            "stages": ["prompt_expand", "image_generate", "tts_synthesize", "ken_burns", "assemble"],
            "motion_mode": "multi_pose",
        })
        ckpt = CheckpointManager(tmpdir)
        captured: list[list[str]] = []

        def fake_instantiate(self, names):
            captured.append(list(names))
            return []

        with patch.object(PipelineEngine, "_warmup_mlx", lambda self: None), \
             patch("custom_nodes4macos.pipeline.engine.ModelManager") as MockMM, \
             patch.object(PipelineEngine, "_instantiate_stages", fake_instantiate):
            MockMM.return_value = MagicMock()
            engine._execute(ctx, ckpt)

        self.assertTrue(captured, "_instantiate_stages was not called")
        resolved = captured[0]
        self.assertIn("multi_pose", resolved)
        self.assertNotIn("ken_burns", resolved)
        self.assertEqual(resolved.count("multi_pose"), 1)

    def test_motion_mode_default_keeps_ken_burns(self):
        from custom_nodes4macos.pipeline.engine import PipelineEngine, _auto_discover_stages
        from custom_nodes4macos.pipeline.context import PipelineContext
        from custom_nodes4macos.pipeline.checkpoint import CheckpointManager

        _auto_discover_stages()
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "content_type": "puppet_show",
            "stages": ["prompt_expand", "image_generate", "tts_synthesize", "ken_burns", "assemble"],
        })
        ckpt = CheckpointManager(tmpdir)
        captured: list[list[str]] = []

        def fake_instantiate(self, names):
            captured.append(list(names))
            return []

        with patch.object(PipelineEngine, "_warmup_mlx", lambda self: None), \
             patch("custom_nodes4macos.pipeline.engine.ModelManager") as MockMM, \
             patch.object(PipelineEngine, "_instantiate_stages", fake_instantiate):
            MockMM.return_value = MagicMock()
            engine._execute(ctx, ckpt)

        resolved = captured[0]
        self.assertIn("ken_burns", resolved)
        self.assertNotIn("multi_pose", resolved)


if __name__ == "__main__":
    unittest.main()
