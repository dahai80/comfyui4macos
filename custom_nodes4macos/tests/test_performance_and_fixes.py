from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.checkpoint import CheckpointManager, CheckpointData
from custom_nodes4macos.pipeline.stage import Stage, StageInfo
from custom_nodes4macos.pipeline.engine import PipelineEngine, register_stage, _STAGE_REGISTRY


class TestPipelineContextRaceCondition(unittest.TestCase):

    def test_has_artifact_on_disk_missing_file(self):
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        self.assertFalse(ctx.has_artifact_on_disk(1, "image"))

    def test_has_artifact_on_disk_file_deleted_between_checks(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        fpath = os.path.join(tmpdir, "scene_001_image.png")
        ctx.set_artifact(1, "image", fpath)
        with open(fpath, "w") as f:
            f.write("x")
        self.assertTrue(ctx.has_artifact_on_disk(1, "image"))
        os.remove(fpath)
        self.assertFalse(ctx.has_artifact_on_disk(1, "image"))

    def test_artifact_path_creates_consistent_paths(self):
        ctx = PipelineContext(job_id="test", job_dir="/output/test", config={})
        p1 = ctx.artifact_path(1, "image")
        p2 = ctx.artifact_path(1, "image")
        self.assertEqual(p1, p2)
        self.assertTrue(p1.endswith("scene_001_image.png"))

    def test_artifact_path_clip_extension(self):
        ctx = PipelineContext(job_id="test", job_dir="/output/test", config={})
        p = ctx.artifact_path(1, "clip")
        self.assertTrue(p.endswith("scene_001_clip.mp4"))

    def test_should_checkpoint_scene_disabled(self):
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={})
        self.assertFalse(ctx.should_checkpoint_scene(5))

    def test_should_checkpoint_scene_enabled(self):
        ctx = PipelineContext(job_id="test", job_dir="/tmp", config={"checkpoint_every_n_scenes": 3})
        self.assertFalse(ctx.should_checkpoint_scene(1))
        self.assertTrue(ctx.should_checkpoint_scene(3))
        self.assertTrue(ctx.should_checkpoint_scene(6))


class TestCheckpointManagerOverrides(unittest.TestCase):

    def test_save_and_restore_with_overrides(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="test_job", job_dir=tmpdir, config={
            "content_type": "short_drama",
            "template_name": "horror",
            "scene_count": 8,
            "story_seed": "鬼故事",
            "overrides": {"story_seed": "鬼故事", "scene_count": 8},
        })
        ctx.created_at = "2026-07-04T12:00:00"
        ctx.scenes = [{"scene_id": 1, "visual_prompt": "dark forest"}]
        ctx.set_artifact(1, "image", "/tmp/scene_001_image.png")
        ctx.completed_stages = ["prompt_expand"]

        cp = CheckpointManager(tmpdir)
        cp.save(ctx)

        loaded = cp.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.job_id, "test_job")
        self.assertEqual(loaded.completed_stages, ["prompt_expand"])
        self.assertEqual(loaded.config_overrides["story_seed"], "鬼故事")

    def test_restore_context(self):
        tmpdir = tempfile.mkdtemp()
        cp_data = CheckpointData(
            job_id="test",
            content_type="short_drama",
            template_name="horror",
            completed_stages=["prompt_expand", "image_generate"],
            scenes=[{"scene_id": 1}],
            artifacts={"1_image": "/tmp/img.png"},
            config_overrides={"scene_count": 5},
            created_at="2026-07-04T10:00:00",
            updated_at="2026-07-04T10:30:00",
        )
        path = os.path.join(tmpdir, "_checkpoint.json")
        import dataclasses
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(cp_data), f)

        ctx = PipelineContext(job_id="test", job_dir=tmpdir, config={})
        cp = CheckpointManager(tmpdir)
        restored = cp.restore_context(ctx)
        self.assertTrue(restored)
        self.assertEqual(ctx.completed_stages, ["prompt_expand", "image_generate"])
        self.assertEqual(ctx.scenes, [{"scene_id": 1}])
        self.assertEqual(ctx.artifacts, {"1_image": "/tmp/img.png"})


class TestEngineOverridesNoRecursion(unittest.TestCase):

    def test_overrides_no_self_reference(self):
        tmpdir = tempfile.mkdtemp()
        orig_registry = dict(_STAGE_REGISTRY)

        class _NoopStage(Stage):
            @classmethod
            def info(cls) -> StageInfo:
                return StageInfo(name="noop_test", description="noop", model_requirements=[], memory_estimate_gb=0.0)
            def process(self, ctx, model_manager) -> None:
                pass

        _STAGE_REGISTRY.clear()
        register_stage(_NoopStage)

        try:
            engine = PipelineEngine(output_root=os.path.join(tmpdir, "output"))
            engine._templates = {
                "test_ct": {
                    "name": "test",
                    "content_type": "test_ct",
                    "stages": ["noop_test"],
                    "defaults": {"scene_count": 3},
                }
            }
            engine._loaded = True

            result = engine.run("test_ct", story_seed="hello", scene_count=5)
            self.assertIsNotNone(result)
            cp_path = os.path.join(result.job_dir, "_checkpoint.json")
            self.assertTrue(os.path.exists(cp_path))
            with open(cp_path) as f:
                cp = json.load(f)
            self.assertIn("config_overrides", cp)
            self.assertEqual(cp["config_overrides"]["story_seed"], "hello")
            self.assertEqual(cp["config_overrides"]["scene_count"], 5)
        finally:
            _STAGE_REGISTRY.clear()
            _STAGE_REGISTRY.update(orig_registry)


class TestEngineWarmup(unittest.TestCase):

    def test_warmup_mlx_succeeds(self):
        PipelineEngine._warmup_mlx()

    def test_warmup_handles_import_error(self):
        with patch.dict("sys.modules", {"mlx.core": None}):
            PipelineEngine._warmup_mlx()


class TestFFmpegUtilVideoEncoder(unittest.TestCase):

    def test_has_videotoolbox_returns_bool(self):
        from custom_nodes4macos.ffmpeg_util import has_videotoolbox
        result = has_videotoolbox()
        self.assertIsInstance(result, bool)

    def test_video_encoder_args_returns_list(self):
        from custom_nodes4macos.ffmpeg_util import video_encoder_args
        args = video_encoder_args()
        self.assertIsInstance(args, list)
        self.assertTrue(len(args) >= 4)

    def test_thread_args_default(self):
        from custom_nodes4macos import ffmpeg_util
        orig = ffmpeg_util._FFMPEG_THREADS
        ffmpeg_util._FFMPEG_THREADS = 0
        self.assertEqual(ffmpeg_util.thread_args(), [])
        ffmpeg_util._FFMPEG_THREADS = 4
        self.assertEqual(ffmpeg_util.thread_args(), ["-threads", "4"])
        ffmpeg_util._FFMPEG_THREADS = orig


class TestPromptExpandSceneIdNoCollision(unittest.TestCase):

    def test_renumber_scenes_global_offset(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        scenes_ep1 = [
            {"scene_id": 1, "visual_prompt": "a"},
            {"scene_id": 2, "visual_prompt": "b"},
        ]
        scenes_ep2 = [
            {"scene_id": 1, "visual_prompt": "c"},
            {"scene_id": 2, "visual_prompt": "d"},
        ]
        offset = PromptExpandStage._renumber_scenes(scenes_ep1, 0)
        self.assertEqual(offset, 2)
        self.assertEqual(scenes_ep1[0]["scene_id"], 1)
        self.assertEqual(scenes_ep1[1]["scene_id"], 2)

        offset = PromptExpandStage._renumber_scenes(scenes_ep2, offset)
        self.assertEqual(offset, 4)
        self.assertEqual(scenes_ep2[0]["scene_id"], 3)
        self.assertEqual(scenes_ep2[1]["scene_id"], 4)

    def test_parse_and_validate_raw_extracts_global_style(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        json_str = '{"global_style": "ink wash dark", "scenes": [{"visual_prompt": "x"}]}'
        parsed = PromptExpandStage._parse_and_validate_raw(json_str)
        self.assertEqual(parsed["global_style"], "ink wash dark")
        self.assertEqual(len(parsed["scenes"]), 1)

    def test_parse_and_validate_backwards_compat(self):
        from custom_nodes4macos.pipeline.stages.prompt_expand import PromptExpandStage
        json_str = '{"scenes": [{"visual_prompt": "a"}, {"visual_prompt": "b"}]}'
        scenes = PromptExpandStage._parse_and_validate(json_str)
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0]["scene_id"], 1)
        self.assertEqual(scenes[1]["scene_id"], 2)


if __name__ == "__main__":
    unittest.main()
