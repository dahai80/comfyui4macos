"""端到端管线集成测试。

覆盖 PipelineEngine.run 从 run → stages → final_video 的全链路，
以及 resume 续跑完整链路。所有 MLX/HTTP 模型调用全部 mock，
ffmpeg 调用 mock（避免依赖真实 ffmpeg 与 GPU）。

填补 REVIEW_REPORT P1-1：无端到端管线集成测试。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.engine import PipelineEngine, register_stage, _STAGE_REGISTRY
from custom_nodes4macos.pipeline.stage import Stage, StageInfo


class _StubStage(Stage):
    """记录执行顺序、写入占位产物、可控制失败的可控 stage。"""

    execution_log: list[str] = []
    fail_on: str | None = None

    @classmethod
    def reset(cls, fail_on: str | None = None):
        cls.execution_log = []
        cls.fail_on = fail_on

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="stub_e2e",
            description="e2e stub",
            model_requirements=[],
            memory_estimate_gb=0.0,
        )

    def process(self, ctx, model_manager) -> None:
        name = self.info().name
        _StubStage.execution_log.append(name)
        if _StubStage.fail_on == name:
            raise RuntimeError(f"stub forced fail at {name}")
        final_path = ctx.artifact_path(0, "final")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(b"fake-final-mp4")
        ctx.set_artifact(0, "final", final_path)


class _RecordingStage(Stage):
    """带 name 参数的参数化 stage，用于测试多 stage 顺序执行。"""

    def __init__(self, name: str = "rec"):
        self._name = name

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(name="rec", description="rec", model_requirements=[], memory_estimate_gb=0.0)

    def process(self, ctx, model_manager) -> None:
        _StubStage.execution_log.append(self._name)


class TestPipelineEndToEnd(unittest.TestCase):
    """全链路：run → stage 执行 → checkpoint 持久化 → final_video 返回。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        register_stage(_StubStage)
        _StubStage.reset()

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_full_run_produces_final_video(self, _warmup):
        """run 全链路：返回 PipelineResult，final_video 存在，checkpoint 记录完成。"""
        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "out"))
        engine._loaded = True
        engine._templates = {
            "e2e_content": {
                "name": "e2e_tpl",
                "content_type": "e2e_content",
                "stages": ["stub_e2e"],
                "defaults": {"scene_count": 1},
            }
        }
        result = engine.run("e2e_content", story_seed="seed")
        self.assertTrue(os.path.isdir(result.job_dir))
        self.assertIsNotNone(result.final_video)
        self.assertTrue(os.path.exists(result.final_video))
        self.assertEqual(_StubStage.execution_log, ["stub_e2e"])

        cp_path = os.path.join(result.job_dir, "_checkpoint.json")
        self.assertTrue(os.path.exists(cp_path))
        with open(cp_path) as f:
            cp = json.load(f)
        self.assertIn("stub_e2e", cp["completed_stages"])

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_final_video_taken_from_artifact_zero_final(self, _warmup):
        """engine 返回的 final_video 来自 ctx.get_artifact(0, 'final')。"""
        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "out"))
        engine._loaded = True
        engine._templates = {
            "e2e_content": {
                "content_type": "e2e_content",
                "stages": ["stub_e2e"],
            }
        }
        result = engine.run("e2e_content", story_seed="seed")
        self.assertTrue(result.final_video.endswith("scene_000_final.mp4"))

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_stage_failure_saves_checkpoint_and_reraises(self, _warmup):
        """stage 抛异常时，engine 先存 checkpoint 再 re-raise。"""
        _StubStage.reset(fail_on="stub_e2e")
        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "out"))
        engine._loaded = True
        engine._templates = {
            "e2e_content": {
                "content_type": "e2e_content",
                "stages": ["stub_e2e"],
            }
        }
        with self.assertRaises(RuntimeError):
            engine.run("e2e_content", story_seed="seed")
        # 失败前 checkpoint 已写入（job_dir 创建即写入）
        out_root = os.path.join(self._tmpdir, "out")
        jobs = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
        self.assertEqual(len(jobs), 1)
        cp_path = os.path.join(out_root, jobs[0], "_checkpoint.json")
        self.assertTrue(os.path.exists(cp_path))


class TestPipelineResumeEndToEnd(unittest.TestCase):
    """resume 全链路：恢复 checkpoint → 跳过已完成 stage → 继续执行。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        register_stage(_StubStage)
        _StubStage.reset()

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_resume_skips_completed_stage(self, _warmup):
        """已完成的 stage 在 resume 时不重跑，execution_log 为空。"""
        job_id = "20260704_e2e0001"
        job_dir = os.path.join(self._tmpdir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        # 预置 checkpoint：stub_e2e 已完成
        cp_data = {
            "job_id": job_id,
            "content_type": "e2e_content",
            "template_name": "e2e_tpl",
            "completed_stages": ["stub_e2e"],
            "scenes": [],
            "artifacts": {},
            "config_overrides": {"story_seed": "resumed"},
            "created_at": "2026-07-04T07:00:00",
            "updated_at": "2026-07-04T07:00:00",
        }
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump(cp_data, f)

        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        engine._templates = {
            "e2e_content": {
                "content_type": "e2e_content",
                "stages": ["stub_e2e"],
            }
        }
        result = engine.run("e2e_content", resume_from=job_id)
        # stub_e2e 已完成，不应执行
        self.assertEqual(_StubStage.execution_log, [])

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_resume_restores_artifacts_and_scenes(self, _warmup):
        """resume 后 ctx.artifacts 与 scenes 从 checkpoint 恢复。"""
        job_id = "20260704_e2e0002"
        job_dir = os.path.join(self._tmpdir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        cp_data = {
            "job_id": job_id,
            "content_type": "e2e_content",
            "template_name": "e2e_tpl",
            "completed_stages": ["stub_e2e"],
            "scenes": [{"scene_id": 1, "visual_prompt": "v"}],
            "artifacts": {"1_image": "/tmp/scene_001_image.png"},
            "config_overrides": {},
            "created_at": "2026-07-04T07:00:00",
            "updated_at": "2026-07-04T07:00:00",
        }
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump(cp_data, f)

        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        engine._templates = {"e2e_content": {"content_type": "e2e_content", "stages": ["stub_e2e"]}}

        # 用 _resume 直接验证 ctx 恢复（避免 engine.run 二次包装）
        ctx_holder = {}

        original_execute = engine._execute

        def capture_execute(ctx, checkpoint):
            ctx_holder["ctx"] = ctx
            return original_execute(ctx, checkpoint)

        with patch.object(engine, "_execute", side_effect=capture_execute):
            engine.run("e2e_content", resume_from=job_id)

        ctx = ctx_holder["ctx"]
        self.assertEqual(ctx.scenes, [{"scene_id": 1, "visual_prompt": "v"}])
        self.assertEqual(ctx.artifacts, {"1_image": "/tmp/scene_001_image.png"})
        self.assertEqual(ctx.completed_stages, ["stub_e2e"])

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_resume_unknown_job_raises(self, _warmup):
        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        with self.assertRaises(ValueError) as cm:
            engine.run("e2e_content", resume_from="nonexistent_job")
        self.assertIn("job not found", str(cm.exception))


class TestPipelineMultiStageOrder(unittest.TestCase):
    """验证多 stage 严格按模板声明顺序执行。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        # 注册多个不同名的 stub stage
        for n in ["s1", "s2", "s3"]:
            cls = type(f"Stage{n}", (_StubStage,), {
                "info": classmethod(lambda cls, _n=n: StageInfo(name=_n, description=_n, model_requirements=[], memory_estimate_gb=0.0)),
                "process": lambda self, ctx, mm, _n=n: _StubStage.execution_log.append(_n),
            })
            register_stage(cls)
        _StubStage.reset()

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine._warmup_mlx")
    def test_stages_execute_in_template_order(self, _warmup):
        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "out"))
        engine._loaded = True
        engine._templates = {
            "multi": {
                "content_type": "multi",
                "stages": ["s3", "s1", "s2"],  # 故意非字母序
            }
        }
        engine.run("multi", story_seed="x")
        self.assertEqual(_StubStage.execution_log, ["s3", "s1", "s2"])


class TestPipelineListJobs(unittest.TestCase):
    """list_jobs 端到端：扫描 output_root，返回 job 元数据。"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    def test_list_jobs_returns_completed_stages(self):
        job_id = "20260704_list0001"
        job_dir = os.path.join(self._tmpdir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump({
                "job_id": job_id,
                "content_type": "short_drama",
                "completed_stages": ["prompt_expand", "image_generate"],
                "updated_at": "2026-07-04T07:30:00",
            }, f)

        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        jobs = engine.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["job_id"], job_id)
        self.assertEqual(jobs[0]["content_type"], "short_drama")
        self.assertEqual(jobs[0]["completed_stages"], ["prompt_expand", "image_generate"])

    def test_list_jobs_skips_dir_without_checkpoint(self):
        os.makedirs(os.path.join(self._tmpdir, "stray_dir"), exist_ok=True)
        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        self.assertEqual(engine.list_jobs(), [])

    def test_list_jobs_skips_corrupt_checkpoint(self):
        job_dir = os.path.join(self._tmpdir, "20260704_bad0001")
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            f.write("{corrupt")
        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        # 腐败 checkpoint 应被跳过而非抛异常
        self.assertEqual(engine.list_jobs(), [])


if __name__ == "__main__":
    unittest.main()
