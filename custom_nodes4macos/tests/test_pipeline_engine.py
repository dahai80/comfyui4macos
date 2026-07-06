from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stage import Stage, StageInfo
from custom_nodes4macos.pipeline.engine import PipelineEngine, register_stage, _STAGE_REGISTRY
from custom_nodes4macos.pipeline.model_manager import ModelManager
from custom_nodes4macos.pipeline.result import PipelineResult


class _DummyStage(Stage):
    process_called = False

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="test_dummy",
            description="test dummy stage",
            model_requirements=[],
            memory_estimate_gb=0.0,
        )

    def process(self, ctx, model_manager) -> None:
        _DummyStage.process_called = True


class TestPipelineEngineRun(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        register_stage(_DummyStage)

        self._tpl_dir = os.path.join(self._tmpdir, "templates")
        os.makedirs(self._tpl_dir, exist_ok=True)
        tpl = {
            "name": "test_tpl",
            "content_type": "test_content",
            "stages": ["test_dummy"],
            "defaults": {"scene_count": 3},
        }
        with open(os.path.join(self._tpl_dir, "test_content.yaml"), "w") as f:
            json.dump(tpl, f)

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    @patch("custom_nodes4macos.pipeline.engine._TEMPLATE_DIR")
    def test_run_creates_job_dir(self, mock_tpl_dir):
        mock_tpl_dir.__str__ = lambda s: self._tpl_dir
        mock_tpl_dir.exists.return_value = True
        mock_tpl_dir.glob = MagicMock(return_value=[])

        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "output"))
        engine._templates = {
            "test_content": {
                "name": "test_tpl",
                "content_type": "test_content",
                "stages": ["test_dummy"],
                "defaults": {"scene_count": 3},
            }
        }
        engine._loaded = True

        _DummyStage.process_called = False
        result = engine.run("test_content", story_seed="hello")
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(os.path.isdir(result.job_dir))
        self.assertTrue(_DummyStage.process_called)

    def test_run_unknown_content_type_raises(self):
        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        with self.assertRaises(ValueError):
            engine.run("nonexistent_type", story_seed="x")

    @patch("custom_nodes4macos.pipeline.engine._TEMPLATE_DIR")
    def test_run_passes_model_overrides_from_config(self, mock_tpl_dir):
        mock_tpl_dir.__str__ = lambda s: self._tpl_dir
        mock_tpl_dir.exists.return_value = True
        mock_tpl_dir.glob = MagicMock(return_value=[])

        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "output"))
        engine._templates = {
            "test_content": {
                "name": "test_tpl",
                "content_type": "test_content",
                "stages": ["test_dummy"],
                "defaults": {"scene_count": 3},
            }
        }
        engine._loaded = True

        captured = {}

        class _SpyMgr(ModelManager):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

        with patch("custom_nodes4macos.pipeline.engine.ModelManager", _SpyMgr):
            _DummyStage.process_called = False
            engine.run(
                "test_content",
                story_seed="hello",
                llm_model="MyLLM-8bit",
                flux_model="MyFlux-dev",
            )

        self.assertEqual(
            captured.get("model_overrides"),
            {"llm": "MyLLM-8bit", "flux": "MyFlux-dev"},
        )
        self.assertTrue(_DummyStage.process_called)


class TestPipelineEngineResume(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        register_stage(_DummyStage)

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    def test_resume_missing_job_raises(self):
        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        with self.assertRaises(ValueError):
            engine.run("test_content", resume_from="nonexistent_job")

    def test_resume_restores_checkpoint(self):
        job_id = "20260704_test0001"
        job_dir = os.path.join(self._tmpdir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        cp_data = {
            "job_id": job_id,
            "content_type": "test_content",
            "template_name": "test_tpl",
            "completed_stages": ["test_dummy"],
            "scenes": [],
            "artifacts": {},
            "config_overrides": {},
            "created_at": "2026-07-04T10:00:00",
            "updated_at": "2026-07-04T10:00:00",
        }
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump(cp_data, f)

        engine = PipelineEngine(output_root=self._tmpdir)
        engine._loaded = True
        engine._templates = {
            "test_content": {
                "name": "test_tpl",
                "content_type": "test_content",
                "stages": ["test_dummy"],
                "defaults": {},
            }
        }

        _DummyStage.process_called = False
        result = engine.run("test_content", resume_from=job_id)
        self.assertIsInstance(result, PipelineResult)
        self.assertFalse(_DummyStage.process_called)


class TestPipelineEngineListJobs(unittest.TestCase):

    def test_list_jobs_empty(self):
        engine = PipelineEngine(output_root="/tmp/nonexistent_test_dir_12345")
        engine._loaded = True
        self.assertEqual(engine.list_jobs(), [])


if __name__ == "__main__":
    unittest.main()
