from __future__ import annotations

import json
import os
import tempfile
import unittest

from custom_nodes4macos.pipeline.checkpoint import CheckpointManager, CheckpointData
from custom_nodes4macos.pipeline.context import PipelineContext


class TestCheckpointSaveLoad(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_save_and_load(self):
        ctx = PipelineContext(
            job_id="20260704_abc12345",
            job_dir=self._tmpdir,
            config={"content_type": "short_drama", "template_name": "horror_short_drama"},
        )
        ctx.completed_stages = ["prompt_expand", "image_generate"]
        ctx.scenes = [
            {"scene_id": 1, "visual_prompt": "test", "audio_script": "hello"},
        ]
        ctx.artifacts = {"1_image": "/tmp/scene_001_image.png"}
        ctx.created_at = "2026-07-04T10:00:00"

        self._mgr.save(ctx)

        loaded = self._mgr.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.job_id, "20260704_abc12345")
        self.assertEqual(loaded.completed_stages, ["prompt_expand", "image_generate"])
        self.assertEqual(len(loaded.scenes), 1)
        self.assertEqual(loaded.artifacts["1_image"], "/tmp/scene_001_image.png")

    def test_load_missing_returns_none(self):
        empty_dir = tempfile.mkdtemp()
        mgr = CheckpointManager(empty_dir)
        self.assertIsNone(mgr.load())

    def test_load_corrupt_returns_none(self):
        cp_path = os.path.join(self._tmpdir, "_checkpoint.json")
        with open(cp_path, "w") as f:
            f.write("{invalid json")
        self.assertIsNone(self._mgr.load())


class TestCheckpointRestore(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_restore_context(self):
        ctx = PipelineContext(
            job_id="20260704_abc12345",
            job_dir=self._tmpdir,
            config={"content_type": "short_drama"},
        )
        ctx.completed_stages = ["prompt_expand"]
        ctx.scenes = [{"scene_id": 1}]
        ctx.artifacts = {"1_image": "/tmp/img.png"}
        ctx.created_at = "2026-07-04T10:00:00"
        self._mgr.save(ctx)

        new_ctx = PipelineContext(
            job_id="20260704_abc12345",
            job_dir=self._tmpdir,
            config={},
        )
        result = self._mgr.restore_context(new_ctx)
        self.assertTrue(result)
        self.assertEqual(new_ctx.completed_stages, ["prompt_expand"])
        self.assertEqual(len(new_ctx.scenes), 1)
        self.assertEqual(new_ctx.artifacts["1_image"], "/tmp/img.png")

    def test_restore_empty_returns_false(self):
        ctx = PipelineContext(job_id="empty", job_dir=self._tmpdir, config={})
        result = self._mgr.restore_context(ctx)
        self.assertFalse(result)


class TestCheckpointData(unittest.TestCase):

    def test_defaults(self):
        data = CheckpointData()
        self.assertEqual(data.job_id, "")
        self.assertEqual(data.completed_stages, [])
        self.assertEqual(data.scenes, [])
        self.assertEqual(data.artifacts, {})


if __name__ == "__main__":
    unittest.main()
