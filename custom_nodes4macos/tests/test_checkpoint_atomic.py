"""CheckpointManager 原子性与边界测试。

覆盖腐败容忍、字段白名单、未知字段忽略、
默认值、restore_context 边界。

补充 REVIEW_REPORT P2-1：checkpoint 非原子写入（测试现状，标记 todo）。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from custom_nodes4macos.pipeline.checkpoint import CheckpointManager, CheckpointData
from custom_nodes4macos.pipeline.context import PipelineContext


class TestCheckpointSaveLoadRoundtrip(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_full_roundtrip_preserves_all_fields(self):
        ctx = PipelineContext(
            job_id="rt_test",
            job_dir=self._tmpdir,
            config={
                "content_type": "series",
                "template_name": "series_drama",
                "overrides": {"episode_count": 10},
            },
        )
        ctx.completed_stages = ["story_ingest", "prompt_expand"]
        ctx.scenes = [
            {"scene_id": 1, "visual_prompt": "v1", "episode_id": 1},
            {"scene_id": 2, "visual_prompt": "v2", "episode_id": 1},
        ]
        ctx.artifacts = {"1_image": "/tmp/1.png", "2_image": "/tmp/2.png"}
        ctx.created_at = "2026-07-04T08:00:00"

        self._mgr.save(ctx)
        loaded = self._mgr.load()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.job_id, "rt_test")
        self.assertEqual(loaded.content_type, "series")
        self.assertEqual(loaded.template_name, "series_drama")
        self.assertEqual(loaded.completed_stages, ["story_ingest", "prompt_expand"])
        self.assertEqual(len(loaded.scenes), 2)
        self.assertEqual(loaded.scenes[0]["episode_id"], 1)
        self.assertEqual(loaded.artifacts["1_image"], "/tmp/1.png")
        self.assertEqual(loaded.config_overrides["episode_count"], 10)
        self.assertEqual(loaded.created_at, "2026-07-04T08:00:00")

    def test_updated_at_changes_on_each_save(self):
        import time
        ctx = PipelineContext(job_id="ts", job_dir=self._tmpdir, config={"content_type": "x"})
        self._mgr.save(ctx)
        first = self._mgr.load().updated_at
        time.sleep(0.05)
        self._mgr.save(ctx)
        second = self._mgr.load().updated_at
        self.assertNotEqual(first, second)


class TestCheckpointCorruptionTolerance(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_empty_file_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            f.write("")
        self.assertIsNone(self._mgr.load())

    def test_partial_json_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            f.write('{"job_id": "partial"')  # 缺闭合
        self.assertIsNone(self._mgr.load())

    def test_json_array_root_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump(["not", "a", "dict"], f)
        self.assertIsNone(self._mgr.load())

    def test_json_string_root_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump("just a string", f)
        self.assertIsNone(self._mgr.load())

    def test_json_number_root_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump(42, f)
        self.assertIsNone(self._mgr.load())

    def test_null_root_returns_none(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump(None, f)
        self.assertIsNone(self._mgr.load())


class TestCheckpointFieldWhitelist(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_unknown_fields_ignored(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump({
                "job_id": "test",
                "unknown_field": "ignored",
                "another_unknown": 123,
                "completed_stages": ["a"],
            }, f)
        loaded = self._mgr.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.job_id, "test")
        self.assertFalse(hasattr(loaded, "unknown_field"))

    def test_missing_optional_fields_use_defaults(self):
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump({"job_id": "minimal"}, f)
        loaded = self._mgr.load()
        self.assertEqual(loaded.job_id, "minimal")
        self.assertEqual(loaded.completed_stages, [])
        self.assertEqual(loaded.scenes, [])
        self.assertEqual(loaded.artifacts, {})
        self.assertEqual(loaded.config_overrides, {})

    def test_extra_nested_data_ignored(self):
        """嵌套的未知字段不破坏已知字段。"""
        with open(os.path.join(self._tmpdir, "_checkpoint.json"), "w") as f:
            json.dump({
                "job_id": "test",
                "completed_stages": ["a", "b"],
                "scenes": [{"scene_id": 1, "extra": "ok"}],
            }, f)
        loaded = self._mgr.load()
        self.assertEqual(loaded.completed_stages, ["a", "b"])
        # scenes 内部的 extra 字段保留（dataclass 只过滤顶层）
        self.assertEqual(loaded.scenes[0]["extra"], "ok")


class TestCheckpointRestoreContext(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CheckpointManager(self._tmpdir)

    def test_restore_overwrites_empty_ctx(self):
        ctx = PipelineContext(job_id="test", job_dir=self._tmpdir, config={})
        ctx.completed_stages = ["old"]
        ctx.scenes = [{"scene_id": 99}]

        # 先存一个不同的状态
        cp_ctx = PipelineContext(job_id="test", job_dir=self._tmpdir, config={"content_type": "x"})
        cp_ctx.completed_stages = ["new"]
        cp_ctx.scenes = [{"scene_id": 1}]
        self._mgr.save(cp_ctx)

        result = self._mgr.restore_context(ctx)
        self.assertTrue(result)
        self.assertEqual(ctx.completed_stages, ["new"])
        self.assertEqual(ctx.scenes, [{"scene_id": 1}])

    def test_restore_returns_false_when_no_checkpoint(self):
        empty_dir = tempfile.mkdtemp()
        mgr = CheckpointManager(empty_dir)
        ctx = PipelineContext(job_id="x", job_dir=empty_dir, config={})
        self.assertFalse(mgr.restore_context(ctx))

    def test_restore_returns_false_when_empty_completed_stages(self):
        """completed_stages 为空时 restore_context 返回 False。"""
        ctx = PipelineContext(job_id="t", job_dir=self._tmpdir, config={"content_type": "x"})
        self._mgr.save(ctx)  # completed_stages 为空
        new_ctx = PipelineContext(job_id="t", job_dir=self._tmpdir, config={})
        result = self._mgr.restore_context(new_ctx)
        self.assertFalse(result)

    def test_restore_returns_true_when_completed_stages_present(self):
        ctx = PipelineContext(job_id="t", job_dir=self._tmpdir, config={"content_type": "x"})
        ctx.completed_stages = ["prompt_expand"]
        self._mgr.save(ctx)
        new_ctx = PipelineContext(job_id="t", job_dir=self._tmpdir, config={})
        result = self._mgr.restore_context(new_ctx)
        self.assertTrue(result)


class TestCheckpointDataDefaults(unittest.TestCase):

    def test_all_defaults(self):
        d = CheckpointData()
        self.assertEqual(d.job_id, "")
        self.assertEqual(d.content_type, "")
        self.assertEqual(d.template_name, "")
        self.assertEqual(d.completed_stages, [])
        self.assertEqual(d.scenes, [])
        self.assertEqual(d.artifacts, {})
        self.assertEqual(d.config_overrides, {})
        self.assertEqual(d.created_at, "")
        self.assertEqual(d.updated_at, "")

    def test_defaults_are_independent_instances(self):
        """dataclass field(default_factory) 应为独立实例。"""
        d1 = CheckpointData()
        d2 = CheckpointData()
        d1.completed_stages.append("x")
        self.assertEqual(d2.completed_stages, [])  # d2 不受影响


class TestCheckpointSaveCreatesDir(unittest.TestCase):

    def test_save_creates_missing_job_dir(self):
        """job_dir 不存在时 save 自动创建。"""
        tmpdir = tempfile.mkdtemp()
        nested = os.path.join(tmpdir, "a", "b", "c")
        mgr = CheckpointManager(nested)
        ctx = PipelineContext(job_id="t", job_dir=nested, config={"content_type": "x"})
        mgr.save(ctx)
        self.assertTrue(os.path.exists(os.path.join(nested, "_checkpoint.json")))


if __name__ == "__main__":
    unittest.main()
