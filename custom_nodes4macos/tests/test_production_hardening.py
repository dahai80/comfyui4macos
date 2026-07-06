from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml


class TestDreamFactoryNode(unittest.TestCase):

    def test_output_node_true(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        self.assertTrue(getattr(FusionMLXDreamFactory, "OUTPUT_NODE", False))

    def test_input_types_has_all_content_types(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        inputs = FusionMLXDreamFactory.INPUT_TYPES()
        ct = inputs["required"]["content_type"][0]
        expected = [
            "short_drama", "ad_drama", "puppet_show",
            "medium_video", "series", "digital_human", "digital_human_live",
        ]
        for t in expected:
            self.assertIn(t, ct)

    def test_return_types(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        self.assertEqual(FusionMLXDreamFactory.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(FusionMLXDreamFactory.RETURN_NAMES, ("video_path", "scenes_json"))

    def test_produce_requires_seed_or_file(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        node = FusionMLXDreamFactory()
        with patch("custom_nodes4macos.pipeline.engine.PipelineEngine") as mock_engine:
            with self.assertRaises(ValueError):
                node.produce(content_type="short_drama", story_seed="")

    def test_produce_with_seed_calls_engine(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        node = FusionMLXDreamFactory()
        mock_result = MagicMock()
        mock_result.job_id = "test_job"
        mock_result.job_dir = tempfile.mkdtemp()
        mock_result.final_video = "/tmp/test.mp4"
        with patch("custom_nodes4macos.pipeline.engine.PipelineEngine") as mock_cls:
            mock_cls.return_value.run.return_value = mock_result
            cp_path = os.path.join(mock_result.job_dir, "_checkpoint.json")
            with open(cp_path, "w") as f:
                json.dump({"scenes": [], "completed_stages": []}, f)
            result = node.produce(
                content_type="short_drama",
                story_seed="测试故事",
            )
            self.assertEqual(result[0], "/tmp/test.mp4")
            self.assertIn("test_job", result[1])

    def test_config_overrides_bad_json_ignored(self):
        from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory
        node = FusionMLXDreamFactory()
        mock_result = MagicMock()
        mock_result.job_id = "test2"
        mock_result.job_dir = tempfile.mkdtemp()
        mock_result.final_video = ""
        with patch("custom_nodes4macos.pipeline.engine.PipelineEngine") as mock_cls:
            mock_cls.return_value.run.return_value = mock_result
            cp_path = os.path.join(mock_result.job_dir, "_checkpoint.json")
            with open(cp_path, "w") as f:
                json.dump({"scenes": [], "completed_stages": []}, f)
            result = node.produce(
                content_type="short_drama",
                story_seed="测试",
                config_overrides="{invalid json",
            )
            mock_cls.return_value.run.assert_called_once()


class TestTemplateYAMLValidation(unittest.TestCase):

    def _template_dir(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "pipeline", "templates",
        )

    def test_all_templates_parseable(self):
        td = self._template_dir()
        for fname in os.listdir(td):
            if not fname.endswith(".yaml"):
                continue
            with open(os.path.join(td, fname), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.assertIsInstance(data, dict, f"{fname} is not a dict")
            self.assertIn("content_type", data, f"{fname} missing content_type")
            self.assertIn("stages", data, f"{fname} missing stages")
            self.assertIsInstance(data["stages"], list, f"{fname} stages not a list")

    def test_short_drama_scene_count(self):
        td = self._template_dir()
        with open(os.path.join(td, "short_drama.yaml"), "r") as f:
            data = yaml.safe_load(f)
        sc = data["defaults"]["scene_count"]
        self.assertGreaterEqual(sc, 12, f"short_drama scene_count={sc} too low for 2min target")

    def test_series_scenes_per_episode(self):
        td = self._template_dir()
        with open(os.path.join(td, "series.yaml"), "r") as f:
            data = yaml.safe_load(f)
        defaults = data["defaults"]
        self.assertNotIn(
            "scenes_per_episode", defaults,
            "series.yaml 应使用 scene_count（每集场景数），scenes_per_episode 为死配置已移除",
        )
        sc = defaults["scene_count"]
        self.assertGreaterEqual(sc, 8, f"series scene_count={sc} 低于每集 8 场景下限")
        self.assertLessEqual(sc, 12, f"series scene_count={sc} 高于每集 12 场景上限")
        ep = defaults["episode_count"]
        self.assertGreaterEqual(ep, 1, f"series episode_count={ep} 应为多集")
        total = sc * ep
        self.assertGreaterEqual(total, 8, f"series 总场景数 {total} 过少")


class TestCheckpointValidation(unittest.TestCase):

    def test_load_corrupt_json_returns_none(self):
        from custom_nodes4macos.pipeline.checkpoint import CheckpointManager
        tmpdir = tempfile.mkdtemp()
        cp_path = os.path.join(tmpdir, "_checkpoint.json")
        with open(cp_path, "w") as f:
            f.write("{invalid json")
        mgr = CheckpointManager(tmpdir)
        result = mgr.load()
        self.assertIsNone(result)

    def test_load_non_dict_root_returns_none(self):
        from custom_nodes4macos.pipeline.checkpoint import CheckpointManager
        tmpdir = tempfile.mkdtemp()
        cp_path = os.path.join(tmpdir, "_checkpoint.json")
        with open(cp_path, "w") as f:
            json.dump(["not", "a", "dict"], f)
        mgr = CheckpointManager(tmpdir)
        result = mgr.load()
        self.assertIsNone(result)

    def test_load_unknown_fields_ignored(self):
        from custom_nodes4macos.pipeline.checkpoint import CheckpointManager
        tmpdir = tempfile.mkdtemp()
        cp_path = os.path.join(tmpdir, "_checkpoint.json")
        with open(cp_path, "w") as f:
            json.dump({
                "job_id": "test",
                "unknown_field": "should be ignored",
                "completed_stages": ["prompt_expand"],
            }, f)
        mgr = CheckpointManager(tmpdir)
        result = mgr.load()
        self.assertIsNotNone(result)
        self.assertEqual(result.job_id, "test")
        self.assertFalse(hasattr(result, "unknown_field") and result.unknown_field == "should be ignored")


class TestFusionClientRetry(unittest.TestCase):

    def test_429_retry(self):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_429.headers = {"retry-after": "0"}
            resp_200 = MagicMock()
            resp_200.status_code = 200
            mock_client.request.side_effect = [resp_429, resp_200]
            client = FusionMLXClient()
            result = client._request("GET", "/test")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(mock_client.request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
