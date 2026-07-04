"""DreamFactory 节点 produce 全流程测试。

覆盖 config_overrides 注入、resume 路径、checkpoint 读取、
错误路径、参数传递。

补充 REVIEW_REPORT P2：nodes/dream_factory 仅测元数据，未测 produce 全流程。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.nodes.dream_factory import FusionMLXDreamFactory


class TestDreamFactoryInputTypes(unittest.TestCase):

    def test_all_seven_content_types_declared(self):
        inputs = FusionMLXDreamFactory.INPUT_TYPES()
        ct_list = inputs["required"]["content_type"][0]
        expected = {
            "short_drama", "ad_drama", "puppet_show",
            "medium_video", "series", "digital_human", "digital_human_live",
        }
        self.assertEqual(set(ct_list), expected)

    def test_optional_inputs_present(self):
        inputs = FusionMLXDreamFactory.INPUT_TYPES()
        opt = inputs["optional"]
        for key in ["episode_title", "scene_count", "style_preset", "story_file",
                    "episode_count", "avatar_reference", "resume_job_id", "config_overrides"]:
            self.assertIn(key, opt)

    def test_class_attributes(self):
        self.assertEqual(FusionMLXDreamFactory.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(FusionMLXDreamFactory.RETURN_NAMES, ("video_path", "scenes_json"))
        self.assertEqual(FusionMLXDreamFactory.FUNCTION, "produce")
        self.assertTrue(FusionMLXDreamFactory.OUTPUT_NODE)


class _FakeResult:
    """真实属性对象，避免 MagicMock 自动属性在 json.dumps 时抛 TypeError。"""
    def __init__(self, job_id, job_dir, final_video):
        self.job_id = job_id
        self.job_dir = job_dir
        self.final_video = final_video


class TestProduceNewJob(unittest.TestCase):

    def _setup_mock_engine(self, mock_cls, job_id="job_test", final_video="/tmp/out.mp4"):
        job_dir = tempfile.mkdtemp()
        mock_result = _FakeResult(job_id=job_id, job_dir=job_dir, final_video=final_video)
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance
        cp_path = os.path.join(job_dir, "_checkpoint.json")
        with open(cp_path, "w") as f:
            json.dump({"scenes": [{"scene_id": 1}], "completed_stages": ["prompt_expand"]}, f)
        return mock_instance, mock_result

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_with_seed_returns_video_and_scenes(self, mock_cls):
        mock_instance, mock_result = self._setup_mock_engine(mock_cls)
        node = FusionMLXDreamFactory()
        video_path, scenes_json = node.produce(
            content_type="short_drama",
            story_seed="深夜破庙",
        )
        self.assertEqual(video_path, "/tmp/out.mp4")
        data = json.loads(scenes_json)
        self.assertEqual(data["job_id"], "job_test")
        self.assertEqual(data["content_type"], "short_drama")
        self.assertEqual(len(data["scenes"]), 1)

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_passes_overrides_to_engine(self, mock_cls):
        mock_instance, _ = self._setup_mock_engine(mock_cls)
        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="seed",
            episode_title="测试集",
            scene_count=10,
            style_preset="水墨悬疑",
            episode_count=30,
            avatar_reference="/tmp/a.png",
        )
        call_kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(call_kwargs["story_seed"], "seed")
        self.assertEqual(call_kwargs["episode_title"], "测试集")
        self.assertEqual(call_kwargs["scene_count"], 10)
        self.assertEqual(call_kwargs["style_preset"], "水墨悬疑")
        self.assertEqual(call_kwargs["episode_count"], 30)
        self.assertEqual(call_kwargs["avatar_reference"], "/tmp/a.png")

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_with_config_overrides_json(self, mock_cls):
        mock_instance, _ = self._setup_mock_engine(mock_cls)
        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="seed",
            config_overrides='{"custom_key": "custom_val", "scene_count": 20}',
        )
        call_kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(call_kwargs["custom_key"], "custom_val")
        self.assertEqual(call_kwargs["scene_count"], 20)

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_config_overrides_bad_json_ignored(self, mock_cls):
        mock_instance, _ = self._setup_mock_engine(mock_cls)
        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="seed",
            config_overrides="{invalid json",
        )
        mock_instance.run.assert_called_once()

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_config_overrides_non_dict_ignored(self, mock_cls):
        mock_instance, _ = self._setup_mock_engine(mock_cls)
        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="seed",
            config_overrides='["not", "a", "dict"]',
        )
        mock_instance.run.assert_called_once()


class TestProduceResume(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_with_resume_job_id_uses_resume_from(self, mock_cls):
        job_dir = tempfile.mkdtemp()
        mock_result = _FakeResult("resumed", job_dir, "/tmp/resumed.mp4")
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump({"scenes": [], "completed_stages": []}, f)

        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="",  # resume 时 story_seed 可空
            resume_job_id="20260704_abc123",
        )
        call_kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(call_kwargs["resume_from"], "20260704_abc123")

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_resume_strips_whitespace(self, mock_cls):
        job_dir = tempfile.mkdtemp()
        mock_result = _FakeResult("r", job_dir, "")
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump({"scenes": [], "completed_stages": []}, f)

        node = FusionMLXDreamFactory()
        node.produce(
            content_type="short_drama",
            story_seed="",
            resume_job_id="  20260704_xyz  ",
        )
        call_kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(call_kwargs["resume_from"], "20260704_xyz")


class TestProduceErrorPaths(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_no_seed_no_file_raises(self, _):
        """新任务但无 story_seed 且无 story_file 时抛 ValueError。"""
        node = FusionMLXDreamFactory()
        with self.assertRaises(ValueError) as cm:
            node.produce(content_type="short_drama", story_seed="", story_file="")
        self.assertIn("story_seed or story_file", str(cm.exception))

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_seed_only_whitespace_raises(self, _):
        """story_seed 仅空白时视为空。"""
        node = FusionMLXDreamFactory()
        with self.assertRaises(ValueError):
            node.produce(content_type="short_drama", story_seed="   \n\t  ")

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_file_substitutes_for_seed(self, mock_cls):
        """有 story_file 时即使 story_seed 空，也能调用引擎。"""
        job_dir = tempfile.mkdtemp()
        mock_result = _FakeResult("f", job_dir, "/tmp/f.mp4")
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump({"scenes": [], "completed_stages": []}, f)

        node = FusionMLXDreamFactory()
        node.produce(
            content_type="series",
            story_seed="",
            story_file="/tmp/story.pdf",
        )
        mock_instance.run.assert_called_once()


class TestProduceCheckpointRead(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_reads_scenes_from_checkpoint(self, mock_cls):
        """produce 从 _checkpoint.json 读取 scenes 和 completed_stages。"""
        job_dir = tempfile.mkdtemp()
        mock_result = _FakeResult("cp_test", job_dir, "/tmp/cp.mp4")
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance

        cp_data = {
            "scenes": [{"scene_id": 1, "visual_prompt": "v1"}, {"scene_id": 2, "visual_prompt": "v2"}],
            "completed_stages": ["prompt_expand", "image_generate"],
        }
        with open(os.path.join(job_dir, "_checkpoint.json"), "w") as f:
            json.dump(cp_data, f)

        node = FusionMLXDreamFactory()
        _, scenes_json = node.produce(content_type="short_drama", story_seed="x")
        data = json.loads(scenes_json)
        self.assertEqual(len(data["scenes"]), 2)
        self.assertEqual(data["scenes"][0]["scene_id"], 1)
        self.assertEqual(data["completed_stages"], ["prompt_expand", "image_generate"])

    @patch("custom_nodes4macos.pipeline.engine.PipelineEngine")
    def test_produce_missing_checkpoint_returns_empty_scenes(self, mock_cls):
        """checkpoint 文件不存在/损坏时 scenes 为空，不抛异常。"""
        job_dir = tempfile.mkdtemp()  # 无 _checkpoint.json
        mock_result = _FakeResult("no_cp", job_dir, "/tmp/x.mp4")
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_cls.return_value = mock_instance

        node = FusionMLXDreamFactory()
        _, scenes_json = node.produce(content_type="short_drama", story_seed="x")
        data = json.loads(scenes_json)
        self.assertEqual(data["scenes"], [])


if __name__ == "__main__":
    unittest.main()
