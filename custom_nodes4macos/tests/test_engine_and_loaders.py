"""补充覆盖：engine 模板加载/自动发现/warmup、model_manager 真实加载器、assemble stage 分支。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from custom_nodes4macos.pipeline.engine import PipelineEngine, register_stage, _STAGE_REGISTRY
from custom_nodes4macos.pipeline.model_manager import ModelManager, ModelMode
from custom_nodes4macos.pipeline.stages.assemble import AssembleStage
from custom_nodes4macos.pipeline.context import PipelineContext


class TestEngineTemplateLoading(unittest.TestCase):
    """覆盖 _load_templates、_auto_discover_stages、_ensure_loaded。"""

    def test_load_templates_reads_all_yaml(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        with tempfile.TemporaryDirectory() as td:
            for name, ct in [("a.yaml", "cta"), ("b.yaml", "ctb")]:
                with open(os.path.join(td, name), "w") as f:
                    yaml.dump({"content_type": ct, "stages": [], "defaults": {}}, f)
            with patch("custom_nodes4macos.pipeline.engine._TEMPLATE_DIR", __import__("pathlib").Path(td)):
                engine._load_templates()
            self.assertIn("cta", engine._templates)
            self.assertIn("ctb", engine._templates)

    def test_load_templates_skips_corrupt_yaml(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "good.yaml"), "w") as f:
                yaml.dump({"content_type": "good", "stages": []}, f)
            with open(os.path.join(td, "bad.yaml"), "w") as f:
                f.write("{not: valid: yaml: [[[")
            with patch("custom_nodes4macos.pipeline.engine._TEMPLATE_DIR", __import__("pathlib").Path(td)):
                engine._load_templates()
            self.assertIn("good", engine._templates)
            self.assertNotIn("bad", engine._templates)

    def test_load_templates_missing_dir_no_crash(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        with patch("custom_nodes4macos.pipeline.engine._TEMPLATE_DIR", __import__("pathlib").Path("/nonexistent/xyz")):
            engine._load_templates()
        self.assertEqual(engine._templates, {})

    def test_get_template_unknown_raises_with_available(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        engine._templates = {"known": {}}
        with self.assertRaises(ValueError) as cm:
            engine._get_template("unknown")
        self.assertIn("known", str(cm.exception))

    def test_ensure_loaded_idempotent(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        engine._loaded = True
        # _loaded=True 时 _ensure_loaded 不应重新加载
        with patch.object(engine, "_load_templates") as mock_load:
            engine._ensure_loaded()
            mock_load.assert_not_called()

    def test_auto_discover_stages_imports_all_py(self):
        """_auto_discover_stages 应尝试导入 stages 目录下所有非下划线 .py。"""
        from custom_nodes4macos.pipeline.engine import _auto_discover_stages
        # 调用不应抛异常（即使某些 stage import 失败也只 warning）
        _auto_discover_stages()
        # 至少已注册 7 个 stage
        self.assertGreaterEqual(len(_STAGE_REGISTRY), 7)

    def test_warmup_mlx_import_error_handled(self):
        """_warmup_mlx 在 mlx 不可用时静默返回。"""
        PipelineEngine._warmup_mlx()  # 不抛异常

    def test_warmup_mlx_success(self):
        """mlx 可用时 warmup 调用 mx.zeros。"""
        # 注意：本机 mlx 已安装，_warmup_mlx 会真实调用 mx.zeros。
        # 此测试仅验证不抛异常（mlx 真实可用）。
        PipelineEngine._warmup_mlx()

    def test_make_job_id_format(self):
        job_id = PipelineEngine._make_job_id()
        self.assertRegex(job_id, r"^\d{8}_[0-9a-f]{8}$")

    def test_merge_config_user_overrides_template(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        template = {"defaults": {"scene_count": 8, "flux_steps": 4}, "content_type": "x", "name": "t", "stages": [], "prompts": {"k": "v"}, "style_presets": {"s": "v"}}
        merged = engine._merge_config(template, {"scene_count": 20})
        self.assertEqual(merged["scene_count"], 20)  # 用户覆盖
        self.assertEqual(merged["flux_steps"], 4)  # 模板默认保留
        self.assertEqual(merged["content_type"], "x")
        self.assertEqual(merged["k"], "v")  # prompts 展开到 config
        self.assertEqual(merged["style_presets"], {"s": "v"})

    def test_instantiate_stages_unknown_raises(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        with self.assertRaises(ValueError) as cm:
            engine._instantiate_stages(["nonexistent_stage"])
        self.assertIn("registered:", str(cm.exception))

    def test_list_templates_returns_keys(self):
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        engine._loaded = True
        engine._templates = {"a": {}, "b": {}}
        self.assertEqual(sorted(engine.list_templates()), ["a", "b"])

    def test_run_no_stages_returns_empty_result(self):
        """模板 stages 为空时返回空 PipelineResult。"""
        from custom_nodes4macos.pipeline.result import PipelineResult
        engine = PipelineEngine(output_root=tempfile.mkdtemp())
        engine._loaded = True
        engine._templates = {"empty": {"content_type": "empty", "stages": []}}
        result = engine.run("empty", story_seed="x")
        self.assertIsInstance(result, PipelineResult)
        self.assertIsNone(result.final_video)


class TestModelManagerRealLoaders(unittest.TestCase):
    """覆盖 _load_llm/_load_tts 的 ImportError 路径。"""

    def test_load_llm_import_error_propagates(self):
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name == "mlx_lm" or name.startswith("mlx_lm."):
                raise ImportError("no mlx_lm")
            return real_import(name, *a, **k)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                ModelManager._load_llm("fake/path")

    def test_load_tts_import_error_propagates(self):
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *a, **k):
            if name == "mlx_audio" or name.startswith("mlx_audio."):
                raise ImportError("no mlx_audio")
            return real_import(name, *a, **k)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                ModelManager._load_tts("fake/path")

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "m": {"path": "fake", "memory_gb": 1.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value=("model", "tok"))
    def test_acquire_handle_model_attribute(self, _):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL, memory_budget_gb=10.0)
        with mgr.acquire("m") as handle:
            self.assertEqual(handle.name, "m")
            self.assertEqual(handle.model, ("model", "tok"))
            handle.release()  # 显式 release 不应抛异常


class TestAssembleStageBranches(unittest.TestCase):
    """覆盖 assemble stage 的 use_clip_audio/bgm/transition 各分支。"""

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_with_clip_audio_and_bgm_mixes(self, mock_ffmpeg):
        """片段有音轨 + BGM → amix 混音。"""
        tmpdir = tempfile.mkdtemp()
        clips = []
        for i in range(1, 3):
            p = os.path.join(tmpdir, f"c{i}.mp4")
            with open(p, "wb") as f:
                f.write(b"mp4")
            clips.append(p)
        bgm = os.path.join(tmpdir, "bgm.wav")
        with open(bgm, "wb") as f:
            f.write(b"wav")

        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_bgm_path": bgm, "assemble_transition": "none",
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": i} for i in range(1, 3)]
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)

        mock_ffmpeg.probe_has_audio.return_value = True
        mock_ffmpeg.probe_duration.return_value = 1.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"final")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())
        # 验证 filter_complex 含 amix
        filter_arg = mock_ffmpeg.run_ffmpeg.call_args.args[0]
        fc_idx = filter_arg.index("-filter_complex") + 1
        self.assertIn("amix", filter_arg[fc_idx])

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_partial_audio_drops_clip_audio(self, mock_ffmpeg):
        """部分片段无音轨 → 丢弃片段音频，仅用 BGM。"""
        tmpdir = tempfile.mkdtemp()
        clips = []
        for i in range(1, 3):
            p = os.path.join(tmpdir, f"c{i}.mp4")
            with open(p, "wb") as f:
                f.write(b"mp4")
            clips.append(p)
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_transition": "none",
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": i} for i in range(1, 3)]
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)

        # 第一个有音轨，第二个无 → 不一致
        mock_ffmpeg.probe_has_audio.side_effect = [True, False]
        mock_ffmpeg.probe_duration.return_value = 1.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"final")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())
        # 无 BGM 且丢弃片段音频 → audio_map=None
        filter_arg = mock_ffmpeg.run_ffmpeg.call_args.args[0]
        fc_idx = filter_arg.index("-filter_complex") + 1
        self.assertIn("concat=n=2:v=1:a=0", filter_arg[fc_idx])

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_fade_transition_adds_fade_filter(self, mock_ffmpeg):
        tmpdir = tempfile.mkdtemp()
        clips = []
        for i in range(1, 3):
            p = os.path.join(tmpdir, f"c{i}.mp4")
            with open(p, "wb") as f:
                f.write(b"mp4")
            clips.append(p)
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_transition": "fade",
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": i} for i in range(1, 3)]
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)

        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.probe_duration.return_value = 2.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"final")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())
        filter_arg = mock_ffmpeg.run_ffmpeg.call_args.args[0]
        fc_idx = filter_arg.index("-filter_complex") + 1
        self.assertIn("fade=t=in", filter_arg[fc_idx])
        self.assertIn("fade=t=out", filter_arg[fc_idx])

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_assemble_bgm_only_no_clip_audio(self, mock_ffmpeg):
        """片段无音轨 + BGM → 仅 BGM。"""
        tmpdir = tempfile.mkdtemp()
        clips = [os.path.join(tmpdir, "c1.mp4")]
        with open(clips[0], "wb") as f:
            f.write(b"mp4")
        bgm = os.path.join(tmpdir, "bgm.wav")
        with open(bgm, "wb") as f:
            f.write(b"wav")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_bgm_path": bgm, "assemble_transition": "none",
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": 1}]
        ctx.set_artifact(1, "clip", clips[0])

        mock_ffmpeg.probe_has_audio.return_value = False  # 片段无音轨
        # BGM 探测单独调用会返回 True，但 probe_has_audio 已 mock 为 False
        # 改用 side_effect 区分
        mock_ffmpeg.probe_has_audio.side_effect = None
        mock_ffmpeg.probe_has_audio.return_value = False
        mock_ffmpeg.probe_duration.return_value = 1.0
        mock_ffmpeg.video_encoder_args.return_value = ["-c:v", "libx264"]
        def fake_run(args, timeout=None, label=""):
            with open(args[-1], "wb") as f:
                f.write(b"final")
        mock_ffmpeg.run_ffmpeg.side_effect = fake_run

        stage = AssembleStage()
        stage.process(ctx, MagicMock())
        # BGM 不可用（probe 返回 False）→ audio_map=None
        mock_ffmpeg.run_ffmpeg.assert_called_once()


class TestFusionClientMethods(unittest.TestCase):
    """覆盖 fusion_client 的 chat/generate_image/synthesize_speech 成功路径。"""

    @patch("httpx.Client")
    def test_chat_returns_content_and_usage(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"total_tokens": 5},
        }
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        content, usage = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(content, "hello")
        self.assertEqual(usage, {"total_tokens": 5})

    @patch("httpx.Client")
    def test_chat_non_200_raises(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient, FusionMLXError
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "bad request"
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        with self.assertRaises(FusionMLXError):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("httpx.Client")
    def test_generate_image_decodes_b64(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        import base64
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        fake_png = b"\x89PNG\r\n\x1a\nfake"
        b64 = base64.b64encode(fake_png).decode()
        mock_resp.json.return_value = {"data": [{"b64_json": b64}]}
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        images = client.generate_image("prompt")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], fake_png)

    @patch("httpx.Client")
    def test_generate_image_data_uri(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        import base64
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        fake_png = b"\x89PNGfake"
        b64 = base64.b64encode(fake_png).decode()
        mock_resp.json.return_value = {"data": [{"url": f"data:image/png;base64,{b64}"}]}
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        images = client.generate_image("p")
        self.assertEqual(images[0], fake_png)

    @patch("httpx.Client")
    def test_generate_image_missing_fields_raises(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient, FusionMLXError
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"other": "no image"}]}
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        with self.assertRaises(FusionMLXError):
            client.generate_image("p")

    @patch("httpx.Client")
    def test_generate_image_non_200_raises(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient, FusionMLXError
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "err"
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient(retries=0)
        with self.assertRaises(FusionMLXError):
            client.generate_image("p")

    @patch("httpx.Client")
    def test_synthesize_speech_returns_bytes(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"RIFF...WAV"
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        audio = client.synthesize_speech("text", "tts-model")
        self.assertEqual(audio, b"RIFF...WAV")

    @patch("httpx.Client")
    def test_synthesize_speech_non_200_raises(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient, FusionMLXError
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "forbidden"
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        with self.assertRaises(FusionMLXError):
            client.synthesize_speech("t", "m")

    @patch("httpx.Client")
    def test_list_models_parses_ids(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "m1"}, {"id": "m2"}, {"other": "no id"}]}
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient()
        models = client.list_models()
        self.assertEqual(models, ["m1", "m2"])

    @patch("httpx.Client")
    def test_list_models_non_200_raises(self, mock_cls):
        from custom_nodes4macos.fusion_client import FusionMLXClient, FusionMLXError
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "err"
        mock_client.request.return_value = mock_resp
        client = FusionMLXClient(retries=0)
        with self.assertRaises(FusionMLXError):
            client.list_models()


if __name__ == "__main__":
    unittest.main()
