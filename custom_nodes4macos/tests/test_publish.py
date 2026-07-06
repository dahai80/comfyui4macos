from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.engine import (
    PipelineEngine,
    _STAGE_REGISTRY,
    register_stage,
)
from custom_nodes4macos.pipeline.stage import Stage, StageInfo
from custom_nodes4macos.pipeline.stages.publish import PublishStage
from custom_nodes4macos.publisher import (
    DouyinPublisher,
    PublishConfigError,
    PublishDependencyError,
    PublishMeta,
)


def _make_ctx(config: dict, final_video: str | None = None) -> PipelineContext:
    tmpdir = tempfile.mkdtemp(prefix="publish_test_")
    ctx = PipelineContext(job_id="t", job_dir=tmpdir, config=dict(config))
    if final_video:
        ctx.set_artifact(0, "final", final_video)
    return ctx


def _make_video(tmpdir: str, name: str = "ep.mp4", size: int = 128) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    return path


class TestPublishStageInfo(unittest.TestCase):

    def test_info(self):
        info = PublishStage.info()
        self.assertEqual(info.name, "publish")
        self.assertEqual(info.input_kinds, ["final"])
        self.assertEqual(info.output_kinds, ["publish"])
        self.assertEqual(info.model_requirements, [])


class TestPublishStageDisabled(unittest.TestCase):

    def test_disabled_skips_without_manifest(self):
        ctx = _make_ctx({"publish_enabled": False})
        PublishStage().process(ctx, MagicMock())
        self.assertIsNone(ctx.get_artifact(0, "publish"))
        manifest = os.path.join(ctx.job_dir, "publish_manifest.json")
        self.assertFalse(os.path.exists(manifest))

    def test_skip_if_completed(self):
        ctx = _make_ctx({"publish_enabled": True})
        ctx.completed_stages = ["publish"]
        PublishStage().process(ctx, MagicMock())
        self.assertIsNone(ctx.get_artifact(0, "publish"))

    def test_unsupported_platform_skips(self):
        ctx = _make_ctx({"publish_enabled": True, "publish_platform": "xiaohongshu"})
        PublishStage().process(ctx, MagicMock())
        self.assertIsNone(ctx.get_artifact(0, "publish"))

    def test_no_final_video_skips(self):
        ctx = _make_ctx({"publish_enabled": True, "publish_dry_run": True})
        PublishStage().process(ctx, MagicMock())
        self.assertIsNone(ctx.get_artifact(0, "publish"))


class TestPublishStageDryRun(unittest.TestCase):

    def test_dry_run_writes_manifest_and_artifact(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": True,
            "story_title": "画皮",
            "publish_tags": ["鬼故事", "民间传说"],
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        PublishStage().process(ctx, MagicMock())

        manifest_path = ctx.get_artifact(0, "publish")
        self.assertTrue(manifest_path and os.path.exists(manifest_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["platform"], "douyin")
        self.assertEqual(len(payload["results"]), 1)
        r = payload["results"][0]
        self.assertEqual(r["status"], "dry_run")
        self.assertEqual(r["title"], "画皮")
        self.assertEqual(r["tags"], ["鬼故事", "民间传说"])
        self.assertEqual(r["video_path"], video)
        self.assertTrue(os.path.exists(r["manifest_path"]))

    def test_dry_run_title_falls_back_to_filename(self):
        ctx = _make_ctx({"publish_enabled": True, "publish_dry_run": True})
        video = _make_video(ctx.job_dir, name="alone.mp4")
        ctx.set_artifact(0, "final", video)

        PublishStage().process(ctx, MagicMock())

        manifest_path = ctx.get_artifact(0, "publish")
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["results"][0]["title"], "alone")

    def test_publish_title_overrides_story_title(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": True,
            "story_title": "ignored",
            "publish_title": "自定义标题",
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["results"][0]["title"], "自定义标题")


class TestPublishStageLiveGuard(unittest.TestCase):

    def test_live_without_cookies_skips_video(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": False,
            "publish_cookies_path": "",
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["results"][0]["status"], "skipped")
        self.assertIn("cookies", payload["results"][0]["error"])

    def test_dep_missing_propagates_to_manifest(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": False,
            "publish_cookies_path": "/whatever.json",
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        with patch.object(
            DouyinPublisher, "upload_draft",
            side_effect=PublishDependencyError("playwright not installed"),
        ):
            PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertTrue(payload["dep_missing"])
        self.assertEqual(payload["results"][0]["status"], "skipped")
        self.assertIn("playwright", payload["results"][0]["error"])

    def test_live_launch_failure_does_not_crash_pipeline(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": False,
            "publish_cookies_path": "/whatever.json",
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        with patch.object(
            DouyinPublisher, "upload_draft",
            side_effect=RuntimeError("browser launch failed"),
        ):
            PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertFalse(payload["dep_missing"])
        self.assertEqual(payload["results"][0]["status"], "failed")
        self.assertIn("browser launch", payload["results"][0]["error"])


class TestPublishStageMultiEpisode(unittest.TestCase):

    def test_publish_all_episodes_iterates_episode_finals(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": True,
            "publish_all_episodes": True,
            "story_title": "山村诡事",
        })
        ep1 = _make_video(ctx.job_dir, name="ep1.mp4")
        ep2 = _make_video(ctx.job_dir, name="ep2.mp4")
        ep3 = _make_video(ctx.job_dir, name="ep3.mp4")
        ctx.config["_episode_finals"] = [ep1, ep2, ep3]

        PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(len(payload["results"]), 3)
        titles = [r["title"] for r in payload["results"]]
        self.assertEqual(titles, ["山村诡事 第1集", "山村诡事 第2集", "山村诡事 第3集"])
        for r in payload["results"]:
            self.assertEqual(r["status"], "dry_run")

    def test_publish_all_episodes_falls_back_to_final(self):
        ctx = _make_ctx({
            "publish_enabled": True,
            "publish_dry_run": True,
            "publish_all_episodes": True,
        })
        video = _make_video(ctx.job_dir)
        ctx.set_artifact(0, "final", video)

        PublishStage().process(ctx, MagicMock())

        with open(ctx.get_artifact(0, "publish"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["video_path"], video)


class TestDouyinPublisherUnit(unittest.TestCase):

    def test_dry_run_writes_manifest(self):
        tmpdir = tempfile.mkdtemp(prefix="dh_pub_")
        video = _make_video(tmpdir, name="clip.mp4")
        result = DouyinPublisher().upload_draft(
            video_path=video,
            meta=PublishMeta(title="测试标题", tags=["tag1"]),
            cookies_path="",
            dry_run=True,
            manifest_dir=tmpdir,
        )
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.title, "测试标题")
        self.assertTrue(os.path.exists(result.manifest_path))
        with open(result.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "dry_run")
        self.assertEqual(data["title"], "测试标题")

    def test_missing_video_raises_config_error(self):
        with self.assertRaises(PublishConfigError):
            DouyinPublisher().upload_draft(
                video_path="/nonexistent/x.mp4",
                meta=PublishMeta(title="x"),
                cookies_path="",
                dry_run=True,
            )

    def test_live_without_cookies_raises_config_error(self):
        tmpdir = tempfile.mkdtemp(prefix="dh_pub_")
        video = _make_video(tmpdir)
        with self.assertRaises(PublishConfigError):
            DouyinPublisher().upload_draft(
                video_path=video,
                meta=PublishMeta(title="x"),
                cookies_path="/nonexistent/cookies.json",
                dry_run=False,
            )

    def test_live_playwright_import_blocked_raises_dep_error(self):
        tmpdir = tempfile.mkdtemp(prefix="dh_pub_")
        cookies_path = os.path.join(tmpdir, "c.json")
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump([{"name": "sid", "value": "x", "domain": ".douyin.com"}], f)
        video = _make_video(tmpdir)

        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("playwright"):
                raise ImportError("simulated missing playwright")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked):
            with self.assertRaises(PublishDependencyError):
                DouyinPublisher().upload_draft(
                    video_path=video,
                    meta=PublishMeta(title="x"),
                    cookies_path=cookies_path,
                    dry_run=False,
                )

    def test_load_cookies_normalizes_domain(self):
        tmpdir = tempfile.mkdtemp(prefix="dh_cookie_")
        cookies_path = os.path.join(tmpdir, "c.json")
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump([
                {"name": "a", "value": "1"},
                {"name": "b", "value": "2", "domain": ".douyin.com", "path": "/", "expires": 999},
            ], f)
        cookies = DouyinPublisher._load_cookies(cookies_path)
        self.assertEqual(len(cookies), 2)
        self.assertEqual(cookies[0]["domain"], ".douyin.com")
        self.assertEqual(cookies[1]["expires"], 999)


class _StubFinalStage(Stage):
    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="stub_final",
            description="stub final producer for publish integration test",
            output_kinds=["final"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return
        final_path = ctx.artifact_path(0, "final")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
        ctx.set_artifact(0, "final", final_path)
        ctx.update_progress("stub_final", 1, 1)


class TestPublishEngineIntegration(unittest.TestCase):

    def setUp(self):
        self._orig_registry = dict(_STAGE_REGISTRY)
        _STAGE_REGISTRY.clear()
        register_stage(_StubFinalStage)
        register_stage(PublishStage)
        self._tmpdir = tempfile.mkdtemp(prefix="publish_e2e_")

    def tearDown(self):
        _STAGE_REGISTRY.clear()
        _STAGE_REGISTRY.update(self._orig_registry)

    def _engine_with_template(self, stages: list[str], defaults: dict) -> PipelineEngine:
        engine = PipelineEngine(output_root=os.path.join(self._tmpdir, "output"))
        engine._templates = {
            "publish_e2e": {
                "name": "publish_e2e_tpl",
                "content_type": "publish_e2e",
                "stages": stages,
                "defaults": defaults,
            }
        }
        engine._loaded = True
        return engine

    def test_engine_run_publish_dry_run_produces_manifest(self):
        engine = self._engine_with_template(
            stages=["stub_final", "publish"],
            defaults={
                "scene_count": 1,
                "publish_enabled": True,
                "publish_dry_run": True,
                "story_title": "集成测试鬼故事",
                "publish_tags": ["测试"],
            },
        )

        result = engine.run("publish_e2e", story_seed="seed")

        manifest_path = os.path.join(result.job_dir, "publish_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "publish_manifest.json missing")
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["status"], "dry_run")
        self.assertEqual(payload["results"][0]["title"], "集成测试鬼故事")

        cp_path = os.path.join(result.job_dir, "_checkpoint.json")
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        self.assertIn("publish", cp.get("completed_stages", []))

    def test_engine_run_publish_disabled_skips_manifest(self):
        engine = self._engine_with_template(
            stages=["stub_final", "publish"],
            defaults={"scene_count": 1, "publish_enabled": False},
        )

        result = engine.run("publish_e2e", story_seed="seed")

        manifest_path = os.path.join(result.job_dir, "publish_manifest.json")
        self.assertFalse(os.path.exists(manifest_path))

        cp_path = os.path.join(result.job_dir, "_checkpoint.json")
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        self.assertIn("publish", cp.get("completed_stages", []))


if __name__ == "__main__":
    unittest.main()
