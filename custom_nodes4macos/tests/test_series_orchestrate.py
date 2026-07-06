from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stage import Stage, StageInfo
from custom_nodes4macos.pipeline.stages.series_orchestrate import (
    SeriesOrchestratorStage,
    _cn_numeral,
)


class _FakeRecorder:
    calls = []
    seen_registries = []

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.seen_registries = []


class _FakeEpisodeStage(Stage):
    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(name="fake_episode", description="fake per-episode stage", output_kinds=["final"])

    def process(self, ctx, model_manager) -> None:
        _FakeRecorder.calls.append(ctx.job_dir)
        reg = ctx.config.get("character_registry", [])
        _FakeRecorder.seen_registries.append(list(reg))
        if ctx.job_dir.endswith("episode_01") and not any(
            c.get("name") == "书生" for c in reg
        ):
            reg.append({"name": "书生", "appearance": "young scholar in blue robe"})
        final_path = ctx.artifact_path(0, "final")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(b"fake_episode_final")
        ctx.set_artifact(0, "final", final_path)
        for kind in ("image", "audio", "clip"):
            p = ctx.artifact_path(0, kind)
            with open(p, "wb") as f:
                f.write(b"intermediate_" + kind.encode())
        with open(ctx.artifact_path(0, "subtitle"), "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\ntest\n")


def _make_ctx(tmpdir, episodes, completed=None, story_title="青溪渡阴"):
    config = {
        "content_type": "series",
        "episodes": episodes,
        "story_title": story_title,
        "scene_count": 2,
        "character_registry": [],
        "_completed_episodes": completed or [],
    }
    ctx = PipelineContext(job_id="job_test", job_dir=tmpdir, config=config)
    return ctx


class TestCnNumeral(unittest.TestCase):
    def test_single_digits(self):
        self.assertEqual(_cn_numeral(1), "一")
        self.assertEqual(_cn_numeral(2), "二")
        self.assertEqual(_cn_numeral(3), "三")

    def test_ten_and_teens(self):
        self.assertEqual(_cn_numeral(10), "十")
        self.assertEqual(_cn_numeral(11), "十一")
        self.assertEqual(_cn_numeral(12), "十二")

    def test_twenties(self):
        self.assertEqual(_cn_numeral(25), "二十五")

    def test_fallback(self):
        self.assertEqual(_cn_numeral(100), "100")
        self.assertEqual(_cn_numeral(0), "0")


class TestSeriesOrchestrator(unittest.TestCase):

    def setUp(self):
        self._orig_stages = SeriesOrchestratorStage._PER_EPISODE_STAGES
        SeriesOrchestratorStage._PER_EPISODE_STAGES = (_FakeEpisodeStage,)
        _FakeRecorder.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="series_orch_test_")

    def tearDown(self):
        SeriesOrchestratorStage._PER_EPISODE_STAGES = self._orig_stages
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _episodes(self, n=2):
        return [
            {"episode_id": i + 1, "title": f"第{i+1}回", "synopsis": f"synopsis {i+1}"}
            for i in range(n)
        ]

    def test_produces_per_episode_files(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(2))
        SeriesOrchestratorStage().process(ctx, None)
        ep1 = os.path.join(self.tmpdir, "青溪渡阴_第一集.mp4")
        ep2 = os.path.join(self.tmpdir, "青溪渡阴_第二集.mp4")
        self.assertTrue(os.path.exists(ep1), f"missing {ep1}")
        self.assertTrue(os.path.exists(ep2), f"missing {ep2}")
        self.assertEqual(ctx.config["_completed_episodes"], [1, 2])
        self.assertEqual(len(ctx.config["_episode_finals"]), 2)
        self.assertEqual(ctx.get_artifact(0, "final"), ep2)
        self.assertEqual(len(_FakeRecorder.calls), 2)

    def test_resume_skips_completed(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(3), completed=[1, 2])
        SeriesOrchestratorStage().process(ctx, None)
        self.assertEqual(len(_FakeRecorder.calls), 1)
        self.assertIn(os.path.join(self.tmpdir, "episode_03"), _FakeRecorder.calls[0])
        self.assertEqual(ctx.config["_completed_episodes"], [1, 2, 3])
        ep3 = os.path.join(self.tmpdir, "青溪渡阴_第三集.mp4")
        self.assertTrue(os.path.exists(ep3))

    def test_empty_episodes_noop(self):
        ctx = _make_ctx(self.tmpdir, [])
        SeriesOrchestratorStage().process(ctx, None)
        self.assertEqual(_FakeRecorder.calls, [])
        self.assertIsNone(ctx.get_artifact(0, "final"))

    def test_character_registry_shared_across_episodes(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(2))
        SeriesOrchestratorStage().process(ctx, None)
        ep1_reg, ep2_reg = _FakeRecorder.seen_registries
        self.assertEqual(len(ep1_reg), 0)
        self.assertEqual(len(ep2_reg), 1)
        self.assertEqual(ep2_reg[0]["name"], "书生")
        self.assertEqual(len(ctx.config["character_registry"]), 1)

    def test_episode_dirs_created(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(2))
        SeriesOrchestratorStage().process(ctx, None)
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "episode_01")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "episode_02")))

    def test_intermediates_cleaned_after_episode(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(2))
        SeriesOrchestratorStage().process(ctx, None)
        for ep in ("episode_01", "episode_02"):
            ep_dir = os.path.join(self.tmpdir, ep)
            self.assertFalse(
                os.path.exists(os.path.join(ep_dir, "scene_000_image.png")),
                f"image intermediate should be cleaned in {ep}",
            )
            self.assertFalse(
                os.path.exists(os.path.join(ep_dir, "scene_000_audio.wav")),
                f"audio intermediate should be cleaned in {ep}",
            )
            self.assertFalse(
                os.path.exists(os.path.join(ep_dir, "scene_000_clip.mp4")),
                f"clip intermediate should be cleaned in {ep}",
            )
            self.assertTrue(
                os.path.exists(os.path.join(ep_dir, "scene_000_final.mp4")),
                f"final should be kept in {ep}",
            )
            self.assertTrue(
                os.path.exists(os.path.join(ep_dir, "scene_000_subtitle.srt")),
                f"subtitle should be kept in {ep}",
            )

    def test_cleanup_disabled_keeps_intermediates(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(1))
        ctx.config["cleanup_episode_intermediates"] = False
        SeriesOrchestratorStage().process(ctx, None)
        ep_dir = os.path.join(self.tmpdir, "episode_01")
        self.assertTrue(os.path.exists(os.path.join(ep_dir, "scene_000_image.png")))
        self.assertTrue(os.path.exists(os.path.join(ep_dir, "scene_000_audio.wav")))


class _DHRecorder:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []


class _FakeDigitalHumanStage(Stage):
    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(name="fake_dh", description="fake digital_human stage", output_kinds=["final"])

    def process(self, ctx, model_manager) -> None:
        _DHRecorder.calls.append((
            "dh",
            ctx.job_dir,
            ctx.config.get("avatar_package", ""),
            ctx.config.get("voice_ref_audio", ""),
        ))
        final_path = ctx.artifact_path(0, "final")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(b"fake_dh_final")
        ctx.set_artifact(0, "final", final_path)


class _FakeImageStage(Stage):
    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(name="fake_img", description="fake image stage", output_kinds=["final"])

    def process(self, ctx, model_manager) -> None:
        _DHRecorder.calls.append(("img", ctx.job_dir, "", ""))
        final_path = ctx.artifact_path(0, "final")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "wb") as f:
            f.write(b"fake_img_final")
        ctx.set_artifact(0, "final", final_path)


class TestSeriesDigitalHumanMode(unittest.TestCase):

    def setUp(self):
        self._orig_per = SeriesOrchestratorStage._PER_EPISODE_STAGES
        self._orig_dh = SeriesOrchestratorStage._DIGITAL_HUMAN_STAGES
        SeriesOrchestratorStage._PER_EPISODE_STAGES = (_FakeImageStage,)
        SeriesOrchestratorStage._DIGITAL_HUMAN_STAGES = (_FakeDigitalHumanStage,)
        _DHRecorder.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="series_dh_test_")

    def tearDown(self):
        SeriesOrchestratorStage._PER_EPISODE_STAGES = self._orig_per
        SeriesOrchestratorStage._DIGITAL_HUMAN_STAGES = self._orig_dh
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _episodes(self, n=1):
        return [
            {"episode_id": i + 1, "title": f"第{i+1}回", "synopsis": f"synopsis {i+1}"}
            for i in range(n)
        ]

    def test_is_digital_human_mode_avatar_package_dir(self):
        pkg = tempfile.mkdtemp(prefix="avatar_pkg_")
        try:
            ctx = _make_ctx(self.tmpdir, self._episodes(1))
            ctx.config["avatar_package"] = pkg
            self.assertTrue(SeriesOrchestratorStage._is_digital_human_mode(ctx))
        finally:
            shutil.rmtree(pkg, ignore_errors=True)

    def test_is_digital_human_mode_avatar_reference_file(self):
        ref = os.path.join(self.tmpdir, "ref.png")
        with open(ref, "wb") as f:
            f.write(b"x")
        ctx = _make_ctx(self.tmpdir, self._episodes(1))
        ctx.config["avatar_reference"] = ref
        self.assertTrue(SeriesOrchestratorStage._is_digital_human_mode(ctx))

    def test_is_digital_human_mode_false_without_avatar(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(1))
        self.assertFalse(SeriesOrchestratorStage._is_digital_human_mode(ctx))

    def test_is_digital_human_mode_false_for_nonexistent_path(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(1))
        ctx.config["avatar_reference"] = "/nonexistent/avatar_ref.png"
        ctx.config["avatar_package"] = "/nonexistent/avatar_pkg"
        self.assertFalse(SeriesOrchestratorStage._is_digital_human_mode(ctx))

    def test_avatar_package_triggers_digital_human_stages(self):
        pkg = tempfile.mkdtemp(prefix="avatar_pkg_")
        try:
            ctx = _make_ctx(self.tmpdir, self._episodes(1))
            ctx.config["avatar_package"] = pkg
            SeriesOrchestratorStage().process(ctx, None)
            self.assertEqual(len(_DHRecorder.calls), 1)
            self.assertEqual(_DHRecorder.calls[0][0], "dh")
            self.assertEqual(_DHRecorder.calls[0][2], pkg)
        finally:
            shutil.rmtree(pkg, ignore_errors=True)

    def test_no_avatar_uses_image_stages(self):
        ctx = _make_ctx(self.tmpdir, self._episodes(1))
        SeriesOrchestratorStage().process(ctx, None)
        self.assertEqual(len(_DHRecorder.calls), 1)
        self.assertEqual(_DHRecorder.calls[0][0], "img")

    def test_avatar_package_propagated_to_all_episodes(self):
        pkg = tempfile.mkdtemp(prefix="avatar_pkg_")
        voice_ref = os.path.join(self.tmpdir, "voice.wav")
        with open(voice_ref, "wb") as f:
            f.write(b"voice")
        try:
            ctx = _make_ctx(self.tmpdir, self._episodes(3))
            ctx.config["avatar_package"] = pkg
            ctx.config["voice_ref_audio"] = voice_ref
            SeriesOrchestratorStage().process(ctx, None)
            self.assertEqual(len(_DHRecorder.calls), 3)
            for tag, _job_dir, av_pkg, vref in _DHRecorder.calls:
                self.assertEqual(tag, "dh")
                self.assertEqual(av_pkg, pkg, "avatar_package must be shared across episodes")
                self.assertEqual(vref, voice_ref, "voice_ref_audio must be shared across episodes")
        finally:
            shutil.rmtree(pkg, ignore_errors=True)


class TestEpisodeCleanupDirect(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="series_cleanup_test_")
    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_removes_image_audio_clip_keeps_final_subtitle(self):
        keepers = ("scene_000_final.mp4", "scene_000_subtitle.srt")
        intermediates = ("scene_000_image.png", "scene_000_audio.wav", "scene_000_clip.mp4")
        for name in keepers + intermediates:
            with open(os.path.join(self.tmpdir, name), "wb") as f:
                f.write(b"data_" + name.encode())
        SeriesOrchestratorStage._cleanup_episode_intermediates(self.tmpdir)
        for name in intermediates:
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, name)), f"{name} should be removed")
        for name in keepers:
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, name)), f"{name} should be kept")

    def test_multiple_scenes_all_cleaned(self):
        exts = {"image": ".png", "audio": ".wav", "clip": ".mp4"}
        for sid in (0, 1, 2):
            for kind, ext in exts.items():
                with open(os.path.join(self.tmpdir, f"scene_{sid:03d}_{kind}{ext}"), "wb") as f:
                    f.write(b"x")
        SeriesOrchestratorStage._cleanup_episode_intermediates(self.tmpdir)
        remaining = [n for n in os.listdir(self.tmpdir) if n.startswith("scene_")]
        self.assertEqual(remaining, [], "all scene intermediates should be removed")

    def test_non_scene_files_untouched(self):
        for name in ("config.json", "manifest.yaml", "scene_meta.json"):
            with open(os.path.join(self.tmpdir, name), "w") as f:
                f.write("keep")
        SeriesOrchestratorStage._cleanup_episode_intermediates(self.tmpdir)
        for name in ("config.json", "manifest.yaml", "scene_meta.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, name)), f"{name} should be untouched")


if __name__ == "__main__":
    unittest.main()
