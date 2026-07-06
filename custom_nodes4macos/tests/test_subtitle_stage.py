from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages import subtitle as subtitle_mod
from custom_nodes4macos.pipeline.stages.subtitle import SubtitleStage


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class TestSplitIntoCues(unittest.TestCase):

    def test_short_sentence_single_cue(self):
        cues = SubtitleStage._split_into_cues("夜晚的森林")
        self.assertEqual(cues, ["夜晚的森林"])

    def test_multiple_sentences_split(self):
        cues = SubtitleStage._split_into_cues("夜晚的森林。远处传来脚步声。")
        self.assertEqual(cues, ["夜晚的森林。", "远处传来脚步声。"])

    def test_long_line_split_by_comma(self):
        text = "月光洒在破庙的屋顶上，风吹动木门发出嘎吱的声响。"
        cues = SubtitleStage._split_into_cues(text)
        self.assertTrue(len(cues) >= 2)
        for c in cues:
            self.assertLessEqual(len(c), 18, f"cue too long: {c!r}")

    def test_empty_returns_empty(self):
        self.assertEqual(SubtitleStage._split_into_cues(""), [])
        self.assertEqual(SubtitleStage._split_into_cues("   "), [])


class TestFormatTimestamp(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(SubtitleStage._format_timestamp(0.0), "00:00:00,000")

    def test_simple_seconds(self):
        self.assertEqual(SubtitleStage._format_timestamp(1.5), "00:00:01,500")

    def test_minute_overflow(self):
        self.assertEqual(SubtitleStage._format_timestamp(65.25), "00:01:05,250")

    def test_negative_clamped(self):
        self.assertEqual(SubtitleStage._format_timestamp(-3.0), "00:00:00,000")

    def test_rounding_overflow(self):
        self.assertEqual(SubtitleStage._format_timestamp(1.9999), "00:00:02,000")


class TestBuildSrt(unittest.TestCase):

    def test_basic_two_scenes_cumulative_timeline(self):
        scenes = [
            {"scene_id": 1, "audio_script": "夜晚的森林。", "duration_seconds": 2.0},
            {"scene_id": 2, "audio_script": "远处传来脚步声。", "duration_seconds": 3.0},
        ]
        srt = SubtitleStage._build_srt(scenes)
        self.assertIn("1\n00:00:00,000 --> 00:00:02,000", srt)
        self.assertIn("2\n00:00:02,000 --> 00:00:05,000", srt)
        self.assertIn("夜晚的森林。", srt)
        self.assertIn("远处传来脚步声。", srt)

    def test_skips_empty_audio_script_but_timeline_advances(self):
        scenes = [
            {"scene_id": 1, "audio_script": "", "duration_seconds": 2.0},
            {"scene_id": 2, "audio_script": "脚步声。", "duration_seconds": 2.0},
        ]
        srt = SubtitleStage._build_srt(scenes)
        self.assertNotIn("脚步声", srt.split("\n")[3] if len(srt.split("\n")) > 3 else "")
        self.assertIn("00:00:02,000 --> 00:00:04,000", srt)

    def test_missing_duration_fallback(self):
        scenes = [{"scene_id": 1, "audio_script": "测试。"}]
        srt = SubtitleStage._build_srt(scenes)
        self.assertIn("00:00:00,000", srt)
        self.assertIn("测试。", srt)

    def test_multiple_cues_within_scene_distribute(self):
        scenes = [
            {"scene_id": 1, "audio_script": "第一句。第二句。第三句。", "duration_seconds": 6.0},
        ]
        srt = SubtitleStage._build_srt(scenes)
        self.assertEqual(srt.count(" --> "), 3)
        self.assertIn("00:00:00,000", srt)
        self.assertIn("00:00:06,000", srt)

    def test_no_scenes_returns_empty(self):
        self.assertEqual(SubtitleStage._build_srt([]), "")


class TestSubtitleStageInfo(unittest.TestCase):

    def test_info(self):
        info = SubtitleStage.info()
        self.assertEqual(info.name, "subtitle")
        self.assertEqual(info.model_requirements, [])
        self.assertIn("final", info.output_kinds)
        self.assertIn("subtitle", info.output_kinds)

    def test_skip_if_completed(self):
        stage = SubtitleStage()
        ctx = PipelineContext(job_id="t", job_dir="/tmp", config={})
        ctx.completed_stages = ["subtitle"]
        self.assertTrue(stage._skip_if_completed(ctx))


class TestSubtitleStageProcess(unittest.TestCase):

    def setUp(self):
        self._prev_burn = subtitle_mod._BURN_CAPABLE
        subtitle_mod._BURN_CAPABLE = False

    def tearDown(self):
        subtitle_mod._BURN_CAPABLE = self._prev_burn

    def test_process_skips_when_no_final(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"subtitle_mode": "none"})
        ctx.scenes = [{"scene_id": 1, "audio_script": "x。", "duration_seconds": 1.0}]
        stage = SubtitleStage()
        stage.process(ctx, None)
        self.assertIsNone(ctx.get_artifact(0, "subtitle"))
        self.assertEqual(ctx.progress.get("stage"), "subtitle")

    def test_process_writes_srt_only_when_mode_none(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"subtitle_mode": "none"})
        final_path = os.path.join(tmpdir, "scene_000_final.mp4")
        with open(final_path, "wb") as f:
            f.write(b"fake_final")
        ctx.set_artifact(0, "final", final_path)
        ctx.scenes = [{"scene_id": 1, "audio_script": "测试字幕。", "duration_seconds": 2.0}]
        stage = SubtitleStage()
        stage.process(ctx, None)
        srt_path = ctx.get_artifact(0, "subtitle")
        self.assertIsNotNone(srt_path)
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("测试字幕。", content)
        with open(final_path, "rb") as f:
            self.assertEqual(f.read(), b"fake_final")


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe not available")
class TestSubtitleSoftMuxIntegration(unittest.TestCase):

    def _make_tiny_video(self, path: str, seconds: float = 2.0):
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=black:s=320x240:d={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", path,
            ],
            check=True,
        )

    def _has_subtitle_stream(self, path: str) -> bool:
        proc = subprocess.run(
            [
                "ffprobe", "-hide_banner", "-loglevel", "error",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "csv=p=0", path,
            ],
            capture_output=True, text=True, check=True,
        )
        return "subtitle" in proc.stdout

    def setUp(self):
        self._prev_burn = subtitle_mod._BURN_CAPABLE
        subtitle_mod._BURN_CAPABLE = False

    def tearDown(self):
        subtitle_mod._BURN_CAPABLE = self._prev_burn

    def test_soft_mux_produces_subtitled_mp4(self):
        tmpdir = tempfile.mkdtemp()
        final_path = os.path.join(tmpdir, "scene_000_final.mp4")
        self._make_tiny_video(final_path, seconds=2.0)
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"subtitle_mode": "auto"})
        ctx.set_artifact(0, "final", final_path)
        ctx.scenes = [
            {"scene_id": 1, "audio_script": "夜晚的森林。", "duration_seconds": 1.0},
            {"scene_id": 2, "audio_script": "远处传来脚步声。", "duration_seconds": 1.0},
        ]
        stage = SubtitleStage()
        stage.process(ctx, None)

        self.assertTrue(os.path.exists(final_path))
        self.assertTrue(os.path.getsize(final_path) > 0)
        self.assertTrue(self._has_subtitle_stream(final_path))
        srt_path = ctx.get_artifact(0, "subtitle")
        self.assertTrue(os.path.exists(srt_path))
        with open(srt_path, encoding="utf-8") as f:
            self.assertIn("夜晚的森林", f.read())


if __name__ == "__main__":
    unittest.main()
