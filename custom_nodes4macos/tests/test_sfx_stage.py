from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.assemble import AssembleStage
from custom_nodes4macos.pipeline.stages.sfx import SFXStage


def _make_wav(path: str, freq: int = 440, dur: float = 0.5):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur}",
         "-c:a", "pcm_s16le", path],
        check=True, timeout=60,
    )


class TestCollectSfx(unittest.TestCase):

    def test_matches_keyword_and_accumulates_timestamp(self):
        tmpdir = tempfile.mkdtemp()
        sfx_a = os.path.join(tmpdir, "wind.wav")
        sfx_b = os.path.join(tmpdir, "thunder.wav")
        open(sfx_a, "wb").write(b"w")
        open(sfx_b, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [
            {"scene_id": 1, "duration_seconds": 4.0, "sound_effect": "秋风萧瑟"},
            {"scene_id": 2, "duration_seconds": 3.0, "sound_effect": "远处雷鸣"},
            {"scene_id": 3, "duration_seconds": 2.0, "sound_effect": "静默无音"},
        ]
        sfx_map = {"风": sfx_a, "雷": sfx_b}
        picks = SFXStage._collect_sfx(ctx, sfx_map)
        self.assertEqual(picks, [(sfx_a, 0.0), (sfx_b, 4.0)])

    def test_missing_sfx_file_skipped(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 4.0, "sound_effect": "风声"}]
        picks = SFXStage._collect_sfx(ctx, {"风": "/nonexistent.wav"})
        self.assertEqual(picks, [])

    def test_no_sound_effect_text(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 4.0}]
        picks = SFXStage._collect_sfx(ctx, {"风": "/whatever.wav"})
        self.assertEqual(picks, [])

    def test_default_duration_when_missing(self):
        tmpdir = tempfile.mkdtemp()
        sfx_a = os.path.join(tmpdir, "wind.wav")
        open(sfx_a, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.scenes = [
            {"scene_id": 1, "sound_effect": "风声"},
            {"scene_id": 2, "sound_effect": "风声"},
        ]
        picks = SFXStage._collect_sfx(ctx, {"风": sfx_a})
        self.assertEqual(len(picks), 2)
        self.assertAlmostEqual(picks[1][1], 8.0, delta=0.01)


class TestSfxStageProcess(unittest.TestCase):

    def test_skip_when_no_final(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"sfx_map": {"风": "/x"}})
        stage = SFXStage()
        with patch.object(stage, "_render_sfx") as mock_render:
            stage.process(ctx, MagicMock())
        mock_render.assert_not_called()

    def test_skip_when_no_sfx_map(self):
        tmpdir = tempfile.mkdtemp()
        final = os.path.join(tmpdir, "final.mp4")
        open(final, "wb").write(b"mp4")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={})
        ctx.set_artifact(0, "final", final)
        stage = SFXStage()
        with patch.object(stage, "_render_sfx") as mock_render:
            stage.process(ctx, MagicMock())
        mock_render.assert_not_called()

    def test_skip_when_no_match(self):
        tmpdir = tempfile.mkdtemp()
        final = os.path.join(tmpdir, "final.mp4")
        open(final, "wb").write(b"mp4")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"sfx_map": {"雷": "/x.wav"}})
        ctx.set_artifact(0, "final", final)
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 4.0, "sound_effect": "风声"}]
        stage = SFXStage()
        with patch.object(stage, "_render_sfx") as mock_render:
            stage.process(ctx, MagicMock())
        mock_render.assert_not_called()

    def test_render_called_when_matched(self):
        tmpdir = tempfile.mkdtemp()
        final = os.path.join(tmpdir, "final.mp4")
        open(final, "wb").write(b"mp4")
        sfx_a = os.path.join(tmpdir, "wind.wav")
        open(sfx_a, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"sfx_map": {"风": sfx_a}})
        ctx.set_artifact(0, "final", final)
        ctx.scenes = [{"scene_id": 1, "duration_seconds": 4.0, "sound_effect": "风声"}]
        stage = SFXStage()
        with patch.object(stage, "_render_sfx") as mock_render:
            stage.process(ctx, MagicMock())
        mock_render.assert_called_once()
        picks = mock_render.call_args.args[1]
        self.assertEqual(picks, [(sfx_a, 0.0)])


class TestSfxRenderFilter(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.sfx.ffmpeg_util")
    def test_filter_has_adelay_and_amix(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        final = os.path.join(tmpdir, "final.mp4")
        open(final, "wb").write(b"mp4")
        sfx_a = os.path.join(tmpdir, "a.wav")
        sfx_b = os.path.join(tmpdir, "b.wav")
        open(sfx_a, "wb").write(b"w")
        open(sfx_b, "wb").write(b"w")
        mock_ff.probe_has_audio.return_value = True
        mock_ff.video_encoder_args.return_value = ["-c:v", "libx264"]
        captured = {}

        def fake_run(args, timeout=None, label=""):
            captured["args"] = args
            open(args[-1], "wb").write(b"out")

        mock_ff.run_ffmpeg.side_effect = fake_run
        SFXStage._render_sfx(final, [(sfx_a, 0.0), (sfx_b, 4.0)])
        fc = captured["args"]
        idx = fc.index("-filter_complex") + 1
        graph = fc[idx]
        self.assertIn("adelay=0|0", graph)
        self.assertIn("adelay=4000|4000", graph)
        self.assertIn("amix=inputs=3", graph)
        self.assertIn("[orig_a]", graph)
        self.assertIn("-c:v", fc)
        self.assertIn("copy", fc)


class TestAssembleDucking(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_duck_uses_sidechaincompress(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        clips = []
        for i in range(1, 3):
            p = os.path.join(tmpdir, f"c{i}.mp4")
            open(p, "wb").write(b"mp4")
            clips.append(p)
        bgm = os.path.join(tmpdir, "bgm.wav")
        open(bgm, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_bgm_path": bgm, "assemble_transition": "none",
            "assemble_duck_bgm": True,
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": i} for i in range(1, 3)]
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)
        mock_ff.probe_has_audio.return_value = True
        mock_ff.probe_duration.return_value = 1.0
        mock_ff.video_encoder_args.return_value = ["-c:v", "libx264"]

        def fake_run(args, timeout=None, label=""):
            open(args[-1], "wb").write(b"final")

        mock_ff.run_ffmpeg.side_effect = fake_run
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertIn("sidechaincompress", graph)
        self.assertIn("asplit=2", graph)
        self.assertIn("amix=inputs=2:duration=first:normalize=0", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_no_duck_uses_plain_amix(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        clips = []
        for i in range(1, 3):
            p = os.path.join(tmpdir, f"c{i}.mp4")
            open(p, "wb").write(b"mp4")
            clips.append(p)
        bgm = os.path.join(tmpdir, "bgm.wav")
        open(bgm, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "assemble_bgm_path": bgm, "assemble_transition": "none",
            "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
        })
        ctx.scenes = [{"scene_id": i} for i in range(1, 3)]
        for i, p in enumerate(clips, 1):
            ctx.set_artifact(i, "clip", p)
        mock_ff.probe_has_audio.return_value = True
        mock_ff.probe_duration.return_value = 1.0
        mock_ff.video_encoder_args.return_value = ["-c:v", "libx264"]

        def fake_run(args, timeout=None, label=""):
            open(args[-1], "wb").write(b"final")

        mock_ff.run_ffmpeg.side_effect = fake_run
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertNotIn("sidechaincompress", graph)
        self.assertIn("amix=inputs=2:duration=first:dropout_transition=0", graph)


class TestResolveBgmByMood(unittest.TestCase):

    def test_resolves_by_explicit_mood(self):
        tmpdir = tempfile.mkdtemp()
        bgm = os.path.join(tmpdir, "horror.wav")
        open(bgm, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "bgm_mood_map": {"恐怖": bgm}, "bgm_mood": "恐怖",
        })
        self.assertEqual(AssembleStage._resolve_bgm_by_mood(ctx), bgm)

    def test_falls_back_to_style_preset(self):
        tmpdir = tempfile.mkdtemp()
        bgm = os.path.join(tmpdir, "ink.wav")
        open(bgm, "wb").write(b"w")
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "bgm_mood_map": {"水墨悬疑": bgm}, "style_preset": "水墨悬疑",
        })
        self.assertEqual(AssembleStage._resolve_bgm_by_mood(ctx), bgm)

    def test_missing_file_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={
            "bgm_mood_map": {"恐怖": "/nope.wav"}, "bgm_mood": "恐怖",
        })
        self.assertEqual(AssembleStage._resolve_bgm_by_mood(ctx), "")

    def test_no_mood_map_returns_empty(self):
        tmpdir = tempfile.mkdtemp()
        ctx = PipelineContext(job_id="t", job_dir=tmpdir, config={"bgm_mood": "恐怖"})
        self.assertEqual(AssembleStage._resolve_bgm_by_mood(ctx), "")


HAS_FFMPEG = bool(shutil.which("ffmpeg"))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestSfxIntegration(unittest.TestCase):

    def test_layer_sfx_onto_final(self):
        tmpdir = tempfile.mkdtemp()
        final = os.path.join(tmpdir, "final.mp4")
        sfx = os.path.join(tmpdir, "beep.wav")
        _make_wav(sfx, freq=880, dur=0.5)
        # build a 2s final mp4 with tone audio
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x128:d=2:r=15",
             "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-shortest", final],
            check=True, timeout=120,
        )
        SFXStage._render_sfx(final, [(sfx, 1.0)])
        self.assertTrue(os.path.getsize(final) > 0)
        # verify audio still present
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", final],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("audio", probe.stdout)


if __name__ == "__main__":
    unittest.main()
