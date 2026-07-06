from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.context import PipelineContext
from custom_nodes4macos.pipeline.stages.assemble import AssembleStage, _XFADE_MAP


def _ctx_with_clips(tmpdir, n, transition, extra=None):
    clips = []
    for i in range(1, n + 1):
        p = os.path.join(tmpdir, f"c{i}.mp4")
        open(p, "wb").write(b"mp4")
        clips.append(p)
    cfg = {
        "assemble_transition": transition,
        "ken_burns_width": 64, "ken_burns_height": 128, "ken_burns_fps": 15,
    }
    if extra:
        cfg.update(extra)
    ctx = PipelineContext(job_id="t", job_dir=tmpdir, config=cfg)
    ctx.scenes = [{"scene_id": i} for i in range(1, n + 1)]
    for i, p in enumerate(clips, 1):
        ctx.set_artifact(i, "clip", p)
    return ctx, clips


def _mock_ff(mock_ff):
    mock_ff.probe_has_audio.return_value = True
    mock_ff.probe_duration.return_value = 2.0
    mock_ff.video_encoder_args.return_value = ["-c:v", "libx264"]

    def fake_run(args, timeout=None, label=""):
        open(args[-1], "wb").write(b"final")

    mock_ff.run_ffmpeg.side_effect = fake_run


class TestXfadeTrigger(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_crossfade_uses_xfade_filter(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 3, "crossfade")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=fade", graph)
        self.assertIn("acrossfade=d=0.500", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_horror_maps_to_fadeblack(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 2, "horror")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=fadeblack", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_dissolve_uses_dissolve(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 2, "dissolve")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=dissolve", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_none_does_not_use_xfade(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 3, "none")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertNotIn("xfade=", graph)
        self.assertIn("concat=", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_fade_does_not_use_xfade(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 3, "fade")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        fc = mock_ff.run_ffmpeg.call_args.args[0]
        graph = fc[fc.index("-filter_complex") + 1]
        self.assertNotIn("xfade=", graph)
        self.assertIn("fade=t=in", graph)


class TestXfadeOffset(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_offset_accumulates(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 3, "crossfade")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        graph = mock_ff.run_ffmpeg.call_args.args[0]
        graph = graph[graph.index("-filter_complex") + 1]
        self.assertIn("offset=1.500", graph)
        self.assertIn("offset=3.000", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_short_clips_clamp_duration(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 2, "crossfade")
        mock_ff.probe_has_audio.return_value = True
        mock_ff.probe_duration.return_value = 0.4
        mock_ff.video_encoder_args.return_value = ["-c:v", "libx264"]

        def fake_run(args, timeout=None, label=""):
            open(args[-1], "wb").write(b"final")

        mock_ff.run_ffmpeg.side_effect = fake_run
        AssembleStage().process(ctx, MagicMock())
        graph = mock_ff.run_ffmpeg.call_args.args[0]
        graph = graph[graph.index("-filter_complex") + 1]
        # D = min(0.5, 0.4*0.5=0.2) = 0.2
        self.assertIn("duration=0.200", graph)
        self.assertIn("acrossfade=d=0.200", graph)


class TestXfadeDucking(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_xfade_with_duck_bgm(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        bgm = os.path.join(tmpdir, "bgm.wav")
        open(bgm, "wb").write(b"w")
        ctx, _ = _ctx_with_clips(tmpdir, 2, "crossfade", {"assemble_bgm_path": bgm, "assemble_duck_bgm": True})
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        graph = mock_ff.run_ffmpeg.call_args.args[0]
        graph = graph[graph.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=fade", graph)
        self.assertIn("sidechaincompress", graph)
        self.assertIn("asplit=2", graph)

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_xfade_without_duck_plain_amix(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        bgm = os.path.join(tmpdir, "bgm.wav")
        open(bgm, "wb").write(b"w")
        ctx, _ = _ctx_with_clips(tmpdir, 2, "crossfade", {"assemble_bgm_path": bgm})
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        graph = mock_ff.run_ffmpeg.call_args.args[0]
        graph = graph[graph.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=fade", graph)
        self.assertNotIn("sidechaincompress", graph)
        self.assertIn("amix=inputs=2:duration=first:dropout_transition=0", graph)


class TestXfadeFallback(unittest.TestCase):

    @patch("custom_nodes4macos.pipeline.stages.assemble.ffmpeg_util")
    def test_single_clip_xfade_falls_back_to_concat(self, mock_ff):
        tmpdir = tempfile.mkdtemp()
        ctx, _ = _ctx_with_clips(tmpdir, 1, "crossfade")
        _mock_ff(mock_ff)
        AssembleStage().process(ctx, MagicMock())
        graph = mock_ff.run_ffmpeg.call_args.args[0]
        graph = graph[graph.index("-filter_complex") + 1]
        self.assertNotIn("xfade=", graph)
        self.assertIn("concat=", graph)


class TestXfadeMap(unittest.TestCase):

    def test_horror_maps_to_fadeblack(self):
        self.assertEqual(_XFADE_MAP["horror"], "fadeblack")

    def test_crossfade_maps_to_fade(self):
        self.assertEqual(_XFADE_MAP["crossfade"], "fade")

    def test_all_modes_present(self):
        for key in ["crossfade", "dissolve", "wipeleft", "wiperight",
                    "fadeblack", "fadewhite", "circleopen", "circleclose", "horror"]:
            self.assertIn(key, _XFADE_MAP)


HAS_FFMPEG = bool(shutil.which("ffmpeg"))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestXfadeIntegration(unittest.TestCase):

    def test_render_three_clips_crossfade(self):
        tmp = tempfile.mkdtemp()
        clips = []
        for i in range(3):
            p = os.path.join(tmp, f"c{i}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"color=c=black:s=64x128:d=2:r=15",
                 "-f", "lavfi", "-i", f"sine=frequency={300+i*100}:duration=2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-shortest", p],
                check=True, timeout=120,
            )
            clips.append(p)
        out = os.path.join(tmp, "out.mp4")
        AssembleStage._render_xfade(clips, 64, 128, 15, "fade", "", False, out)
        self.assertGreater(os.path.getsize(out), 1000)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", out],
            capture_output=True, text=True, check=True,
        )
        dur = float(probe.stdout.strip())
        self.assertAlmostEqual(dur, 5.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
