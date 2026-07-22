from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.stages import ken_burns as kb
from custom_nodes4macos.pipeline.stages.ken_burns import (
    KenBurnsStage,
    _build_zoompan_multishot,
    _cut_frames_to_seg_lengths,
    _detect_silence,
    _even_pick,
    _pick_multishot_presets,
    _silence_to_cut_frames,
)


class TestDetectSilence(unittest.TestCase):

    def _mock_proc(self, stderr: str):
        m = MagicMock()
        m.stderr = stderr
        return m

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.subprocess.run")
    def test_parses_pairs(self, mock_run):
        stderr = (
            "[silencedetect @ 0x1] silence_start: 2.0\n"
            "[silencedetect @ 0x1] silence_end: 2.4 | silence_duration: 0.4\n"
            "[silencedetect @ 0x1] silence_start: 5.0\n"
            "[silencedetect @ 0x1] silence_end: 5.3 | silence_duration: 0.3\n"
        )
        mock_run.return_value = self._mock_proc(stderr)
        result = _detect_silence("/fake/audio.wav", -30, 0.3)
        self.assertEqual(result, [(2.0, 2.4), (5.0, 5.3)])

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.subprocess.run")
    def test_unpaired_start_dropped(self, mock_run):
        stderr = (
            "[silencedetect @ 0x1] silence_start: 2.0\n"
            "[silencedetect @ 0x1] silence_end: 2.4 | silence_duration: 0.4\n"
            "[silencedetect @ 0x1] silence_start: 9.0\n"
        )
        mock_run.return_value = self._mock_proc(stderr)
        result = _detect_silence("/fake/audio.wav", -30, 0.3)
        self.assertEqual(result, [(2.0, 2.4)])

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.subprocess.run")
    def test_no_silence_returns_empty(self, mock_run):
        mock_run.return_value = self._mock_proc("no markers here\n")
        result = _detect_silence("/fake/audio.wav", -30, 0.3)
        self.assertEqual(result, [])

    @patch("custom_nodes4macos.pipeline.stages.ken_burns.subprocess.run")
    def test_exception_returns_empty(self, mock_run):
        mock_run.side_effect = OSError("boom")
        result = _detect_silence("/fake/audio.wav", -30, 0.3)
        self.assertEqual(result, [])


class TestSilenceToCutFrames(unittest.TestCase):

    def test_two_silences(self):
        silences = [(2.0, 2.4), (5.0, 5.4)]
        cuts = _silence_to_cut_frames(silences, 8.0, 30, 4)
        self.assertEqual(cuts, [66, 156])

    def test_empty_returns_empty(self):
        self.assertEqual(_silence_to_cut_frames([], 8.0, 30, 4), [])

    def test_edge_silences_excluded(self):
        silences = [(0.1, 0.5), (7.6, 8.0)]
        cuts = _silence_to_cut_frames(silences, 8.0, 30, 4)
        self.assertEqual(cuts, [])

    def test_capped_to_max_shots(self):
        silences = [(1.0, 1.2), (3.0, 3.2), (5.0, 5.2), (6.5, 6.7)]
        cuts = _silence_to_cut_frames(silences, 8.0, 30, 2)
        self.assertEqual(len(cuts), 1)
        total = round(8.0 * 30)
        self.assertTrue(0 < cuts[0] < total)

    def test_zero_duration_returns_empty(self):
        self.assertEqual(_silence_to_cut_frames([(1.0, 1.2)], 0.0, 30, 4), [])


class TestEvenPick(unittest.TestCase):

    def test_k_two_picks_ends(self):
        self.assertEqual(_even_pick([10, 20, 30, 40, 50], 2), [10, 50])

    def test_k_one_picks_middle(self):
        self.assertEqual(_even_pick([10, 20, 30, 40, 50], 1), [30])

    def test_k_three_even(self):
        self.assertEqual(_even_pick([10, 20, 30, 40, 50], 3), [10, 30, 50])

    def test_k_zero_empty(self):
        self.assertEqual(_even_pick([10, 20, 30], 0), [])

    def test_items_fewer_than_k_returns_all(self):
        self.assertEqual(_even_pick([10, 20], 5), [10, 20])


class TestCutFramesToSegLengths(unittest.TestCase):

    def test_two_cuts_three_segments(self):
        self.assertEqual(_cut_frames_to_seg_lengths([66, 156], 240), [66, 90, 84])

    def test_no_cuts_single_segment(self):
        self.assertEqual(_cut_frames_to_seg_lengths([], 240), [240])


class TestPickMultishotPresets(unittest.TestCase):

    def test_fixed_preset_repeated(self):
        self.assertEqual(_pick_multishot_presets(3, "zoom-in"), ["zoom-in"] * 3)

    def test_random_uses_rotation(self):
        self.assertEqual(
            _pick_multishot_presets(3, "random"),
            ["zoom-in", "pan-right", "zoom-out"],
        )

    def test_random_wraps_rotation(self):
        self.assertEqual(
            _pick_multishot_presets(5, "random"),
            ["zoom-in", "pan-right", "zoom-out", "pan-left", "zoom-in"],
        )


class TestBuildZoompanMultishot(unittest.TestCase):

    def test_two_segments_has_if_and_boundary(self):
        result = _build_zoompan_multishot(
            ["zoom-in", "pan-right"], [120, 120], 240, 1080, 1920, 30, 30,
        )
        self.assertIn("zoompan", result)
        self.assertIn("if(lt(on,120)", result)
        self.assertIn("1+0.25*on/120", result)
        self.assertIn("(on-120)", result)

    def test_three_segments_nested_ifs(self):
        result = _build_zoompan_multishot(
            ["zoom-in", "zoom-out", "pan-left"], [80, 80, 80], 240, 1080, 1920, 30, 30,
        )
        self.assertIn("if(lt(on,80)", result)
        self.assertIn("if(lt(on,160)", result)
        self.assertIn("(on-80)", result)
        self.assertIn("(on-160)", result)
        self.assertIn("1+0.25*on/80", result)
        self.assertIn("1.25-0.25*(on-80)/80", result)

    def test_uses_local_segment_duration(self):
        result = _build_zoompan_multishot(
            ["zoom-in", "zoom-in"], [100, 140], 240, 1080, 1920, 30, 30,
        )
        self.assertIn("1+0.25*on/100", result)
        self.assertIn("1+0.25*(on-100)/140", result)


class TestRenderScene(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img = os.path.join(self.tmpdir, "img.png")
        self.audio = os.path.join(self.tmpdir, "audio.wav")
        self.out = os.path.join(self.tmpdir, "out.mp4")
        with open(self.img, "wb") as f:
            f.write(b"png")
        with open(self.audio, "wb") as f:
            f.write(b"wav")
        self.cut_cfg = (True, -30, 0.3, 4, 3.0)

    def test_audio_cut_off_calls_render_clip(self):
        stage = KenBurnsStage()
        cfg = (False, -30, 0.3, 4, 3.0)
        with patch.object(KenBurnsStage, "_render_clip") as mock_clip, \
                patch.object(KenBurnsStage, "_run_zoompan_render") as mock_run, \
                patch.object(kb, "_detect_silence") as mock_silence:
            stage._render_scene(self.img, self.audio, 5.0, "random", 1080, 1920, 30, 30, 1, self.out, cfg)
        mock_clip.assert_called_once()
        self.assertEqual(mock_clip.call_args.args[8], 1)
        mock_run.assert_not_called()
        mock_silence.assert_not_called()

    def test_no_silence_falls_back_to_single(self):
        stage = KenBurnsStage()
        with patch.object(KenBurnsStage, "_render_clip") as mock_clip, \
                patch.object(KenBurnsStage, "_run_zoompan_render") as mock_run, \
                patch.object(kb, "_detect_silence", return_value=[]):
            stage._render_scene(self.img, self.audio, 5.0, "random", 1080, 1920, 30, 30, 2, self.out, self.cut_cfg)
        mock_clip.assert_called_once()
        self.assertEqual(mock_clip.call_args.args[8], 2)
        mock_run.assert_not_called()

    def test_short_duration_falls_back(self):
        stage = KenBurnsStage()
        with patch.object(KenBurnsStage, "_render_clip") as mock_clip, \
                patch.object(KenBurnsStage, "_run_zoompan_render") as mock_run, \
                patch.object(kb, "_detect_silence", return_value=[(1.0, 1.2)]):
            stage._render_scene(self.img, self.audio, 2.0, "random", 1080, 1920, 30, 30, 3, self.out, self.cut_cfg)
        mock_clip.assert_called_once()
        mock_run.assert_not_called()

    def test_silence_triggers_multishot(self):
        stage = KenBurnsStage()
        with patch.object(KenBurnsStage, "_render_clip") as mock_clip, \
                patch.object(KenBurnsStage, "_run_zoompan_render") as mock_run, \
                patch.object(kb, "_detect_silence", return_value=[(2.0, 2.4)]):
            stage._render_scene(self.img, self.audio, 5.0, "random", 1080, 1920, 30, 30, 4, self.out, self.cut_cfg)
        mock_run.assert_called_once()
        zoompan_arg = mock_run.call_args.args[3]
        self.assertIn("if(", zoompan_arg)
        mock_clip.assert_not_called()

    def test_multishot_segment_count_matches_cuts(self):
        stage = KenBurnsStage()
        silences = [(2.0, 2.4), (3.5, 3.7)]
        with patch.object(KenBurnsStage, "_render_clip"), \
                patch.object(KenBurnsStage, "_run_zoompan_render") as mock_run, \
                patch.object(kb, "_detect_silence", return_value=silences):
            stage._render_scene(self.img, self.audio, 6.0, "random", 1080, 1920, 30, 30, 5, self.out, self.cut_cfg)
        mock_run.assert_called_once()
        zoompan_arg = mock_run.call_args.args[3]
        self.assertEqual(zoompan_arg.count("if("), 6)
        self.assertIn("lt(on,66)", zoompan_arg)
        self.assertIn("lt(on,108)", zoompan_arg)


HAS_FFMPEG = bool(shutil.which("ffmpeg"))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg 未安装")
class TestAudioCutIntegration(unittest.TestCase):

    def _make_tone_silence_tone(self, path: str):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-f", "lavfi", "-i", "aevalsrc=0:d=0.5",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1",
             "-c:a", "pcm_s16le", path],
            check=True, timeout=60,
        )

    def _make_png(self, path: str):
        from PIL import Image
        import numpy
        arr = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
        arr[:, :, 0] = 120
        arr[16:48, 16:48, 1] = 200
        Image.fromarray(arr).save(path)

    def test_detect_silence_on_real_audio(self):
        tmp = tempfile.mkdtemp()
        wav = os.path.join(tmp, "a.wav")
        self._make_tone_silence_tone(wav)
        intervals = _detect_silence(wav, -30, 0.3)
        self.assertTrue(len(intervals) >= 1)
        s, e = intervals[0]
        self.assertAlmostEqual(s, 2.0, delta=0.1)
        self.assertAlmostEqual(e, 2.5, delta=0.1)

    def test_render_scene_multishot_produces_mp4(self):
        tmp = tempfile.mkdtemp()
        img = os.path.join(tmp, "img.png")
        audio = os.path.join(tmp, "audio.wav")
        out = os.path.join(tmp, "out.mp4")
        self._make_png(img)
        self._make_tone_silence_tone(audio)
        stage = KenBurnsStage()
        cut_cfg = (True, -30, 0.3, 4, 3.0)
        stage._render_scene(img, audio, 4.5, "random", 1080, 1920, 30, 30, 1, out, cut_cfg)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 1000)


if __name__ == "__main__":
    unittest.main()
