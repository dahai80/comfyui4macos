"""ffmpeg_util 边界测试。

覆盖未安装场景、timeout、probe 解析失败、
VideoToolbox 缓存、threads 参数。

补充 REVIEW_REPORT：ffmpeg_util 边界条件未充分覆盖。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos import ffmpeg_util
from custom_nodes4macos.fusion_client import FusionMLXError


class TestEnsureFfmpeg(unittest.TestCase):

    def test_ensure_ffmpeg_returns_path_when_set(self):
        with patch.object(ffmpeg_util, "_FFMPEG_PATH", "/custom/ffmpeg"):
            self.assertEqual(ffmpeg_util.ensure_ffmpeg(), "/custom/ffmpeg")

    def test_ensure_ffmpeg_raises_when_missing(self):
        with patch.object(ffmpeg_util, "_FFMPEG_PATH", ""):
            with self.assertRaises(FusionMLXError) as cm:
                ffmpeg_util.ensure_ffmpeg()
            self.assertIn("ffmpeg", str(cm.exception))

    def test_ensure_ffprobe_returns_path_when_set(self):
        with patch.object(ffmpeg_util, "_FFPROBE_PATH", "/custom/ffprobe"):
            self.assertEqual(ffmpeg_util.ensure_ffprobe(), "/custom/ffprobe")

    def test_ensure_ffprobe_raises_when_missing(self):
        with patch.object(ffmpeg_util, "_FFPROBE_PATH", ""):
            with self.assertRaises(FusionMLXError) as cm:
                ffmpeg_util.ensure_ffprobe()
            self.assertIn("ffprobe", str(cm.exception))


class TestVideoToolboxCache(unittest.TestCase):

    def setUp(self):
        ffmpeg_util._VT_CACHE = None  # 重置缓存

    def tearDown(self):
        ffmpeg_util._VT_CACHE = None

    def test_has_videotoolbox_caches_result(self):
        """第二次调用不再 subprocess.run。"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="h264_videotoolbox encoder")
            r1 = ffmpeg_util.has_videotoolbox()
            r2 = ffmpeg_util.has_videotoolbox()
            self.assertTrue(r1)
            self.assertTrue(r2)
            self.assertEqual(mock_run.call_count, 1)

    def test_has_videotoolbox_false_when_not_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="libx264 encoder only")
            self.assertFalse(ffmpeg_util.has_videotoolbox())

    def test_has_videotoolbox_false_on_exception(self):
        with patch("subprocess.run", side_effect=Exception("fail")):
            self.assertFalse(ffmpeg_util.has_videotoolbox())
            # 缓存了 False
            self.assertFalse(ffmpeg_util._VT_CACHE)

    def test_video_encoder_args_vt(self):
        with patch.object(ffmpeg_util, "has_videotoolbox", return_value=True):
            args = ffmpeg_util.video_encoder_args(quality=50)
            self.assertIn("h264_videotoolbox", args)
            self.assertIn("50", args)

    def test_video_encoder_args_libx264_fallback(self):
        with patch.object(ffmpeg_util, "has_videotoolbox", return_value=False):
            args = ffmpeg_util.video_encoder_args()
            self.assertIn("libx264", args)
            self.assertIn("ultrafast", args)


class TestThreadArgs(unittest.TestCase):

    def test_zero_threads_returns_empty(self):
        with patch.object(ffmpeg_util, "_FFMPEG_THREADS", 0):
            self.assertEqual(ffmpeg_util.thread_args(), [])

    def test_positive_threads_returns_flag(self):
        with patch.object(ffmpeg_util, "_FFMPEG_THREADS", 8):
            self.assertEqual(ffmpeg_util.thread_args(), ["-threads", "8"])


class TestRunFfmpeg(unittest.TestCase):

    @patch.object(ffmpeg_util, "_FFMPEG_PATH", "/fake/ffmpeg")
    @patch.object(ffmpeg_util, "thread_args", return_value=[])
    def test_run_ffmpeg_success(self, _):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            ffmpeg_util.run_ffmpeg(["-version"], label="test")
            mock_run.assert_called_once()

    @patch.object(ffmpeg_util, "_FFMPEG_PATH", "/fake/ffmpeg")
    @patch.object(ffmpeg_util, "thread_args", return_value=["-threads", "4"])
    def test_run_ffmpeg_includes_thread_flags(self, _):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ffmpeg_util.run_ffmpeg(["-i", "in.mp4", "out.mp4"])
            cmd = mock_run.call_args.args[0]
            self.assertIn("-threads", cmd)
            self.assertIn("4", cmd)

    @patch.object(ffmpeg_util, "_FFMPEG_PATH", "/fake/ffmpeg")
    def test_run_ffmpeg_failure_raises_with_stderr(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="decoder error")
            with self.assertRaises(FusionMLXError) as cm:
                ffmpeg_util.run_ffmpeg(["bad"])
            self.assertIn("decoder error", str(cm.exception))

    @patch.object(ffmpeg_util, "_FFMPEG_PATH", "/fake/ffmpeg")
    def test_run_ffmpeg_timeout_raises(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)):
            with self.assertRaises(FusionMLXError) as cm:
                ffmpeg_util.run_ffmpeg(["-i", "x"], timeout=5, label="t")
            self.assertIn("超时", str(cm.exception))

    @patch.object(ffmpeg_util, "_FFMPEG_PATH", "/fake/ffmpeg")
    def test_run_ffmpeg_includes_y_and_loglevel(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ffmpeg_util.run_ffmpeg(["-i", "in.mp4"])
            cmd = mock_run.call_args.args[0]
            self.assertIn("-y", cmd)
            self.assertIn("-loglevel", cmd)
            self.assertIn("error", cmd)


class TestProbeDuration(unittest.TestCase):

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_duration_parses_float(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="12.5\n", stderr="")
            self.assertEqual(ffmpeg_util.probe_duration("/x.mp4"), 12.5)

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_duration_raises_on_non_numeric(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="N/A\n", stderr="")
            with self.assertRaises(FusionMLXError) as cm:
                ffmpeg_util.probe_duration("/x.mp4")
            self.assertIn("无法解析", str(cm.exception))

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_duration_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="file not found")
            with self.assertRaises(FusionMLXError):
                ffmpeg_util.probe_duration("/missing.mp4")

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_duration_raises_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=15)):
            with self.assertRaises(FusionMLXError):
                ffmpeg_util.probe_duration("/x.mp4")


class TestProbeHasAudio(unittest.TestCase):

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_has_audio_true(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="audio\n", stderr="")
            self.assertTrue(ffmpeg_util.probe_has_audio("/x.mp4"))

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_has_audio_false(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="\n", stderr="")
            self.assertFalse(ffmpeg_util.probe_has_audio("/x.mp4"))

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_has_audio_false_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=15)):
            self.assertFalse(ffmpeg_util.probe_has_audio("/x.mp4"))

    @patch.object(ffmpeg_util, "_FFPROBE_PATH", "/fake/ffprobe")
    def test_probe_has_audio_false_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="err")
            self.assertFalse(ffmpeg_util.probe_has_audio("/x.mp4"))


if __name__ == "__main__":
    unittest.main()
