import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_nodes4macos.nodes import ken_burns as kb
from custom_nodes4macos import ffmpeg_util

HAS_FFMPEG = bool(shutil.which("ffmpeg"))
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 未安装")


def _tiny_image_tensor():
    import numpy

    arr = numpy.zeros((64, 64, 3), dtype=numpy.uint8)
    arr[:, :, 0] = 120
    arr[16:48, 16:48, 1] = 200
    return arr


def _make_wav(path: str, dur: float = 0.6) -> str:
    ffmpeg_util.run_ffmpeg(
        ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", path],
        timeout=30, label="test_wav",
    )
    return path


def test_build_zoompan_shapes():
    for preset in ["zoom-in", "zoom-out", "pan-right", "pan-left"]:
        f = kb._build_zoompan(preset, 1080, 1920, 30, 150)
        assert "zoompan" in f and "1080x1920" in f and "d=150" in f, f
    fr = kb._build_zoompan("random", 1080, 1920, 30, 150)
    assert "zoompan" in fr


def test_render_no_audio(tmp_path):
    node = kb.FusionMLXKenBurns()
    out = node.render(
        image=_tiny_image_tensor(),
        duration_seconds=0.5,
        motion_preset="zoom-in",
        width=64, height=128, fps=10,
        filename_prefix="kb_test",
        audio_path="",
    )[0]
    assert os.path.exists(out) and os.path.getsize(out) > 0
    dur = ffmpeg_util.probe_duration(out)
    assert 0.3 <= dur <= 0.9, dur
    assert not ffmpeg_util.probe_has_audio(out)


def test_render_all_presets(tmp_path):
    for preset in ["zoom-in", "zoom-out", "pan-right", "pan-left", "random"]:
        node = kb.FusionMLXKenBurns()
        out = node.render(
            image=_tiny_image_tensor(),
            duration_seconds=0.4,
            motion_preset=preset,
            width=64, height=128, fps=10,
            filename_prefix=f"kb_{preset}",
            audio_path="",
        )[0]
        assert os.path.exists(out) and os.path.getsize(out) > 0, preset


def test_render_with_audio(tmp_path):
    wav = _make_wav(str(tmp_path / "narr.wav"), dur=0.8)
    node = kb.FusionMLXKenBurns()
    out = node.render(
        image=_tiny_image_tensor(),
        duration_seconds=0.5,
        motion_preset="zoom-in",
        width=64, height=128, fps=10,
        filename_prefix="kb_audio",
        audio_path=wav,
    )[0]
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert ffmpeg_util.probe_has_audio(out), "应嵌入旁白音轨"


def test_render_missing_audio_falls_back(tmp_path):
    node = kb.FusionMLXKenBurns()
    out = node.render(
        image=_tiny_image_tensor(),
        duration_seconds=0.4,
        motion_preset="zoom-in",
        width=64, height=128, fps=10,
        filename_prefix="kb_noaudio_path",
        audio_path=str(tmp_path / "nope.wav"),
    )[0]
    assert os.path.exists(out)
    assert not ffmpeg_util.probe_has_audio(out)
