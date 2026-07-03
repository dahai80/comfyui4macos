import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_nodes4macos.nodes import assemble as asm
from custom_nodes4macos import ffmpeg_util

HAS_FFMPEG = bool(shutil.which("ffmpeg"))
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg 未安装")


def _make_clip(path: str, color: str = "red", dur: float = 0.4, with_audio: bool = True) -> str:
    args = ["-f", "lavfi", "-i", f"color=c={color}:s=64x128:r=15"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    args += ["-t", f"{dur:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    args.append(path)
    ffmpeg_util.run_ffmpeg(args, timeout=30, label="test_clip")
    return path


def _make_wav(path: str, dur: float = 2.0) -> str:
    ffmpeg_util.run_ffmpeg(
        ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", path],
        timeout=30, label="test_bgm",
    )
    return path


def test_parse_clips_skips_missing(tmp_path):
    a = _make_clip(str(tmp_path / "a.mp4"))
    text = f"{a}\n\n{tmp_path / 'nope.mp4'}\n  \n"
    clips = asm._parse_clips(text)
    assert clips == [a]


def test_parse_clips_empty():
    assert asm._parse_clips("") == []
    assert asm._parse_clips("   \n  \n") == []


def test_assemble_no_audio_clips(tmp_path):
    c1 = _make_clip(str(tmp_path / "c1.mp4"), "red", 0.4, with_audio=False)
    c2 = _make_clip(str(tmp_path / "c2.mp4"), "blue", 0.4, with_audio=False)
    node = asm.FusionMLXAssemble()
    out = node.render(
        clips=f"{c1}\n{c2}", transition="none",
        width=64, height=128, fps=15,
        filename_prefix="asm_test", bgm_path="",
    )[0]
    assert os.path.exists(out) and os.path.getsize(out) > 0
    dur = ffmpeg_util.probe_duration(out)
    assert 0.6 <= dur <= 1.0, dur
    assert not ffmpeg_util.probe_has_audio(out)


def test_assemble_with_clip_audio(tmp_path):
    c1 = _make_clip(str(tmp_path / "c1.mp4"), "red", 0.4, with_audio=True)
    c2 = _make_clip(str(tmp_path / "c2.mp4"), "blue", 0.4, with_audio=True)
    node = asm.FusionMLXAssemble()
    out = node.render(
        clips=f"{c1}\n{c2}", transition="none",
        width=64, height=128, fps=15,
        filename_prefix="asm_aud", bgm_path="",
    )[0]
    assert os.path.exists(out)
    assert ffmpeg_util.probe_has_audio(out), "应保留片段音轨"


def test_assemble_with_bgm(tmp_path):
    c1 = _make_clip(str(tmp_path / "c1.mp4"), "red", 0.4, with_audio=False)
    c2 = _make_clip(str(tmp_path / "c2.mp4"), "blue", 0.4, with_audio=False)
    bgm = _make_wav(str(tmp_path / "bgm.wav"), dur=2.0)
    node = asm.FusionMLXAssemble()
    out = node.render(
        clips=f"{c1}\n{c2}", transition="none",
        width=64, height=128, fps=15,
        filename_prefix="asm_bgm", bgm_path=bgm,
    )[0]
    assert os.path.exists(out)
    assert ffmpeg_util.probe_has_audio(out), "应混入 BGM"


def test_assemble_fade_transition(tmp_path):
    c1 = _make_clip(str(tmp_path / "c1.mp4"), "red", 0.4, with_audio=False)
    c2 = _make_clip(str(tmp_path / "c2.mp4"), "blue", 0.4, with_audio=False)
    node = asm.FusionMLXAssemble()
    out = node.render(
        clips=f"{c1}\n{c2}", transition="fade",
        width=64, height=128, fps=15,
        filename_prefix="asm_fade", bgm_path="",
    )[0]
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_assemble_empty_clips_raises(tmp_path):
    node = asm.FusionMLXAssemble()
    with pytest.raises(RuntimeError):
        node.render(clips="", transition="none", width=64, height=128, fps=15,
                    filename_prefix="asm_empty", bgm_path="")
