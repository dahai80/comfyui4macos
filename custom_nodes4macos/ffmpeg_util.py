import logging
import os
import shutil
import subprocess

from .fusion_client import FusionMLXError

logger = logging.getLogger("custom_nodes4macos.ffmpeg_util")

_FFMPEG_PATH = os.environ.get("FFMPEG_BIN", "") or shutil.which("ffmpeg")
_FFPROBE_PATH = os.environ.get("FFPROBE_BIN", "") or shutil.which("ffprobe")
_DEFAULT_TIMEOUT = float(os.environ.get("FFMPEG_TIMEOUT", "300"))
_FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", "0"))

_VT_CACHE: bool | None = None


def has_videotoolbox() -> bool:
    global _VT_CACHE
    if _VT_CACHE is not None:
        return _VT_CACHE
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _VT_CACHE = "h264_videotoolbox" in result.stdout
        return _VT_CACHE
    except Exception:
        _VT_CACHE = False
        return False


def video_encoder_args(quality: int = 65) -> list[str]:
    if has_videotoolbox():
        return ["-c:v", "h264_videotoolbox", "-q:v", str(quality)]
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]


def thread_args() -> list[str]:
    if _FFMPEG_THREADS > 0:
        return ["-threads", str(_FFMPEG_THREADS)]
    return []


def ensure_ffmpeg() -> str:
    if not _FFMPEG_PATH:
        raise FusionMLXError("ffmpeg 未安装/不在 PATH；安装: brew install ffmpeg")
    return _FFMPEG_PATH


def ensure_ffprobe() -> str:
    if not _FFPROBE_PATH:
        raise FusionMLXError("ffprobe 未安装/不在 PATH；安装: brew install ffmpeg")
    return _FFPROBE_PATH


def run_ffmpeg(args: list[str], timeout: float | None = None, label: str = "") -> None:
    ffmpeg = ensure_ffmpeg()
    thread_flags = thread_args()
    cmd = [ffmpeg, "-y", "-loglevel", "error", *thread_flags, *args]
    logger.info("ffmpeg %s args=%s", label or "run", " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout or _DEFAULT_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise FusionMLXError(f"ffmpeg 超时 ({label}): {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-800:]
        logger.error("ffmpeg failed rc=%d stderr=%s", proc.returncode, tail)
        raise FusionMLXError(f"ffmpeg 失败 ({label}) rc={proc.returncode}: {tail}")


def probe_duration(path: str) -> float:
    ffprobe = ensure_ffprobe()
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise FusionMLXError(f"ffprobe 超时: {path}") from exc
    if proc.returncode != 0:
        raise FusionMLXError(f"ffprobe 失败: {proc.stderr.strip()}")
    out = (proc.stdout or "").strip()
    try:
        return float(out)
    except ValueError as exc:
        raise FusionMLXError(f"ffprobe 无法解析时长 {out!r}: {path}") from exc


def probe_has_audio(path: str) -> bool:
    ffprobe = ensure_ffprobe()
    cmd = [ffprobe, "-v", "error", "-select_streams", "a",
           "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    return "audio" in (proc.stdout or "").lower()
