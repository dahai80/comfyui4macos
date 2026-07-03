import logging
import os
import time
import uuid

from ..fusion_client import FusionMLXClient, FusionMLXError, list_models_safe

logger = logging.getLogger("custom_nodes4macos.horror_tts")

_DEFAULT_INSTRUCTIONS = (
    "低沉、压抑、略带颤抖的中式恐怖旁白语气；语速偏慢，停顿处留白以制造悬念"
)

_AUDIO_EXTS = ("wav",)


def _resolve_tts_model_from(models: list[str], model: str = "(auto)") -> str:
    if model and model != "(auto)":
        return model
    for m in models:
        if m == "(auto)":
            continue
        if "tts" in m.lower():
            return m
    return "tts-1"


def _resolve_tts_model(model: str) -> str:
    return _resolve_tts_model_from(list_models_safe(), model)


def _output_directory() -> str:
    try:
        from folder_paths import get_output_directory

        out = get_output_directory()
        if out:
            os.makedirs(out, exist_ok=True)
            return out
    except Exception as exc:
        logger.debug("folder_paths unavailable, fallback to tempfile: %s", exc)
    import tempfile

    return tempfile.gettempdir()


def _save_audio(audio_bytes: bytes, filename_prefix: str, ext: str) -> str:
    ext = ext.lower()
    if ext not in _AUDIO_EXTS:
        ext = "wav"
    out_dir = _output_directory()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    fname = f"{filename_prefix or 'horror_tts'}_{stamp}_{short}.{ext}"
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as fh:
        fh.write(audio_bytes)
    return path


class FusionMLXHorrorTTS:
    @classmethod
    def INPUT_TYPES(cls):
        models = list_models_safe()
        return {
            "required": {
                "audio_script": ("STRING", {"multiline": True, "default": ""}),
                "voice": ("STRING", {"default": ""}),
                "instructions": ("STRING", {"multiline": True, "default": _DEFAULT_INSTRUCTIONS}),
                "model": (models, {"default": "(auto)"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "response_format": (["wav"], {"default": "wav"}),
                "filename_prefix": ("STRING", {"default": "horror_tts"}),
            },
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "synthesize"
    CATEGORY = "FusionMLX/Horror"

    def synthesize(
        self,
        audio_script: str,
        voice: str,
        instructions: str,
        model: str,
        speed: float,
        response_format: str,
        filename_prefix: str,
        base_url: str = "",
        api_key: str = "",
    ):
        text = (audio_script or "").strip()
        if not text:
            raise ValueError("audio_script 不能为空")
        model_name = _resolve_tts_model(model)
        logger.info(
            "synthesize text_len=%d model=%s voice=%s speed=%.2f",
            len(text), model_name, voice or "(default)", speed,
        )
        with FusionMLXClient(base_url=base_url or None, api_key=api_key or None) as client:
            if not client.health():
                raise FusionMLXError(f"fusion-mlx 不可达: {client.base_url}")
            audio_bytes = client.synthesize_speech(
                text=text,
                model=model_name,
                voice=voice or None,
                instructions=instructions or None,
                speed=speed,
                response_format=response_format or "wav",
            )
        if not audio_bytes:
            raise FusionMLXError("synthesize_speech 返回空结果")
        path = _save_audio(audio_bytes, filename_prefix, response_format or "wav")
        logger.info("synthesize done path=%s bytes=%d", path, len(audio_bytes))
        return (path,)
