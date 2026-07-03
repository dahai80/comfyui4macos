import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_nodes4macos.nodes import horror_tts as ht
from custom_nodes4macos.fusion_client import FusionMLXError


class FakeClient:
    base_url = "http://fake:11434"

    def __init__(self, *args, **kwargs):
        self.captured = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def health(self, timeout=1.5):
        return True

    def synthesize_speech(self, text, model, voice=None, instructions=None,
                          speed=1.0, response_format="wav", stream=False, timeout=None):
        self.captured = {
            "text": text, "model": model, "voice": voice, "instructions": instructions,
            "speed": speed, "response_format": response_format,
        }
        return b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav-bytes"

    def close(self):
        pass


def test_resolve_tts_model_picks_tts_from_list():
    models = ["(auto)", "Qwen3-0.6B-4bit", "Qwen3-TTS-12Hz-1.7B-Base-8bit"]
    assert ht._resolve_tts_model_from(models) == "Qwen3-TTS-12Hz-1.7B-Base-8bit"


def test_resolve_tts_model_falls_back_when_no_tts():
    assert ht._resolve_tts_model_from(["(auto)", "Qwen3-0.6B-4bit"]) == "tts-1"


def test_resolve_tts_model_keeps_explicit():
    assert ht._resolve_tts_model_from(["(auto)", "Qwen3-TTS-x"], "cosyvoice-v1") == "cosyvoice-v1"


def test_node_synthesize_offline(monkeypatch, tmp_path):
    fake = FakeClient()
    monkeypatch.setattr(ht, "FusionMLXClient", lambda *a, **k: fake)
    monkeypatch.setattr(ht, "_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(ht, "list_models_safe", lambda: ["(auto)", "FakeTTS-Model-8bit"])
    node = ht.FusionMLXHorrorTTS()
    (path,) = node.synthesize(
        audio_script="夜深，张三独行至破庙。",
        voice="",
        instructions="低沉压抑",
        model="(auto)",
        speed=0.9,
        response_format="wav",
        filename_prefix="tts_test",
        base_url="",
        api_key="",
    )
    assert os.path.exists(path)
    assert os.path.basename(path).startswith("tts_test_")
    assert path.endswith(".wav")
    with open(path, "rb") as fh:
        assert fh.read().startswith(b"RIFF")
    assert fake.captured["model"] == "FakeTTS-Model-8bit"
    assert fake.captured["voice"] is None
    assert fake.captured["speed"] == 0.9


def test_node_named_model_passed_through(monkeypatch, tmp_path):
    fake = FakeClient()
    monkeypatch.setattr(ht, "FusionMLXClient", lambda *a, **k: fake)
    monkeypatch.setattr(ht, "_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(ht, "list_models_safe", lambda: ["(auto)"])
    node = ht.FusionMLXHorrorTTS()
    node.synthesize(
        audio_script="test",
        voice="narrator",
        instructions="",
        model="cosyvoice-v1",
        speed=1.0,
        response_format="wav",
        filename_prefix="tts_test",
        base_url="",
        api_key="",
    )
    assert fake.captured["model"] == "cosyvoice-v1"
    assert fake.captured["voice"] == "narrator"


def test_node_rejects_empty_script(monkeypatch):
    monkeypatch.setattr(ht, "FusionMLXClient", FakeClient)
    node = ht.FusionMLXHorrorTTS()
    with pytest.raises(ValueError):
        node.synthesize("", "", "", "(auto)", 1.0, "wav", "tts", "", "")


def test_node_unreachable_raises(monkeypatch):
    class DeadClient(FakeClient):
        def health(self, timeout=1.5):
            return False
    monkeypatch.setattr(ht, "FusionMLXClient", DeadClient)
    node = ht.FusionMLXHorrorTTS()
    with pytest.raises(FusionMLXError):
        node.synthesize("有台词", "", "", "(auto)", 1.0, "wav", "tts", "", "")


@pytest.mark.live
def test_node_synthesize_live():
    node = ht.FusionMLXHorrorTTS()
    with ht.FusionMLXClient() as probe:
        if not probe.health():
            pytest.skip("fusion-mlx not running; start e.g. `fusion-mlx serve` on port 11434")
        try:
            probe.list_models()
        except Exception:
            pytest.skip("fusion-mlx requires API key; set FUSION_MLX_API_KEY")
    try:
        (path,) = node.synthesize(
            audio_script="夜深了，破庙里传来一声叹息。",
            voice="",
            instructions="低沉、压抑、略带颤抖的中式恐怖旁白",
            model="(auto)",
            speed=0.9,
            response_format="wav",
            filename_prefix="horror_tts_live",
            base_url="",
            api_key="",
        )
    except FusionMLXError as exc:
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg or "401" in msg or "auth" in msg:
            pytest.skip(f"TTS model not available in this fusion-mlx instance: {exc}")
        raise
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
