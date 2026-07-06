import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_nodes4macos.nodes import flux_image as fi
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

    def generate_image(self, prompt, model=None, width=1024, height=1024, steps=4,
                       seed=None, guidance=4.0, n=1, response_format="b64_json", timeout=None):
        self.captured = {
            "prompt": prompt, "model": model, "width": width, "height": height,
            "steps": steps, "seed": seed, "guidance": guidance, "n": n,
        }
        return [b"\x89PNG\r\n\x1a\nfake-image-bytes"]

    def close(self):
        pass


def test_build_prompt_empty_raises():
    with pytest.raises(ValueError):
        fi._build_prompt("", "some style")


def test_build_prompt_with_style():
    out = fi._build_prompt("破庙", "ink-wash, 8k")
    assert out == "破庙, ink-wash, 8k"


def test_build_prompt_without_style():
    assert fi._build_prompt("破庙", "  ") == "破庙"


def test_resolve_flux_model_picks_flux_from_list():
    models = ["(auto)", "Qwen3-0.6B-4bit", "Flux-1.lite-8B-MLX-Q4"]
    assert fi._resolve_flux_model_from(models) == "Flux-1.lite-8B-MLX-Q4"


def test_resolve_flux_model_falls_back_when_no_flux():
    assert fi._resolve_flux_model_from(["(auto)", "Qwen3-0.6B-4bit"]) is None


def test_resolve_flux_model_keeps_explicit():
    assert fi._resolve_flux_model_from(["(auto)", "Flux-x"], "my-flux-manual") == "my-flux-manual"


def test_node_generate_offline(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(fi, "FusionMLXClient", lambda *a, **k: fake)
    monkeypatch.setattr(fi, "list_models_safe", lambda: ["(auto)", "Flux-Test-Model"])
    monkeypatch.setattr(
        fi, "_bytes_to_image_tensor",
        lambda b: SimpleNamespace(shape=(1, 8, 8, 3), raw=b),
    )
    node = fi.FusionMLXFluxImage()
    (tensor,) = node.generate(
        visual_prompt="破庙夜景，白衣老头",
        global_style="ink-wash dark fantasy, 8k",
        model="(auto)",
        width=768,
        height=1024,
        steps=4,
        guidance=4.0,
        seed=0,
        base_url="",
        api_key="",
    )
    assert tensor.shape == (1, 8, 8, 3)
    assert tensor.raw == b"\x89PNG\r\n\x1a\nfake-image-bytes"
    assert fake.captured["model"] == "Flux-Test-Model"
    assert fake.captured["width"] == 768
    assert fake.captured["seed"] is None
    assert fake.captured["prompt"].startswith("破庙夜景")


def test_node_rejects_empty_prompt(monkeypatch):
    monkeypatch.setattr(fi, "FusionMLXClient", FakeClient)
    node = fi.FusionMLXFluxImage()
    with pytest.raises(ValueError):
        node.generate("", "style", "(auto)", 1024, 1024, 4, 4.0, 0, "", "")


def test_node_unreachable_raises(monkeypatch):
    class DeadClient(FakeClient):
        def health(self, timeout=1.5):
            return False
    monkeypatch.setattr(fi, "FusionMLXClient", DeadClient)
    node = fi.FusionMLXFluxImage()
    with pytest.raises(FusionMLXError):
        node.generate("破庙", "style", "(auto)", 1024, 1024, 4, 4.0, 0, "", "")


@pytest.mark.live
def test_node_generate_live():
    node = fi.FusionMLXFluxImage()
    with fi.FusionMLXClient() as probe:
        if not probe.health():
            pytest.skip("fusion-mlx not running; start e.g. `fusion-mlx serve` on port 11434")
        try:
            probe.list_models()
        except Exception:
            pytest.skip("fusion-mlx requires API key; set FUSION_MLX_API_KEY")
    try:
        (tensor,) = node.generate(
            visual_prompt="中式破庙夜景，月光，白衣老者，水墨风",
            global_style="ink-wash dark fantasy, cinematic, 8k",
            model="(auto)",
            width=512,
            height=512,
            steps=4,
            guidance=4.0,
            seed=0,
            base_url="",
            api_key="",
        )
    except FusionMLXError as exc:
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg or "401" in msg or "auth" in msg:
            pytest.skip(f"image generation not available in this fusion-mlx instance: {exc}")
        raise
    assert tensor is not None
    assert hasattr(tensor, "shape")
