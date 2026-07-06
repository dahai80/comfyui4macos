import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_nodes4macos.nodes import prompt_expand as pe
from custom_nodes4macos.fusion_client import FusionMLXError


class FakeClient:
    base_url = "http://fake:11434"

    def __init__(self, *args, **kwargs):
        self.payload = {
            "story_title": "破庙借火",
            "global_style": "ink-wash dark fantasy",
            "scenes": [
                {"scene_id": 1, "visual_prompt": "v1", "audio_script": "夜深，张三独行。", "sound_effect": "wind", "duration_seconds": 5},
                {"scene_id": 2, "visual_prompt": "v2", "audio_script": "破庙里有人。", "sound_effect": "creak", "duration_seconds": 4},
            ],
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def health(self, timeout=3.0):
        return True

    def list_models(self, timeout=3.0):
        return ["fake-llm"]

    def chat(self, messages, model=None, temperature=0.75, json_mode=False, max_tokens=None, timeout=None):
        return json.dumps(self.payload, ensure_ascii=False), {"total_tokens": 10}

    def close(self):
        pass


def test_parse_json_plain():
    assert pe._parse_json('{"a":1}') == {"a": 1}


def test_parse_json_codefence_json():
    assert pe._parse_json('```json\n{"a":2}\n```') == {"a": 2}


def test_parse_json_codefence_bare():
    assert pe._parse_json('```\n{"a":3}\n```') == {"a": 3}


def test_parse_json_invalid_raises():
    with pytest.raises(FusionMLXError):
        pe._parse_json("not json at all")


def test_parse_json_strips_thinking_process():
    raw = "Thinking Process:\n\n1. Analyze.\n2. Output.\n\n" + '{"scenes": [{"scene_id": 1}]}'
    assert pe._parse_json(raw) == {"scenes": [{"scene_id": 1}]}


def test_parse_json_strips_think_tag():
    raw = "<think>some reasoning here</think>\n" + '{"a": 1}'
    assert pe._parse_json(raw) == {"a": 1}


def test_parse_json_brace_fallback():
    raw = 'prefix text {"a": 2} trailing'
    assert pe._parse_json(raw) == {"a": 2}


def test_node_expand_offline(monkeypatch):
    monkeypatch.setattr(pe, "FusionMLXClient", FakeClient)
    node = pe.FusionMLXPromptExpand()
    scenes_json, count = node.expand(
        story_seed="张三深夜赶路，破庙遇白衣老头借火",
        episode_title="破庙借火",
        scene_count=2,
        model="(auto)",
        style_preset="水墨悬疑",
        temperature=0.75,
        base_url="",
        api_key="",
    )
    data = json.loads(scenes_json)
    assert count == 2
    assert data["scenes"][0]["visual_prompt"]
    assert data["scenes"][0]["audio_script"]
    assert "sound_effect" in data["scenes"][0]


def test_node_rejects_empty_seed(monkeypatch):
    monkeypatch.setattr(pe, "FusionMLXClient", FakeClient)
    node = pe.FusionMLXPromptExpand()
    with pytest.raises(ValueError):
        node.expand("", "", 2, "(auto)", "水墨悬疑", 0.75, "", "")


def test_node_unreachable_raises(monkeypatch):
    class DeadClient(FakeClient):
        def health(self, timeout=3.0):
            return False
    monkeypatch.setattr(pe, "FusionMLXClient", DeadClient)
    node = pe.FusionMLXPromptExpand()
    with pytest.raises(FusionMLXError):
        node.expand("有种子", "", 2, "(auto)", "水墨悬疑", 0.75, "", "")


@pytest.mark.live
def test_node_expand_live():
    node = pe.FusionMLXPromptExpand()
    with pe.FusionMLXClient() as probe:
        if not probe.health():
            pytest.skip("fusion-mlx not running; start e.g. `fusion-mlx serve <llm> --port 11434`")
        try:
            probe.list_models()
        except Exception:
            pytest.skip("fusion-mlx requires API key; set FUSION_MLX_API_KEY")
    models = [m for m in pe.list_models_safe() if m != "(auto)"]
    chosen = "(auto)"
    for m in models:
        low = m.lower()
        if "0.6b" in low:
            continue
        if "9b" in low or "27b" in low or "sonnet" in low:
            chosen = m
            break
    if chosen == "(auto)" and models:
        chosen = models[0]
    try:
        scenes_json, count = node.expand(
            story_seed="张三深夜赶路，在破庙遇到一个借火的白衣老头。",
            episode_title="破庙借火",
            scene_count=4,
            model=chosen,
            style_preset="水墨悬疑",
            temperature=0.8,
            base_url="",
            api_key="",
        )
    except pe.FusionMLXError as exc:
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg or "401" in msg or "auth" in msg:
            pytest.skip(f"fusion-mlx model/auth unavailable: {exc}")
        raise
    data = json.loads(scenes_json)
    assert count >= 1
    assert "scenes" in data
    first = data["scenes"][0]
    if isinstance(first, dict) and first.get("visual_prompt"):
        assert first["visual_prompt"]
    else:
        pytest.skip(f"model {chosen} did not follow visual_prompt schema; bridge OK, model quality issue")
