from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.model_manager import (
    ModelManager,
    ModelMode,
    RemoteHandle,
)


class TestRemoteHandle(unittest.TestCase):

    def test_properties_expose_name_client_model(self):
        client = MagicMock()
        handle = RemoteHandle("llm", client, "Qwen3.5-9B-4bit")
        self.assertEqual(handle.name, "llm")
        self.assertIs(handle.client, client)
        self.assertEqual(handle.model_name, "Qwen3.5-9B-4bit")

    def test_release_is_noop(self):
        client = MagicMock()
        handle = RemoteHandle("tts", client, "Qwen3-TTS")
        handle.release()
        client.close.assert_not_called()


class TestModelManagerAcquire(unittest.TestCase):

    def test_acquire_known_returns_remote_handle(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with patch.object(mgr._client, "health", return_value=True):
            with mgr.acquire("llm") as handle:
                self.assertIsInstance(handle, RemoteHandle)
                self.assertEqual(handle.name, "llm")
                self.assertIs(handle.client, mgr._client)

    def test_acquire_unknown_raises(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with self.assertRaises(ValueError):
            with mgr.acquire("nonexistent") as handle:
                pass

    def test_acquire_unreachable_still_returns_handle(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with patch.object(mgr._client, "health", return_value=False):
            with mgr.acquire("llm") as handle:
                self.assertIsInstance(handle, RemoteHandle)
                self.assertEqual(handle.name, "llm")

    def test_model_overrides_override_registry(self):
        mgr = ModelManager(
            mode=ModelMode.SEQUENTIAL,
            model_overrides={"llm": "Custom-LLM"},
        )
        with patch.object(mgr._client, "health", return_value=True):
            with mgr.acquire("llm") as handle:
                self.assertEqual(handle.model_name, "Custom-LLM")

    def test_current_usage_always_zero(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        self.assertEqual(mgr.current_usage_gb, 0.0)
        with patch.object(mgr._client, "health", return_value=True):
            with mgr.acquire("llm"):
                pass
        self.assertEqual(mgr.current_usage_gb, 0.0)

    def test_shutdown_closes_client(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with patch.object(mgr._client, "close") as mock_close:
            mgr.shutdown()
            mock_close.assert_called_once()

    def test_release_is_noop(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        mgr.release("llm")
        self.assertEqual(mgr.current_usage_gb, 0.0)


if __name__ == "__main__":
    unittest.main()
