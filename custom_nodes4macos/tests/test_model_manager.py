from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.model_manager import ModelManager, ModelHandle, ModelMode


class TestModelManagerAcquireRelease(unittest.TestCase):

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "test_model": {"path": "fake", "memory_gb": 1.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value=("model_obj", "tokenizer_obj"))
    def test_sequential_acquire_and_release(self, mock_load):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with mgr.acquire("test_model") as handle:
            self.assertIsInstance(handle, ModelHandle)
            self.assertEqual(handle.model, ("model_obj", "tokenizer_obj"))
        self.assertEqual(mgr.current_usage_gb, 0.0)
        self.assertNotIn("test_model", mgr._loaded)

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "test_model": {"path": "fake", "memory_gb": 2.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="model_obj")
    def test_resident_mode_caches(self, mock_load):
        mgr = ModelManager(mode=ModelMode.RESIDENT)
        with mgr.acquire("test_model") as handle:
            self.assertEqual(handle.model, "model_obj")
        self.assertIn("test_model", mgr._loaded)
        self.assertEqual(mgr.current_usage_gb, 2.0)

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "test_model": {"path": "fake", "memory_gb": 3.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m1")
    def test_resident_reuses_cache(self, mock_load):
        mgr = ModelManager(mode=ModelMode.RESIDENT)
        with mgr.acquire("test_model") as h1:
            pass
        mock_load.return_value = "m2"
        with mgr.acquire("test_model") as h2:
            self.assertEqual(h2.model, "m1")
        self.assertEqual(mock_load.call_count, 1)

    def test_unknown_model_raises(self):
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL)
        with self.assertRaises(ValueError):
            mgr._acquire_handle("nonexistent")

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "test_model": {"path": "fake", "memory_gb": 5.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="obj")
    def test_release_clears_memory(self, mock_load):
        mgr = ModelManager(mode=ModelMode.RESIDENT)
        mgr._acquire_handle("test_model")
        self.assertEqual(mgr.current_usage_gb, 5.0)
        mgr.release("test_model")
        self.assertEqual(mgr.current_usage_gb, 0.0)
        self.assertNotIn("test_model", mgr._loaded)


if __name__ == "__main__":
    unittest.main()
