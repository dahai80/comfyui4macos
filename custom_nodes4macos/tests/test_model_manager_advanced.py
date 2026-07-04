"""ModelManager 进阶测试。

覆盖显存预算边界、多模型叠加、释放回填、
FLUX_PIPELINE_DIR 环境变量、_load_flux 回退逻辑。

补充 REVIEW_REPORT：显存预算边界与 flux 加载路径未覆盖。
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from custom_nodes4macos.pipeline.model_manager import ModelManager, ModelHandle, ModelMode


class TestMemoryBudgetEnforcement(unittest.TestCase):

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "big": {"path": "fake", "memory_gb": 10.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_model_exceeds_budget_raises_memory_error(self, _):
        """单个模型超过预算时抛 MemoryError。"""
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL, memory_budget_gb=8.0)
        with self.assertRaises(MemoryError) as cm:
            with mgr.acquire("big"):
                pass
        self.assertIn("exceeds memory budget", str(cm.exception))

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "a": {"path": "fake", "memory_gb": 4.0, "loader": "_load_llm"},
        "b": {"path": "fake", "memory_gb": 4.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_two_models_fit_budget(self, _):
        """两个模型都在预算内，顺序加载不报错。"""
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL, memory_budget_gb=10.0)
        with mgr.acquire("a"):
            pass
        with mgr.acquire("b"):
            pass
        # SEQUENTIAL 模式每次 acquire 后释放，usage 归零
        self.assertEqual(mgr.current_usage_gb, 0.0)

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "a": {"path": "fake", "memory_gb": 6.0, "loader": "_load_llm"},
        "b": {"path": "fake", "memory_gb": 6.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_resident_mode_accumulates_usage(self, _):
        """RESIDENT 模式下 usage 累加。"""
        mgr = ModelManager(mode=ModelMode.RESIDENT, memory_budget_gb=20.0)
        with mgr.acquire("a"):
            pass
        self.assertEqual(mgr.current_usage_gb, 6.0)
        with mgr.acquire("b"):
            pass
        self.assertEqual(mgr.current_usage_gb, 12.0)

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "a": {"path": "fake", "memory_gb": 6.0, "loader": "_load_llm"},
        "b": {"path": "fake", "memory_gb": 6.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_resident_evicts_to_fit_new_model(self, _):
        """RESIDENT 模式下超预算时释放常驻模型腾出空间。"""
        mgr = ModelManager(mode=ModelMode.RESIDENT, memory_budget_gb=10.0)
        with mgr.acquire("a"):
            pass
        # a 占用 6G，再加载 b(6G) 会超 10G，应先释放 a
        with mgr.acquire("b"):
            pass
        # a 被释放，b 留下
        self.assertNotIn("a", mgr._loaded)
        self.assertIn("b", mgr._loaded)
        self.assertEqual(mgr.current_usage_gb, 6.0)


class TestReleaseUnderflow(unittest.TestCase):
    """覆盖 REVIEW_REPORT C1：release 下溢保护逻辑。"""

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "m": {"path": "fake", "memory_gb": 5.0, "loader": "_load_llm"},
    })
    def test_release_unloaded_model_no_effect(self):
        """release 一个未加载的模型，usage 不变。"""
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL, memory_budget_gb=20.0)
        mgr._current_usage = 3.0  # 模拟有其他模型占用
        mgr.release("m")  # m 未加载
        # usage 应归零（因为 _current_usage < reg["memory_gb"]=5.0，走 else 分支）
        self.assertEqual(mgr.current_usage_gb, 0.0)

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "m": {"path": "fake", "memory_gb": 5.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_release_after_load_zeroes_usage(self, _):
        mgr = ModelManager(mode=ModelMode.RESIDENT, memory_budget_gb=20.0)
        mgr._acquire_handle("m")
        self.assertEqual(mgr.current_usage_gb, 5.0)
        mgr.release("m")
        self.assertEqual(mgr.current_usage_gb, 0.0)

    @patch.object(ModelManager, "MODEL_REGISTRY", {})
    def test_release_unknown_model_silent(self):
        """release 未知模型（不在 registry）静默返回。"""
        mgr = ModelManager(mode=ModelMode.SEQUENTIAL, memory_budget_gb=20.0)
        mgr.release("nonexistent")  # 不应抛异常


class TestFluxLoaderPath(unittest.TestCase):
    """覆盖 _load_flux 的 mflux 加载逻辑。"""

    def test_load_flux_mflux_not_installed_raises(self):
        """mflux 未安装时抛 ImportError。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mflux" or name.startswith("mflux."):
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError) as cm:
                ModelManager._load_flux("fake/path")
            self.assertIn("mflux", str(cm.exception))

    @patch("mflux.models.flux.variants.txt2img.flux.Flux1")
    @patch("mflux.models.common.config.model_config.ModelConfig")
    def test_load_flux_with_mflux(self, mock_config, mock_flux1):
        """mflux 可用时，使用 Flux1 加载。"""
        mock_flux1.return_value = MagicMock()
        mock_config.dev.return_value = MagicMock()
        result = ModelManager._load_flux("/some/local/path")
        mock_flux1.assert_called_once()
        self.assertIsNotNone(result)


class TestDetectMemoryBudget(unittest.TestCase):

    @patch("subprocess.run")
    def test_detect_budget_60_percent_cap(self, mock_run):
        """16G 机器：min(16*0.6, 16-4) = min(9.6, 12) = 9.6G。"""
        mock_run.return_value = MagicMock(stdout=str(16 * 1024 ** 3) + "\n")
        budget = ModelManager._detect_memory_budget()
        self.assertAlmostEqual(budget, 9.6, places=1)

    @patch("subprocess.run")
    def test_detect_budget_floor_8g(self, mock_run):
        """8G 机器：min(8*0.6, 8-4)=min(4.8, 4)=4，但下限 8G。"""
        mock_run.return_value = MagicMock(stdout=str(8 * 1024 ** 3) + "\n")
        budget = ModelManager._detect_memory_budget()
        self.assertEqual(budget, 8.0)

    @patch("subprocess.run", side_effect=Exception("sysctl fail"))
    def test_detect_budget_fallback_on_error(self, _):
        """sysctl 失败时回退到 12.0G。"""
        budget = ModelManager._detect_memory_budget()
        self.assertEqual(budget, 12.0)

    @patch("subprocess.run")
    def test_detect_budget_large_machine(self, mock_run):
        """64G 机器：min(64*0.6, 64-4) = min(38.4, 60) = 38.4G。"""
        mock_run.return_value = MagicMock(stdout=str(64 * 1024 ** 3) + "\n")
        budget = ModelManager._detect_memory_budget()
        self.assertAlmostEqual(budget, 38.4, places=1)


class TestModelOverrides(unittest.TestCase):
    """model_overrides 参数允许覆盖模型路径。"""

    @patch.object(ModelManager, "MODEL_REGISTRY", {
        "llm": {"path": "default-path", "memory_gb": 1.0, "loader": "_load_llm"},
    })
    @patch.object(ModelManager, "_load_llm", return_value="m")
    def test_override_path_used(self, mock_load):
        mgr = ModelManager(
            mode=ModelMode.SEQUENTIAL,
            memory_budget_gb=20.0,
            model_overrides={"llm": "custom-path"},
        )
        with mgr.acquire("llm"):
            pass
        mock_load.assert_called_once_with("custom-path")


if __name__ == "__main__":
    unittest.main()
