from __future__ import annotations

import gc
import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger("custom_nodes4macos.pipeline.model_manager")

_HF_CACHE_ROOT = os.path.expanduser("~/.cache/huggingface/hub")


def _resolve_local_path(repo_id: str) -> str | None:
    if os.path.isdir(repo_id):
        return repo_id
    org_name = repo_id.replace("/", "--")
    model_dir = os.path.join(_HF_CACHE_ROOT, f"models--{org_name}")
    if not os.path.isdir(model_dir):
        return None
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snap_dir):
        return None
    commits = sorted(os.listdir(snap_dir))
    if not commits:
        return None
    latest = os.path.join(snap_dir, commits[-1])
    if os.path.isdir(latest):
        logger.info("resolved %s -> %s", repo_id, latest)
        return latest
    return None


class ModelMode(Enum):
    SEQUENTIAL = "sequential"
    RESIDENT = "resident"


class ModelHandle:

    def __init__(self, name: str, model: Any, manager: ModelManager):
        self._name = name
        self._model = model
        self._manager = manager

    @property
    def model(self) -> Any:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    def release(self):
        self._manager.release(self._name)


class _AcquireContext:

    def __init__(self, mgr: ModelManager, name: str):
        self._mgr = mgr
        self._name = name
        self._handle: ModelHandle | None = None

    def __enter__(self) -> ModelHandle:
        self._handle = self._mgr._acquire_handle(self._name)
        return self._handle

    def __exit__(self, *exc):
        if self._mgr._mode == ModelMode.SEQUENTIAL:
            self._mgr.release(self._name)
        return False


class ModelManager:

    MODEL_REGISTRY = {
        "llm": {
            "path": "mlx-community/Qwen3.5-9B-4bit",
            "memory_gb": 5.6,
            "loader": "_load_llm",
        },
        "flux": {
            "path": "mlx-community/Flux-1.lite-8B-MLX-Q4",
            "memory_gb": 7.0,
            "loader": "_load_flux",
        },
        "tts": {
            "path": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            "memory_gb": 2.9,
            "loader": "_load_tts",
        },
    }

    def __init__(
        self,
        mode: ModelMode = ModelMode.SEQUENTIAL,
        model_overrides: dict | None = None,
        memory_budget_gb: float | None = None,
    ):
        self._mode = mode
        self._loaded: dict[str, Any] = {}
        self._model_overrides = model_overrides or {}
        if memory_budget_gb is not None:
            self._memory_budget = memory_budget_gb
        else:
            self._memory_budget = self._detect_memory_budget()
        self._current_usage = 0.0

    @property
    def mode(self) -> ModelMode:
        return self._mode

    @property
    def current_usage_gb(self) -> float:
        return self._current_usage

    def acquire(self, name: str) -> _AcquireContext:
        return _AcquireContext(self, name)

    def _acquire_handle(self, name: str) -> ModelHandle:
        if name in self._loaded:
            logger.info("model %s returned from cache (resident)", name)
            return ModelHandle(name, self._loaded[name], self)

        reg = self.MODEL_REGISTRY.get(name)
        if reg is None:
            raise ValueError(f"unknown model: {name}")

        needed = reg["memory_gb"]
        if self._current_usage + needed > self._memory_budget:
            logger.warning(
                "model %s (%.1fG) would exceed budget (%.1fG used / %.1fG max), "
                "attempting to release resident models",
                name, needed, self._current_usage, self._memory_budget,
            )
            for loaded_name in list(self._loaded.keys()):
                self.release(loaded_name)
                if self._current_usage + needed <= self._memory_budget:
                    break
            if self._current_usage + needed > self._memory_budget:
                raise MemoryError(
                    f"model {name} ({needed:.1f}G) exceeds memory budget "
                    f"({self._current_usage:.1f}G used / {self._memory_budget:.1f}G max)"
                )

        repo_id_or_path = self._model_overrides.get(name, reg["path"])
        path = _resolve_local_path(repo_id_or_path) or repo_id_or_path
        loader = getattr(self, reg["loader"])
        logger.info("loading model %s from %s ...", name, path)
        model = loader(path)
        logger.info("model %s loaded (%.1fG)", name, reg["memory_gb"])

        if self._mode == ModelMode.RESIDENT:
            self._loaded[name] = model
        self._current_usage += reg["memory_gb"]

        return ModelHandle(name, model, self)

    def release(self, name: str) -> None:
        reg = self.MODEL_REGISTRY.get(name)
        if not reg:
            return
        if name in self._loaded:
            del self._loaded[name]
        if self._current_usage >= reg["memory_gb"]:
            self._current_usage -= reg["memory_gb"]
        else:
            self._current_usage = 0.0
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
            logger.info("model %s released, mx.clear_cache() done", name)
        except ImportError:
            logger.info("model %s released (mlx not available for cache clear)", name)

    @staticmethod
    def _load_llm(path: str):
        from mlx_lm import load
        return load(path)

    @staticmethod
    def _load_flux(path: str):
        try:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config.model_config import ModelConfig
        except ImportError:
            raise ImportError(
                "mflux not installed. Run: pip install mflux"
            )
        return Flux1(
            quantize=4,
            model_path=path,
            model_config=ModelConfig.dev(),
        )

    @staticmethod
    def _load_tts(path: str):
        from mlx_audio.tts.utils import fetch_from_hub
        model, config = fetch_from_hub(path)
        model._tts_config = config
        return model

    @staticmethod
    def _detect_memory_budget() -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / (1024 ** 3)
            budget = min(total_gb * 0.6, total_gb - 4.0)
            logger.info(
                "auto-detected memory budget: %.1fG (system %.1fG, 60%% cap - 4G reserve)",
                budget, total_gb,
            )
            return max(budget, 8.0)
        except Exception:
            logger.info("memory auto-detect failed, defaulting to 12.0G")
            return 12.0
