from __future__ import annotations

import logging
import os
from enum import Enum

from ..fusion_client import FusionMLXClient

logger = logging.getLogger("custom_nodes4macos.pipeline.model_manager")


class ModelMode(Enum):
    SEQUENTIAL = "sequential"
    RESIDENT = "resident"


class RemoteHandle:

    def __init__(self, name: str, client: FusionMLXClient, model_name: str):
        self._name = name
        self._client = client
        self._model_name = model_name

    @property
    def client(self) -> FusionMLXClient:
        return self._client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def name(self) -> str:
        return self._name

    def release(self):
        pass


class _AcquireContext:

    def __init__(self, mgr: "ModelManager", name: str):
        self._mgr = mgr
        self._name = name

    def __enter__(self) -> RemoteHandle:
        return self._mgr._acquire_handle(self._name)

    def __exit__(self, *exc):
        return False


class ModelManager:

    MODEL_REGISTRY = {
        "llm": {
            "model_name": os.environ.get("FUSION_LLM_MODEL", "Qwen3.5-9B-4bit"),
        },
        "flux": {
            "model_name": os.environ.get("FUSION_FLUX_MODEL", ""),
        },
        "tts": {
            "model_name": os.environ.get("FUSION_TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-Base-8bit"),
        },
    }

    def __init__(
        self,
        mode: ModelMode = ModelMode.SEQUENTIAL,
        model_overrides: dict | None = None,
        memory_budget_gb: float | None = None,
    ):
        self._mode = mode
        self._model_overrides = model_overrides or {}
        self._client = FusionMLXClient()
        self._health_checked = False

    @property
    def mode(self) -> ModelMode:
        return self._mode

    @property
    def current_usage_gb(self) -> float:
        return 0.0

    def acquire(self, name: str) -> _AcquireContext:
        return _AcquireContext(self, name)

    def _acquire_handle(self, name: str) -> RemoteHandle:
        reg = self.MODEL_REGISTRY.get(name)
        if reg is None:
            raise ValueError(f"unknown model: {name}")
        model_name = self._model_overrides.get(name, reg["model_name"])
        if not self._health_checked:
            if not self._client.health():
                logger.warning(
                    "fusion-mlx unreachable at acquire(%s); HTTP calls will fail", name,
                )
            self._health_checked = True
        logger.info("acquire %s -> remote model=%s", name, model_name or "(unset)")
        return RemoteHandle(name, self._client, model_name)

    def shutdown(self) -> None:
        self._client.close()
        logger.info("ModelManager shutdown: fusion-mlx client closed")

    def release(self, name: str) -> None:
        pass
