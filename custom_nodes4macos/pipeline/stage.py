from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger("custom_nodes4macos.pipeline.stage")


@dataclass(frozen=True)
class StageInfo:
    name: str
    description: str
    model_requirements: list[str] = field(default_factory=list)
    memory_estimate_gb: float = 0.0
    input_kinds: list[str] = field(default_factory=list)
    output_kinds: list[str] = field(default_factory=list)


class Stage(ABC):

    @classmethod
    @abstractmethod
    def info(cls) -> StageInfo:
        ...

    @abstractmethod
    def process(self, ctx, model_manager) -> None:
        ...

    def _skip_if_completed(self, ctx) -> bool:
        if self.info().name in ctx.completed_stages:
            logger.info("stage %s skipped (checkpoint)", self.info().name)
            return True
        return False
