from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .context import PipelineContext

logger = logging.getLogger("custom_nodes4macos.pipeline.checkpoint")

_CHECKPOINT_FILE = "_checkpoint.json"


@dataclass
class CheckpointData:
    job_id: str = ""
    content_type: str = ""
    template_name: str = ""
    completed_stages: list[str] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    config_overrides: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class CheckpointManager:

    def __init__(self, job_dir: str):
        self._job_dir = job_dir
        self._path = os.path.join(job_dir, _CHECKPOINT_FILE)

    def save(self, ctx: PipelineContext) -> None:
        data = CheckpointData(
            job_id=ctx.job_id,
            content_type=ctx.config.get("content_type", ""),
            template_name=ctx.config.get("template_name", ""),
            completed_stages=list(ctx.completed_stages),
            scenes=list(ctx.scenes),
            artifacts=dict(ctx.artifacts),
            config_overrides=ctx.config.get("overrides", dict(ctx.config)),
            created_at=ctx.created_at or datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        os.makedirs(self._job_dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(asdict(data), f, ensure_ascii=False, indent=2)
        logger.info(
            "checkpoint saved: job=%s stages=%s artifacts=%d",
            ctx.job_id, ctx.completed_stages, len(ctx.artifacts),
        )

    def load(self) -> CheckpointData | None:
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError(f"checkpoint root is {type(raw).__name__}, expected dict")
            required_fields = {"job_id"}
            missing = required_fields - set(raw.keys())
            if missing:
                logger.warning("checkpoint missing fields: %s", missing)
            safe = {k: v for k, v in raw.items() if k in CheckpointData.__dataclass_fields__}
            return CheckpointData(**safe)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("checkpoint load failed (corrupt): %s", exc)
            return None
        except TypeError as exc:
            logger.error("checkpoint load failed (schema mismatch): %s", exc)
            return None

    def restore_context(self, ctx: PipelineContext) -> bool:
        cp = self.load()
        if cp is None:
            return False
        ctx.completed_stages = cp.completed_stages
        ctx.scenes = cp.scenes
        ctx.artifacts = cp.artifacts
        ctx.created_at = cp.created_at
        logger.info(
            "checkpoint restored: job=%s stages=%s scenes=%d artifacts=%d",
            ctx.job_id, cp.completed_stages, len(cp.scenes), len(cp.artifacts),
        )
        return bool(cp.completed_stages)
