from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .checkpoint import CheckpointManager
from .context import PipelineContext
from .model_manager import ModelManager, ModelMode
from .result import PipelineResult
from .stage import Stage

logger = logging.getLogger("custom_nodes4macos.pipeline.engine")

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_OUTPUT_ROOT = Path(os.path.expanduser("~/output/dream_factory"))

_STAGE_REGISTRY: dict[str, type[Stage]] = {}


def register_stage(cls: type[Stage]) -> type[Stage]:
    name = cls.info().name
    _STAGE_REGISTRY[name] = cls
    logger.debug("stage registered: %s → %s", name, cls.__name__)
    return cls


def _auto_discover_stages() -> None:
    from . import stages as _stages_pkg
    pkg_dir = Path(_stages_pkg.__file__).resolve().parent
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = f"{_stages_pkg.__name__}.{py.stem}"
        try:
            __import__(mod_name)
        except Exception as exc:
            logger.warning("stage import failed: %s → %s", mod_name, exc)


class PipelineEngine:

    def __init__(self, output_root: str | None = None):
        self._output_root = Path(output_root) if output_root else _OUTPUT_ROOT
        self._templates: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        _auto_discover_stages()
        self._load_templates()
        self._loaded = True

    def _load_templates(self) -> None:
        if not _TEMPLATE_DIR.exists():
            logger.warning("template dir missing: %s", _TEMPLATE_DIR)
            return
        for yf in sorted(_TEMPLATE_DIR.glob("*.yaml")):
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    tpl = yaml.safe_load(f)
                ct = tpl.get("content_type", yf.stem)
                self._templates[ct] = tpl
                logger.info("template loaded: %s → %s", ct, yf.name)
            except Exception as exc:
                logger.error("template load failed: %s → %s", yf.name, exc)

    def _get_template(self, content_type: str) -> dict:
        if content_type in self._templates:
            return self._templates[content_type]
        raise ValueError(
            f"unknown content_type: {content_type}, available: {list(self._templates.keys())}"
        )

    @staticmethod
    def _make_job_id() -> str:
        ts = time.strftime("%Y%m%d")
        uid = uuid.uuid4().hex[:8]
        return f"{ts}_{uid}"

    def _merge_config(self, template: dict, user_config: dict) -> dict:
        config = dict(template.get("defaults", {}))
        config["content_type"] = template.get("content_type", "")
        config["template_name"] = template.get("name", "")
        config["stages"] = template.get("stages", [])
        prompts = template.get("prompts", {})
        for k, v in prompts.items():
            config[k] = v
        style_presets = template.get("style_presets", {})
        if style_presets:
            config["style_presets"] = style_presets
        config.update(user_config)
        return config

    def _instantiate_stages(self, stage_names: list[str]) -> list[Stage]:
        stages = []
        for name in stage_names:
            cls = _STAGE_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"unknown stage: {name}, registered: {list(_STAGE_REGISTRY.keys())}")
            stages.append(cls())
        return stages

    def run(
        self,
        content_type: str,
        resume_from: str | None = None,
        **user_config: Any,
    ) -> PipelineResult:
        self._ensure_loaded()

        if resume_from:
            return self._resume(resume_from)

        template = self._get_template(content_type)
        config = self._merge_config(template, user_config)

        job_id = self._make_job_id()
        job_dir = str(self._output_root / job_id)
        os.makedirs(job_dir, exist_ok=True)

        ctx = PipelineContext(job_id=job_id, job_dir=job_dir, config=config)
        ctx.created_at = datetime.now().isoformat()

        checkpoint = CheckpointManager(job_dir)
        checkpoint.save(ctx)

        return self._execute(ctx, checkpoint)

    def _resume(self, job_id: str) -> PipelineResult:
        job_dir = str(self._output_root / job_id)
        if not os.path.isdir(job_dir):
            raise ValueError(f"job not found: {job_id} (dir: {job_dir})")

        checkpoint = CheckpointManager(job_dir)
        cp = checkpoint.load()
        if cp is None:
            raise ValueError(f"checkpoint not found for job: {job_id}")

        content_type = cp.content_type
        template = self._get_template(content_type) if content_type else {}
        config = self._merge_config(template, cp.config_overrides)
        config["content_type"] = content_type

        ctx = PipelineContext(job_id=job_id, job_dir=job_dir, config=config)
        ctx.created_at = cp.created_at
        checkpoint.restore_context(ctx)

        logger.info(
            "resume job=%s completed_stages=%s artifacts=%d",
            job_id, ctx.completed_stages, len(ctx.artifacts),
        )
        return self._execute(ctx, checkpoint)

    def _execute(
        self,
        ctx: PipelineContext,
        checkpoint: CheckpointManager,
    ) -> PipelineResult:
        stage_names = ctx.config.get("stages", [])
        if not stage_names:
            logger.warning("no stages configured for content_type=%s", ctx.config.get("content_type"))
            return PipelineResult(job_id=ctx.job_id, job_dir=ctx.job_dir)

        stages = self._instantiate_stages(stage_names)
        memory_budget = ctx.config.get("memory_budget_gb", None)
        model_mgr = ModelManager(
            mode=ModelMode.SEQUENTIAL,
            memory_budget_gb=memory_budget,
        )

        for stage in stages:
            info = stage.info()
            if info.name in ctx.completed_stages:
                logger.info("stage %s skipped (completed)", info.name)
                continue

            logger.info(
                "stage %s starting (memory_est=%.1fG models=%s)",
                info.name, info.memory_estimate_gb, info.model_requirements,
            )
            try:
                stage.process(ctx, model_mgr)
            except Exception as exc:
                logger.error("stage %s failed: %s", info.name, exc)
                checkpoint.save(ctx)
                raise

            ctx.completed_stages.append(info.name)
            checkpoint.save(ctx)
            logger.info("stage %s completed, checkpoint saved", info.name)

        final_video = ctx.get_artifact(0, "final")
        return PipelineResult(
            job_id=ctx.job_id,
            job_dir=ctx.job_dir,
            final_video=final_video,
        )

    def list_templates(self) -> list[str]:
        self._ensure_loaded()
        return list(self._templates.keys())

    def list_jobs(self) -> list[dict]:
        jobs = []
        if not self._output_root.exists():
            return jobs
        for d in sorted(self._output_root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            cp_path = d / "_checkpoint.json"
            if not cp_path.exists():
                continue
            try:
                checkpoint = CheckpointManager(str(d))
                cp = checkpoint.load()
                if cp:
                    jobs.append({
                        "job_id": cp.job_id,
                        "content_type": cp.content_type,
                        "completed_stages": cp.completed_stages,
                        "updated_at": cp.updated_at,
                    })
            except Exception as exc:
                logger.warning("list_jobs skip %s: %s", d.name, exc)
        return jobs
