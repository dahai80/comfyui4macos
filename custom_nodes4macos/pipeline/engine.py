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

_FINAL_STAGE: dict[str, str] = {
    "series": "series_orchestrate",
    "short_drama": "subtitle",
    "ad_drama": "assemble",
    "puppet_show": "assemble",
    "medium_video": "assemble",
    "digital_human": "assemble",
    "digital_human_live": "assemble",
}

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
        config["overrides"] = dict(user_config)

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

        if ctx.scenes and not ctx.config.get("episodes"):
            ctx.config["episodes"] = ctx.scenes
            logger.info("resume: synced %d episodes from ctx.scenes to config", len(ctx.scenes))

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
        if ctx.config.get("motion_mode") == "multi_pose" and "ken_burns" in stage_names:
            stage_names = ["multi_pose" if s == "ken_burns" else s for s in stage_names]
            logger.info("motion_mode=multi_pose: swapped ken_burns → multi_pose")
        if not stage_names:
            logger.warning("no stages configured for content_type=%s", ctx.config.get("content_type"))
            return PipelineResult(job_id=ctx.job_id, job_dir=ctx.job_dir)

        stages = self._instantiate_stages(stage_names)
        memory_budget = ctx.config.get("memory_budget_gb", None)
        model_overrides = {}
        for _role in ("llm", "flux", "tts"):
            _val = ctx.config.get(f"{_role}_model")
            if _val:
                model_overrides[_role] = _val
        if model_overrides:
            logger.info("model overrides from config: %s", model_overrides)
        model_mgr = ModelManager(
            mode=ModelMode.SEQUENTIAL,
            model_overrides=model_overrides,
            memory_budget_gb=memory_budget,
        )

        self._warmup_mlx()

        # P4: Chunked rendering — process scenes in batches to save memory
        chunks = ctx.config.get("render_chunks", 0)
        if chunks > 0 and ctx.scenes:
            chunk_size = max(1, len(ctx.scenes) // chunks)
            logger.info("P4: chunked rendering %d scenes in %d chunks (%d/scene)", 
                       len(ctx.scenes), chunks, chunk_size)
            for chunk_start in range(0, len(ctx.scenes), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(ctx.scenes))
                ctx.config["_chunk_range"] = (chunk_start, chunk_end)
                logger.info("P4: processing chunk %d-%d", chunk_start, chunk_end)
                self._run_stages(stages, ctx, model_mgr, checkpoint)
                # Clear memory between chunks (fusion-mlx manages GPU server-side)
                import gc; gc.collect()
        else:
            self._run_stages(stages, ctx, model_mgr, checkpoint)

        return self.finalize_and_return(ctx)

    def _run_stages(
        self,
        stages: list[Stage],
        ctx: PipelineContext,
        model_mgr: ModelManager,
        checkpoint: CheckpointManager,
    ) -> None:
        for stage in stages:
            info = stage.info()
            if info.name in ctx.completed_stages:
                logger.info("stage %s skipped (completed)", info.name)
                continue

            logger.info(
                "stage %s starting (memory_est=%.1fG models=%s model_usage=%.1fG)",
                info.name, info.memory_estimate_gb, info.model_requirements,
                model_mgr.current_usage_gb,
            )
            t_stage = time.time()
            try:
                stage.process(ctx, model_mgr)
            except Exception as exc:
                logger.error("stage %s failed: %s", info.name, exc)
                checkpoint.save(ctx)
                model_mgr.shutdown()
                raise
            t_stage_elapsed = time.time() - t_stage

            ctx.completed_stages.append(info.name)
            checkpoint.save(ctx)
            logger.info(
                "stage %s completed in %.1fs (%.1fmin) model_usage=%.1fG checkpoint saved",
                info.name, t_stage_elapsed, t_stage_elapsed / 60,
                model_mgr.current_usage_gb,
            )

        model_mgr.shutdown()

    def finalize_and_return(self, ctx: PipelineContext) -> PipelineResult:
        final_video = ctx.get_artifact(0, "final")
        return PipelineResult(
            job_id=ctx.job_id,
            job_dir=ctx.job_dir,
            final_video=final_video,
        )

    def list_templates(self) -> list[str]:
        self._ensure_loaded()
        return list(self._templates.keys())

    @staticmethod
    def _warmup_mlx() -> None:
        try:
            from ..fusion_client import FusionMLXClient
            if FusionMLXClient().health():
                logger.info("fusion-mlx reachable (warmup ok)")
            else:
                logger.warning("fusion-mlx unreachable at warmup; pipeline will fail on first acquire")
        except Exception as exc:
            logger.warning("fusion-mlx warmup check failed: %s", exc)

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
                    jobs.append(self._job_summary(cp, str(d)))
            except Exception as exc:
                logger.warning("list_jobs skip %s: %s", d.name, exc)
        return jobs

    def get_job(self, job_id: str) -> dict | None:
        job_dir = self._output_root / job_id
        if not job_dir.is_dir():
            return None
        cp_path = job_dir / "_checkpoint.json"
        if not cp_path.exists():
            return None
        checkpoint = CheckpointManager(str(job_dir))
        cp = checkpoint.load()
        if not cp:
            return None
        summary = self._job_summary(cp, str(job_dir))
        summary["episode_finals"] = self._episode_finals_detail(cp, str(job_dir))
        summary["artifacts"] = cp.artifacts
        summary["character_registry_count"] = len(cp.character_registry)
        summary["global_style"] = cp.global_style
        return summary

    def resolve_job_file(self, job_id: str, filename: str) -> Path | None:
        job_dir = (self._output_root / job_id).resolve()
        if not job_dir.is_dir():
            return None
        target = (job_dir / filename).resolve()
        try:
            target.relative_to(job_dir)
        except ValueError:
            logger.warning("resolve_job_file rejected traversal: job=%s file=%s", job_id, filename)
            return None
        if not target.is_file():
            return None
        return target

    @staticmethod
    def _derive_status(
        content_type: str,
        completed_stages: list[str],
        completed_episodes: list[int],
        total_episodes: int,
    ) -> str:
        final = _FINAL_STAGE.get(content_type)
        if final and final in completed_stages:
            return "done"
        if content_type == "series" and total_episodes and len(completed_episodes) >= total_episodes:
            return "done"
        if completed_stages:
            return "in_progress"
        return "pending"

    def _job_summary(self, cp, job_dir: str) -> dict:
        content_type = cp.content_type
        total_episodes = len(cp.scenes) if content_type == "series" else 0
        completed_eps = list(cp.completed_episodes or [])
        status = self._derive_status(content_type, cp.completed_stages, completed_eps, total_episodes)
        if content_type == "series" and total_episodes:
            progress_label = f"{len(completed_eps)}/{total_episodes} 集"
        else:
            progress_label = f"{len(cp.completed_stages)} stages"
        return {
            "job_id": cp.job_id,
            "content_type": content_type,
            "story_title": (cp.config_overrides or {}).get("story_title", ""),
            "status": status,
            "completed_stages": list(cp.completed_stages),
            "completed_episodes": completed_eps,
            "total_episodes": total_episodes,
            "episode_final_count": len(cp.episode_finals or []),
            "progress_label": progress_label,
            "created_at": cp.created_at,
            "updated_at": cp.updated_at,
        }

    @staticmethod
    def _episode_finals_detail(cp, job_dir: str) -> list[dict]:
        finals = []
        for idx, path in enumerate(cp.episode_finals or [], start=1):
            import os as _os
            exists = _os.path.isfile(path)
            size = _os.path.getsize(path) if exists else 0
            finals.append({
                "episode": idx,
                "path": path,
                "basename": _os.path.basename(path),
                "size_bytes": size,
                "exists": exists,
            })
        return finals
