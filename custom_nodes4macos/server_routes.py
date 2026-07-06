from __future__ import annotations

import logging
import os

logger = logging.getLogger("custom_nodes4macos.server_routes")

_REGISTERED = False

try:
    from server import PromptServer
    from aiohttp import web

    _COMFYUI_AVAILABLE = True
except Exception:
    _COMFYUI_AVAILABLE = False
    PromptServer = None
    web = None


def _engine():
    from .pipeline.engine import PipelineEngine

    output_root = os.environ.get("DREAM_FACTORY_OUTPUT_ROOT")
    return PipelineEngine(output_root=output_root) if output_root else PipelineEngine()


def list_jobs() -> list[dict]:
    return _engine().list_jobs()


def get_job(job_id: str) -> dict | None:
    return _engine().get_job(job_id)


def resolve_job_file(job_id: str, filename: str):
    return _engine().resolve_job_file(job_id, filename)


def register_routes() -> None:
    global _REGISTERED
    if not _COMFYUI_AVAILABLE:
        logger.info("ComfyUI PromptServer unavailable, skip route registration")
        return
    if _REGISTERED:
        return
    routes = PromptServer.instance.routes

    @routes.get("/dream_factory/jobs")
    async def _jobs_route(request):
        try:
            return web.json_response({"jobs": list_jobs()})
        except Exception as exc:
            logger.exception("list_jobs failed: %s", exc)
            return web.json_response({"error": "list_jobs failed"}, status=500)

    @routes.get("/dream_factory/jobs/{job_id}")
    async def _job_detail_route(request):
        job_id = request.match_info["job_id"]
        try:
            job = get_job(job_id)
        except Exception as exc:
            logger.exception("get_job failed: %s", exc)
            return web.json_response({"error": "get_job failed"}, status=500)
        if job is None:
            return web.json_response({"error": "job not found", "job_id": job_id}, status=404)
        return web.json_response({"job": job})

    @routes.get("/dream_factory/preview/{job_id}/{filename:.+}")
    async def _preview_route(request):
        job_id = request.match_info["job_id"]
        filename = request.match_info["filename"]
        try:
            path = resolve_job_file(job_id, filename)
        except Exception as exc:
            logger.exception("resolve_job_file failed: %s", exc)
            return web.json_response({"error": "resolve failed"}, status=500)
        if path is None:
            return web.json_response(
                {"error": "file not found", "job_id": job_id, "filename": filename},
                status=404,
            )
        return web.FileResponse(str(path))

    _REGISTERED = True
    logger.info("registered ComfyUI routes: /dream_factory/jobs, /jobs/{id}, /preview/{id}/{file}")


register_routes()
