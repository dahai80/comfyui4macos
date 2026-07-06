from __future__ import annotations

import json
import logging
import os

from ..stage import Stage, StageInfo
from ...publisher import (
    DouyinPublisher,
    PublishConfigError,
    PublishDependencyError,
    PublishMeta,
)

logger = logging.getLogger("custom_nodes4macos.pipeline.stages.publish")


class PublishStage(Stage):

    @classmethod
    def info(cls) -> StageInfo:
        return StageInfo(
            name="publish",
            description="发布成品到平台草稿箱（仅 Douyin；默认 dry_run 安全清单，opt-in）",
            model_requirements=[],
            memory_estimate_gb=0.0,
            input_kinds=["final"],
            output_kinds=["publish"],
        )

    def process(self, ctx, model_manager) -> None:
        if self._skip_if_completed(ctx):
            return

        if not ctx.config.get("publish_enabled", False):
            logger.info("publish: disabled (publish_enabled=false), skip")
            ctx.update_progress("publish", 1, 1)
            return

        dry_run = ctx.config.get("publish_dry_run", True)
        platform = ctx.config.get("publish_platform", "douyin")
        if platform != "douyin":
            logger.warning("publish: unsupported platform '%s', skip", platform)
            ctx.update_progress("publish", 1, 1)
            return

        videos = self._collect_videos(ctx)
        if not videos:
            logger.warning("publish: no final video to publish, skip")
            ctx.update_progress("publish", 1, 1)
            return

        cookies_path = ctx.config.get("publish_cookies_path", "") or ""
        publisher = DouyinPublisher()
        results: list[dict] = []
        dep_missing = False

        for idx, video_path in enumerate(videos):
            meta = self._build_meta(ctx, video_path, idx, len(videos))
            try:
                result = publisher.upload_draft(
                    video_path=video_path,
                    meta=meta,
                    cookies_path=cookies_path,
                    dry_run=dry_run,
                    manifest_dir=ctx.job_dir,
                )
                results.append(result.to_dict())
                logger.info("publish: [%d/%d] %s → %s", idx + 1, len(videos), video_path, result.status)
            except PublishConfigError as exc:
                logger.warning("publish: [%d/%d] config error: %s (skip this video)", idx + 1, len(videos), exc)
                results.append({"video_path": video_path, "status": "skipped", "error": str(exc)})
            except PublishDependencyError as exc:
                logger.warning(
                    "publish: [%d/%d] dependency missing: %s (skip; install to enable live upload)",
                    idx + 1, len(videos), exc,
                )
                results.append({"video_path": video_path, "status": "skipped", "error": str(exc)})
                dep_missing = True
                break
            except Exception as exc:
                logger.error("publish: [%d/%d] upload failed: %s (skip, pipeline continues)", idx + 1, len(videos), exc)
                results.append({"video_path": video_path, "status": "failed", "error": str(exc)})

        manifest_path = os.path.join(ctx.job_dir, "publish_manifest.json")
        payload = {
            "platform": platform,
            "dry_run": dry_run,
            "dep_missing": dep_missing,
            "results": results,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        ctx.set_artifact(0, "publish", manifest_path)
        logger.info("publish: done, %d video(s), dep_missing=%s, manifest=%s", len(results), dep_missing, manifest_path)
        ctx.update_progress("publish", 1, 1)

    @staticmethod
    def _collect_videos(ctx) -> list[str]:
        if ctx.config.get("publish_all_episodes", False):
            episode_finals = ctx.config.get("_episode_finals", []) or []
            if episode_finals:
                return [v for v in episode_finals if v and os.path.isfile(v)]
        final_path = ctx.get_artifact(0, "final")
        if final_path and os.path.isfile(final_path):
            return [final_path]
        return []

    @staticmethod
    def _build_meta(ctx, video_path: str, idx: int, total: int) -> PublishMeta:
        base_title = (
            ctx.config.get("publish_title", "")
            or ctx.config.get("story_title", "")
            or os.path.splitext(os.path.basename(video_path))[0]
        )
        if total > 1:
            title = f"{base_title} 第{idx + 1}集"
        else:
            title = base_title
        tags = list(ctx.config.get("publish_tags", []) or [])
        cover_path = ctx.config.get("publish_cover_path", "") or ""
        return PublishMeta(title=title, tags=tags, cover_path=cover_path)
