from __future__ import annotations

import logging
import os

logger = logging.getLogger("custom_nodes4macos.pipeline.context")

_ARTIFACT_EXTENSIONS = {
    "image": "png",
    "audio": "wav",
    "clip": "mp4",
    "final": "mp4",
}


class PipelineContext:

    def __init__(self, job_id: str, job_dir: str, config: dict):
        self.job_id = job_id
        self.job_dir = job_dir
        self.config = config
        self.scenes: list[dict] = []
        self.artifacts: dict[str, str] = {}
        self.completed_stages: list[str] = []
        self.progress: dict = {}
        self.created_at: str = ""

    def artifact_path(self, scene_id: int, kind: str) -> str:
        ext = _ARTIFACT_EXTENSIONS.get(kind, "bin")
        return os.path.join(self.job_dir, f"scene_{scene_id:03d}_{kind}.{ext}")

    def set_artifact(self, scene_id: int, kind: str, path: str):
        key = f"{scene_id}_{kind}"
        self.artifacts[key] = path
        logger.debug("artifact set: %s → %s", key, path)

    def get_artifact(self, scene_id: int, kind: str) -> str | None:
        key = f"{scene_id}_{kind}"
        if key in self.artifacts:
            return self.artifacts[key]
        return None

    def has_artifact_on_disk(self, scene_id: int, kind: str) -> bool:
        path = self.get_artifact(scene_id, kind)
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        path = self.artifact_path(scene_id, kind)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self.artifacts[f"{scene_id}_{kind}"] = path
            return True
        return False

    def update_progress(self, stage_name: str, scene: int = 0, total: int = 0):
        pct = (scene / total * 100) if total > 0 else 0.0
        self.progress = {
            "stage": stage_name,
            "scene": scene,
            "total": total,
            "pct": round(pct, 1),
        }
        logger.info("progress: stage=%s scene=%d/%d (%.0f%%)", stage_name, scene, total, pct)

    def should_checkpoint_scene(self, scene_idx: int) -> bool:
        interval = self.config.get("checkpoint_every_n_scenes", 0)
        if interval <= 0:
            return False
        return scene_idx > 0 and scene_idx % interval == 0
