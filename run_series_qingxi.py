#!/usr/bin/env python3
"""Run series pipeline for 青溪渡阴 with performance monitoring."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import resource

os.environ.setdefault("PYTHONPATH", ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_series_qingxi")


def get_memory_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0


def monitor_stage(stage_name, ctx, model_mgr):
    stage_start = time.time()
    mem_before = get_memory_mb()
    logger.info(
        "=== STAGE %s START === mem=%.0fMB model_usage=%.1fG",
        stage_name, mem_before, model_mgr.current_usage_gb,
    )
    return stage_start, mem_before


def monitor_scene(stage_name, scene_id, total, t0):
    elapsed = time.time() - t0
    mem = get_memory_mb()
    logger.info(
        "--- %s scene %d/%d done in %.1fs mem=%.0fMB ---",
        stage_name, scene_id, total, elapsed, mem,
    )


def main():
    story_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "input", "青溪渡阴.md",
    )
    if not os.path.isfile(story_file):
        logger.error("story file not found: %s", story_file)
        sys.exit(1)

    logger.info("story file: %s (%d bytes)", story_file, os.path.getsize(story_file))

    from custom_nodes4macos.pipeline import PipelineEngine

    engine = PipelineEngine()

    t_total = time.time()
    mem_start = get_memory_mb()
    logger.info("=== PIPELINE START === mem=%.0fMB", mem_start)

    episode_count = int(os.environ.get("EPISODE_COUNT", "2"))
    scene_count = int(os.environ.get("SCENE_COUNT", "8"))

    logger.info(
        "running with episode_count=%d scene_count=%d", episode_count, scene_count,
    )

    result = engine.run(
        content_type="series",
        story_file=story_file,
        episode_count=episode_count,
        scene_count=scene_count,
        style_preset="电影叙事",
        flux_steps=8,
        flux_guidance=4.0,
        flux_width=1024,
        flux_height=1024,
        consistency_check=True,
        checkpoint_every_n_scenes=5,
        ken_burns_workers=3,
    )

    t_elapsed = time.time() - t_total
    mem_end = get_memory_mb()

    logger.info("=== PIPELINE COMPLETE ===")
    logger.info("job_id: %s", result.job_id)
    logger.info("job_dir: %s", result.job_dir)
    logger.info("final_video: %s", result.final_video)
    logger.info("total time: %.1fs (%.1fmin)", t_elapsed, t_elapsed / 60)
    logger.info("peak memory: %.0fMB", mem_end)
    logger.info("memory delta: %.0fMB", mem_end - mem_start)

    cp_path = os.path.join(result.job_dir, "_checkpoint.json")
    if os.path.isfile(cp_path):
        with open(cp_path, "r") as f:
            cp = json.load(f)
        logger.info("completed_stages: %s", cp.get("completed_stages", []))
        logger.info("total artifacts: %d", len(cp.get("artifacts", {})))
        logger.info("total scenes: %d", len(cp.get("scenes", [])))


if __name__ == "__main__":
    main()
