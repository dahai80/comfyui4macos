#!/usr/bin/env python3
"""Full series generation test: 青溪渡阴 — 6 chapters, 6 episodes, 8 scenes each."""
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from custom_nodes4macos.pipeline import PipelineEngine

STORY_FILE = "/Users/dahai/solution/comfyui4macos/input/青溪渡阴.md"


def main():
    engine = PipelineEngine()
    t0 = time.time()

    result = engine.run(
        "series",
        story_file=STORY_FILE,
        episode_count=6,
        scene_count=8,
        style_preset="电影叙事",
        flux_steps=6,
        flux_scheduler="flow_match_euler_discrete",
        flux_steps_auto=True,
        consistency_check=True,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESULT: job_id={result.job_id}")
    print(f"  job_dir={result.job_dir}")
    print(f"  final_video={result.final_video}")
    print(f"  total_time={elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
