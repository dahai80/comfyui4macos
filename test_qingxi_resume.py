#!/usr/bin/env python3
"""Resume the 6-episode series generation from checkpoint."""
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

JOB_ID = "20260704_f2dbfa1e"


def main():
    engine = PipelineEngine()
    t0 = time.time()

    result = engine.run("series", resume_from=JOB_ID)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESULT: job_id={result.job_id}")
    print(f"  job_dir={result.job_dir}")
    print(f"  final_video={result.final_video}")
    print(f"  total_time={elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
