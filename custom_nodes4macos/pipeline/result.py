from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineResult:
    job_id: str
    job_dir: str
    final_video: str | None = None
