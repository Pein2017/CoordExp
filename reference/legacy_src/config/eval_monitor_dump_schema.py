from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvalMonitorDumpConfig:
    enabled: bool = False
    every_evals: int = 1
    only_world_process_zero: bool = True
    max_events: int = 20
    max_samples: int = 1
    max_text_chars: int = 4000
    async_write: bool = True
    max_pending_writes: int = 2
    min_free_gb: float = 2.0
    out_dir: Optional[str] = None
    write_markdown: bool = True
