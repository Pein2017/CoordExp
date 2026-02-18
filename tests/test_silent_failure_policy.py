from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Authoritative allowlist for intentional suppression sinks.
# Key format: "relative/path.py:line".
# Add entries only when failures cannot affect model inputs/labels/metrics artifacts.
ALLOWLIST: dict[str, str] = {}

# Phase-1 policy patterns.
PHASE1_PATTERNS = [
    re.compile(r"except\s+Exception\s*:\s*\n\s*pass\b"),
    re.compile(r"except\s*:\s*\n\s*pass\b"),
    re.compile(r"except\s+BaseException\s*:\s*\n\s*pass\b"),
]

CORE_POLICY_FILES = [
    "src/datasets/dense_caption.py",
    "src/datasets/unified_fusion_dataset.py",
    "src/eval/detection.py",
    "src/infer/pipeline.py",
    "src/trainers/stage2_ab/scheduler.py",
]


def _line_number(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def test_core_paths_forbid_blanket_exception_suppression_phase1() -> None:
    violations: list[str] = []

    for rel_path in CORE_POLICY_FILES:
        file_path = REPO_ROOT / rel_path
        text = file_path.read_text(encoding="utf-8")

        for pattern in PHASE1_PATTERNS:
            for match in pattern.finditer(text):
                line = _line_number(text, match.start())
                allowlist_key = f"{rel_path}:{line}"
                if allowlist_key in ALLOWLIST:
                    continue

                snippet = match.group(0).splitlines()[0].strip()
                violations.append(f"{allowlist_key}: {snippet}")

    assert not violations, (
        "Forbidden blanket suppression found outside the explicit allowlist. "
        "Either remove/narrow the catch or add an allowlist entry with a one-line "
        "justification for why this site is a safe best-effort sink.\n"
        + "\n".join(violations)
    )
