from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

# Directory-wide policy applies to all Python modules under src/.
POLICY_GLOB = "src/**/*.py"

# Temporary wave-2 debt ledger for existing blanket-suppression sites.
# Format: path -> (expected_count, one-line justification)
ALLOWLIST_FILE_COUNTS: dict[str, tuple[int, str]] = {
    "src/common/prediction_parsing.py": (
        5,
        "Legacy tolerant parsing fallbacks pending wave-2 cleanup.",
    ),
    "src/data_collators/batch_extras_collator.py": (
        2,
        "Legacy optional batch-extras handling pending wave-2 cleanup.",
    ),
    "src/data_collators/enrichers.py": (
        1,
        "Legacy optional enricher fallback pending wave-2 cleanup.",
    ),
    "src/datasets/preprocessors/augmentation.py": (
        2,
        "Legacy augmentation best-effort paths pending wave-2 cleanup.",
    ),
    "src/datasets/preprocessors/resize.py": (
        1,
        "Legacy resize probing fallback pending wave-2 cleanup.",
    ),
    "src/datasets/preprocessors/sequential.py": (
        2,
        "Legacy sequential-preprocessor fallback pending wave-2 cleanup.",
    ),
    "src/datasets/wrappers/packed_caption.py": (
        1,
        "Legacy packed-wrapper fallback pending wave-2 cleanup.",
    ),
    "src/metrics/coord_monitors.py": (
        2,
        "Legacy metrics-only fallback pending wave-2 cleanup.",
    ),
    "src/sft.py": (
        10,
        "Legacy SFT transitional fallback paths pending wave-2 cleanup.",
    ),
    "src/trainers/batch_extras.py": (
        1,
        "Legacy optional batch extras fallback pending wave-2 cleanup.",
    ),
    "src/trainers/gkd_monitor.py": (
        3,
        "Legacy telemetry fallback paths pending wave-2 cleanup.",
    ),
    "src/trainers/metrics/mixins.py": (
        8,
        "Legacy trainer metrics fallback paths pending wave-2 cleanup.",
    ),
    "src/trainers/monitoring/instability.py": (
        7,
        "Legacy instability-monitor fallback paths pending wave-2 cleanup.",
    ),
    "src/trainers/rollout_matching_sft.py": (
        27,
        "Legacy rollout-matching fallback paths pending wave-2 cleanup.",
    ),
    "src/trainers/stage2_ab_training.py": (
        14,
        "Legacy stage2-ab fallback paths pending wave-2 cleanup.",
    ),
    "src/utils/logger.py": (
        6,
        "Explicit logging sink fallback paths.",
    ),
}


def _iter_policy_files() -> list[str]:
    out: list[str] = []
    for path in sorted(REPO_ROOT.glob(POLICY_GLOB)):
        if not path.is_file():
            continue
        out.append(str(path.relative_to(REPO_ROOT)))
    return out


def _is_named_exception(node: ast.expr, names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in names
    return False


def _is_blanket_exception_type(node: Optional[ast.expr]) -> bool:
    if node is None:
        return True  # bare except

    blanket_names = {"Exception", "BaseException"}
    if isinstance(node, ast.Tuple):
        return any(_is_named_exception(elem, blanket_names) for elem in node.elts)
    return _is_named_exception(node, blanket_names)


def _handler_label(node: Optional[ast.expr]) -> str:
    if node is None:
        return "except"
    if isinstance(node, ast.Name):
        return f"except {node.id}"
    if isinstance(node, ast.Attribute):
        return f"except {node.attr}"
    if isinstance(node, ast.Tuple):
        labels = []
        for elem in node.elts:
            if isinstance(elem, ast.Name):
                labels.append(elem.id)
            elif isinstance(elem, ast.Attribute):
                labels.append(elem.attr)
            else:
                labels.append(type(elem).__name__)
        return "except (" + ", ".join(labels) + ")"
    return f"except {type(node).__name__}"


def _iter_blanket_pass_handlers(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _is_blanket_exception_type(handler.type):
                continue
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                yield int(handler.lineno), _handler_label(handler.type)


def test_src_paths_forbid_blanket_exception_suppression_phase1() -> None:
    violations: list[str] = []
    observed_counts: dict[str, int] = {}

    for rel_path in _iter_policy_files():
        file_path = REPO_ROOT / rel_path
        source = file_path.read_text(encoding="utf-8")
        handlers = list(_iter_blanket_pass_handlers(source))
        if not handlers:
            continue

        observed_counts[rel_path] = len(handlers)
        allow = ALLOWLIST_FILE_COUNTS.get(rel_path)
        if allow is None:
            for lineno, label in handlers:
                violations.append(f"{rel_path}:{lineno}: {label}: pass")
            continue

        expected_count, reason = allow
        if len(handlers) != int(expected_count):
            violations.append(
                f"{rel_path}: expected {expected_count} allowlisted blanket suppressions "
                f"({reason}), found {len(handlers)}"
            )

    for rel_path, (expected_count, reason) in ALLOWLIST_FILE_COUNTS.items():
        actual = observed_counts.get(rel_path, 0)
        if actual == 0:
            violations.append(
                f"{rel_path}: allowlist expects {expected_count} blanket suppressions "
                f"({reason}), but found none; remove/update allowlist entry"
            )

    assert not violations, (
        "Forbidden blanket suppression found outside the explicit wave-2 allowlist, "
        "or allowlist drift detected.\n"
        + "\n".join(violations)
    )
