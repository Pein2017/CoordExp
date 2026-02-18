from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

# Authoritative allowlist for intentional suppression sinks.
# Key format: "relative/path.py:line".
# Add entries only for explicit best-effort sinks where failures cannot affect
# model inputs, labels, or metrics artifacts.
ALLOWLIST: dict[str, str] = {}

# Phase-1 scope: infer/eval modules plus core dataset/trainer entry paths.
CORE_POLICY_FILES = [
    "src/datasets/dense_caption.py",
    "src/datasets/unified_fusion_dataset.py",
    "src/trainers/stage2_ab/scheduler.py",
]
CORE_POLICY_GLOBS = [
    "src/infer/**/*.py",
    "src/eval/**/*.py",
]


def _iter_policy_files() -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for rel in CORE_POLICY_FILES:
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)

    for pattern in CORE_POLICY_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if rel in seen:
                continue
            seen.add(rel)
            out.append(rel)

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


def test_core_paths_forbid_blanket_exception_suppression_phase1() -> None:
    violations: list[str] = []

    for rel_path in _iter_policy_files():
        file_path = REPO_ROOT / rel_path
        source = file_path.read_text(encoding="utf-8")

        for lineno, label in _iter_blanket_pass_handlers(source):
            allowlist_key = f"{rel_path}:{lineno}"
            if allowlist_key in ALLOWLIST:
                continue
            violations.append(f"{allowlist_key}: {label}: pass")

    assert not violations, (
        "Forbidden blanket suppression found outside the explicit allowlist. "
        "Either remove/narrow the catch or add an allowlist entry with a one-line "
        "justification for why this site is a safe best-effort sink.\n"
        + "\n".join(violations)
    )
