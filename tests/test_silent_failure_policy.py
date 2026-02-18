from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_GLOB = "src/**/*.py"


def _iter_policy_files() -> list[str]:
    return [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(REPO_ROOT.glob(POLICY_GLOB))
        if path.is_file()
    ]


def _is_named_exception(node: ast.expr, names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return node.attr in names
    return False


def _is_blanket_exception_type(node: Optional[ast.expr]) -> bool:
    if node is None:
        return True

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


def _iter_blanket_pass_handlers(source: str, rel_path: str):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(
            f"{rel_path}:{exc.lineno}: syntax error while enforcing fail-fast policy: {exc.msg}"
        ) from exc

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _is_blanket_exception_type(handler.type):
                continue
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                yield int(handler.lineno), _handler_label(handler.type)


def test_src_paths_forbid_blanket_exception_suppression() -> None:
    violations: list[str] = []

    for rel_path in _iter_policy_files():
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for lineno, label in _iter_blanket_pass_handlers(source, rel_path):
            violations.append(f"{rel_path}:{lineno}: {label}: pass")

    assert not violations, (
        "Blanket suppression is forbidden in src/. Replace with fail-fast handling.\n"
        + "\n".join(violations)
    )
