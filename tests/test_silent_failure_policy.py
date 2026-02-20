import ast
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_GLOB = "src/**/*.py"

# Option A (strict): blanket catch-alls are forbidden everywhere in src/ except
# an explicit allowlist for best-effort sinks / diagnostic wrappers.
ALLOWLIST_BLANKET_EXCEPT = {
    "src/metrics/reporter.py",
    "src/utils/logger.py",
}


class Violation(NamedTuple):
    rel_path: str
    lineno: int
    label: str
    action: str


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
    """Return True for bare except + Exception/BaseException catch-alls."""

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


def _iter_try_handlers(tree: ast.AST) -> Iterable[ast.ExceptHandler]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            yield from node.handlers


def _last_stmt(stmts: list[ast.stmt]) -> ast.stmt | None:
    if not stmts:
        return None
    return stmts[-1]


def _iter_tier0_violations(source: str, rel_path: str) -> Iterator[Violation]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(
            f"{rel_path}:{exc.lineno}: syntax error while enforcing silent-failure policy: {exc.msg}"
        ) from exc

    for handler in _iter_try_handlers(tree):
        if not _is_blanket_exception_type(handler.type):
            continue
        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
            yield Violation(
                rel_path,
                int(handler.lineno),
                _handler_label(handler.type),
                "pass",
            )


def _iter_tier1_violations(source: str, rel_path: str) -> Iterator[Violation]:
    """Tier 1: blanket suppression via continue/break/return."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(
            f"{rel_path}:{exc.lineno}: syntax error while enforcing silent-failure policy: {exc.msg}"
        ) from exc

    for handler in _iter_try_handlers(tree):
        if not _is_blanket_exception_type(handler.type):
            continue

        last = _last_stmt(handler.body)
        if isinstance(last, ast.Continue):
            yield Violation(rel_path, int(last.lineno), _handler_label(handler.type), "continue")
        elif isinstance(last, ast.Break):
            yield Violation(rel_path, int(last.lineno), _handler_label(handler.type), "break")
        elif isinstance(last, ast.Return):
            yield Violation(rel_path, int(last.lineno), _handler_label(handler.type), "return")


def _iter_tier2_violations(source: str, rel_path: str) -> Iterator[Violation]:
    """Tier 2: blanket except handlers are forbidden outside allowlist."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(
            f"{rel_path}:{exc.lineno}: syntax error while enforcing silent-failure policy: {exc.msg}"
        ) from exc

    for handler in _iter_try_handlers(tree):
        if _is_blanket_exception_type(handler.type):
            yield Violation(
                rel_path,
                int(handler.lineno),
                _handler_label(handler.type),
                "blanket",
            )


def test_src_paths_forbid_blanket_exception_pass_tier0() -> None:
    violations: list[str] = []

    for rel_path in _iter_policy_files():
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for v in _iter_tier0_violations(source, rel_path):
            violations.append(f"{v.rel_path}:{v.lineno}: {v.label}: {v.action}")

    assert not violations, (
        "Tier 0 (blocking): blanket pass is forbidden in src/.\n"
        "Fix by narrowing exception types and either re-raising with context, "
        "or using an explicitly-defined model-output consumer path with structured errors + counters.\n"
        + "\n".join(violations)
    )


def test_src_paths_forbid_blanket_suppression_tier1() -> None:
    violations: list[str] = []

    for rel_path in _iter_policy_files():
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for v in _iter_tier1_violations(source, rel_path):
            violations.append(f"{v.rel_path}:{v.lineno}: {v.label}: {v.action}")

    assert not violations, (
        "Tier 1 (blocking): blanket Exception/BaseException handlers must not suppress via continue/break/return.\n"
        "Fix by narrowing exception types, validating inputs preflight, and avoiding semantics-changing defaults in core paths.\n"
        + "\n".join(violations)
    )


def test_src_paths_forbid_blanket_exception_handlers_tier2() -> None:
    violations: list[str] = []

    for rel_path in _iter_policy_files():
        if rel_path in ALLOWLIST_BLANKET_EXCEPT:
            continue
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for v in _iter_tier2_violations(source, rel_path):
            violations.append(f"{v.rel_path}:{v.lineno}: {v.label}: {v.action}")

    assert not violations, (
        "Tier 2 (blocking): blanket except handlers are forbidden in src/ (except allowlist).\n"
        "Fix by narrowing exception types (ImportError/OSError/ValueError/etc), or removing the try/except and letting errors fail fast.\n"
        + "\n".join(violations)
    )


def test_policy_scan_detects_known_patterns() -> None:
    bad_pass = """
try:
    x = 1
except Exception:
    pass
"""
    bad_continue = """
for _ in range(1):
    try:
        x = 1
    except Exception:
        continue
"""
    bad_return = """
def f():
    try:
        x = 1
    except Exception:
        return 0.0
    return 1.0
"""
    ok_narrow = """
for _ in range(1):
    try:
        int("nope")
    except (TypeError, ValueError):
        continue
"""

    assert list(_iter_tier0_violations(bad_pass, "<mem>"))
    assert not list(_iter_tier0_violations(ok_narrow, "<mem>"))

    assert list(_iter_tier1_violations(bad_continue, "<mem>"))
    assert list(_iter_tier1_violations(bad_return, "<mem>"))
    assert not list(_iter_tier1_violations(ok_narrow, "<mem>"))

    assert list(_iter_tier2_violations(bad_pass, "<mem>"))
    assert list(_iter_tier2_violations(bad_continue, "<mem>"))
    assert list(_iter_tier2_violations(bad_return, "<mem>"))
    assert not list(_iter_tier2_violations(ok_narrow, "<mem>"))
