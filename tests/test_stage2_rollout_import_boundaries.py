from __future__ import annotations

import ast
from pathlib import Path


def test_stage2_ab_does_not_import_private_rollout_symbols() -> None:
    """Regression guard: Stage-2 AB must not import underscore-private rollout symbols.

    This uses AST inspection (not regex) so formatting/comment changes do not
    introduce false positives.
    """

    repo_root = Path(__file__).resolve().parents[1]
    stage2_paths = [repo_root / "src" / "trainers" / "stage2_ab_training.py"]
    stage2_pkg = repo_root / "src" / "trainers" / "stage2_ab"
    stage2_paths.extend(sorted(stage2_pkg.rglob("*.py")))

    violations: list[tuple[str, str, str, int]] = []
    for stage2_path in stage2_paths:
        text = stage2_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(stage2_path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module:
                continue

            if "rollout_matching" not in node.module:
                continue

            for alias in node.names:
                if alias.name.startswith("_"):
                    rel_path = str(stage2_path.relative_to(repo_root))
                    violations.append(
                        (
                            rel_path,
                            node.module,
                            alias.name,
                            int(getattr(node, "lineno", 0) or 0),
                        )
                    )

    if violations:
        formatted = "\n".join(
            f"- {path}: {mod}: {name} (line {lineno})"
            for path, mod, name, lineno in violations
        )
        raise AssertionError(
            "Stage-2 AB reintroduced underscore-private rollout imports.\n" + formatted
        )
