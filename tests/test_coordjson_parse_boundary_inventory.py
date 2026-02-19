from __future__ import annotations

from pathlib import Path


def test_coordjson_parse_boundary_requires_transpile_before_json_loads() -> None:
    """Inventory gate: never `json.loads(model_text)` on CoordJSON-like outputs.

    CoordJSON is intentionally *not* strict JSON (bare `<|coord_k|>` literals), so
    any direct `json.loads(...)` on model-generated text is almost certainly a bug.

    This test is intentionally narrow to avoid false positives for JSONL reading or
    deep-copy patterns.
    """

    import re

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    # Common variable names used for model-generated strings.
    suspicious_names = (
        "assistant_text",
        "response_text",
        "raw_text",
        "pred_text",
        "output_text",
        "generated_text",
        "model_text",
    )

    pattern = re.compile(
        r"json\\.loads\\(\\s*(%s)\\s*\\)" % "|".join(suspicious_names)
    )

    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(rel)

    assert not offenders, (
        "Found direct json.loads(...) on model-text-like variables (CoordJSON must be transpiled first): "
        + ", ".join(offenders)
    )
