import re
from pathlib import Path


def test_no_blanket_except_exception_pass_in_src() -> None:
    """Policy: avoid `except Exception: pass` in core code paths.

    Best-effort behavior should be explicit (logging, narrow exception types, or
    structured fallbacks). This test is intentionally narrow to reduce false
    positives while still catching truly-silent suppression.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    pattern = re.compile(
        r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n\s+pass\b",
        flags=re.MULTILINE,
    )

    offenders: list[str] = []
    for path in sorted(src_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(repo_root)))

    assert not offenders, "Found silent exception suppression: " + ", ".join(offenders)
