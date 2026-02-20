import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_prediction_parsing_transpiles_before_json_loads() -> None:
    text = _read("src/common/prediction_parsing.py")

    assert "coordjson_to_strict_json_with_meta(" in text
    assert re.search(r"json\.loads\(strict_text\)", text)
    assert not re.search(r"json\.loads\(text\)", text)
    assert not re.search(r"json\.loads\(response_text\)", text)
    assert not re.search(r"json\.loads\(assistant_text\)", text)


def test_inventory_gate_for_raw_coordjson_json_loads() -> None:
    pattern = re.compile(
        r"json\.loads\((text|response_text|raw_text|assistant_text|pred_text|prediction_text|rollout_text)\)"
    )
    allowlist = {
        "src/datasets/fusion.py",
    }

    hits: list[str] = []
    for base in ("src", "scripts", "tests"):
        for path in sorted((ROOT / base).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in allowlist:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}:{line.strip()}")

    assert hits == []


def test_inventory_gate_no_object_n_keys_in_repo_surfaces() -> None:
    pattern = re.compile(r'"object_[0-9]+"')
    text_suffixes = {
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".txt",
        ".sh",
    }

    hits: list[str] = []
    for base in ("src", "tests", "scripts", "configs", "docs", "progress"):
        for path in sorted((ROOT / base).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            rel = path.relative_to(ROOT).as_posix()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}:{line.strip()}")

    assert hits == []
