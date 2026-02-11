from __future__ import annotations

from pathlib import Path

from src.common.paths import resolve_image_path_best_effort, resolve_image_path_strict


def test_best_effort_absolute_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "img.png"
    got = resolve_image_path_best_effort(str(abs_path), jsonl_dir=tmp_path)
    assert got == abs_path
    assert got.is_absolute()


def test_best_effort_root_image_dir_precedence(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path / "root"
    jsonl_dir = tmp_path / "jsonl"
    env_root = tmp_path / "env_root"
    root_dir.mkdir()
    jsonl_dir.mkdir()
    env_root.mkdir()

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(env_root))

    got = resolve_image_path_best_effort(
        "a/b.png",
        jsonl_dir=jsonl_dir,
        root_image_dir=root_dir,
    )
    assert got == (root_dir / "a/b.png").resolve()


def test_best_effort_env_root_precedence(tmp_path: Path, monkeypatch) -> None:
    jsonl_dir = tmp_path / "jsonl"
    env_root = tmp_path / "env_root"
    jsonl_dir.mkdir()
    env_root.mkdir()

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(env_root))

    got = resolve_image_path_best_effort(
        "rel.jpg",
        jsonl_dir=jsonl_dir,
        root_image_dir=None,
        env_root_var="ROOT_IMAGE_DIR",
    )
    assert got == (env_root / "rel.jpg").resolve()


def test_best_effort_jsonl_dir_fallback(tmp_path: Path, monkeypatch) -> None:
    jsonl_dir = tmp_path / "jsonl"
    jsonl_dir.mkdir()

    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    got = resolve_image_path_best_effort(
        "rel.jpg",
        jsonl_dir=jsonl_dir,
        root_image_dir=None,
    )
    assert got == (jsonl_dir / "rel.jpg").resolve()


def test_best_effort_cwd_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    got = resolve_image_path_best_effort(
        "rel.jpg",
        jsonl_dir=None,
        root_image_dir=None,
        env_root_var=None,
    )
    assert got == (tmp_path / "rel.jpg").resolve()


def test_strict_returns_none_for_empty_image_field(tmp_path: Path) -> None:
    got = resolve_image_path_strict(None, jsonl_dir=tmp_path)
    assert got is None


def test_strict_resolves_existing_under_root_image_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    img = root_dir / "img.png"
    img.write_bytes(b"x")

    got = resolve_image_path_strict(
        "img.png",
        jsonl_dir=tmp_path,
        root_image_dir=root_dir,
    )
    assert got == img


def test_strict_resolves_existing_under_jsonl_dir(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    jsonl_dir.mkdir()
    img = jsonl_dir / "img.png"
    img.write_bytes(b"x")

    got = resolve_image_path_strict(
        "img.png",
        jsonl_dir=jsonl_dir,
        root_image_dir=None,
    )
    assert got == img


def test_strict_absolute_existing_passthrough(tmp_path: Path) -> None:
    img = tmp_path / "img.png"
    img.write_bytes(b"x")

    got = resolve_image_path_strict(
        str(img),
        jsonl_dir=None,
        root_image_dir=None,
    )
    assert got == img


def test_strict_missing_returns_none(tmp_path: Path) -> None:
    root_dir = tmp_path / "root"
    root_dir.mkdir()

    got = resolve_image_path_strict(
        "missing.png",
        jsonl_dir=None,
        root_image_dir=root_dir,
    )
    assert got is None
