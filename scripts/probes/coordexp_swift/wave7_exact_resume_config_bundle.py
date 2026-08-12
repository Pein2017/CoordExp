#!/usr/bin/env python3
"""Atomically author the canonical Wave 7 r7 three-config bundle."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.fingerprint import sha256_json  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402


SCHEMA = "coordexp-swift-wave7-r7-config-bundle-v1"
CANONICAL_BASE_CONFIG = (
    REPO_ROOT / "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
).resolve()
CANONICAL_BASE_SHA256 = (
    "43f46f3df390cea3e6fe117654d9d82822a448a06e71b36d83d46c5a4d22f313"
)
CANONICAL_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4"
).resolve()
ROLE_FILENAMES = {
    "uninterrupted": "uninterrupted.yaml",
    "interrupted_parent": "interrupted-parent.yaml",
    "resume_child": "resume-child.yaml",
}
RECEIPT_NAME = "config-bundle-receipt.json"


class ConfigBundleError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigBundleError("config bundle JSON is not strict") from exc


def _signed(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "receipt_payload_sha256": _sha256_bytes(_canonical_json_bytes(body)),
    }


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _assert_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    absolute = path.expanduser().absolute()
    rows = [absolute, *absolute.parents] if include_leaf else list(absolute.parents)
    for candidate in rows:
        info = _lstat_or_none(candidate)
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise ConfigBundleError(
                f"config bundle path contains a symlink component: {candidate}"
            )


def _canonical_base(path_value: str | Path) -> Path:
    requested = Path(path_value).expanduser()
    if requested.is_symlink():
        raise ConfigBundleError("base config is not the canonical base file")
    resolved = requested.resolve()
    if resolved != CANONICAL_BASE_CONFIG:
        raise ConfigBundleError("base config is not the canonical base file")
    _assert_no_symlink_components(requested, include_leaf=True)
    info = resolved.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ConfigBundleError("canonical base config is not a regular file")
    if _sha256_file(resolved) != CANONICAL_BASE_SHA256:
        raise ConfigBundleError("canonical base config file identity drifted")
    return resolved


def _canonical_absent_output(path_value: str | Path) -> Path:
    requested = Path(path_value).expanduser()
    if requested.is_symlink():
        raise ConfigBundleError("canonical r7 root must be absent, not a symlink")
    resolved = requested.resolve(strict=False)
    if resolved != CANONICAL_OUTPUT_ROOT.resolve(strict=False):
        raise ConfigBundleError("output is not the canonical r7 root")
    _assert_no_symlink_components(requested, include_leaf=False)
    if _lstat_or_none(requested) is not None:
        raise ConfigBundleError("canonical r7 output root is not absent")
    parent = resolved.parent
    parent_info = _lstat_or_none(parent)
    if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
        raise ConfigBundleError("canonical r7 output parent is unavailable")
    return resolved


def _config_payload(role: str, output_root: Path) -> dict[str, Any]:
    if role not in ROLE_FILENAMES:
        raise ConfigBundleError(f"unsupported config role: {role}")
    checkpoint_dir = (
        str(output_root / "runs/interrupted_parent/checkpoints/step-3")
        if role == "resume_child"
        else None
    )
    return {
        "schema_version": 1,
        "extends": str(CANONICAL_BASE_CONFIG),
        "run": {
            "name": role,
            "artifact_root": str(output_root / "runs"),
            "collision_policy": "fail",
        },
        "training": {
            "max_steps": 5,
            "forward_input_provider_mode": "synchronous",
        },
        "runtime": {
            "seed": 17,
            "determinism": {"mode": "strict_cuda_replay_v1"},
        },
        "eval": {"forward": {"steps": [3]}},
        "checkpoint": {
            "steps": [3, 5],
            "save_final": True,
        },
        "resume": {
            "mode": "exact_same_world_size",
            "checkpoint_dir": checkpoint_dir,
        },
    }


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "config bundle write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigBundleError("authored config is not strict UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ConfigBundleError("authored config root is not a mapping")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigBundleError("config bundle receipt is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ConfigBundleError("config bundle receipt root is not an object")
    _canonical_json_bytes(value)
    return value


def _validate_resolved(role: str, config: dict[str, Any], output_root: Path) -> None:
    expected_checkpoint = (
        str(output_root / "runs/interrupted_parent/checkpoints/step-3")
        if role == "resume_child"
        else None
    )
    try:
        exact = (
            config["schema_version"] == 1
            and config["run"]["name"] == role
            and config["run"]["artifact_root"] == str(output_root / "runs")
            and config["run"]["collision_policy"] == "fail"
            and config["training"]["max_steps"] == 5
            and config["training"]["forward_input_provider_mode"] == "synchronous"
            and config["runtime"]["seed"] == 17
            and config["runtime"]["determinism"]["mode"] == "strict_cuda_replay_v1"
            and config["eval"]["forward"]["every_fraction"] is None
            and config["eval"]["forward"]["steps"] == [3]
            and config["checkpoint"]["every_fraction"] is None
            and config["checkpoint"]["steps"] == [3, 5]
            and config["checkpoint"]["save_final"] is True
            and config["resume"]["mode"] == "exact_same_world_size"
            and config["resume"]["checkpoint_dir"] == expected_checkpoint
        )
    except (KeyError, TypeError):
        exact = False
    if not exact:
        raise ConfigBundleError(f"{role} resolved config semantics drifted")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_directory_no_replace(stage: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace config bundle publication is unavailable",
            target,
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def _write_and_validate_stage(
    stage: Path, *, base_config: Path, output_root: Path
) -> dict[str, Any]:
    config_dir = stage / "configs"
    config_dir.mkdir(mode=0o755)
    records: dict[str, Any] = {}
    projections: list[dict[str, Any]] = []
    for role, filename in ROLE_FILENAMES.items():
        payload = _config_payload(role, output_root)
        path = config_dir / filename
        encoded = yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=False,
        ).encode("utf-8")
        _write_new_file(path, encoded)
        if _strict_yaml(path) != payload:
            raise ConfigBundleError(f"{role} authored YAML failed exact reload")
        try:
            resolved = load_train_config(path)
        except Exception as exc:
            raise ConfigBundleError(
                f"{role} resolved config semantics are invalid"
            ) from exc
        _validate_resolved(role, resolved.config_dict, output_root)
        projection = dict(resolved.config_dict)
        del projection["run"]
        del projection["resume"]
        projections.append(projection)
        records[role] = {
            "path": str(output_root / "configs" / filename),
            "file_sha256": _sha256_file(path),
            "resolved_config_fingerprint": resolved.fingerprint,
        }
    if not projections or any(
        projection != projections[0] for projection in projections
    ):
        raise ConfigBundleError("authored config semantic projections differ")
    body = {
        "schema": SCHEMA,
        "status": "passed",
        "base_config": {
            "path": str(base_config),
            "file_sha256": CANONICAL_BASE_SHA256,
        },
        "output_root": str(output_root),
        "configs": records,
        "semantic_projection_sha256": sha256_json(projections[0]),
    }
    receipt = _signed(body)
    receipt_path = stage / RECEIPT_NAME
    _write_new_file(receipt_path, _canonical_json_bytes(receipt) + b"\n")
    if _strict_json(receipt_path) != receipt:
        raise ConfigBundleError("config bundle receipt failed exact reload")
    _fsync_directory(config_dir)
    _fsync_directory(stage)
    return receipt


def author(base_config: str | Path, output_root: str | Path) -> Path:
    base = _canonical_base(base_config)
    target = _canonical_absent_output(output_root)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=str(target.parent))
    )
    try:
        _write_and_validate_stage(stage, base_config=base, output_root=target)
        try:
            _install_directory_no_replace(stage, target)
        except (OSError, FileExistsError) as exc:
            raise ConfigBundleError("atomic config bundle publication failed") from exc
        # Successful RENAME_NOREPLACE is the explicit commit point: the complete,
        # signed bundle is now authoritative. Parent-directory fsync is only a
        # best-effort durability hint and cannot reclassify that commit as failure.
        try:
            _fsync_directory(target.parent)
        except OSError:
            pass
        return target / RECEIPT_NAME
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    author_parser = subparsers.add_parser(
        "author", help="Publish the canonical absent-only r7 config bundle."
    )
    author_parser.add_argument("--base-config", required=True)
    author_parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "author":
        raise ConfigBundleError("unsupported config bundle command")
    receipt = author(args.base_config, args.output_root)
    print(json.dumps({"receipt": str(receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
