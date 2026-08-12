#!/usr/bin/env python3
"""Strictly validate and deterministically project a Wave 9 transition packet.

This module is intentionally read-only.  It has no authoring, sealing, cache
preparation, launch, retry, or artifact-publication surface.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


PACKET_SCHEMA = "coordexp-swift-wave9-transition-packet-v1"
PROJECTION_SCHEMA = "coordexp-swift-wave9-transition-projection-v1"
CACHE_VERSION = "coordexp-swift-pack-cache-v3"
CACHE_ROOT_ENV = "COORDEXP_SWIFT_PACK_CACHE_ROOT"
STRICT_DETERMINISM_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
}
WORLD_SIZE = 8
MAX_JSON_BYTES = 16 * 1024 * 1024

_PACKET_KEYS = {
    "schema",
    "lifecycle",
    "missing_inputs",
    "gates",
    "production",
    "cache",
    "budgets",
    "rollback",
    "launch",
    "packet_payload_sha256",
}
_REQUIRED_INPUT_PATHS = (
    "/gates/findings",
    "/gates/wave7",
    "/gates/wave8",
    "/production/config",
    "/production/policy",
    "/production/identities/source",
    "/production/identities/provenance",
    "/production/identities/runtime",
    "/production/identities/determinants",
    "/production/identities/model",
    "/cache/root_env",
    "/cache/splits/train",
    "/cache/splits/eval.forward",
    "/cache/build",
    "/cache/materialization",
    "/budgets/cpu",
    "/budgets/io",
    "/budgets/time",
    "/budgets/storage",
    "/budgets/gpu",
    "/budgets/host",
    "/budgets/artifact",
    "/rollback/bindings",
    "/rollback/route",
    "/rollback/immutable",
    "/rollback/delete_old_caches",
    "/rollback/mutate_old_caches",
    "/launch/argv",
    "/launch/env",
    "/launch/devices",
    "/launch/targets",
    "/launch/attempt_marker",
    "/launch/terminal_receipt",
    "/launch/retry",
)
_BUDGET_UNITS = {
    "cpu": "cpu_core_seconds",
    "io": "bytes",
    "time": "seconds",
    "storage": "bytes",
    "gpu": "gpu_device_seconds",
    "host": "bytes",
    "artifact": "bytes",
}
_CACHE_ENV_ALLOWLIST = {
    CACHE_ROOT_ENV,
    *STRICT_DETERMINISM_ENVIRONMENT,
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}
_LAUNCH_ENV_ALLOWLIST = {
    CACHE_ROOT_ENV,
    "CUDA_VISIBLE_DEVICES",
    "FLASH_ATTENTION_DETERMINISTIC",
    "CUBLAS_WORKSPACE_CONFIG",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}
_PLACEHOLDER = re.compile(
    r"(?:\b(?:tbd|todo|placeholder|unknown|changeme)\b|\{[^{}]+\}|<[^<>]+>)",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class TransitionPacketError(ValueError):
    """The packet cannot safely describe a Wave 9 transition."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical byte representation used for all identities."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TransitionPacketError("value is not canonical finite JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _parse_strict_json(raw: bytes, *, owner: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TransitionPacketError(f"{owner} is not strict finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TransitionPacketError(f"{owner} strict JSON root must be an object")
    return value


def _absolute_path(value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TransitionPacketError(f"{owner} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.normpath(value)) != path:
        raise TransitionPacketError(f"{owner} must be a normalized absolute path")
    return path


def _assert_no_symlink_components(path: Path, *, owner: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TransitionPacketError(f"{owner} path cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise TransitionPacketError(f"{owner} path contains a symlink")


def _regular_file(value: Any, *, owner: str) -> Path:
    path = _absolute_path(value, owner=owner)
    _assert_no_symlink_components(path, owner=owner)
    try:
        info = path.lstat()
    except OSError as exc:
        raise TransitionPacketError(
            f"{owner} is not an available regular file"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise TransitionPacketError(f"{owner} is not a regular file")
    if info.st_size > MAX_JSON_BYTES and path.suffix == ".json":
        raise TransitionPacketError(f"{owner} exceeds the bounded JSON size")
    return path


def _existing_directory(value: Any, *, owner: str) -> Path:
    path = _absolute_path(value, owner=owner)
    _assert_no_symlink_components(path, owner=owner)
    try:
        info = path.lstat()
    except OSError as exc:
        raise TransitionPacketError(f"{owner} is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise TransitionPacketError(f"{owner} must be an existing directory")
    return path


def _absent_path(value: Any, *, owner: str) -> Path:
    path = _absolute_path(value, owner=owner)
    _assert_no_symlink_components(path, owner=owner)
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise TransitionPacketError(f"{owner} absence cannot be inspected") from exc
    raise TransitionPacketError(f"{owner} must be absent")


def load_transition_packet(path: str | Path) -> dict[str, Any]:
    """Load one canonical strict-JSON packet without changing any filesystem state."""

    packet_path = _regular_file(str(Path(path).absolute()), owner="transition packet")
    try:
        raw = packet_path.read_bytes()
    except OSError as exc:
        raise TransitionPacketError("transition packet cannot be read") from exc
    value = _parse_strict_json(raw, owner="transition packet")
    if raw != canonical_json_bytes(value) + b"\n":
        raise TransitionPacketError("transition packet bytes are not canonical JSON")
    return value


def _exact_keys(
    value: Any,
    expected: set[str],
    *,
    owner: str,
    allow_missing: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransitionPacketError(f"{owner} must be an object")
    actual = set(value)
    extra = actual - expected
    missing = expected - actual
    if extra or (missing and not allow_missing):
        raise TransitionPacketError(
            f"{owner} keys mismatch; missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def _reject_nulls_placeholders(value: Any, *, path: str = "") -> None:
    if value is None:
        raise TransitionPacketError(f"null is forbidden at {path or '/'}")
    if isinstance(value, float) and not math.isfinite(value):
        raise TransitionPacketError(f"non-finite value is forbidden at {path or '/'}")
    if isinstance(value, str) and _PLACEHOLDER.search(value):
        raise TransitionPacketError(f"placeholder is forbidden at {path or '/'}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nulls_placeholders(child, path=f"{path}/{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_nulls_placeholders(child, path=f"{path}/{index}")


def _require_sha256(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TransitionPacketError(f"{owner} must be a lowercase SHA-256")
    return value


def _signed_payload_sha256(value: Mapping[str, Any], *, owner: str) -> str:
    candidates = [
        key
        for key in value
        if key.endswith("_payload_sha256")
        or key in {"attestation_sha256", "receipt_sha256", "plan_sha256"}
    ]
    valid: list[str] = []
    for key in candidates:
        claimed = value.get(key)
        if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
            continue
        body = {name: child for name, child in value.items() if name != key}
        if _sha256_bytes(canonical_json_bytes(body)) == claimed:
            valid.append(claimed)
    if len(valid) != 1:
        raise TransitionPacketError(f"{owner} lacks one canonical signed identity")
    return valid[0]


def _validate_binding(
    value: Any,
    *,
    owner: str,
    schema_token: str | None = None,
    extra_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    keys = {
        "path",
        "file_sha256",
        "payload_sha256",
        "schema",
        "status",
        "current",
    } | (extra_keys or set())
    row = _exact_keys(value, keys, owner=owner)
    schema = row.get("schema")
    if not isinstance(schema, str):
        raise TransitionPacketError(f"{owner} schema is missing")
    lowered = schema.lower()
    if "legacy" in lowered or lowered.endswith("-v0"):
        raise TransitionPacketError(f"{owner} uses a legacy gate or identity schema")
    if schema_token is not None and schema_token not in lowered:
        raise TransitionPacketError(
            f"{owner} schema is not the current {schema_token} binding"
        )
    if row.get("status") != "passed":
        raise TransitionPacketError(f"{owner} must be passed")
    if row.get("current") is not True:
        raise TransitionPacketError(f"{owner} must be explicitly current")
    path = _regular_file(row.get("path"), owner=owner)
    expected_file_sha = _require_sha256(
        row.get("file_sha256"), owner=f"{owner} file_sha256"
    )
    if _sha256_file(path) != expected_file_sha:
        raise TransitionPacketError(f"{owner} file_sha256 mutation detected")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TransitionPacketError(f"{owner} cannot be read") from exc
    artifact = _parse_strict_json(raw, owner=owner)
    if raw != canonical_json_bytes(artifact) + b"\n":
        raise TransitionPacketError(f"{owner} is not canonical JSON")
    if artifact.get("schema") != schema or artifact.get("status") != "passed":
        raise TransitionPacketError(f"{owner} schema or passed status drifted")
    payload_sha = _require_sha256(
        row.get("payload_sha256"), owner=f"{owner} payload_sha256"
    )
    if _signed_payload_sha256(artifact, owner=owner) != payload_sha:
        raise TransitionPacketError(f"{owner} payload_sha256 mutation detected")
    return row


def _lookup(value: Mapping[str, Any], pointer: str) -> tuple[bool, Any]:
    current: Any = value
    for part in pointer.lstrip("/").split("/"):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _actual_missing_inputs(packet: Mapping[str, Any]) -> list[str]:
    return sorted(
        pointer for pointer in _REQUIRED_INPUT_PATHS if not _lookup(packet, pointer)[0]
    )


def _validate_gates(value: Any, *, draft: bool) -> None:
    gates = _exact_keys(
        value,
        {"wave7", "wave8", "findings"},
        owner="gates",
        allow_missing=draft,
    )
    if "wave7" in gates:
        _validate_binding(gates["wave7"], owner="Wave 7 gate", schema_token="wave7")
    if "wave8" in gates:
        _validate_binding(gates["wave8"], owner="Wave 8 gate", schema_token="wave8")
    if "findings" in gates:
        findings = _exact_keys(gates["findings"], {"p0", "p1"}, owner="findings")
        if findings.get("p0") != [] or findings.get("p1") != []:
            raise TransitionPacketError("production findings must contain zero P0/P1")


def _validate_config(value: Any) -> None:
    config = _exact_keys(
        value,
        {"path", "file_sha256", "resolved_fingerprint"},
        owner="production config",
    )
    path = _regular_file(config.get("path"), owner="selected production config")
    expected = _require_sha256(
        config.get("file_sha256"), owner="production config file_sha256"
    )
    if _sha256_file(path) != expected:
        raise TransitionPacketError("production config file_sha256 mutation detected")
    _require_sha256(
        config.get("resolved_fingerprint"), owner="resolved config fingerprint"
    )


def _validate_policy(value: Any) -> None:
    policy = _exact_keys(
        value,
        {
            "packing_policy",
            "input_provider",
            "cache_version",
            "dependency_change",
            "world_size",
        },
        owner="production policy",
    )
    expected = {
        "packing_policy": "source_order_next_fit",
        "input_provider": "synchronous",
        "cache_version": CACHE_VERSION,
        "dependency_change": "none",
        "world_size": WORLD_SIZE,
    }
    if dict(policy) != expected:
        raise TransitionPacketError(
            "production policy drifted from the retained defaults"
        )


def _validate_identities(value: Any, *, draft: bool) -> None:
    identities = _exact_keys(
        value,
        {"source", "provenance", "runtime", "determinants", "model"},
        owner="production identities",
        allow_missing=draft,
    )
    tokens = {
        "provenance": "provenance",
        "runtime": "runtime",
        "determinants": "determinant",
        "model": "model",
    }
    if "source" in identities:
        source = _validate_binding(
            identities["source"],
            owner="dirty source identity",
            schema_token="source",
            extra_keys={"repository_commit", "dirty", "dirty_sha256"},
        )
        if source.get("dirty") is not True:
            raise TransitionPacketError(
                "source identity must explicitly bind dirty state"
            )
        commit = source.get("repository_commit")
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
            raise TransitionPacketError(
                "source identity requires an exact repository commit"
            )
        _require_sha256(source.get("dirty_sha256"), owner="dirty source digest")
    for name, token in tokens.items():
        if name in identities:
            _validate_binding(
                identities[name], owner=f"{name} identity", schema_token=token
            )


def _validate_production(value: Any, *, draft: bool) -> None:
    production = _exact_keys(
        value,
        {"config", "policy", "identities"},
        owner="production",
        allow_missing=draft,
    )
    if "config" in production:
        _validate_config(production["config"])
    if "policy" in production:
        _validate_policy(production["policy"])
    if "identities" in production:
        _validate_identities(production["identities"], draft=draft)


def _argv_value(argv: Sequence[str], option: str, *, owner: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise TransitionPacketError(f"{owner} must contain exactly one {option}")
    return argv[positions[0] + 1]


def _validate_argv(value: Any, *, owner: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item or "\0" in item or "\n" in item
            for item in value
        )
    ):
        raise TransitionPacketError(f"{owner} must be an exact nonempty argv array")
    return value


def _validate_env(value: Any, *, owner: str, allowlist: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(child, str)
        for key, child in value.items()
    ):
        raise TransitionPacketError(f"{owner} must be a string environment object")
    unexpected = set(value) - allowlist
    if unexpected:
        raise TransitionPacketError(f"{owner} violates the environment allowlist")
    return value


def _validate_cache(value: Any, *, config_path: str, draft: bool) -> None:
    cache = _exact_keys(
        value,
        {"root_env", "splits", "build", "materialization"},
        owner="cache transition",
        allow_missing=draft,
    )
    root: Path | None = None
    if "root_env" in cache:
        root_env = _exact_keys(
            cache["root_env"], {"name", "value"}, owner="cache root environment"
        )
        if root_env.get("name") != CACHE_ROOT_ENV:
            raise TransitionPacketError(
                f"cache root must explicitly name {CACHE_ROOT_ENV}"
            )
        root = _existing_directory(root_env.get("value"), owner="production cache root")
    if "splits" in cache:
        splits = _exact_keys(
            cache["splits"],
            {"train", "eval.forward"},
            owner="cache splits",
            allow_missing=draft,
        )
        seen: set[str] = set()
        for split in ("train", "eval.forward"):
            if split not in splits:
                continue
            row = _exact_keys(
                splits[split], {"target", "fingerprint"}, owner=f"{split} cache"
            )
            fingerprint = _require_sha256(
                row.get("fingerprint"), owner=f"{split} cache fingerprint"
            )
            target = _absent_path(row.get("target"), owner=f"{split} cache target")
            if root is None:
                raise TransitionPacketError(
                    "cache splits require the explicit cache root"
                )
            canonical = root / CACHE_VERSION / fingerprint
            if target != canonical:
                raise TransitionPacketError(f"{split} cache target is not canonical v3")
            if fingerprint in seen:
                raise TransitionPacketError(
                    "train and eval cache fingerprints must be distinct"
                )
            seen.add(fingerprint)
    if "build" in cache:
        build = _exact_keys(
            cache["build"],
            {"argv", "env", "payload_validation", "receipt_target"},
            owner="cache build",
        )
        argv = _validate_argv(build.get("argv"), owner="cache build argv")
        try:
            module_index = argv.index("-m")
        except ValueError as exc:
            raise TransitionPacketError(
                "cache build argv must use module execution"
            ) from exc
        if (
            module_index + 1 >= len(argv)
            or argv[module_index + 1] != "src.prepare_train_cache"
        ):
            raise TransitionPacketError(
                "cache build argv must execute src.prepare_train_cache"
            )
        if _argv_value(argv, "--config", owner="cache build argv") != config_path:
            raise TransitionPacketError("cache build argv config path drifted")
        receipt_target = _absent_path(
            build.get("receipt_target"), owner="cache build receipt target"
        )
        if _argv_value(argv, "--receipt", owner="cache build argv") != str(
            receipt_target
        ):
            raise TransitionPacketError("cache build argv receipt target drifted")
        env = _validate_env(
            build.get("env"),
            owner="cache build environment",
            allowlist=_CACHE_ENV_ALLOWLIST,
        )
        if CACHE_ROOT_ENV not in env:
            raise TransitionPacketError(
                f"cache build environment lacks {CACHE_ROOT_ENV}"
            )
        if root is None or env[CACHE_ROOT_ENV] != str(root):
            raise TransitionPacketError("cache build environment cache root drifted")
        if any(
            env.get(name) != expected
            for name, expected in STRICT_DETERMINISM_ENVIRONMENT.items()
        ):
            raise TransitionPacketError(
                "cache build strict determinism environment is missing or conflicting"
            )
        if build.get("payload_validation") != "payloads":
            raise TransitionPacketError("cache build must require payloads validation")
    if "materialization" in cache:
        materialization = _exact_keys(
            cache["materialization"],
            {"strategy", "workers", "resolved_planner_workers"},
            owner="cache materialization",
        )
        workers = materialization.get("workers")
        planner_workers = materialization.get("resolved_planner_workers")
        if materialization.get("strategy") != "fork_process_pool":
            raise TransitionPacketError(
                "cache materialization must use fork_process_pool"
            )
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 1:
            raise TransitionPacketError(
                "cache materialization workers must be greater than one"
            )
        if (
            isinstance(planner_workers, bool)
            or not isinstance(planner_workers, int)
            or planner_workers <= 0
        ):
            raise TransitionPacketError(
                "resolved planner workers must be a positive integer"
            )


def _validate_budgets(value: Any, *, draft: bool) -> None:
    budgets = _exact_keys(
        value,
        set(_BUDGET_UNITS),
        owner="budgets",
        allow_missing=draft,
    )
    for name, unit in _BUDGET_UNITS.items():
        if name not in budgets:
            continue
        row = _exact_keys(
            budgets[name],
            {"expected", "hard", "unit", "evidence"},
            owner=f"{name} budget",
        )
        expected = row.get("expected")
        hard = row.get("hard")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(expected)
            or expected <= 0
        ):
            raise TransitionPacketError(
                f"{name} expected budget must be positive and finite"
            )
        if (
            isinstance(hard, bool)
            or not isinstance(hard, (int, float))
            or not math.isfinite(hard)
            or hard < expected
        ):
            raise TransitionPacketError(
                f"{name} hard budget must cover the expected budget"
            )
        if row.get("unit") != unit:
            raise TransitionPacketError(f"{name} budget unit drifted")
        _validate_binding(row.get("evidence"), owner=f"{name} budget evidence")


def _validate_rollback(value: Any, *, draft: bool) -> None:
    rollback = _exact_keys(
        value,
        {"bindings", "route", "immutable", "delete_old_caches", "mutate_old_caches"},
        owner="rollback",
        allow_missing=draft,
    )
    if "bindings" in rollback:
        bindings = rollback["bindings"]
        if not isinstance(bindings, list) or not bindings:
            raise TransitionPacketError("rollback requires immutable bindings")
        for index, binding in enumerate(bindings):
            _validate_binding(binding, owner=f"rollback binding {index}")
    if "route" in rollback and (
        not isinstance(rollback["route"], str) or not rollback["route"].strip()
    ):
        raise TransitionPacketError("rollback route must be explicit")
    if "immutable" in rollback and rollback["immutable"] is not True:
        raise TransitionPacketError("rollback bindings must be immutable")
    if (
        "delete_old_caches" in rollback and rollback["delete_old_caches"] is not False
    ) or (
        "mutate_old_caches" in rollback and rollback["mutate_old_caches"] is not False
    ):
        raise TransitionPacketError("rollback must never mutate or delete an old cache")


def _validate_launch(
    value: Any,
    *,
    config_path: str,
    cache_root: str,
    draft: bool,
) -> None:
    launch = _exact_keys(
        value,
        {
            "argv",
            "env",
            "devices",
            "targets",
            "attempt_marker",
            "terminal_receipt",
            "retry",
        },
        owner="launch",
        allow_missing=draft,
    )
    devices: list[int] | None = None
    if "argv" in launch:
        argv = _validate_argv(launch["argv"], owner="launch argv")
        if _argv_value(argv, "--num_processes", owner="launch argv") != str(WORLD_SIZE):
            raise TransitionPacketError("launch argv world size drifted")
        if _argv_value(argv, "--module", owner="launch argv") != "src.train":
            raise TransitionPacketError("launch argv must use module src.train")
        if _argv_value(argv, "--config", owner="launch argv") != config_path:
            raise TransitionPacketError("launch argv config path drifted")
    if "devices" in launch:
        devices = launch["devices"]
        if devices != list(range(WORLD_SIZE)):
            raise TransitionPacketError(
                "launch devices must be the exact eight-device set"
            )
    if "env" in launch:
        env = _validate_env(
            launch["env"], owner="launch environment", allowlist=_LAUNCH_ENV_ALLOWLIST
        )
        required = {
            CACHE_ROOT_ENV: cache_root,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        if any(env.get(key) != expected for key, expected in required.items()):
            raise TransitionPacketError(
                "launch environment drifted from the exact contract"
            )
    if "targets" in launch:
        targets = _exact_keys(
            launch["targets"],
            {"run_root", "checkpoint_root", "evaluation_root", "resume_root"},
            owner="launch targets",
        )
        for name, target in targets.items():
            _absent_path(target, owner=f"launch {name}")
    if "attempt_marker" in launch:
        _absent_path(launch["attempt_marker"], owner="launch attempt marker")
    if "terminal_receipt" in launch:
        _absent_path(launch["terminal_receipt"], owner="launch terminal receipt")
    if "retry" in launch:
        retry = _exact_keys(
            launch["retry"],
            {"allowed", "automatic", "max_attempts"},
            owner="launch retry",
        )
        if dict(retry) != {"allowed": False, "automatic": False, "max_attempts": 1}:
            raise TransitionPacketError("launch retry is forbidden")


def _validate_sections(packet: Mapping[str, Any], *, draft: bool) -> None:
    if "gates" in packet:
        _validate_gates(packet["gates"], draft=draft)
    if "production" in packet:
        _validate_production(packet["production"], draft=draft)
    config_path = ""
    if _lookup(packet, "/production/config/path")[0]:
        config_path = str(packet["production"]["config"]["path"])
    if "cache" in packet:
        _validate_cache(packet["cache"], config_path=config_path, draft=draft)
    if "budgets" in packet:
        _validate_budgets(packet["budgets"], draft=draft)
    if "rollback" in packet:
        _validate_rollback(packet["rollback"], draft=draft)
    if "launch" in packet:
        found_root, root_value = _lookup(packet, "/cache/root_env/value")
        _validate_launch(
            packet["launch"],
            config_path=config_path,
            cache_root=str(root_value) if found_root else "",
            draft=draft,
        )


def _validate_packet_signature(packet: Mapping[str, Any]) -> str:
    claimed = _require_sha256(
        packet.get("packet_payload_sha256"), owner="packet payload signature"
    )
    body = {
        key: value for key, value in packet.items() if key != "packet_payload_sha256"
    }
    observed = _sha256_bytes(canonical_json_bytes(body))
    if observed != claimed:
        raise TransitionPacketError("packet payload signature mismatch")
    return claimed


def _sealed_projection(packet: Mapping[str, Any]) -> dict[str, Any]:
    identities = packet["production"]["identities"]
    projection: dict[str, Any] = {
        "schema": PROJECTION_SCHEMA,
        "packet_schema": PACKET_SCHEMA,
        "packet_payload_sha256": packet["packet_payload_sha256"],
        "lifecycle": packet["lifecycle"],
        "executable": True,
        "missing_inputs": [],
        "gates": {
            "wave7_payload_sha256": packet["gates"]["wave7"]["payload_sha256"],
            "wave8_payload_sha256": packet["gates"]["wave8"]["payload_sha256"],
            "p0_count": 0,
            "p1_count": 0,
        },
        "config": deepcopy(packet["production"]["config"]),
        "policy": deepcopy(packet["production"]["policy"]),
        "identities": {
            name: {
                "schema": row["schema"],
                "file_sha256": row["file_sha256"],
                "payload_sha256": row["payload_sha256"],
            }
            for name, row in identities.items()
        },
        "cache": deepcopy(packet["cache"]),
        "budgets": {
            name: {
                "expected": row["expected"],
                "hard": row["hard"],
                "unit": row["unit"],
                "evidence_payload_sha256": row["evidence"]["payload_sha256"],
            }
            for name, row in packet["budgets"].items()
        },
        "rollback": {
            "route": packet["rollback"]["route"],
            "immutable": packet["rollback"]["immutable"],
            "delete_old_caches": packet["rollback"]["delete_old_caches"],
            "mutate_old_caches": packet["rollback"]["mutate_old_caches"],
            "binding_payload_sha256s": [
                row["payload_sha256"] for row in packet["rollback"]["bindings"]
            ],
        },
        "launch": deepcopy(packet["launch"]),
    }
    projection["identities"]["source"].update(
        {
            "repository_commit": identities["source"]["repository_commit"],
            "dirty": True,
            "dirty_sha256": identities["source"]["dirty_sha256"],
        }
    )
    return projection


def validate_transition_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a packet and return its deterministic, signed projection."""

    if not isinstance(packet, Mapping):
        raise TransitionPacketError("transition packet must be an object")
    allowed_missing = packet.get("lifecycle") == "draft"
    row = _exact_keys(
        packet,
        _PACKET_KEYS,
        owner="transition packet",
        allow_missing=allowed_missing,
    )
    for mandatory in ("schema", "lifecycle", "missing_inputs", "packet_payload_sha256"):
        if mandatory not in row:
            raise TransitionPacketError(f"transition packet is missing {mandatory}")
    _reject_nulls_placeholders(row)
    _validate_packet_signature(row)
    if row.get("schema") != PACKET_SCHEMA:
        raise TransitionPacketError("transition packet schema is not v1")
    lifecycle = row.get("lifecycle")
    if lifecycle not in {"draft", "sealed"}:
        raise TransitionPacketError(
            "transition packet lifecycle must be draft or sealed"
        )
    missing_inputs = row.get("missing_inputs")
    if (
        not isinstance(missing_inputs, list)
        or any(not isinstance(item, str) for item in missing_inputs)
        or missing_inputs != sorted(set(missing_inputs))
    ):
        raise TransitionPacketError(
            "missing_inputs must be an exact sorted unique list"
        )
    actual_missing = _actual_missing_inputs(row)
    if missing_inputs != actual_missing:
        raise TransitionPacketError(
            f"missing_inputs mismatch; expected={actual_missing} observed={missing_inputs}"
        )
    draft = lifecycle == "draft"
    if not draft and missing_inputs:
        raise TransitionPacketError("sealed packet cannot have missing_inputs")
    _validate_sections(row, draft=draft)
    if draft:
        projection: dict[str, Any] = {
            "schema": PROJECTION_SCHEMA,
            "packet_schema": PACKET_SCHEMA,
            "packet_payload_sha256": row["packet_payload_sha256"],
            "lifecycle": "draft",
            "executable": False,
            "missing_inputs": list(missing_inputs),
        }
    else:
        projection = _sealed_projection(row)
    projection["projection_sha256"] = _sha256_bytes(canonical_json_bytes(projection))
    return projection


def load_and_validate_transition_packet(path: str | Path) -> dict[str, Any]:
    return validate_transition_packet(load_transition_packet(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Wave 9 production-transition packet validator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate and project a packet")
    validate.add_argument(
        "--packet", required=True, help="Canonical signed packet JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse owns this boundary.
        raise TransitionPacketError("unsupported command")
    try:
        projection = load_and_validate_transition_packet(args.packet)
    except TransitionPacketError as exc:
        print(f"invalid Wave 9 transition packet: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(projection).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
