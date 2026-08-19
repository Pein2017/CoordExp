"""One generic identity owner behind two import paths.

Wave 1 of ``decompose-coordexp-swift-training-orchestration`` moves the
domain-neutral identity machinery out of ``src.qwen.parity`` (design decision
7).  The move must be a *move*: the historical parity path and the neutral
``src.artifacts.identity`` path must resolve to the same function objects, the
same exception type, and the same ``qwen.parity.*`` contract codes, so every
historical reader that catches those errors keeps working and every published
artifact keeps comparing byte-for-byte.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

from src.artifacts import identity as neutral_identity
from src.common.errors import RuntimeContractError
from src.qwen import parity as parity_identity


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Public generic identity operations named by design decision 7.
MOVED_PUBLIC_NAMES: tuple[str, ...] = (
    "canonical_json_bytes",
    "sha256_json",
    "sha256_file",
    "assert_absent_artifact_target",
    "write_strict_json_atomic",
    "base_model_weight_identity",
    "base_model_weight_identity_with_execution_policy",
    "validate_model_weight_identity",
    "assert_model_weight_identity_equal",
    "repo_identity",
    "source_owner_identity",
)

#: Bound and schema constants that travel with the moved implementation.
MOVED_CONSTANT_NAMES: tuple[str, ...] = (
    "MODEL_WEIGHT_IDENTITY_SCHEMA",
    "MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA",
    "MAX_WEIGHT_INDEX_BYTES",
    "MAX_WEIGHT_DECLARATIONS",
    "MAX_WEIGHT_SHARDS",
    "MAX_WEIGHT_SHARD_BYTES",
    "MAX_WEIGHT_TOTAL_BYTES",
)


def _standalone_model_root(root: Path, *, payload: bytes = b"weights\n") -> Path:
    model_root = root / "base-model"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "model.safetensors").write_bytes(payload)
    return model_root


def _bounded_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "context_keys": sorted(getattr(exc, "context", {})),
    }


# ---------------------------------------------------------------------------
# One owner, two import paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MOVED_PUBLIC_NAMES)
def test_historical_and_neutral_identity_imports_share_one_object(name: str) -> None:
    assert getattr(parity_identity, name) is getattr(neutral_identity, name)


@pytest.mark.parametrize("name", MOVED_CONSTANT_NAMES)
def test_historical_and_neutral_identity_constants_are_equal(name: str) -> None:
    assert getattr(parity_identity, name) == getattr(neutral_identity, name)


def test_contract_error_type_is_shared_and_keeps_its_historical_name() -> None:
    assert parity_identity.ParityContractError is neutral_identity.ParityContractError
    assert issubclass(neutral_identity.ParityContractError, RuntimeContractError)
    assert neutral_identity.ParityContractError.__name__ == "ParityContractError"


def test_moved_implementations_are_no_longer_defined_inside_parity() -> None:
    """The move deletes the parity bodies; a duplicate would silently diverge."""

    tree = ast.parse(
        (REPO_ROOT / "src" / "qwen" / "parity.py").read_text(encoding="utf-8"),
        filename="src/qwen/parity.py",
    )
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    assert defined.isdisjoint(set(MOVED_PUBLIC_NAMES) | {"ParityContractError"})


def test_neutral_owner_does_not_import_the_historical_parity_module() -> None:
    tree = ast.parse(
        (REPO_ROOT / "src" / "artifacts" / "identity.py").read_text(encoding="utf-8"),
        filename="src/artifacts/identity.py",
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert "src.qwen.parity" not in modules


# ---------------------------------------------------------------------------
# Byte, digest, and schema equivalence
# ---------------------------------------------------------------------------


def test_canonical_json_and_digests_are_byte_equivalent(tmp_path: Path) -> None:
    payload = {"b": 2, "a": [1, {"z": None, "y": True}], "unicode": "é"}
    source = tmp_path / "source.txt"
    source.write_bytes(b"characterized-source-bytes\n")

    encoded = neutral_identity.canonical_json_bytes(payload)

    assert encoded == parity_identity.canonical_json_bytes(payload)
    assert encoded == b'{"a":[1,{"y":true,"z":null}],"b":2,"unicode":"\xc3\xa9"}'
    assert neutral_identity.sha256_json(payload) == parity_identity.sha256_json(payload)
    assert neutral_identity.sha256_file(source) == parity_identity.sha256_file(source)


def test_absent_target_validation_and_atomic_publication_agree(tmp_path: Path) -> None:
    target = tmp_path / "published.json"

    resolved = neutral_identity.assert_absent_artifact_target(target)
    published = neutral_identity.write_strict_json_atomic(target, {"published": True})

    assert resolved == target.resolve()
    assert published == target.resolve()
    assert published.read_bytes() == b'{"published":true}\n'
    with pytest.raises(parity_identity.ParityContractError) as occupied:
        parity_identity.assert_absent_artifact_target(target)
    assert _bounded_failure(occupied.value) == {
        "type": "ParityContractError",
        "code": "qwen.parity.artifact_collision",
        "context_keys": ["path"],
    }
    with pytest.raises(neutral_identity.ParityContractError) as collision:
        neutral_identity.write_strict_json_atomic(target, {"published": True})
    assert collision.value.code == "qwen.parity.artifact_collision"


def test_base_model_weight_identity_and_execution_policy_agree(tmp_path: Path) -> None:
    model_root = _standalone_model_root(tmp_path)

    identity = neutral_identity.base_model_weight_identity(model_root)
    with_policy, policy = (
        neutral_identity.base_model_weight_identity_with_execution_policy(
            model_root, max_workers=1
        )
    )

    assert identity == parity_identity.base_model_weight_identity(model_root)
    assert identity == with_policy
    assert identity["schema"] == neutral_identity.MODEL_WEIGHT_IDENTITY_SCHEMA
    assert identity["mode"] == "standalone_safetensors"
    assert identity["bounds"] == {
        "max_index_bytes": neutral_identity.MAX_WEIGHT_INDEX_BYTES,
        "max_declarations": neutral_identity.MAX_WEIGHT_DECLARATIONS,
        "max_shards": neutral_identity.MAX_WEIGHT_SHARDS,
        "max_shard_bytes": neutral_identity.MAX_WEIGHT_SHARD_BYTES,
        "max_total_bytes": neutral_identity.MAX_WEIGHT_TOTAL_BYTES,
    }
    assert policy == {
        "schema": neutral_identity.MODEL_WEIGHT_HASH_EXECUTION_POLICY_SCHEMA,
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": 1,
        "payload_file_count": 1,
    }
    assert neutral_identity.validate_model_weight_identity(identity) == identity
    assert parity_identity.validate_model_weight_identity(identity) == identity
    neutral_identity.assert_model_weight_identity_equal(identity, dict(identity))


def test_weight_identity_drift_raises_the_historical_contract_error(
    tmp_path: Path,
) -> None:
    identity = neutral_identity.base_model_weight_identity(
        _standalone_model_root(tmp_path)
    )
    drifted_body = copy.deepcopy(identity)
    del drifted_body["aggregate_sha256"]
    drifted_body["shards"][0]["sha256"] = "1" * 64
    drifted = {
        **drifted_body,
        "aggregate_sha256": neutral_identity.sha256_json(drifted_body),
    }

    with pytest.raises(parity_identity.ParityContractError) as drift:
        neutral_identity.assert_model_weight_identity_equal(identity, drifted)
    with pytest.raises(parity_identity.ParityContractError) as corrupt:
        neutral_identity.assert_model_weight_identity_equal(
            identity, {**identity, "aggregate_sha256": "0" * 64}
        )

    assert _bounded_failure(drift.value) == {
        "type": "ParityContractError",
        "code": "qwen.parity.weight_identity_drift",
        "context_keys": ["expected_aggregate_sha256", "observed_aggregate_sha256"],
    }
    assert _bounded_failure(corrupt.value) == {
        "type": "ParityContractError",
        "code": "qwen.parity.weight_fingerprint",
        "context_keys": ["expected", "observed"],
    }


def test_repository_and_source_owner_identity_agree(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "source.txt").write_bytes(b"characterized-source-bytes\n")

    rows = neutral_identity.source_owner_identity(tmp_path, ["owned/source.txt"])
    repository = neutral_identity.repo_identity(REPO_ROOT)

    assert rows == parity_identity.source_owner_identity(
        tmp_path, ["owned/source.txt"]
    )
    assert rows == [
        {
            "path": "owned/source.txt",
            "sha256": neutral_identity.sha256_file(owned / "source.txt"),
        }
    ]
    assert sorted(repository) == [
        "dirty",
        "head",
        "root",
        "status_sha256",
        "tracked_diff_sha256",
    ]
    assert repository == parity_identity.repo_identity(REPO_ROOT)


# ---------------------------------------------------------------------------
# Bounded failure contracts historical readers catch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_context_keys"),
    (
        ("canonical_json_bytes_non_json", "qwen.parity.strict_json", ["error", "value_type"]),
        (
            "canonical_json_bytes_non_finite",
            "qwen.parity.strict_json",
            ["error", "value_type"],
        ),
        ("sha256_file_missing", "qwen.parity.identity_file", ["error", "path"]),
        ("absent_target_missing_parent", "qwen.parity.artifact_parent", ["path"]),
        (
            "validate_model_weight_identity_incomplete",
            "qwen.parity.fields",
            ["missing", "owner", "unknown"],
        ),
        ("source_owner_identity_escape", "qwen.parity.source_owner_path", ["path"]),
    ),
)
def test_moved_failures_keep_their_namespaced_codes(
    tmp_path: Path,
    case: str,
    expected_code: str,
    expected_context_keys: list[str],
) -> None:
    bodies = {
        "canonical_json_bytes_non_json": lambda: neutral_identity.canonical_json_bytes(
            {"value": object()}
        ),
        "canonical_json_bytes_non_finite": (
            lambda: neutral_identity.canonical_json_bytes({"value": float("nan")})
        ),
        "sha256_file_missing": lambda: neutral_identity.sha256_file(
            tmp_path / "absent.txt"
        ),
        "absent_target_missing_parent": (
            lambda: neutral_identity.assert_absent_artifact_target(
                tmp_path / "absent-dir" / "target.json"
            )
        ),
        "validate_model_weight_identity_incomplete": (
            lambda: neutral_identity.validate_model_weight_identity(
                {"schema": neutral_identity.MODEL_WEIGHT_IDENTITY_SCHEMA}
            )
        ),
        "source_owner_identity_escape": (
            lambda: neutral_identity.source_owner_identity(tmp_path, ["../escape.txt"])
        ),
    }

    # Historical readers catch the parity-owned type; it must still be raised.
    with pytest.raises(parity_identity.ParityContractError) as failure:
        bodies[case]()

    assert _bounded_failure(failure.value) == {
        "type": "ParityContractError",
        "code": expected_code,
        "context_keys": expected_context_keys,
    }
