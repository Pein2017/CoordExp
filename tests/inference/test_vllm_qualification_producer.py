from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.errors import RuntimeContractError


SHA = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_engine_kwargs(model_path: str, max_num_seqs: int) -> dict[str, Any]:
    return {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "seed": 0,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "max_num_seqs": max_num_seqs,
        "disable_custom_all_reduce": True,
        "disable_log_stats": True,
        "generation_config": "vllm",
        "logprobs_mode": "processed_logprobs",
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "mm_processor_kwargs": {"do_resize": False},
    }


def _identity(tmp_path: Path, *, suffix: str = "") -> dict[str, Any]:
    from src.inference.backend import BackendLaunch
    from src.inference.vllm_qualification_producer import build_launch_contract

    config = tmp_path / "vllm.yaml"
    model = tmp_path / "model"
    source = tmp_path / "source.py"
    model.mkdir(exist_ok=True)
    config.write_text(f"config{suffix}\n", encoding="utf-8")
    source.write_text(f"source{suffix}\n", encoding="utf-8")
    identity = {
        "source": {
            "files": [
                {
                    "path": "src/inference/source.py",
                    "sha256": _sha256(source),
                }
            ],
            "fingerprint": hashlib.sha256(
                json.dumps(
                    [{"path": "src/inference/source.py", "sha256": _sha256(source)}],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "config": {
            "entry_path": str(config.resolve()),
            "fingerprint": hashlib.sha256(f"config{suffix}".encode()).hexdigest(),
            "sources": [
                {
                    "path": str(config.resolve()),
                    "sha256": _sha256(config),
                }
            ],
            "launch_contracts": {},
        },
        "model": {
            "mode": "materialized",
            "composition_key": hashlib.sha256(f"composition{suffix}".encode()).hexdigest(),
            "snapshot_fingerprint": _bounded_child_evidence(tmp_path)[
                "bounded_generation"
            ]["snapshot_manifest"]["fingerprint"],
            "receipt_fingerprint": hashlib.sha256(f"receipt{suffix}".encode()).hexdigest(),
            "source_fingerprints": {
                "base": hashlib.sha256(f"base{suffix}".encode()).hexdigest(),
                "adapter": hashlib.sha256(f"adapter{suffix}".encode()).hexdigest(),
                "embedding_delta": hashlib.sha256(f"delta{suffix}".encode()).hexdigest(),
            },
            "target_dtype": "bf16",
        },
        "runtime": {
            "python": "3.12.0",
            "packages": {
                "vllm": "0.14.1",
                "torch": "2.9.1",
                "transformers": "4.57.6",
                "peft": "0.18.1",
                "qwen-vl-utils": "0.0.14",
            },
            "loaded_sources": [
                {
                    "package": "vllm",
                    "relative_path": "vllm/engine/arg_utils.py",
                    "sha256": SHA,
                }
            ],
        },
    }
    model_path = str(model.resolve())
    for max_num_seqs in (1, 4):
        launch = BackendLaunch(
            backend="vllm",
            model_path=model_path,
            model_dtype="bf16",
            batch_size=max_num_seqs,
            generation_config_fingerprint=(
                "generation" if max_num_seqs == 1 else "generation-seq4"
            ),
            backend_options={"vllm": {}},
        )
        identity["config"]["launch_contracts"][f"seq{max_num_seqs}"] = (
            build_launch_contract(
                launch=launch,
                engine_kwargs=_fixture_engine_kwargs(model_path, max_num_seqs),
            )
        )
    return identity


def _child_evidence(kind: str, max_num_seqs: int) -> dict[str, Any]:
    cleanup = {
        "shutdown_completed": True,
        "worker_pid_alive_after_exit": False,
        "owned_children_after": [],
        "gpu_memory_returned_to_baseline": True,
    }
    if kind == "composition":
        return {
            "status": "passed",
            "composition_digest": "b" * 64,
            "prompt_ids_equal": True,
            "selected_rows_equal": True,
            "greedy_ids_equal": True,
            "full_vocab": {
                "allclose": True,
                "max_abs_diff": 0.001,
                "max_rel_diff": 0.002,
                "compared_value_count": 16,
            },
            "selected_vocab": {
                "allclose": True,
                "max_abs_diff": 0.001,
                "max_rel_diff": 0.002,
                "compared_value_count": 8,
            },
            "cleanup": cleanup,
        }
    if kind == "runtime":
        return {
            "status": "passed",
            "max_num_seqs": max_num_seqs,
            "request_count": 1,
            "completed_request_count": 1,
            "generated_token_count": 2,
            "finite_non_positive_policy_logprobs": True,
            "cleanup": cleanup,
        }
    if kind == "concurrency":
        return {
            "status": "passed",
            "max_num_seqs": max_num_seqs,
            "request_count": 4,
            "completed_request_count": 4,
            "ordered_request_ids_sha256": "c" * 64,
            "generated_token_count": 8,
            "finite_non_positive_policy_logprobs": True,
            "cleanup": cleanup,
        }
    assert kind == "forced_replay"
    return {
        "status": "passed",
        "max_num_seqs": max_num_seqs,
        "request_count": 1,
        "completed_request_count": 1,
        "aligned_token_count": 2,
        "processor_source_sha256": "d" * 64,
        "finite_non_positive_raw_logprobs": True,
        "raw_logprob_min": -1.25,
        "raw_logprob_max": -0.25,
        "cleanup": cleanup,
    }


def _dependencies(tmp_path: Path, *, identity: dict[str, Any] | None = None) -> Any:
    from src.inference.vllm_qualification_producer import (
        ChildExecution,
        QualificationDependencies,
    )

    current_identity = identity or _identity(tmp_path)

    def run_child(spec: Any) -> ChildExecution:
        spec.evidence_dir.mkdir(parents=True, exist_ok=False)
        artifact = spec.evidence_dir / "evidence.json"
        artifact.write_text(
            json.dumps({"kind": spec.kind, "max_num_seqs": spec.max_num_seqs}),
            encoding="utf-8",
        )
        return ChildExecution(
            returncode=0,
            evidence=(
                _bounded_child_evidence(tmp_path)
                if spec.kind == "composition"
                else _child_evidence(spec.kind, spec.max_num_seqs)
            ),
            process={
                "worker_pid": 1234,
                "worker_returncode": 0,
                "worker_pid_alive_after_exit": False,
                "owned_children_after": [],
                "process_group_id": 1234,
                "process_group_members_after": [],
                "process_group_termination_required": False,
                "gpu_process_group_memory_after_mib": 0,
                "visible_gpu_selector": "2",
                "gpu_memory_returned_to_baseline": True,
                "gpu_memory_before_mib": 100,
                "gpu_memory_after_mib": 100,
                "gpu_memory_tolerance_mib": 64,
            },
        )

    return QualificationDependencies(
        build_identity=lambda _: current_identity,
        run_child=run_child,
    )


def _produce(tmp_path: Path, *, dependencies: Any | None = None) -> Path:
    from src.inference.vllm_qualification_producer import produce

    root = tmp_path / "candidate"
    produce(
        config_path=tmp_path / "vllm.yaml",
        output_root=root,
        dependencies=dependencies or _dependencies(tmp_path),
    )
    return root


def _bounded_child_evidence(tmp_path: Path) -> dict[str, Any]:
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from src.inference.model_assets import build_model_snapshot_manifest
    from src.qwen.tokens import DEFAULT_COORDINATE_TOKENS, DEFAULT_WRAPPER_TOKENS

    snapshot = tmp_path / "model"
    snapshot.mkdir(exist_ok=True)
    tokens = [
        "[UNK]",
        "<|im_end|>",
        "\n",
        *DEFAULT_WRAPPER_TOKENS,
        *DEFAULT_COORDINATE_TOKENS[::2],
        *DEFAULT_COORDINATE_TOKENS[1::2],
    ]
    tokenizer = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]")
    )
    tokenizer.add_special_tokens(
        [AddedToken(token, normalized=False, special=True) for token in tokens[1:]]
    )
    tokenizer.save(str(snapshot / "tokenizer.json"))
    dynamic = [
        tokenizer.token_to_id(DEFAULT_WRAPPER_TOKENS[0]),
        tokenizer.token_to_id("<|coord_0|>"),
        tokenizer.token_to_id("<|im_end|>"),
    ]
    materialized = [dynamic[0], tokenizer.token_to_id("<|coord_1|>"), dynamic[-1]]
    evidence = _child_evidence("composition", 1)
    evidence["greedy_ids_equal"] = False
    for field in ("full_vocab", "selected_vocab"):
        evidence[field]["allclose"] = False
        evidence[field]["max_abs_diff"] = 1.875
    evidence["bounded_generation"] = {
        "policy": "coordexp-merged-bf16-coordinate-grid1-v1",
        "execution_reference": "materialized_bf16",
        "coordinate_max_abs_delta": 1,
        "snapshot_manifest": build_model_snapshot_manifest(snapshot),
        "dynamic_generated_ids": dynamic,
        "materialized_generated_ids": materialized,
        "token_count_equal": True,
        "coordinate_positions_equal": True,
        "non_coordinate_tokens_equal": True,
        "eos_positions_equal": True,
        "coordinate_count": 1,
        "coordinate_change_count": 1,
        "max_coordinate_value_delta": 1,
        "accepted": True,
    }
    return evidence


def test_bounded_composition_producer_accepts_grid1_with_honest_numeric_drift(
    tmp_path: Path,
) -> None:
    evidence = _bounded_child_evidence(tmp_path)
    identity = _identity(tmp_path)
    identity["model"]["snapshot_fingerprint"] = evidence["bounded_generation"][
        "snapshot_manifest"
    ]["fingerprint"]
    deps = _dependencies(tmp_path, identity=identity)
    original = deps.run_child

    def run_child(spec):
        execution = original(spec)
        return (
            replace(execution, evidence=evidence)
            if spec.kind == "composition"
            else execution
        )

    root = _produce(tmp_path, dependencies=replace(deps, run_child=run_child))
    receipt = json.loads((root / "vllm-bf16-composition.json").read_text())
    assert receipt["evidence"]["bounded_generation"]["accepted"] is True
    assert receipt["evidence"]["greedy_ids_equal"] is False
    assert receipt["evidence"]["full_vocab"]["allclose"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_policy",
        "wrong_policy",
        "two_grids_forged",
        "bool_id",
        "non_coordinate",
        "length",
        "eos",
        "invented_mapping",
    ],
)
def test_bounded_composition_producer_recomputes_acceptance(
    tmp_path: Path, mutation: str
) -> None:
    from src.inference.vllm_qualification_producer import _validate_mode_evidence

    evidence = _bounded_child_evidence(tmp_path)
    bounded = evidence["bounded_generation"]
    if mutation == "missing_policy":
        del bounded["policy"]
    elif mutation == "wrong_policy":
        bounded["coordinate_max_abs_delta"] = 2
    elif mutation == "two_grids_forged":
        # IDs 7 and 8 are neighboring IDs but coordinate VALUES 0 and 2.
        bounded["materialized_generated_ids"][1] = (
            bounded["dynamic_generated_ids"][1] + 1
        )
    elif mutation == "bool_id":
        bounded["materialized_generated_ids"][0] = True
    elif mutation == "non_coordinate":
        bounded["materialized_generated_ids"][0] += 1
    elif mutation == "length":
        bounded["materialized_generated_ids"].append(1)
    elif mutation == "eos":
        bounded["materialized_generated_ids"][-1] = 0
    else:
        bounded["coordinate_token_ids"] = [7, 8]
    with pytest.raises(RuntimeContractError):
        _validate_mode_evidence(
            kind="composition",
            max_num_seqs=1,
            value=evidence,
            expected_snapshot_fingerprint=bounded["snapshot_manifest"]["fingerprint"],
        )


@pytest.mark.parametrize("binding", [None, "f" * 64])
def test_bounded_composition_rejects_unbound_or_swapped_snapshot(
    tmp_path: Path, binding: str | None
) -> None:
    from src.inference.vllm_qualification_producer import _validate_mode_evidence

    evidence = _bounded_child_evidence(tmp_path)
    with pytest.raises(RuntimeContractError):
        _validate_mode_evidence(
            kind="composition",
            max_num_seqs=1,
            value=evidence,
            expected_snapshot_fingerprint=binding,
        )


def test_bounded_composition_rejects_resealed_tokenizer_swap(tmp_path: Path) -> None:
    from src.inference.execution_model_composition import (
        build_bounded_generation_evidence,
    )
    from src.inference.model_assets import build_model_snapshot_manifest
    from src.inference.vllm_qualification_producer import _validate_mode_evidence

    evidence = _bounded_child_evidence(tmp_path)
    original = evidence["bounded_generation"]
    admitted_fingerprint = original["snapshot_manifest"]["fingerprint"]
    tokenizer_path = tmp_path / "model" / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    # Swap complete canonical token assignments and reseal a valid snapshot.
    # An internally valid replacement map must not inherit the admitted model.
    vocab = tokenizer["model"]["vocab"]
    left, right = "<|coord_1|>", "<|coord_2|>"
    vocab[left], vocab[right] = vocab[right], vocab[left]
    for added in tokenizer["added_tokens"]:
        if added["content"] in {left, right}:
            added["id"] = vocab[added["content"]]
    tokenizer_path.write_text(json.dumps(tokenizer))
    swapped = build_model_snapshot_manifest(tokenizer_path.parent)
    evidence["bounded_generation"] = build_bounded_generation_evidence(
        dynamic_generated_ids=original["dynamic_generated_ids"],
        materialized_generated_ids=[
            original["materialized_generated_ids"][0],
            vocab[left],
            original["materialized_generated_ids"][-1],
        ],
        snapshot_manifest=swapped,
        expected_snapshot_fingerprint=swapped["fingerprint"],
    )
    assert evidence["bounded_generation"]["accepted"] is True
    with pytest.raises(RuntimeContractError):
        _validate_mode_evidence(
            kind="composition",
            max_num_seqs=1,
            value=evidence,
            expected_snapshot_fingerprint=admitted_fingerprint,
        )


def test_bounded_composition_requires_new_policy_evidence_and_contract_version() -> (
    None
):
    from src.inference.vllm_qualification_producer import (
        CONTRACT_VERSION,
        SCHEMA_VERSION,
        _validate_mode_evidence,
    )

    with pytest.raises(RuntimeContractError):
        _validate_mode_evidence(
            kind="composition",
            max_num_seqs=1,
            value=_child_evidence("composition", 1),
            expected_snapshot_fingerprint=SHA,
        )
    assert CONTRACT_VERSION != "coordexp-vllm-bf16-qualification-v3"
    assert SCHEMA_VERSION >= 4


def test_producer_requires_absent_output_root_and_bf16_vllm_identity(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_qualification_producer import (
        QualificationDependencies,
        produce,
    )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(RuntimeContractError) as exists:
        produce(
            config_path=tmp_path / "vllm.yaml",
            output_root=occupied,
            dependencies=_dependencies(tmp_path),
        )
    assert exists.value.code == "vllm_qualification.output_root_exists"

    wrong = _identity(tmp_path)
    wrong["model"]["target_dtype"] = "fp32"
    deps = QualificationDependencies(
        build_identity=lambda _: wrong,
        run_child=lambda _: pytest.fail("FP32 reached child execution"),
    )
    with pytest.raises(RuntimeContractError) as dtype:
        produce(
            config_path=tmp_path / "vllm.yaml",
            output_root=tmp_path / "fp32",
            dependencies=deps,
        )
    assert dtype.value.code == "vllm_qualification.bf16_required"


def test_producer_orchestrates_exact_isolated_modes_and_bounded_receipts(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_qualification_producer import (
        EXPECTED_RECEIPT_FILENAMES,
        produce,
    )

    base = _dependencies(tmp_path)
    specs: list[Any] = []

    def run_child(spec: Any) -> Any:
        specs.append(spec)
        return base.run_child(spec)

    root = tmp_path / "candidate"
    receipt = produce(
        config_path=tmp_path / "vllm.yaml",
        output_root=root,
        dependencies=replace(base, run_child=run_child),
    )

    assert [(spec.kind, spec.max_num_seqs) for spec in specs] == [
        ("composition", 1),
        ("runtime", 1),
        ("concurrency", 4),
        ("forced_replay", 1),
    ]
    assert len({spec.evidence_dir for spec in specs}) == 4
    assert {path.name for path in root.glob("*.json")} == set(
        EXPECTED_RECEIPT_FILENAMES.values()
    )
    assert receipt["status"] == "passed"
    for path in root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["identity"]["model"]["target_dtype"] == "bf16"
        assert len(path.read_bytes()) < 64 * 1024
        assert "generated_token_ids" not in payload["evidence"]
        assert "logits" not in payload["evidence"]


@pytest.mark.parametrize("failure", ["nonzero", "partial", "unbounded"])
def test_child_failure_or_partial_or_unbounded_evidence_cannot_pass(
    tmp_path: Path,
    failure: str,
) -> None:
    from src.inference.vllm_qualification_producer import (
        ChildExecution,
        QualificationDependencies,
        produce,
    )

    identity = _identity(tmp_path)

    def run_child(spec: Any) -> ChildExecution:
        spec.evidence_dir.mkdir(parents=True, exist_ok=False)
        (spec.evidence_dir / "evidence.json").write_text("{}", encoding="utf-8")
        evidence = (
            _bounded_child_evidence(tmp_path)
            if spec.kind == "composition"
            else _child_evidence(spec.kind, spec.max_num_seqs)
        )
        returncode = 0
        if spec.kind == "runtime":
            if failure == "nonzero":
                returncode = 7
            elif failure == "partial":
                evidence = {"status": "passed", "max_num_seqs": 1}
            else:
                evidence = {**evidence, "generated_token_ids": list(range(10_000))}
        return ChildExecution(
            returncode=returncode,
            evidence=evidence,
            process={
                "worker_pid": 12,
                "worker_returncode": returncode,
                "worker_pid_alive_after_exit": False,
                "owned_children_after": [],
                "gpu_memory_returned_to_baseline": True,
            },
        )

    root = tmp_path / "candidate"
    with pytest.raises(RuntimeContractError):
        produce(
            config_path=tmp_path / "vllm.yaml",
            output_root=root,
            dependencies=QualificationDependencies(
                build_identity=lambda _: identity,
                run_child=run_child,
            ),
        )
    runtime_receipt = root / "vllm-bf16-runtime-seq1.json"
    assert not runtime_receipt.exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "version", "source", "config", "model", "runtime", "status"],
)
def test_admit_rejects_incomplete_or_drifted_receipt_sets(
    tmp_path: Path,
    mutation: str,
) -> None:
    from src.inference.vllm_qualification_producer import admit

    identity = _identity(tmp_path)
    root = _produce(tmp_path, dependencies=_dependencies(tmp_path, identity=identity))
    deps = _dependencies(tmp_path, identity=identity)
    if mutation == "missing":
        next(root.glob("*.json")).unlink()
    elif mutation == "extra":
        (root / "unexpected.json").write_text("{}", encoding="utf-8")
    elif mutation in {"version", "status"}:
        path = next(root.glob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version" if mutation == "version" else "status"] = (
            999 if mutation == "version" else "failed"
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        changed = json.loads(json.dumps(identity))
        if mutation == "source":
            changed["source"]["fingerprint"] = "f" * 64
        elif mutation == "config":
            changed["config"]["fingerprint"] = "f" * 64
        elif mutation == "model":
            changed["model"]["snapshot_fingerprint"] = "f" * 64
        else:
            changed["runtime"]["packages"]["vllm"] = "9.9.9"
        deps = _dependencies(tmp_path, identity=changed)

    with pytest.raises(RuntimeContractError):
        admit(
            config_path=tmp_path / "vllm.yaml",
            receipts_root=root,
            target_root=tmp_path / "installed",
            dependencies=deps,
        )
    assert not (tmp_path / "installed").exists()


def test_admit_is_all_or_none_absent_or_identical_and_rejects_drift(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_qualification_producer import (
        EXPECTED_RECEIPT_FILENAMES,
        admit,
    )

    identity = _identity(tmp_path)
    deps = _dependencies(tmp_path, identity=identity)
    root = _produce(tmp_path, dependencies=deps)
    target = tmp_path / "installed"

    first = admit(
        config_path=tmp_path / "vllm.yaml",
        receipts_root=root,
        target_root=target,
        dependencies=deps,
    )
    assert first["status"] == "installed"
    assert {path.name for path in target.iterdir()} == set(
        EXPECTED_RECEIPT_FILENAMES.values()
    )
    second = admit(
        config_path=tmp_path / "vllm.yaml",
        receipts_root=root,
        target_root=target,
        dependencies=deps,
    )
    assert second["status"] == "identical"

    changed = target / EXPECTED_RECEIPT_FILENAMES["runtime"]
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeContractError) as drift:
        admit(
            config_path=tmp_path / "vllm.yaml",
            receipts_root=root,
            target_root=target,
            dependencies=deps,
        )
    assert drift.value.code == "vllm_qualification.admission_target_drift"


def test_default_validator_fails_before_admission_and_passes_after_valid_fixture(
    tmp_path: Path,
) -> None:
    from src.inference.backend import BackendLaunch
    from src.inference.vllm_qualification import (
        validate_vllm_forced_replay_qualification,
        validate_vllm_runtime_qualification,
    )
    from src.inference.vllm_qualification_producer import admit

    identity = _identity(tmp_path)
    target = tmp_path / "installed"
    model = identity["model"]
    launch = BackendLaunch(
        backend="vllm",
        model_path=str((tmp_path / "model").resolve()),
        model_dtype="bf16",
        batch_size=1,
        generation_config_fingerprint="generation",
        backend_options={"vllm": {}},
        execution_model_identity={
            **model,
            "source_identity": {
                "base": {"fingerprint": model["source_fingerprints"]["base"]},
                "adapter": {
                    "fingerprint": model["source_fingerprints"]["adapter"]
                },
                "embedding_delta": {
                    "fingerprint": model["source_fingerprints"]["embedding_delta"]
                },
            },
        },
    )
    engine_kwargs = _fixture_engine_kwargs(launch.model_path, 1)
    deps = _dependencies(tmp_path, identity=identity)
    root = _produce(tmp_path, dependencies=deps)

    with pytest.raises(RuntimeContractError) as missing:
        validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=engine_kwargs,
            receipt_root=target,
            identity_builder=lambda _: identity,
        )
    assert missing.value.code == "vllm_backend.qualification_receipt"

    admit(
        config_path=tmp_path / "vllm.yaml",
        receipts_root=root,
        target_root=target,
        dependencies=deps,
    )
    validated = validate_vllm_runtime_qualification(
        launch=launch,
        engine_kwargs=engine_kwargs,
        receipt_root=target,
        identity_builder=lambda _: identity,
    )
    assert validated["status"] == "passed"
    assert validated["max_num_seqs"] == 1
    replay = validate_vllm_forced_replay_qualification(
        launch=launch,
        processor_identity={"source_sha256": "d" * 64},
        receipt_root=target,
        identity_builder=lambda _: identity,
    )
    assert replay["status"] == "passed"

    drifted_engine = {**engine_kwargs, "gpu_memory_utilization": 0.8}
    with pytest.raises(RuntimeContractError) as config_drift:
        validate_vllm_runtime_qualification(
            launch=launch,
            engine_kwargs=drifted_engine,
            receipt_root=target,
            identity_builder=lambda _: identity,
        )
    assert config_drift.value.code == "vllm_backend.qualification_config_identity"


def test_source_identity_binds_the_child_executor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.inference import vllm_qualification_producer as producer

    assert "src/inference/vllm_qualification_runtime.py" in producer._SOURCE_PATHS
    assert "src/adapters/dora.py" in producer._SOURCE_PATHS
    runtime = tmp_path / "src/inference/vllm_qualification_runtime.py"
    adapter = tmp_path / "src/adapters/dora.py"
    runtime.parent.mkdir(parents=True)
    adapter.parent.mkdir(parents=True)
    runtime.write_text("executor-v1\n", encoding="utf-8")
    adapter.write_text("dora-v1\n", encoding="utf-8")
    monkeypatch.setattr(producer, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        producer,
        "_SOURCE_PATHS",
        (
            "src/adapters/dora.py",
            "src/inference/vllm_qualification_runtime.py",
        ),
    )

    first = producer._build_source_identity()
    adapter.write_text("dora-v2\n", encoding="utf-8")
    adapter_drift = producer._build_source_identity()
    runtime.write_text("executor-v2\n", encoding="utf-8")
    runtime_drift = producer._build_source_identity()

    assert first["fingerprint"] != adapter_drift["fingerprint"]
    assert adapter_drift["fingerprint"] != runtime_drift["fingerprint"]
    source_files = first["files"]
    assert isinstance(source_files, list)
    assert all(isinstance(item, Mapping) for item in source_files)
    assert [item["path"] for item in source_files if isinstance(item, Mapping)] == [
        "src/adapters/dora.py",
        "src/inference/vllm_qualification_runtime.py",
    ]


def test_source_identity_covers_static_local_imports_and_package_initializers() -> None:
    from src.inference import vllm_qualification_producer as producer

    covered = set(producer._SOURCE_PATHS)

    def resolve_module(module: str) -> str | None:
        module_path = Path(*module.split("."))
        candidates = (module_path.with_suffix(".py"), module_path / "__init__.py")
        for candidate in candidates:
            if (producer._REPO_ROOT / candidate).is_file():
                return candidate.as_posix()
        return None

    missing: set[str] = set()
    for relative in covered:
        parent = Path(relative).parent
        while parent.parts and parent.parts[0] == "src":
            initializer = parent / "__init__.py"
            if (producer._REPO_ROOT / initializer).is_file():
                if initializer.as_posix() not in covered:
                    missing.add(initializer.as_posix())
            parent = parent.parent

        tree = ast.parse((producer._REPO_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    path = Path(relative)
                    package = list(path.with_suffix("").parts[:-1])
                    if path.name == "__init__.py":
                        package = list(path.parent.parts)
                    package = package[: len(package) - node.level + 1]
                    base = ".".join(
                        [*package, *(node.module or "").split(".")]
                    ).rstrip(".")
                else:
                    base = node.module or ""
                if base:
                    modules.append(base)
                    modules.extend(f"{base}.{alias.name}" for alias in node.names)
            for module in modules:
                if module != "src" and not module.startswith("src."):
                    continue
                resolved = resolve_module(module)
                if resolved is not None and resolved not in covered:
                    missing.add(resolved)

    assert not missing
    assert "src/templates/__init__.py" in covered
    assert "src/templates/renderer.py" in covered


def test_renderer_source_drift_rejects_stale_receipts_through_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    source_root = tmp_path / "checkout"
    original_root = producer._REPO_ROOT
    for relative in producer._SOURCE_PATHS:
        source = original_root / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    renderer = source_root / "src/templates/renderer.py"
    if not renderer.exists():
        renderer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / "src/templates/renderer.py", renderer)
    monkeypatch.setattr(producer, "_REPO_ROOT", source_root)

    identity = _identity(tmp_path)
    identity["source"] = producer._build_source_identity()
    receipts = _produce(
        tmp_path,
        dependencies=_dependencies(tmp_path, identity=identity),
    )

    renderer.write_bytes(renderer.read_bytes() + b"\n# qualification drift probe\n")
    drifted_identity = json.loads(json.dumps(identity))
    drifted_identity["source"] = producer._build_source_identity()

    with pytest.raises(RuntimeContractError) as exc_info:
        producer.admit(
            config_path=tmp_path / "vllm.yaml",
            receipts_root=receipts,
            target_root=tmp_path / "admitted",
            dependencies=producer.QualificationDependencies(
                build_identity=lambda _: drifted_identity,
                run_child=lambda _: pytest.fail("admission launched a child"),
            ),
        )

    assert exc_info.value.code == "vllm_backend.qualification_identity_drift"
    assert not (tmp_path / "admitted").exists()


def test_process_group_cleanup_detects_and_terminates_an_extant_member() -> None:
    from src.inference.vllm_qualification_producer import (
        _process_group_members,
        _terminate_process_group,
    )

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 2
        while child.pid not in _process_group_members(child.pid):
            if time.monotonic() >= deadline:
                pytest.fail("child never appeared in its owned process group")
            time.sleep(0.01)
        _terminate_process_group(child.pid)
        child.wait(timeout=2)
        assert _process_group_members(child.pid) == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


def test_surviving_process_group_member_cannot_publish_a_passed_receipt(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_qualification_producer import (
        ChildExecution,
        QualificationDependencies,
        produce,
    )

    base = _dependencies(tmp_path)

    def run_child(spec: Any) -> ChildExecution:
        execution = base.run_child(spec)
        return ChildExecution(
            returncode=execution.returncode,
            evidence=execution.evidence,
            process={
                **execution.process,
                "process_group_members_after": [4321],
                "gpu_process_group_memory_after_mib": 128,
            },
        )

    with pytest.raises(RuntimeContractError) as exc_info:
        produce(
            config_path=tmp_path / "vllm.yaml",
            output_root=tmp_path / "candidate",
            dependencies=QualificationDependencies(
                build_identity=base.build_identity,
                run_child=run_child,
            ),
        )

    assert exc_info.value.code == "vllm_backend.qualification_receipt_incompatible"


@pytest.mark.parametrize("raw", [None, "", "0,1", "-1", "MIG-deadbeef"])
def test_production_gpu_selector_requires_one_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    if raw is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)

    with pytest.raises(RuntimeContractError) as exc_info:
        producer._visible_physical_gpu_selector()

    assert exc_info.value.code == "vllm_qualification.cuda_visible_devices"


@pytest.mark.parametrize("raw", ["2", "GPU-01234567-89ab-cdef-0123-456789abcdef"])
def test_production_gpu_selector_accepts_one_explicit_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert producer._visible_physical_gpu_selector() == raw


def test_gpu_memory_query_is_scoped_to_the_selected_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    commands: list[list[str]] = []
    timeouts: list[float] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        timeouts.append(kwargs["timeout"])
        return SimpleNamespace(stdout="321\n")

    monkeypatch.setattr(producer.subprocess, "run", fake_run)

    assert producer._gpu_memory_used_mib("2") == 321
    assert commands == [
        [
            "nvidia-smi",
            "--id=2",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert timeouts == [producer.GPU_CENSUS_TIMEOUT_SECONDS]


def test_process_gpu_memory_query_is_scoped_to_the_selected_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(stdout="42, 128\n7, 512\n")

    monkeypatch.setattr(producer.subprocess, "run", fake_run)

    assert producer._gpu_memory_for_pids_mib([42], selector="2") == 128
    assert commands[0][1] == "--id=2"
    assert commands[0][2] == "--query-compute-apps=pid,used_memory"


def test_gpu_census_timeout_preserves_unknown_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    def time_out(command: list[str], **kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(producer.subprocess, "run", time_out)

    assert producer._gpu_memory_used_mib("2") is None
    assert producer._gpu_memory_for_pids_mib([123], selector="2") == -1


@pytest.mark.parametrize(
    "invalid_timeout",
    [0, -1, math.nan, math.inf, -math.inf, True, "1"],
)
def test_produce_rejects_invalid_child_timeout_before_work(
    tmp_path: Path,
    invalid_timeout: object,
) -> None:
    from src.inference.vllm_qualification_producer import (
        QualificationDependencies,
        produce,
    )

    def unexpected(*_: Any, **__: Any) -> Any:
        pytest.fail("invalid timeout reached qualification work")

    output_root = tmp_path / "candidate"
    with pytest.raises(RuntimeContractError) as exc_info:
        produce(
            config_path=tmp_path / "vllm.yaml",
            output_root=output_root,
            child_timeout_seconds=invalid_timeout,
            dependencies=QualificationDependencies(
                build_identity=unexpected,
                run_child=unexpected,
            ),
        )

    assert exc_info.value.code == "vllm_qualification.child_timeout"
    assert not output_root.exists()


def test_production_supervisor_times_out_and_cleans_pipe_holding_descendant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    pid_path = tmp_path / "child-pids.txt"
    child_script = (
        "import os, pathlib, subprocess, sys\n"
        "descendant = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "pathlib.Path(sys.argv[1]).write_text("
        "f'{os.getpid()} {descendant.pid}\\n', encoding='utf-8')\n"
    )
    monkeypatch.setattr(
        producer,
        "_qualification_child_command",
        lambda _: [sys.executable, "-c", child_script, str(pid_path)],
    )
    monkeypatch.setattr(producer, "_visible_physical_gpu_selector", lambda: "0")
    monkeypatch.setattr(producer, "_gpu_memory_used_mib", lambda _: 0)
    monkeypatch.setattr(producer, "_gpu_memory_for_pids_mib", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(producer, "_settle_gpu_memory_used_mib", lambda **_: 0)

    output_root = tmp_path / "candidate"
    child_pid: int | None = None
    descendant_pid: int | None = None
    try:
        started = time.monotonic()
        with pytest.raises(RuntimeContractError) as exc_info:
            producer.produce(
                config_path=tmp_path / "vllm.yaml",
                output_root=output_root,
                child_timeout_seconds=1.0,
                dependencies=producer.QualificationDependencies(
                    build_identity=lambda _: _identity(tmp_path),
                    run_child=producer._run_subprocess_child,
                ),
            )
        elapsed = time.monotonic() - started
        child_pid, descendant_pid = map(
            int,
            pid_path.read_text(encoding="utf-8").split(),
        )

        assert exc_info.value.code == "vllm_qualification.child_timeout"
        assert exc_info.value.context["worker_reaped"] is True
        assert exc_info.value.context["process_group_members_after"] == []
        assert exc_info.value.context["gpu_cleanup_evidence_available"] is True
        assert elapsed < 10
        assert producer._process_group_members(child_pid) == []
        assert not Path(f"/proc/{child_pid}").exists()
        assert not any(
            (output_root / name).exists()
            for name in producer.EXPECTED_RECEIPT_FILENAMES.values()
        )
        assert descendant_pid not in producer._process_group_members(child_pid)
    finally:
        if child_pid is None and pid_path.is_file():
            child_pid, descendant_pid = map(
                int,
                pid_path.read_text(encoding="utf-8").split(),
            )
        if child_pid is not None:
            producer._terminate_process_group(child_pid)
        if descendant_pid is not None and Path(f"/proc/{descendant_pid}").exists():
            try:
                os.kill(descendant_pid, 9)
            except ProcessLookupError:
                pass


def test_gpu_memory_settle_is_bounded_and_requires_a_measured_after_value() -> None:
    from src.inference.vllm_qualification_producer import _settle_gpu_memory_used_mib

    readings = iter([220, None, 164])
    sleeps: list[float] = []
    measured = _settle_gpu_memory_used_mib(
        selector="2",
        before_mib=100,
        tolerance_mib=64,
        attempts=3,
        interval_seconds=0.01,
        read_memory=lambda _: next(readings),
        sleep=sleeps.append,
    )
    assert measured == 164
    assert sleeps == [0.01, 0.01]

    unavailable = _settle_gpu_memory_used_mib(
        selector="2",
        before_mib=100,
        tolerance_mib=64,
        attempts=2,
        interval_seconds=0.01,
        read_memory=lambda _: None,
        sleep=lambda _: None,
    )
    assert unavailable is None


def test_process_receipt_requires_measured_gpu_memory_and_recomputes_claim() -> None:
    from src.inference.vllm_qualification_producer import _validate_process_mapping

    common = {
        "worker_pid": 1234,
        "worker_returncode": 0,
        "worker_pid_alive_after_exit": False,
        "owned_children_after": [],
        "process_group_id": 1234,
        "process_group_members_after": [],
        "process_group_termination_required": False,
        "gpu_process_group_memory_after_mib": 0,
        "gpu_memory_returned_to_baseline": True,
    }
    with pytest.raises(RuntimeContractError) as missing:
        _validate_process_mapping(common, kind="runtime")
    assert missing.value.context["field"] == "process.gpu_memory_measurement"

    inconsistent = {
        **common,
        "visible_gpu_selector": "2",
        "gpu_memory_before_mib": 100,
        "gpu_memory_after_mib": 200,
        "gpu_memory_tolerance_mib": 64,
    }
    with pytest.raises(RuntimeContractError) as drift:
        _validate_process_mapping(inconsistent, kind="runtime")
    assert drift.value.context["field"] == "process.gpu_memory_consistency"

    unavailable_after = {**inconsistent, "gpu_memory_after_mib": None}
    with pytest.raises(RuntimeContractError) as unavailable:
        _validate_process_mapping(unavailable_after, kind="runtime")
    assert unavailable.value.context["field"] == "process.gpu_memory_measurement"


def test_missing_before_measurement_fails_before_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import vllm_qualification_producer as producer

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(producer, "_gpu_memory_used_mib", lambda _: None)
    monkeypatch.setattr(
        producer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("child launched without GPU measurement"),
    )
    spec = producer.ChildSpec(
        kind="runtime",
        max_num_seqs=1,
        config_path=tmp_path / "vllm.yaml",
        output_root=tmp_path / "candidate",
        evidence_dir=tmp_path / "candidate/evidence/runtime",
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        producer._run_subprocess_child(spec)

    assert exc_info.value.code == "vllm_qualification.gpu_memory_measurement"


def test_runtime_child_routes_exact_mode_and_preserves_production_evidence(
    tmp_path: Path,
) -> None:
    from src.inference.vllm_qualification_runtime import (
        RuntimeDependencies,
        run_child,
    )

    calls: list[tuple[object, ...]] = []

    def composition(config: Path, evidence_dir: Path) -> dict[str, Any]:
        calls.append(("composition", config, evidence_dir))
        (evidence_dir / "composition.json").write_text("{}", encoding="utf-8")
        return _bounded_child_evidence(tmp_path)

    def inference(
        kind: str,
        config: Path,
        evidence_dir: Path,
        max_num_seqs: int,
        include_raw_model_logprob: bool,
    ) -> dict[str, Any]:
        calls.append(
            (
                kind,
                config,
                evidence_dir,
                max_num_seqs,
                include_raw_model_logprob,
            )
        )
        (evidence_dir / "run.json").write_text("{}", encoding="utf-8")
        return _child_evidence(kind, max_num_seqs)

    deps = RuntimeDependencies(
        run_composition=composition,
        run_inference=inference,
        owned_child_pids=lambda: [],
    )
    config = tmp_path / "vllm.yaml"
    config.write_text("config\n", encoding="utf-8")
    for kind, max_num_seqs in (
        ("composition", 1),
        ("runtime", 1),
        ("concurrency", 4),
        ("forced_replay", 1),
    ):
        result = run_child(
            kind=kind,
            max_num_seqs=max_num_seqs,
            config_path=config,
            evidence_dir=tmp_path / kind,
            dependencies=deps,
        )
        evidence = result["evidence"]
        process = result["process"]
        assert isinstance(evidence, Mapping) and evidence["status"] == "passed"
        assert isinstance(process, Mapping) and process["owned_children_after"] == []

    assert [call[0] for call in calls] == [
        "composition",
        "runtime",
        "concurrency",
        "forced_replay",
    ]
    assert calls[1][-2:] == (1, False)
    assert calls[2][-2:] == (4, False)
    assert calls[3][-2:] == (1, True)


@pytest.mark.parametrize(
    ("kind", "max_num_seqs"),
    [("composition", 4), ("runtime", 4), ("concurrency", 1), ("forced_replay", 4)],
)
def test_runtime_child_rejects_any_noncanonical_mode_before_work(
    tmp_path: Path,
    kind: str,
    max_num_seqs: int,
) -> None:
    from src.inference.vllm_qualification_runtime import (
        RuntimeDependencies,
        run_child,
    )

    deps = RuntimeDependencies(
        run_composition=lambda *_: pytest.fail("invalid mode reached composition"),
        run_inference=lambda *_: pytest.fail("invalid mode reached inference"),
        owned_child_pids=lambda: [],
    )
    with pytest.raises(RuntimeContractError) as exc_info:
        run_child(
            kind=kind,
            max_num_seqs=max_num_seqs,
            config_path=tmp_path / "vllm.yaml",
            evidence_dir=tmp_path / "evidence",
            dependencies=deps,
        )

    assert exc_info.value.code == "vllm_qualification.child_mode"


def test_qualification_module_help_exposes_run_and_admit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.qualify_vllm import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "run" in output
    assert "admit" in output
    assert "BF16" in output or "bf16" in output
    assert "_child" not in output


@pytest.mark.parametrize(
    ("extra_args", "expected_timeout"),
    [([], 1800.0), (["--child-timeout-seconds", "12.5"], 12.5)],
)
def test_qualification_run_cli_forwards_child_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    extra_args: list[str],
    expected_timeout: float,
) -> None:
    from src import qualify_vllm

    captured: dict[str, object] = {}

    def fake_produce(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(qualify_vllm, "produce", fake_produce)
    assert (
        qualify_vllm.main(
            [
                "run",
                "--config",
                str(tmp_path / "vllm.yaml"),
                "--output-root",
                str(tmp_path / "candidate"),
                *extra_args,
            ]
        )
        == 0
    )

    assert captured["child_timeout_seconds"] == expected_timeout
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "-inf"])
def test_qualification_run_cli_rejects_invalid_child_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: str,
) -> None:
    from src import qualify_vllm

    monkeypatch.setattr(
        qualify_vllm,
        "produce",
        lambda **_: pytest.fail("invalid CLI timeout reached producer"),
    )
    with pytest.raises(SystemExit) as exc_info:
        qualify_vllm.main(
            [
                "run",
                "--config",
                str(tmp_path / "vllm.yaml"),
                "--output-root",
                str(tmp_path / "candidate"),
                "--child-timeout-seconds",
                raw,
            ]
        )

    assert exc_info.value.code == 2
