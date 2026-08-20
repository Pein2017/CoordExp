from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
import yaml

from src.config.loader import load_train_config
from src.training import pack_cache
from src.training.input_attestation import (
    build_training_input_attestations,
    validate_training_input_attestations,
)
from src.training.supervised_trainer import SupervisedMicroStep


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    REPO_ROOT
    / "configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml"
)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _determinants(split: str) -> dict[str, object]:
    semantic: dict[str, object] = {
        "version": pack_cache.PACKING_CACHE_VERSION,
        "split": split,
        "dataset": {"split": split},
        "template": {"version": 1},
        "packing": {"policy": "source_order_next_fit"},
        "processor": {"version": 1},
        "ordering": {"order": "source_order"},
        "augmentation": {"split": split},
        "qwen": {
            "processor_identity": {"version": 1},
            "token_identity": {"version": 1},
            "encoding_identity": {"version": 1},
            "model_config_assets": {"version": 1},
            "processor_assets": {"version": 1},
            "tokenizer_assets": {"version": 1},
        },
        "realized_vocab_groups": {"version": 1},
        "micro_step_runtime_config": {
            "fa2_model_dtype": "bf16",
            "capture_fa2_branch": True,
            "require_fa2_branch_proof": True,
        },
        "micro_step_schema": pack_cache._supervised_micro_step_schema_identity(),
    }
    entries = pack_cache._build_determinant_entries(semantic)
    return {
        **semantic,
        "registry_schema_version": pack_cache.PACKING_CACHE_DETERMINANT_REGISTRY_VERSION,
        "determinants": entries,
        "aggregate_fingerprint": pack_cache._registry_entries_fingerprint(entries),
        "code_identity": pack_cache._registry_code_identity(entries),
    }


def _step(split: str) -> SupervisedMicroStep:
    return SupervisedMicroStep(
        pack=f"{split}-pack",
        encoded_examples=(f"{split}-example",),
        position_inputs=f"{split}-positions",
        token_sequence=f"{split}-tokens",
        vocab_groups=f"{split}-vocab",
        metadata={"pack_id": 0},
    )


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.training import input_attestation as owner

    model_root = tmp_path / "model"
    model_root.mkdir()
    save_file({"weight": torch.ones(1)}, str(model_root / "model.safetensors"))
    base = deepcopy(load_train_config(BASE_CONFIG).config_dict)
    base["model"]["base_model"] = str(model_root)
    base["training"]["max_steps"] = 5
    base["training"]["forward_input_provider_mode"] = "synchronous"
    base["runtime"] = {"seed": 17, "determinism": {"mode": "strict_cuda_replay_v1"}}
    base["eval"]["forward"] = {"every_fraction": None, "steps": [3]}
    base["checkpoint"] = {"every_fraction": None, "steps": [3, 5], "save_final": True}
    run_roots = {
        role: tmp_path / "runs" / role
        for role in ("uninterrupted", "interrupted_parent", "resume_child")
    }
    configs: dict[str, str] = {}
    for role, run_root in run_roots.items():
        payload = deepcopy(base)
        payload["run"] = {
            "name": role,
            "artifact_root": str(run_root.parent),
            "output_dir": run_root.name,
            "collision_policy": "fail",
        }
        payload["resume"] = {
            "mode": "exact_same_world_size",
            "checkpoint_dir": (
                str(run_roots["interrupted_parent"] / "checkpoints" / "step-3")
                if role == "resume_child"
                else None
            ),
        }
        path = tmp_path / f"{role}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        configs[role] = str(path)

    components = SimpleNamespace(
        base_model_path=model_root,
        model=None,
        load_model=False,
        tokenizer=SimpleNamespace(),
        token_identity=object(),
    )
    monkeypatch.setattr(
        owner, "load_qwen_components", lambda config, *, load_model: components
    )
    monkeypatch.setattr(
        owner, "build_token_vocabulary_groups", lambda *_args, **_kwargs: object()
    )
    determinants = {split: _determinants(split) for split in ("train", "eval.forward")}
    monkeypatch.setattr(
        owner,
        "build_packing_cache_determinants",
        lambda *_args, split, **_kwargs: deepcopy(determinants[split]),
    )

    cache_root = tmp_path / "cache"
    cache_rows: dict[str, dict[str, object]] = {}
    for split in ("train", "eval.forward"):
        determinant = determinants[split]
        fingerprint = str(determinant["aggregate_fingerprint"])
        cache_dir = pack_cache.cache_dir_for_fingerprint(cache_root, fingerprint)
        manifest = pack_cache.write_micro_step_cache(
            cache_dir,
            (_step(split),),
            cache_root=cache_root,
            fingerprint=fingerprint,
            determinants=determinant,
            materialization={"strategy": "fork_process_pool", "workers": 16},
            determinant_revalidator=lambda value=determinant: value,
            augmentation={
                "split": split,
                "mode": "disabled",
                "policy": "geometry_flips",
                "enabled": False,
                "seed": 17,
                "input_example_count": 1,
                "output_example_count": 1,
                "presentation_count": 1,
                "object_ordering": "geo_sorted",
            },
        )
        cache_rows[split] = {
            "cache_dir": str(cache_dir),
            "fingerprint": fingerprint,
            "manifest_path": str(cache_dir / "manifest.json"),
            "manifest_sha256": hashlib.sha256(
                (cache_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "micro_step_count": int(manifest["micro_step_count"]),
            "status": "complete",
            "build_status": "built",
            "format_version": pack_cache.PACKING_CACHE_VERSION,
            "phase_receipt": {
                "cache_preparation": {"status": "completed", "duration_seconds": 1.0},
                "cache_publication": {"status": "completed", "duration_seconds": 0.1},
                "cache_admission": {
                    "status": "completed",
                    "duration_seconds": 0.2,
                    "verification_level": "payloads",
                },
            },
        }

    resolved = load_train_config(configs["uninterrupted"])
    result = {
        "entry_config_path": str(Path(configs["uninterrupted"]).resolve()),
        "resolved_config_fingerprint": resolved.fingerprint,
        "model_loaded": False,
        "provenance": {"fixture": True},
        "policy_identities": {
            "upstream_runtime_baseline": {"admitted": True},
            "runtime_determinism": {"mode": "strict_cuda_replay_v1"},
            "packing": {"schema_version": 2},
            "cache": {
                "schema_version": 1,
                "root": {
                    "resolved_root": str(cache_root.resolve()),
                    "source": "COORDEXP_SWIFT_PACK_CACHE_ROOT",
                },
                "train_fingerprint": cache_rows["train"]["fingerprint"],
                "eval_fingerprint": cache_rows["eval.forward"]["fingerprint"],
            },
        },
        "measurement": {"duration_seconds": 1.3},
        "train": cache_rows["train"],
        "eval": cache_rows["eval.forward"],
    }
    body = {
        "schema": "coordexp-swift-pack-cache-preparation-receipt-v1",
        "terminal_status": "completed",
        "config_path": str(Path(configs["uninterrupted"]).resolve()),
        "result": result,
        "failure": None,
    }
    receipt = {**body, "receipt_sha256": _sha256_json(body)}
    receipt_path = tmp_path / "preparation.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(
        owner=owner,
        configs=configs,
        model_root=model_root,
        cache_root=cache_root,
        cache_rows=cache_rows,
        receipt_path=receipt_path,
    )


def _build(inputs, *, bound: int = 10_000_000):
    return build_training_input_attestations(
        config_paths=inputs.configs,
        cache_root=inputs.cache_root,
        cache_preparation_receipt_path=inputs.receipt_path,
        expected_model_root=inputs.model_root,
        max_cache_payload_bytes=bound,
    )


def test_builds_native_split_and_full_model_attestations_with_one_payload_pass(
    inputs, monkeypatch
):
    calls: list[tuple[str, str]] = []
    real = inputs.owner.load_cache_manifest
    monkeypatch.setattr(
        inputs.owner,
        "load_cache_manifest",
        lambda cache_dir, **kwargs: calls.append(
            (Path(cache_dir).name, kwargs["level"])
        )
        or real(cache_dir, **kwargs),
    )
    cache, model = _build(inputs)
    assert cache["schema"] == "coordexp-swift-wave7-r5-cache-input-attestation-v1"
    assert set(cache["splits"]) == {"train", "eval.forward"}
    assert cache["splits"]["train"]["materialization"]["workers"] == 16
    assert (
        cache["splits"]["train"]["determinants_sha256"]
        != cache["splits"]["eval.forward"]["determinants_sha256"]
    )
    assert cache["measured_payload_bytes"] > 0
    assert [level for _, level in calls] == ["payloads", "payloads"]
    assert model["schema"] == "coordexp-swift-wave7-r5-model-input-attestation-v2"
    assert model["base_model_weight_identity"]["root"] == str(
        inputs.model_root.resolve()
    )
    assert model["weight_hash_execution_policy"] == {
        "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": 1,
        "payload_file_count": 1,
    }


def test_rejects_semantic_config_drift_and_bad_preparation_binding(inputs):
    payload = yaml.safe_load(Path(inputs.configs["interrupted_parent"]).read_text())
    payload["optimizer"]["groups"]["adapters"]["language"]["lr"] *= 2
    Path(inputs.configs["interrupted_parent"]).write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )
    with pytest.raises(ValueError, match="semantic"):
        _build(inputs)


def test_accepts_presentation_only_observability_drift_across_the_three_runs(inputs):
    """Task 5.1: `observability.steps` is presentation, not run semantics.

    The three Wave-7 roles may make different rank-zero presentation
    decisions (a production parent at `steps: 10`, a resumed child at
    `steps: 1`) without breaking the semantic-projection equality that binds
    the three runs to one training question.
    """

    # The `uninterrupted` file is left byte-identical on purpose: the frozen
    # cache-preparation receipt pins its resolved-config fingerprint, and this
    # node is about cross-ROLE presentation drift, not receipt rebinding.
    authored = yaml.safe_load(Path(inputs.configs["uninterrupted"]).read_text())
    # Wave 1 made the block required with no default, so it is authored.
    assert isinstance(authored["observability"]["steps"], int)
    for role, steps in (("interrupted_parent", 4), ("resume_child", 7)):
        path = Path(inputs.configs[role])
        payload = yaml.safe_load(path.read_text())
        assert payload["observability"]["steps"] == authored["observability"]["steps"]
        assert steps != authored["observability"]["steps"]
        payload["observability"]["steps"] = steps
        path.write_text(yaml.safe_dump(payload, sort_keys=False))

    cache, model = _build(inputs)
    result = validate_training_input_attestations(
        cache_attestation=cache,
        model_attestation=model,
        config_paths=inputs.configs,
        validate_cache_payloads=True,
        rehash_model_weights=False,
        max_cache_payload_bytes=10_000_000,
    )
    assert result["status"] == "passed"


def test_rejects_preparation_receipt_root_and_split_drift(inputs):
    receipt = json.loads(inputs.receipt_path.read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body["result"]["policy_identities"]["cache"]["root"]["resolved_root"] = "/wrong"
    receipt = {**body, "receipt_sha256": _sha256_json(body)}
    inputs.receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="wrong cache root"):
        _build(inputs)


def test_rejects_wrong_split_root_chunk_mutation_and_aggregate_bound(inputs):
    with pytest.raises(ValueError, match="bound"):
        _build(inputs, bound=1)
    cache, model = _build(inputs)
    train_dir = Path(cache["splits"]["train"]["cache_dir"])
    manifest = json.loads((train_dir / "manifest.json").read_text())
    chunk = train_dir / manifest["chunks"][0]["path"]
    chunk.write_bytes(chunk.read_bytes() + b"mutation")
    with pytest.raises(Exception):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=True,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


def test_rehash_rejects_model_shard_mutation(inputs):
    cache, model = _build(inputs)
    (inputs.model_root / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(Exception):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=True,
            max_cache_payload_bytes=10_000_000,
        )


def test_rehash_rejects_model_index_mutation(inputs):
    (inputs.model_root / "model.safetensors").unlink()
    save_file(
        {"first": torch.ones(1)},
        str(inputs.model_root / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"second": torch.ones(1)},
        str(inputs.model_root / "model-00002-of-00002.safetensors"),
    )
    index_path = inputs.model_root / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        )
    )
    cache, model = _build(inputs)
    index_path.write_text(index_path.read_text() + "\n")
    with pytest.raises(Exception):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=True,
            max_cache_payload_bytes=10_000_000,
        )


def test_rejects_legacy_aggregate_cache_wrapper(inputs):
    cache, model = _build(inputs)
    cache["fingerprint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="fields are not exact"):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


def test_validate_rejects_wrong_split_and_uses_one_payload_pass_per_split(
    inputs, monkeypatch
):
    cache, model = _build(inputs)
    calls: list[str] = []
    real = inputs.owner.load_cache_manifest
    monkeypatch.setattr(
        inputs.owner,
        "load_cache_manifest",
        lambda cache_dir, **kwargs: calls.append(kwargs["level"])
        or real(cache_dir, **kwargs),
    )
    result = validate_training_input_attestations(
        cache_attestation=cache,
        model_attestation=model,
        config_paths=inputs.configs,
        validate_cache_payloads=True,
        rehash_model_weights=False,
        max_cache_payload_bytes=10_000_000,
    )
    assert result["status"] == "passed"
    assert calls == ["payloads", "payloads"]

    wrong = deepcopy(cache)
    wrong["splits"]["train"]["split"] = "eval.forward"
    body = {key: value for key, value in wrong.items() if key != "attestation_sha256"}
    wrong["attestation_sha256"] = _sha256_json(body)
    with pytest.raises(ValueError, match="wrong split"):
        validate_training_input_attestations(
            cache_attestation=wrong,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


_INVALID_R5_MATERIALIZATIONS = (
    {"strategy": "serial", "workers": 16},
    {"strategy": "fork_process_pool", "workers": 1},
    {"strategy": "thread_pool", "workers": 16},
    {"strategy": "fork_process_pool"},
    {"strategy": "fork_process_pool", "workers": True},
)


@pytest.mark.parametrize("materialization", _INVALID_R5_MATERIALIZATIONS)
def test_build_rejects_unauthenticated_r5_materialization_policy(
    inputs, materialization
):
    train_manifest_path = Path(inputs.cache_rows["train"]["manifest_path"])
    manifest = json.loads(train_manifest_path.read_text())
    manifest["materialization"] = materialization
    train_manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    receipt = json.loads(inputs.receipt_path.read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body["result"]["train"]["manifest_sha256"] = hashlib.sha256(
        train_manifest_path.read_bytes()
    ).hexdigest()
    inputs.receipt_path.write_text(
        json.dumps({**body, "receipt_sha256": _sha256_json(body)}, sort_keys=True)
        + "\n"
    )

    with pytest.raises(ValueError, match="authenticated materialization policy"):
        _build(inputs)


@pytest.mark.parametrize("materialization", _INVALID_R5_MATERIALIZATIONS)
def test_validate_rejects_unauthenticated_r5_materialization_policy(
    inputs, materialization
):
    cache, model = _build(inputs)
    cache["splits"]["train"]["materialization"] = materialization
    body = {key: value for key, value in cache.items() if key != "attestation_sha256"}
    cache["attestation_sha256"] = _sha256_json(body)

    with pytest.raises(ValueError, match="authenticated materialization policy"):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


def test_validation_returns_authenticated_split_materialization_projection(inputs):
    cache, model = _build(inputs)
    result = validate_training_input_attestations(
        cache_attestation=cache,
        model_attestation=model,
        config_paths=inputs.configs,
        validate_cache_payloads=False,
        rehash_model_weights=False,
        max_cache_payload_bytes=10_000_000,
    )
    assert result["materialization_policy"] == {
        "train": {"strategy": "fork_process_pool", "workers": 16},
        "eval.forward": {"strategy": "fork_process_pool", "workers": 16},
    }


def _resign_model_attestation(model: dict[str, object]) -> None:
    body = {key: value for key, value in model.items() if key != "attestation_sha256"}
    model["attestation_sha256"] = _sha256_json(body)


@pytest.mark.parametrize(
    "policy",
    (
        {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "serial",
            "resolved_workers": 1,
            "payload_file_count": 1,
        },
        {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": 2,
            "payload_file_count": 1,
        },
        {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": True,
            "payload_file_count": 1,
        },
        {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": 1,
            "payload_file_count": 2,
        },
        {
            "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
            "strategy": "thread_pool_file_sha256",
            "resolved_workers": 1,
            "payload_file_count": True,
        },
    ),
)
def test_model_v2_rejects_resigned_weight_hash_policy_mutations(inputs, policy):
    cache, model = _build(inputs)
    model["weight_hash_execution_policy"] = policy
    _resign_model_attestation(model)
    with pytest.raises(ValueError, match="weight hash execution policy"):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


def test_model_v2_multishard_requires_parallel_hashing(inputs):
    (inputs.model_root / "model.safetensors").unlink()
    for index in (1, 2):
        save_file(
            {f"weight_{index}": torch.ones(1)},
            str(inputs.model_root / f"model-{index:05d}-of-00002.safetensors"),
        )
    (inputs.model_root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight_1": "model-00001-of-00002.safetensors",
                    "weight_2": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    cache, model = _build(inputs)
    assert model["weight_hash_execution_policy"]["resolved_workers"] == 2
    assert model["weight_hash_execution_policy"]["payload_file_count"] == 2

    model["weight_hash_execution_policy"]["resolved_workers"] = 1
    _resign_model_attestation(model)
    with pytest.raises(ValueError, match="multi-file.*parallel"):
        validate_training_input_attestations(
            cache_attestation=cache,
            model_attestation=model,
            config_paths=inputs.configs,
            validate_cache_payloads=False,
            rehash_model_weights=False,
            max_cache_payload_bytes=10_000_000,
        )


def test_model_v2_standalone_policy_passes_and_is_returned(inputs):
    cache, model = _build(inputs)
    result = validate_training_input_attestations(
        cache_attestation=cache,
        model_attestation=model,
        config_paths=inputs.configs,
        validate_cache_payloads=False,
        rehash_model_weights=True,
        max_cache_payload_bytes=10_000_000,
    )
    assert result["weight_hash_execution_policy"] == {
        "schema": "coordexp-swift-base-model-weight-hash-execution-policy-v1",
        "strategy": "thread_pool_file_sha256",
        "resolved_workers": 1,
        "payload_file_count": 1,
    }
