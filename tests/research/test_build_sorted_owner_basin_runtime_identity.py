"""Contracts for the CPU-only sorted owner-basin runtime identity builder."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest

from scripts.research.build_sorted_owner_basin_runtime_identity import (
    COORDINATE_TOKEN_ID_START,
    MODEL_VOCAB_SIZE,
    RuntimeIdentityError,
    SCHEMA_TOKENS,
    SCHEMA_VERSION,
    WRAPPER_TOKEN_IDS,
    build_sorted_owner_basin_runtime_identity,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _component_record(root: Path, component: str, relative: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    return {
        "bytes": path.stat().st_size,
        "component": component,
        "path": str(path),
        "relative_path": relative,
        "sha256": sha256_file(path),
    }


def _synchronize_sources(fixture: dict[str, Any]) -> None:
    manifest = fixture["manifest_document"]
    model = manifest["model_identity"]
    tokenizer = manifest["tokenizer_identity"]
    processor = manifest["processor_identity"]
    manifest["adapter_identity"] = deepcopy(model["adapter"])
    manifest["embedding_delta_identity"] = deepcopy(model["embedding_delta"])
    manifest["model_identity_fingerprint"] = sha256_json(model)
    manifest["processor_identity_fingerprint"] = sha256_json(processor)
    manifest["backend_session"]["model_identity"] = deepcopy(model)
    manifest["backend_session"]["tokenizer_identity"] = deepcopy(tokenizer)
    manifest["backend_session"]["processor_identity"] = deepcopy(processor)
    _write_json(fixture["manifest"], manifest)

    receipt = fixture["receipt_document"]
    identity = receipt["model_tokenizer_processor_identity"]
    exact = deepcopy(manifest["backend_session"])
    exact["effective_settings"].pop("performance", None)
    identity["exact_identity"] = exact
    identity["model_identity_sha256"] = sha256_json(exact)
    identity["tokenizer_identity"] = deepcopy(tokenizer)
    identity["processor_identity"] = deepcopy(processor)
    for bound in receipt["inputs"]["bound_files"]:
        if bound["role"] == "production_rp_1_10_manifest":
            bound.update(
                {
                    "bytes": fixture["manifest"].stat().st_size,
                    "sha256": sha256_file(fixture["manifest"]),
                }
            )
    receipt_without_digest = {
        key: value
        for key, value in receipt.items()
        if key != "execution_receipt_content_sha256"
    }
    receipt["execution_receipt_content_sha256"] = sha256_json(receipt_without_digest)
    _write_json(fixture["receipt"], receipt)


def _fixture(root: Path) -> dict[str, Any]:
    base = root / "base"
    adapter = root / "adapter"
    delta = root / "delta"
    base.mkdir(parents=True)
    adapter.mkdir()
    delta.mkdir()

    coord_strings = [f"<|coord_{index}|>" for index in range(1000)]
    added_tokens = {
        **WRAPPER_TOKEN_IDS,
        **{
            token: COORDINATE_TOKEN_ID_START + index
            for index, token in enumerate(coord_strings)
        },
    }
    _write_json(base / "added_tokens.json", added_tokens)
    _write_json(
        base / "config.json",
        {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": {"dtype": "float32", "vocab_size": MODEL_VOCAB_SIZE},
        },
    )
    _write_json(base / "coord_tokens.json", ["<|coord_*|>", *coord_strings])
    _write_json(base / "special_tokens_map.json", {"fixture": True})
    _write_json(base / "tokenizer.json", {"fixture": "tokenizer"})
    _write_json(base / "tokenizer_config.json", {"fixture": True})
    _write_json(adapter / "adapter_config.json", {"peft_type": "LORA"})
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-adapter")

    token_ids = [*WRAPPER_TOKEN_IDS.values(), *range(151670, 152670)]
    token_strings = [*WRAPPER_TOKEN_IDS.keys(), *coord_strings]
    embedding_metadata = {
        "base_config_sha256": sha256_file(base / "config.json"),
        "base_model_path": str(base.resolve()),
        "semantics": "additive_delta",
        "tensor_dtype": "float32",
        "tensor_key": "shared_embed_delta",
        "tensor_shape": [1004, 2048],
        "tie_word_embeddings": True,
        "token_ids": token_ids,
        "token_strings": token_strings,
        "tokenizer_sha256": sha256_file(base / "tokenizer.json"),
    }
    _write_json(delta / "special_token_embeddings.json", embedding_metadata)
    (delta / "special_token_embeddings.safetensors").write_bytes(b"fixture-delta")

    roots = {
        "base_model_path": base,
        "adapter_path": adapter,
        "embedding_delta_path": delta,
    }
    component_files = [
        _component_record(component_root, component, path.name)
        for component, component_root in roots.items()
        for path in sorted(component_root.iterdir())
    ]
    components = {key: str(value.resolve()) for key, value in roots.items()}
    tokenizer_identity = {
        "coord_token_count": 1000,
        "coord_token_id_max": 152669,
        "coord_token_id_min": 151670,
        "coord_token_ids_contiguous": True,
        "im_end_newline_split_verified": True,
        "im_end_newline_text": "<|im_end|>\n",
        "im_end_newline_token_ids": [151645, 198],
        "im_end_token_ids": [151645],
        "newline_token_ids": [198],
        "required_token_count": 1004,
        "tokenizer_vocab_size": MODEL_VOCAB_SIZE,
        "wrapper_token_ids": dict(WRAPPER_TOKEN_IDS),
    }
    processor_identity = {
        "image_processor_class": "Qwen2VLImageProcessorFast",
        "merge_size": 2,
        "patch_size": 16,
        "processor_class": "Qwen3VLProcessor",
        "temporal_patch_size": 2,
        "tokenizer_class": "Qwen2TokenizerFast",
    }
    embedding_identity = {
        "identity": {
            "base_model_path": components["base_model_path"],
            "delta_path": components["embedding_delta_path"],
            "metadata": embedding_metadata,
        },
        "load": {
            "loaded": True,
            "metadata_path": str((delta / "special_token_embeddings.json").resolve()),
            "runtime_tensor_dtype": "float32",
            "source_tensor_dtype": "float32",
            "tensor_dtype": "float32",
            "tensor_key": "shared_embed_delta",
            "tensor_path": str(
                (delta / "special_token_embeddings.safetensors").resolve()
            ),
            "tensor_shape": [1004, 2048],
        },
        "status": "loaded",
    }
    model_identity = {
        "adapter": {
            "active_adapters": ["default"],
            "adapter_name": "default",
            "adapter_path": components["adapter_path"],
            "adapter_type": "dora",
            "base_model_path": components["base_model_path"],
            "enabled": True,
            "missing_keys": [],
            "status": "validated",
            "unexpected_keys": [],
        },
        "base": {"path": components["base_model_path"]},
        "embedding_delta": embedding_identity,
        "family": "base-plus-adapter-plus-delta",
        "qwen": {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "config_class": "Qwen3VLConfig",
            "config_dtype": "torch.float32",
            "model_type": "qwen3_vl",
            "text_hidden_size": 2048,
            "text_vocab_size": MODEL_VOCAB_SIZE,
            "tie_word_embeddings": True,
        },
    }
    session = {
        "backend": "hf",
        "backend_mode": "generate",
        "backend_version": "4.57.1",
        "effective_settings": {
            "backend_options": {"hf": {"attn_implementation": "sdpa"}},
            "batch_size": 4,
            "device": "cuda",
            "output_scores": True,
            "performance": {"generated_token_count": 12},
            "raw_output_logits": "per_request",
            "text_padding_side": "left",
        },
        "execution_model_identity": None,
        "generation_config_fingerprint": sha256_json({"fixture": "generation"}),
        "likelihood_semantics": {
            "policy": "fp32_log_softmax_after_active_generation_processors",
            "raw": "fp32_log_softmax_unmodified_lm_head_logits",
            "score_owned_channel": "policy_logprob",
        },
        "model_identity": model_identity,
        "processor_identity": processor_identity,
        "response_family": "hf",
        "tokenizer_identity": tokenizer_identity,
    }
    manifest_document = {
        "adapter_identity": deepcopy(model_identity["adapter"]),
        "backend": "hf",
        "backend_mode": "generate",
        "backend_session": session,
        "embedding_delta_identity": deepcopy(embedding_identity),
        "generation_config_fingerprint": session["generation_config_fingerprint"],
        "model_identity": model_identity,
        "model_identity_fingerprint": sha256_json(model_identity),
        "processor_identity": processor_identity,
        "processor_identity_fingerprint": sha256_json(processor_identity),
        "resolved_config_fingerprints": {
            "infer_config": sha256_json({"fixture": "config"})
        },
        "response_family": "hf",
        "terminal_status": "completed",
        "tokenizer_identity": tokenizer_identity,
    }
    manifest = root / "run_manifest.json"
    _write_json(manifest, manifest_document)
    exact = deepcopy(session)
    exact["effective_settings"].pop("performance")
    receipt_document = {
        "coordinate_contract": {
            "coordinate_bin_max": 999,
            "coordinate_bin_min": 0,
            "source_token_grammar": "<|coord_N|> with integer N in [0, 999]",
        },
        "execution_status": "completed",
        "inputs": {
            "bound_files": [
                {
                    "bytes": manifest.stat().st_size,
                    "path": str(manifest.resolve()),
                    "role": "production_rp_1_10_manifest",
                    "sha256": sha256_file(manifest),
                },
                *[
                    {
                        "bytes": item["bytes"],
                        "path": item["path"],
                        "role": f"model_component:{item['component']}",
                        "sha256": item["sha256"],
                    }
                    for item in component_files
                ],
            ]
        },
        "model_tokenizer_processor_identity": {
            "exact_identity": exact,
            "model_component_files": component_files,
            "model_components": components,
            "model_identity_sha256": sha256_json(exact),
            "processor_identity": processor_identity,
            "tokenizer_identity": tokenizer_identity,
        },
        "schema_version": "sorted-owner-basin-task0-execution-receipt.v2",
    }
    receipt_document["execution_receipt_content_sha256"] = sha256_json(receipt_document)
    receipt = root / "execution-receipt.json"
    _write_json(receipt, receipt_document)
    return {
        "base": base,
        "manifest": manifest,
        "manifest_document": manifest_document,
        "receipt": receipt,
        "receipt_document": receipt_document,
    }


def _build(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return build_sorted_owner_basin_runtime_identity(
        execution_receipt=fixture["receipt"],
        production_manifest=fixture["manifest"],
        output=output,
    )


def test_builds_canonical_identity_and_is_idempotent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    output = tmp_path / "out" / "runtime-identity.json"

    first = _build(fixture, output)
    first_bytes = output.read_bytes()
    first_stat = output.stat()
    second = _build(fixture, output)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert output.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["status"] == "frozen"
    assert first["coordinate_vocabulary"] == {
        "coordinate_min": 0,
        "coordinate_max": 999,
        "token_id_start": 151670,
        "token_id_end_exclusive": 152670,
    }
    assert first["model_vocab_size"] == 152670
    assert first["schema_tokens"] == SCHEMA_TOKENS
    assert first["tokenizer"]["path"] == str(fixture["base"].resolve())
    for section in ("tokenizer", "model", "runtime"):
        assert first[section]["identity_sha256"] == sha256_json(
            first[section]["identity_source"]
        )
    content = {key: value for key, value in first.items() if key != "receipt_digest"}
    assert first["receipt_digest"] == sha256_json(content)


def test_write_once_rejects_conflicting_existing_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    output = tmp_path / "runtime-identity.json"
    output.write_text("user-owned\n", encoding="utf-8")

    with pytest.raises(RuntimeIdentityError, match="refusing to overwrite"):
        _build(fixture, output)

    assert output.read_text(encoding="utf-8") == "user-owned\n"


def test_rejects_component_file_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    (fixture["base"] / "tokenizer.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        RuntimeIdentityError, match="SHA-256 is stale|byte size is stale"
    ):
        _build(fixture, tmp_path / "out.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda manifest: manifest["tokenizer_identity"].update(
                {"coord_token_id_min": 151669}
            ),
            "wrong coord_token_id_min",
        ),
        (
            lambda manifest: manifest["tokenizer_identity"]["wrapper_token_ids"].update(
                {"<|box_end|>": 151650}
            ),
            "wrong wrapper_token_ids",
        ),
        (
            lambda manifest: manifest["model_identity"].update({"family": "base-only"}),
            "base-plus-adapter-plus-delta",
        ),
    ],
)
def test_rejects_wrong_token_or_model_identity(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    fixture = _fixture(tmp_path / "sources")
    mutate(fixture["manifest_document"])
    _synchronize_sources(fixture)

    with pytest.raises(RuntimeIdentityError, match=match):
        _build(fixture, tmp_path / "out.json")


def test_rejects_stale_execution_receipt_content_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    receipt = fixture["receipt_document"]
    receipt["execution_status"] = "changed-after-freeze"
    _write_json(fixture["receipt"], receipt)

    with pytest.raises(RuntimeIdentityError, match="content digest is stale"):
        _build(fixture, tmp_path / "out.json")


def test_rejects_stale_bound_manifest_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    manifest = fixture["manifest_document"]
    manifest["terminal_status"] = "changed-after-binding"
    _write_json(fixture["manifest"], manifest)

    with pytest.raises(RuntimeIdentityError, match="bound production manifest.*stale"):
        _build(fixture, tmp_path / "out.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda manifest: manifest.update(
                {"backend": "vllm", "response_family": "vllm"}
            ),
            "only the production HF",
        ),
        (
            lambda manifest: manifest["model_identity"]["qwen"].update(
                {"config_dtype": "torch.bfloat16"}
            ),
            "only the FP32 production checkpoint",
        ),
    ],
)
def test_rejects_unsupported_backend_or_precision(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    fixture = _fixture(tmp_path / "sources")
    mutate(fixture["manifest_document"])
    _synchronize_sources(fixture)

    with pytest.raises(RuntimeIdentityError, match=match):
        _build(fixture, tmp_path / "out.json")


def test_rejects_missing_adapter_delta_or_token_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    (fixture["base"] / "coord_tokens.json").unlink()

    with pytest.raises(RuntimeIdentityError, match="does not exist"):
        _build(fixture, tmp_path / "out.json")


def test_cli_writes_the_same_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "sources")
    output = tmp_path / "cli" / "runtime-identity.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/build_sorted_owner_basin_runtime_identity.py",
            "--execution-receipt",
            str(fixture["receipt"]),
            "--production-manifest",
            str(fixture["manifest"]),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {
        "output": str(output.resolve()),
        "receipt_digest": document["receipt_digest"],
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
    }
