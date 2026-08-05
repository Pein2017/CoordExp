#!/usr/bin/env python3
"""Admit the historical sorted image-2299 native rollout into Task-0 ledgers.

This module is deliberately CPU-only.  It does not load model weights or run
generation.  It reconstructs the exact prompt with the current local Qwen
processor, reconstructs generated IDs from the per-token trace, validates the
natural stop and parser row, applies the frozen owner-basin matcher, and emits
the four one-image artifacts consumed by the owner-accessibility planner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_sorted_owner_basin_census import (  # noqa: E402
    IOU_THRESHOLD,
    MATCHER_SCHEMA_VERSION,
    _global_assignment_analysis,
    _iou,
    _load_panel,
    _normalize_description,
)
from src.config.models import TemplateConfig  # noqa: E402
from src.data import RawExample, load_raw_examples  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402
from src.inference.prompt import build_prompt_record  # noqa: E402
from src.qwen.images import rgb_image_sha256  # noqa: E402
from src.qwen.runtime_loading import (  # noqa: E402
    QwenLoadOptions,
    load_qwen_components_from_options,
)


SCHEMA_VERSION = "sorted-image2299-native-ledger.v1"
RECEIPT_SCHEMA_VERSION = "sorted-image2299-native-ledger-receipt.v1"
OWNER_SCHEMA_VERSION = "sorted-owner-basin-owner-ledger.v2"
PREDICTION_SCHEMA_VERSION = "sorted-owner-basin-prediction-row-ledger.v2"
ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"

TARGET_IMAGE_ID = "2299"
TARGET_ROW_ID = "coco2017_val_000000002299"
IM_END_TOKEN_ID = 151645
SOURCE_HORIZON = 512
PROSPECTIVE_HORIZON = 3084

ROOT = Path("/data/CoordExp")
DEFAULT_PANEL = (
    ROOT
    / "outputs/research/qwen3-vl-dense-enumeration"
    / "2026-08-04-sorted-prospective-13-image-panel-admission"
    / "evaluation-inputs/human-refined-13.coord.jsonl"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "outputs/research/qwen3-vl-dense-enumeration"
    / "2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen"
    / "smoke-b-v1/ordinary-inference-runs"
    / "qwen3-vl-2b-rollout-calibration-smoke-b-v1-source"
)

BASE_MODEL_PATH = (
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
ADAPTER_PATH = (
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_"
    "accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/adapter"
)
DELTA_PATH = str(Path(ADAPTER_PATH).parent / "special_token_embeddings")

DEFAULT_ARTIFACT_SHA256: Mapping[str, str] = {
    "run_manifest.json": "6c17734be953a40c9686179a6528b178496e6e0a8e68762b34ce177713275ec4",
    "configs/resolved.json": "7656ca51ff9f4b29f30790aeabc5fc1fa905f44ae64d01ef3a4bc1e7e37d383b",
    "gt_vs_pred.jsonl": "7f5d1da6570d6537a3f88d3c43279cd784f589f38ce381d1a64d9c5e94b92b1a",
    "parse_diagnostics.jsonl": "d9126f818b1be4ed12d9d0b8136fee0581eed6eac567c0cf1b9b839fcb7c8d54",
    "pred_token_trace.jsonl": "fee91c938022b96b9dc64a83cedb6656450f8f773a82f259d5944a46524287a2",
    "image_plan.jsonl": "c0ae887e7e06060e9c62ca6181721ffb251648d78c8c23d5562e8be927b6d608",
    "summary.json": "c2546b1c93fd1de2b794f3b821f56693c420d2ce1dd98cf6a6554ed17932202d",
}
DEFAULT_COMPONENT_SHA256: Mapping[str, str] = {
    "adapter_config": "088aab8bec52f40d6261e8c1c4271458bb1e7c3eb69a9c9c3fa016d609b78a9d",
    "adapter_tensor": "7f6ee67a71f4d948f0e89729ad50f3856a7a215b5ba433fdd3f3dd5b329e5a81",
    "embedding_delta_metadata": "7a465c6b7d43c5a46066d99a1076e96fa26dc0fe30f7ca69fc289ed347b30983",
    "embedding_delta_tensor": "33d65d575df398375e72994e2a197aaff0a257e26c088717e3b7689b3f494c3b",
}


class NativeLedgerContractError(ValueError):
    """Raised before incompatible historical evidence can be published."""


@dataclass(frozen=True)
class ExpectedContract:
    panel_sha256: str = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
    panel_receipt_sha256: str = "ad78c174897509dca07c24c57d12c897fdad42724d83af08147d79c9644c5414"
    authority_row_sha256: str = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
    image_sha256: str = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
    model_identity_fingerprint: str = (
        "1338e1568f4d39ee97cea8e861eceb51693f8282c914c5f2a875b07d711755a1"
    )
    processor_identity_fingerprint: str = (
        "1b6a8d3fdd4ee4bc30734aab17d902f1ce89ae2ac208f875b624cdb6cac4d055"
    )
    generation_config_fingerprint: str = (
        "7414955aac7ceb7ad177f3f0f7d1e3dfd297f609dd30c2e5339cb82a4a47db8e"
    )
    prompt_policy_fingerprint: str = (
        "609b96ccb7022de0e35d0b3434483e62e8ce13a6c3f83b3dc68ebe1da928c157"
    )
    prompt_token_ids_sha256: str = (
        "33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb"
    )
    generated_token_ids_sha256: str = (
        "95a89abb1c56501edc2ca521026e9e102708c32f9faed797c8e7790ecc088c5d"
    )
    artifact_sha256: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ARTIFACT_SHA256)
    )
    component_sha256: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_COMPONENT_SHA256)
    )
    require_panel_receipt: bool = True


PromptReconstructor = Callable[
    [RawExample, Mapping[str, Any], int, Mapping[str, Any]], tuple[list[int], Any, Mapping[str, Any]]
]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeLedgerContractError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise NativeLedgerContractError(f"JSON artifact must contain an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise NativeLedgerContractError(f"cannot read JSONL artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise NativeLedgerContractError(f"blank JSONL row in {path} at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NativeLedgerContractError(
                f"invalid JSONL row in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise NativeLedgerContractError(
                f"JSONL row must be an object in {path} at line {line_number}"
            )
        rows.append(row)
    return rows


def _target_rows(rows: Sequence[Mapping[str, Any]], *, source: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get("row_id")) == TARGET_ROW_ID]
    if len(matches) != 1:
        raise NativeLedgerContractError(
            f"{source} must contain exactly one {TARGET_ROW_ID} row, found {len(matches)}"
        )
    return matches[0]


def _validate_file_identities(
    panel_path: Path,
    run_dir: Path,
    expected: ExpectedContract,
) -> dict[str, Any]:
    panel = panel_path.resolve(strict=True)
    actual_panel = _sha256_file(panel)
    if actual_panel != expected.panel_sha256:
        raise NativeLedgerContractError(
            f"panel identity drift: {actual_panel} != {expected.panel_sha256}"
        )
    artifact_files: dict[str, Any] = {}
    for relative, digest in sorted(expected.artifact_sha256.items()):
        source = (run_dir / relative).resolve(strict=True)
        actual = _sha256_file(source)
        if actual != digest:
            raise NativeLedgerContractError(
                f"historical artifact identity drift for {relative}: {actual} != {digest}"
            )
        artifact_files[relative] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": actual,
        }

    admission_receipt: dict[str, Any] | None = None
    if expected.require_panel_receipt:
        receipt_path = panel.parent.parent / "receipt.json"
        if _sha256_file(receipt_path) != expected.panel_receipt_sha256:
            raise NativeLedgerContractError("panel admission receipt identity drift")
        receipt = _read_json(receipt_path)
        content_digest = receipt.get("receipt_content_sha256")
        if content_digest != _json_digest(
            {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
        ):
            raise NativeLedgerContractError("panel admission receipt self-seal mismatch")
        authority = receipt.get("inputs", {}).get("authority", {})
        if authority.get("target_line_sha256") != expected.authority_row_sha256:
            raise NativeLedgerContractError("wrong GT authority row identity")
        output = receipt.get("output", {})
        if output.get("jsonl_sha256") != actual_panel:
            raise NativeLedgerContractError("panel admission receipt does not bind the admitted panel")
        admission_receipt = {
            "path": str(receipt_path.resolve()),
            "sha256": expected.panel_receipt_sha256,
            "receipt_content_sha256": content_digest,
            "authority_row_sha256": expected.authority_row_sha256,
        }
    return {
        "panel": {"path": str(panel), "bytes": panel.stat().st_size, "sha256": actual_panel},
        "panel_admission_receipt": admission_receipt,
        "historical_artifacts": artifact_files,
    }


def _validate_runtime_contract(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    expected: ExpectedContract,
) -> dict[str, Any]:
    config = resolved.get("config")
    if not isinstance(config, Mapping):
        raise NativeLedgerContractError("resolved config lacks config object")
    backend = config.get("backend")
    model = config.get("model")
    generation = config.get("generation")
    template = config.get("template")
    if not all(isinstance(item, Mapping) for item in (backend, model, generation, template)):
        raise NativeLedgerContractError("resolved runtime sections are missing")
    assert isinstance(backend, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(generation, Mapping)
    assert isinstance(template, Mapping)

    expected_generation = {
        "max_new_tokens": SOURCE_HORIZON,
        "temperature": 0,
        "top_p": 1,
        "repetition_penalty": 1,
    }
    if backend.get("type") != "hf" or backend.get("hf") != {
        "attn_implementation": "sdpa",
        "patch_embed_linearization": "enabled",
    }:
        raise NativeLedgerContractError("runtime must be exact HF SDPA with patch linearization")
    if model.get("base_model") != BASE_MODEL_PATH or model.get("dtype") != "fp32":
        raise NativeLedgerContractError("model runtime identity drift")
    if model.get("processor") != {"do_resize": False}:
        raise NativeLedgerContractError("processor resize policy drift")
    for key, value in expected_generation.items():
        if generation.get(key) != value:
            raise NativeLedgerContractError(f"generation policy drift for {key}")
    if generation.get("n") != 1:
        raise NativeLedgerContractError("generation policy must request exactly one completion")
    if template.get("object_field_order") != "desc_first" or template.get(
        "object_ordering"
    ) != "geo_sorted" or template.get("assistant_format") != "object_box_closed":
        raise NativeLedgerContractError("prompt/template identity drift")

    expected_fingerprints = {
        "model_identity_fingerprint": expected.model_identity_fingerprint,
        "processor_identity_fingerprint": expected.processor_identity_fingerprint,
        "generation_config_fingerprint": expected.generation_config_fingerprint,
        "prompt_policy_fingerprint": expected.prompt_policy_fingerprint,
    }
    for key, value in expected_fingerprints.items():
        if manifest.get(key) != value:
            raise NativeLedgerContractError(f"manifest identity drift for {key}")
    if manifest.get("backend") != "hf" or manifest.get("backend_mode") != "generate":
        raise NativeLedgerContractError("manifest backend identity drift")

    model_identity = manifest.get("model_identity")
    if not isinstance(model_identity, Mapping):
        raise NativeLedgerContractError("manifest model identity is missing")
    base = model_identity.get("base")
    adapter = model_identity.get("adapter")
    delta = model_identity.get("embedding_delta")
    if not all(isinstance(item, Mapping) for item in (base, adapter, delta)):
        raise NativeLedgerContractError("manifest checkpoint components are missing")
    assert isinstance(base, Mapping)
    assert isinstance(adapter, Mapping)
    assert isinstance(delta, Mapping)
    delta_identity = delta.get("identity")
    if not isinstance(delta_identity, Mapping):
        raise NativeLedgerContractError("embedding delta identity is missing")
    if base.get("path") != BASE_MODEL_PATH:
        raise NativeLedgerContractError("base checkpoint identity drift")
    if adapter.get("adapter_path") != ADAPTER_PATH or adapter.get("status") != "validated":
        raise NativeLedgerContractError("sorted step-4887 adapter identity drift")
    if delta_identity.get("delta_path") != DELTA_PATH or delta_identity.get(
        "status"
    ) != "validated":
        raise NativeLedgerContractError("sorted step-4887 embedding delta identity drift")

    policy = manifest.get("generation_policy")
    if not isinstance(policy, Mapping):
        raise NativeLedgerContractError("manifest generation policy is missing")
    for key, value in {
        **expected_generation,
        "do_sample": False,
        "stop_policy": "qwen_im_end",
    }.items():
        if policy.get(key) != value:
            raise NativeLedgerContractError(f"manifest generation policy drift for {key}")
    likelihood = manifest.get("likelihood_semantics")
    if not isinstance(likelihood, Mapping) or likelihood.get("policy") != (
        "fp32_log_softmax_after_active_generation_processors"
    ):
        raise NativeLedgerContractError("likelihood/runtime policy identity drift")
    return {
        "backend": "hf",
        "dtype": "fp32",
        "attention": "sdpa",
        "decode": "greedy",
        "repetition_penalty": 1.0,
        "configured_horizon": SOURCE_HORIZON,
        "prospective_horizon": PROSPECTIVE_HORIZON,
        "model_identity_fingerprint": expected.model_identity_fingerprint,
        "processor_identity_fingerprint": expected.processor_identity_fingerprint,
        "generation_config_fingerprint": expected.generation_config_fingerprint,
        "prompt_policy_fingerprint": expected.prompt_policy_fingerprint,
        "model_identity": model_identity,
        "tokenizer_identity": manifest.get("tokenizer_identity"),
        "processor_identity": manifest.get("processor_identity"),
        "backend_session": manifest.get("backend_session"),
        "likelihood_semantics": likelihood,
    }


def _validate_component_files(
    manifest: Mapping[str, Any], expected: ExpectedContract
) -> list[dict[str, Any]]:
    if not expected.component_sha256:
        return []
    adapter = manifest["adapter_identity"]
    delta = manifest["embedding_delta_identity"]
    paths = {
        "adapter_config": adapter["adapter_payload_evidence"]["config_path"],
        "adapter_tensor": adapter["adapter_payload_evidence"]["tensor_path"],
        "embedding_delta_metadata": delta["identity"]["metadata_path"],
        "embedding_delta_tensor": delta["load"]["tensor_path"],
    }
    receipts = []
    for role, expected_digest in sorted(expected.component_sha256.items()):
        source = Path(str(paths.get(role, ""))).resolve(strict=True)
        actual = _sha256_file(source)
        if actual != expected_digest:
            raise NativeLedgerContractError(f"checkpoint component identity drift for {role}")
        receipts.append(
            {
                "role": role,
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": actual,
            }
        )
    return receipts


def _load_authoritative_owners(
    panel_path: Path, expected: ExpectedContract
) -> tuple[Mapping[str, Any], list[dict[str, Any]], RawExample]:
    panel_rows = _read_jsonl(panel_path)
    matches = [row for row in panel_rows if str(row.get("image_id")) == TARGET_IMAGE_ID]
    if len(matches) != 1:
        raise NativeLedgerContractError(
            f"admitted panel must contain exactly one image 2299 row, found {len(matches)}"
        )
    raw_row = matches[0]
    objects = raw_row.get("objects")
    if not isinstance(objects, list):
        raise NativeLedgerContractError("image 2299 panel row lacks objects")
    descriptions = Counter(_normalize_description(obj.get("desc")) for obj in objects)
    if len(objects) != 46 or descriptions != Counter({"person": 38, "tie": 8}):
        raise NativeLedgerContractError(
            "wrong GT: image 2299 authority must contain 46 owners (38 person, 8 tie)"
        )

    _, owners_by_image, _ = _load_panel(panel_path)
    owners = owners_by_image.get(TARGET_IMAGE_ID)
    if owners is None or len(owners) != 46:
        raise NativeLedgerContractError("wrong GT: canonical owner conversion did not yield 46 owners")

    examples = [example for example in load_raw_examples(panel_path) if example.example_id == TARGET_ROW_ID]
    if len(examples) != 1:
        raise NativeLedgerContractError("admitted panel does not reconstruct exactly one target example")
    image_path = examples[0].image.path.resolve(strict=True)
    if _sha256_file(image_path) != expected.image_sha256:
        raise NativeLedgerContractError("image 2299 byte identity drift")
    return raw_row, owners, examples[0]


def _validate_historical_gt(
    historical_row: Mapping[str, Any], panel_row: Mapping[str, Any]
) -> None:
    gt = historical_row.get("gt")
    objects = panel_row.get("objects")
    if not isinstance(gt, list) or not isinstance(objects, list) or len(gt) != len(objects):
        raise NativeLedgerContractError("wrong GT: historical row does not contain the refined 46 owners")
    expected = [
        {
            "description": _normalize_description(obj.get("desc")),
            "bbox_2d": obj.get("bbox_2d"),
            "category_id": obj.get("category_id"),
        }
        for obj in objects
    ]
    observed = []
    for row in gt:
        source = row.get("metadata", {}).get("source", {})
        observed.append(
            {
                "description": _normalize_description(row.get("description")),
                "bbox_2d": source.get("bbox_2d"),
                "category_id": source.get("category_id"),
            }
        )
    if observed != expected:
        raise NativeLedgerContractError(
            "wrong GT: historical row differs from the admitted refined 46-owner authority"
        )


def _default_prompt_reconstructor(
    example: RawExample,
    resolved_config: Mapping[str, Any],
    merged_visual_tokens: int,
    manifest: Mapping[str, Any],
) -> tuple[list[int], Any, Mapping[str, Any]]:
    components = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=BASE_MODEL_PATH,
            dtype="fp32",
            attn_implementation="sdpa",
            load_model=False,
        )
    )
    processor_identity = components.processor_identity.to_artifact_dict()
    tokenizer_identity = components.token_identity.to_artifact_dict()
    if processor_identity != manifest.get("processor_identity"):
        raise NativeLedgerContractError("current exact processor identity differs from the stored run")
    if tokenizer_identity != manifest.get("tokenizer_identity"):
        raise NativeLedgerContractError("current exact tokenizer identity differs from the stored run")
    template = TemplateConfig.model_validate(resolved_config["template"])
    prompt = build_prompt_record(
        example,
        template,
        processor=components.processor,
        row_index=0,
        merged_visual_tokens=merged_visual_tokens,
    )
    return (
        list(prompt.expected_executed_prompt_token_ids),
        components.tokenizer,
        {
            "processor": processor_identity,
            "tokenizer": tokenizer_identity,
            "chat_text_sha256": hashlib.sha256(prompt.chat_text.encode("utf-8")).hexdigest(),
            "input_prompt_token_ids_sha256": token_ids_sha256(prompt.input_prompt_token_ids),
        },
    )


def _validate_prompt(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    image_plan: Mapping[str, Any],
    example: RawExample,
    expected: ExpectedContract,
    reconstructor: PromptReconstructor,
) -> tuple[list[int], Any, Mapping[str, Any]]:
    prompt_rows = manifest.get("prompt_trace")
    if not isinstance(prompt_rows, list):
        raise NativeLedgerContractError("manifest prompt trace is missing")
    prompt = _target_rows(prompt_rows, source="prompt trace")
    stored = prompt.get("backend_executed_prompt_token_ids_sha256")
    if (
        prompt.get("prompt_token_parity") != "verified"
        or stored != prompt.get("expected_executed_prompt_token_ids_sha256")
        or stored != expected.prompt_token_ids_sha256
    ):
        raise NativeLedgerContractError("stored prompt identity or parity drift")
    config = resolved.get("config")
    assert isinstance(config, Mapping)
    prompt_ids, tokenizer, evidence = reconstructor(
        example, config, int(image_plan["merged_visual_tokens"]), manifest
    )
    actual = token_ids_sha256(prompt_ids)
    if actual != stored or len(prompt_ids) != prompt.get("backend_executed_prompt_token_count"):
        raise NativeLedgerContractError("reconstructed prompt token IDs do not match stored prompt hash")
    return prompt_ids, tokenizer, {**dict(evidence), "prompt_token_ids_sha256": actual}


def _reconstruct_generated_ids(
    trace_rows: Sequence[Mapping[str, Any]],
    historical_row: Mapping[str, Any],
    tokenizer: Any,
    expected: ExpectedContract,
) -> tuple[list[int], dict[str, Any]]:
    generated = [
        row
        for row in trace_rows
        if row.get("row_id") == TARGET_ROW_ID and row.get("trace_type") == "generated_token"
    ]
    if not generated:
        raise NativeLedgerContractError("target token trace has no generated-token rows")
    indices = [row.get("generated_step_index") for row in generated]
    if indices != list(range(len(generated))):
        raise NativeLedgerContractError("generated token reconstruction mismatch: non-contiguous steps")
    first_pad = next((index for index, row in enumerate(generated) if row.get("is_pad")), len(generated))
    if any(not row.get("is_pad") for row in generated[first_pad:]):
        raise NativeLedgerContractError("generated token reconstruction mismatch: non-pad after padding")
    natural = generated[:first_pad]
    if not natural:
        raise NativeLedgerContractError("target token trace contains no natural generated IDs")
    stop_rows = [row for row in natural if row.get("is_stop")]
    if (
        len(stop_rows) != 1
        or stop_rows[0] is not natural[-1]
        or natural[-1].get("token_id") != IM_END_TOKEN_ID
        or natural[-1].get("token_text") != "<|im_end|>"
    ):
        raise NativeLedgerContractError("non-natural/truncated stop: terminal im_end is not unique and final")
    if historical_row.get("decode_stop_reason") != "im_end":
        raise NativeLedgerContractError("non-natural/truncated stop: parser row is not im_end-terminated")
    if len(natural) >= SOURCE_HORIZON:
        raise NativeLedgerContractError("historical 512-token horizon is binding")

    full_ids = [int(row["token_id"]) for row in natural]
    ids = full_ids[:-1]
    if token_ids_sha256(ids) != expected.generated_token_ids_sha256:
        raise NativeLedgerContractError("generated token reconstruction mismatch: ID digest drift")
    token_text = "".join(str(row.get("token_text", "")) for row in natural)
    raw_text = historical_row.get("raw_decode_text")
    if token_text != raw_text:
        raise NativeLedgerContractError("generated token reconstruction mismatch: trace text drift")
    decoded = tokenizer.decode(
        full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != raw_text:
        raise NativeLedgerContractError("generated token reconstruction mismatch: tokenizer decode drift")
    return ids, {
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "natural_generated_token_count_including_im_end": len(natural),
        "emitted_row_token_count_excluding_im_end": len(ids),
        "natural_im_end_step_index": int(natural[-1]["generated_step_index"]),
        "configured_horizon": SOURCE_HORIZON,
        "remaining_tokens_at_natural_stop": SOURCE_HORIZON - len(natural),
        "prospective_horizon": PROSPECTIVE_HORIZON,
        "horizon_nonbinding": True,
        "padding_rows_excluded_from_reconstruction": len(generated) - len(natural),
    }


def _validate_parser_row(
    historical_row: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    predictions = historical_row.get("pred")
    if not isinstance(predictions, list):
        raise NativeLedgerContractError("parser row lacks predictions")
    expected_fields = {
        "parse_status": "accepted",
        "metric_bearing": True,
        "dropped_prediction_count": 0,
        "valid_prediction_count": len(predictions),
    }
    for key, value in expected_fields.items():
        if historical_row.get(key) != value:
            raise NativeLedgerContractError(f"parser row is not fully accepted: {key}")
    for key in ("parse_status", "dropped_prediction_count", "valid_prediction_count"):
        if diagnostic.get(key) != historical_row.get(key):
            raise NativeLedgerContractError(f"parse diagnostics disagree on {key}")
    if diagnostic.get("dropped_predictions") != [] or historical_row.get(
        "dropped_predictions"
    ) != []:
        raise NativeLedgerContractError("parser row contains dropped predictions")
    if [row.get("generated_order") for row in predictions] != list(range(len(predictions))):
        raise NativeLedgerContractError("parser prediction order is not contiguous")
    return predictions


def _prediction_for_match(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise NativeLedgerContractError(f"prediction {index} has invalid pixel bbox")
    coords = [float(value) for value in bbox]
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise NativeLedgerContractError(f"prediction {index} has degenerate pixel bbox")
    description = _normalize_description(row.get("description"))
    if not description:
        raise NativeLedgerContractError(f"prediction {index} has empty description")
    return {
        "pred_row_id": f"pred:sorted:greedy:0:{TARGET_IMAGE_ID}:{index}",
        "original_row_index": index,
        "normalized_description": description,
        "bbox_xyxy": coords,
    }


def _build_ledgers(
    owners: Sequence[Mapping[str, Any]],
    parsed_predictions: Sequence[Mapping[str, Any]],
    source_path: Path,
    panel_sha256: str,
    source_sha256: str,
    admission_contract_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    predictions = [
        _prediction_for_match(row, index) for index, row in enumerate(parsed_predictions)
    ]
    analysis = _global_assignment_analysis(predictions, owners, {})
    if analysis["ambiguity_classes"]:
        raise NativeLedgerContractError(
            "matcher ambiguity: inherited global optimum contains neutral owners/predictions"
        )
    matches = {row["pred_row_id"]: row for row in analysis["matches"]}
    matched_owner_ids = {row["gt_owner_id"] for row in analysis["matches"]}
    trajectory_id = f"trajectory:sorted:rp1.00:greedy:0:{TARGET_IMAGE_ID}"
    owner_ledger = []
    for owner in owners:
        owner_id = str(owner["gt_owner_id"])
        owner_ledger.append(
            {
                "schema_version": OWNER_SCHEMA_VERSION,
                "schema_compatibility": {
                    "replaces_schema_version": "sorted-owner-basin-owner-ledger.v1",
                    "migration": "honor decision_eligibility and ambiguity_receipt_ids",
                },
                "gt_owner_id": owner_id,
                "diagnostic_owner_id": f"diagnostic:{owner_id}",
                "diagnostic_owner_kind": "frozen_gt",
                "mapped_gt_owner_id": owner_id,
                "image_id": TARGET_IMAGE_ID,
                "original_annotation_index": owner["original_annotation_index"],
                "description": owner["description"],
                "normalized_description": owner["normalized_description"],
                "official_coco_category_id": owner["official_coco_category_id"],
                "bbox_xyxy": list(owner["bbox_xyxy"]),
                "ambiguity_receipt_ids": [],
                "decision_eligibility": {
                    "greedy_natural": {"eligible": True, "status": "eligible"},
                    "greedy_k16_paired": {
                        "eligible": False,
                        "status": "not_materialized_single_greedy_slice",
                    },
                    "k16_any_hit": {
                        "eligible": False,
                        "status": "not_materialized_single_greedy_slice",
                    },
                },
                "native_greedy_match_status": (
                    "matched" if owner_id in matched_owner_ids else "false_negative"
                ),
                "foreign_keys": {"panel_image_id": TARGET_IMAGE_ID, "mapped_gt_owner_id": owner_id},
                "null_status": {
                    "aux_owner_mapping": "not_materialized_no_review_input",
                    "unresolved_mapping": "not_materialized_no_review_input",
                },
                "execution_receipt_content_sha256": admission_contract_sha256,
                "source_digests": {
                    "panel": panel_sha256,
                    "execution_receipt_content": admission_contract_sha256,
                },
            }
        )

    prediction_ledger = []
    for index, (parsed, prediction) in enumerate(zip(parsed_predictions, predictions, strict=True)):
        pred_id = prediction["pred_row_id"]
        match = matches.get(pred_id)
        compatible_receipts = sorted(
            (
                {
                    "gt_owner_id": str(owner["gt_owner_id"]),
                    "intersection_over_union": _iou(prediction["bbox_xyxy"], owner["bbox_xyxy"]),
                }
                for owner in owners
                if owner["normalized_description"] == prediction["normalized_description"]
            ),
            key=lambda item: (-item["intersection_over_union"], item["gt_owner_id"]),
        )
        matched_owner = None if match is None else str(match["gt_owner_id"])
        prediction_ledger.append(
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "schema_compatibility": {
                    "replaces_schema_version": "sorted-owner-basin-prediction-row-ledger.v1",
                    "migration": "treat ambiguous_neutral rows as excluded",
                },
                "pred_row_id": pred_id,
                "trajectory_id": trajectory_id,
                "policy_stratum": "primary_rp_1.00",
                "image_id": TARGET_IMAGE_ID,
                "example_id": TARGET_ROW_ID,
                "decode_mode": "greedy",
                "seed": 0,
                "original_row_index": index,
                "row_kind": "complete_prediction",
                "description": parsed["description"],
                "normalized_description": prediction["normalized_description"],
                "bbox_xyxy": prediction["bbox_xyxy"],
                "object_span_id": parsed.get("object_span_id"),
                "raw_span_sha256": parsed.get("raw_span_sha256"),
                "strict_match_gt_owner_id": matched_owner,
                "strict_match_status": "matched" if match is not None else "unmatched",
                "strict_match_iou": None if match is None else match["intersection_over_union"],
                "semantic_relation": None if match is None else match["semantic_relation"],
                "eligible_owner_iou_receipts": compatible_receipts,
                "max_any_owner_iou": max(
                    _iou(prediction["bbox_xyxy"], owner["bbox_xyxy"]) for owner in owners
                ),
                "ambiguous_same_description_gt_owner_ids": [],
                "globally_optimal_edge_receipts": [
                    row for row in analysis["globally_optimal_edges"] if row["pred_row_id"] == pred_id
                ],
                "ambiguity_receipt_ids": [],
                "foreign_keys": {"trajectory_id": trajectory_id, "gt_owner_id": matched_owner},
                "null_status": {
                    "strict_match_gt_owner_id": "present" if match is not None else "unmatched",
                    "diagnostic_owner_id": "not_assigned_task0_1",
                },
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_sha256,
                "execution_receipt_content_sha256": admission_contract_sha256,
                "source_digests": {
                    "rollout_artifact": source_sha256,
                    "execution_receipt_content": admission_contract_sha256,
                },
            }
        )
    return owner_ledger, prediction_ledger, analysis


def build_native_ledger(
    *,
    panel_path: Path = DEFAULT_PANEL,
    run_dir: Path = DEFAULT_RUN_DIR,
    expected: ExpectedContract = ExpectedContract(),
    prompt_reconstructor: PromptReconstructor = _default_prompt_reconstructor,
) -> dict[str, Any]:
    """Validate all inputs and return the four serializable output payloads."""

    run_dir = run_dir.resolve(strict=True)
    identities = _validate_file_identities(panel_path, run_dir, expected)
    manifest = _read_json(run_dir / "run_manifest.json")
    resolved = _read_json(run_dir / "configs/resolved.json")
    summary = _read_json(run_dir / "summary.json")
    if summary.get("terminal_status") != "completed":
        raise NativeLedgerContractError("historical inference run is not terminally completed")
    runtime = _validate_runtime_contract(manifest, resolved, expected)
    components = _validate_component_files(manifest, expected)
    panel_row, owners, example = _load_authoritative_owners(panel_path, expected)

    historical_row = _target_rows(_read_jsonl(run_dir / "gt_vs_pred.jsonl"), source="gt_vs_pred")
    diagnostic = _target_rows(
        _read_jsonl(run_dir / "parse_diagnostics.jsonl"), source="parse diagnostics"
    )
    image_plan = _target_rows(_read_jsonl(run_dir / "image_plan.jsonl"), source="image plan")
    from PIL import Image

    with Image.open(example.image.path) as image:
        executed_media_sha256 = rgb_image_sha256(image.convert("RGB"))
    if (
        image_plan.get("status") != "ok"
        or image_plan.get("image_content_sha256") != expected.image_sha256
        or image_plan.get("executed_media_sha256") != executed_media_sha256
        or int(image_plan.get("decoded_width", -1)) != int(panel_row["width"])
        or int(image_plan.get("decoded_height", -1)) != int(panel_row["height"])
    ):
        raise NativeLedgerContractError("image/input execution identity drift")
    _validate_historical_gt(historical_row, panel_row)
    parsed_predictions = _validate_parser_row(historical_row, diagnostic)
    prompt_ids, tokenizer, prompt_evidence = _validate_prompt(
        manifest,
        resolved,
        image_plan,
        example,
        expected,
        prompt_reconstructor,
    )
    generated_ids, stop_evidence = _reconstruct_generated_ids(
        _read_jsonl(run_dir / "pred_token_trace.jsonl"),
        historical_row,
        tokenizer,
        expected,
    )

    source_sha256 = expected.artifact_sha256["gt_vs_pred.jsonl"]
    admission_contract = {
        "schema_version": SCHEMA_VERSION,
        "target_image_id": TARGET_IMAGE_ID,
        "target_row_id": TARGET_ROW_ID,
        "input_identities": identities,
        "runtime_identity": runtime,
        "checkpoint_component_files": components,
        "prompt_identity": prompt_evidence,
        "generated_identity": stop_evidence,
        "matcher": {
            "matcher_id": MATCHER_SCHEMA_VERSION,
            "iou_threshold": IOU_THRESHOLD,
            "category_join": "normalized_description_exact_no_aliases",
            "assignment_objective": ["maximum_cardinality", "maximum_total_iou"],
            "ambiguity_policy": "fail_closed",
        },
    }
    admission_contract_sha256 = _json_digest(admission_contract)
    owner_ledger, prediction_ledger, matching = _build_ledgers(
        owners,
        parsed_predictions,
        run_dir / "gt_vs_pred.jsonl",
        expected.panel_sha256,
        source_sha256,
        admission_contract_sha256,
    )

    predictions_envelope = {
        "parse_status": "accepted",
        "metric_bearing": True,
        "dropped_prediction_count": 0,
        "dropped_predictions": [],
        "valid_prediction_count": len(parsed_predictions),
        "predictions": [dict(row) for row in parsed_predictions],
    }
    greedy = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollout_count": 1,
        "config": {
            "decode_mode": "greedy",
            "image_ids": [TARGET_ROW_ID],
            "max_new_tokens": SOURCE_HORIZON,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "seeds": [0],
            "historical_horizon_admitted_as_nonbinding": True,
            "prospective_horizon": PROSPECTIVE_HORIZON,
            "resolved_fingerprint": manifest["resolved_config_fingerprints"]["infer_config"],
        },
        "model_identity": {
            "backend": manifest["backend"],
            "backend_mode": manifest["backend_mode"],
            "effective_settings": manifest["backend_session"]["effective_settings"],
            "execution_model_identity": manifest.get("execution_model_identity"),
            "generation_config_fingerprint": manifest["generation_config_fingerprint"],
            "likelihood_semantics": manifest["likelihood_semantics"],
            "model_identity": manifest["model_identity"],
            "processor_identity": manifest["processor_identity"],
            "tokenizer_identity": manifest["tokenizer_identity"],
        },
        "prompt_metadata": {
            TARGET_IMAGE_ID: {
                "prompt_token_ids": prompt_ids,
                "chat_text_sha256": prompt_evidence.get("chat_text_sha256"),
                "image_sha256": expected.image_sha256,
            }
        },
        "rollouts": [
            {
                "image_id": TARGET_IMAGE_ID,
                "example_id": TARGET_ROW_ID,
                "seed": 0,
                "decode_mode": "greedy",
                "stop_reason": "im_end",
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                "generated_token_ids": generated_ids,
                "generated_token_ids_sha256": token_ids_sha256(generated_ids),
                "generated_text": str(historical_row["raw_decode_text"]).removesuffix("<|im_end|>"),
                "executed_media_sha256": image_plan["executed_media_sha256"],
                "observed_image_grid_thw": image_plan["observed_image_grid_thw"],
                "predictions": predictions_envelope,
                "admission_contract_sha256": admission_contract_sha256,
                "natural_stop_evidence": stop_evidence,
            }
        ],
    }

    serialized = {
        "owner-ledger.jsonl": _jsonl_bytes(owner_ledger),
        "prediction-row-ledger.jsonl": _jsonl_bytes(prediction_ledger),
        "greedy.json": _json_bytes(greedy),
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": "2026-08-04-sorted-image2299-prospective-mechanism-extension",
        "status": "admitted",
        "scope": {
            "image_id": TARGET_IMAGE_ID,
            "legacy_12_denominator_unchanged": True,
            "forbidden_gt": "official/val200 22-owner row",
            "gpu_execution": "not_performed",
        },
        "admission_contract": admission_contract,
        "admission_contract_sha256": admission_contract_sha256,
        "counts": {
            "owners": len(owner_ledger),
            "predictions": len(prediction_ledger),
            "matched": int(matching["optimum_cardinality"]),
            "false_negatives": len(owner_ledger) - int(matching["optimum_cardinality"]),
            "false_positives": len(prediction_ledger) - int(matching["optimum_cardinality"]),
            "matcher_ambiguity_classes": 0,
        },
        "outputs": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(serialized.items())
        },
        "self_seal_contract": {
            "field": "receipt_content_sha256",
            "canonicalization": "UTF-8 JSON, ensure_ascii=true, sorted keys, compact separators",
            "excluded_top_level_fields": ["receipt_content_sha256"],
        },
    }
    receipt["receipt_content_sha256"] = _json_digest(receipt)
    serialized["receipt.json"] = _json_bytes(receipt)
    return {
        "owner_ledger": owner_ledger,
        "prediction_ledger": prediction_ledger,
        "greedy": greedy,
        "receipt": receipt,
        "serialized": serialized,
    }


def _json_bytes(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def publish_create_or_identical(output_dir: Path, payload: Mapping[str, Any]) -> str:
    serialized = payload.get("serialized")
    if not isinstance(serialized, Mapping):
        raise NativeLedgerContractError("payload lacks serialized artifacts")
    expected_names = set(serialized)
    destination = output_dir.resolve()
    if destination.exists():
        if not destination.is_dir():
            raise NativeLedgerContractError(f"output path is not a directory: {destination}")
        observed_names = {path.name for path in destination.iterdir() if path.is_file()}
        if observed_names != expected_names or any(path.is_dir() for path in destination.iterdir()):
            raise NativeLedgerContractError(
                f"existing output is not identical: artifact set {sorted(observed_names)}"
            )
        for name, data in serialized.items():
            if (destination / str(name)).read_bytes() != data:
                raise NativeLedgerContractError(f"existing output is not identical: {name}")
        return "identical"
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in serialized.items():
        (destination / str(name)).write_bytes(data)
    return "created"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is set")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_native_ledger(panel_path=args.panel, run_dir=args.run_dir)
    status = "validated"
    if not args.validate_only:
        status = publish_create_or_identical(args.output_dir, payload)
    print(
        json.dumps(
            {
                "status": status,
                "receipt_content_sha256": payload["receipt"]["receipt_content_sha256"],
                "counts": payload["receipt"]["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
