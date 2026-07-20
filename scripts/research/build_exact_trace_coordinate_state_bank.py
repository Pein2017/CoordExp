#!/usr/bin/env python3
"""Assemble a reviewed coordinate-boundary state bank from exact trace rows.

The input review packet is deliberately kept separate from review decisions.
This helper only joins already stored token evidence to explicit human review;
it never tokenizes text, repairs a box, or infers a positive label.  Accepted
cases are represented as coordinate-only events matching the live
``src.rollout_calibration.state_bank`` contract.  Rejected cases are retained
in the intermediate rollout/review JSON Lines files and are counted by the
canonical assembler, but do not enter the immutable state bank.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.adapters.dora import inspect_dora_adapter_payload
from src.config.fingerprint import sha256_file, sha256_json
from src.config.inference import load_infer_config
from src.inference.backend import token_ids_sha256
from src.qwen.special_token_embeddings import inspect_special_token_embedding_delta_payload
from src.rollout_calibration import (
    CheckpointIdentity,
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)


SCHEMA_VERSION = "exact_trace_coordinate_state_bank_assembler.v1"
PROMPT_CONTRACT_SCHEMA_VERSION = "coordexp.rollout_calibration.prompt_contract.v1"
REVIEW_SCHEMA_VERSION = "exact_trace_coordinate_visual_review.v1"
PACKET_SCHEMA_VERSION = "exact_trace_coordinate_review_packet.v1"
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670
DEFAULT_IMAGE_PAD_TOKEN_ID = 151655
BLIND_IMAGE_IDS = {
    "1584",
    "2685",
    "4134",
    "5001",
    "6040",
    "7511",
    "10707",
    "13348",
    "13923",
    "14038",
    "14439",
    "16228",
}
COORDINATE_ORDER = ("x1", "y1", "x2", "y2")
COORDINATE_AXIS = {
    "x1": "horizontal",
    "x2": "horizontal",
    "y1": "vertical",
    "y2": "vertical",
}


class AssemblyError(ValueError):
    """Raised when source evidence cannot support a coordinate state bank."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"invalid JSON: {path}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssemblyError(f"immutable assembler output already exists: {path}")
    path.write_bytes(payload)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssemblyError(f"{name} must be a list")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssemblyError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AssemblyError(f"{name} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AssemblyError(f"{name} must be a SHA-256 hex string") from exc
    return value


def _int(value: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssemblyError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise AssemblyError(f"{name} is below {minimum}")
    if maximum is not None and value > maximum:
        raise AssemblyError(f"{name} is above {maximum}")
    return value


def _require_hash(value: Sequence[int], declared: Any, name: str) -> str:
    actual = token_ids_sha256(value)
    if declared != actual:
        raise AssemblyError(f"{name} hash mismatch: expected {actual}, got {declared}")
    return actual


def _image_id(value: Any, name: str) -> str:
    if isinstance(value, bool):
        raise AssemblyError(f"{name} must be an image id")
    text = str(value)
    try:
        return str(int(text))
    except ValueError as exc:
        raise AssemblyError(f"{name} must be an image id") from exc


def _validate_bbox(value: Any, name: str) -> list[int]:
    values = _list(value, name)
    if len(values) != 4:
        raise AssemblyError(f"{name} must have four coordinate bins")
    result: list[int] = []
    for index, item in enumerate(values):
        # Exact traces may carry JSON ``403.0`` values even though the
        # coordinate-bin contract is integer-valued.  Accept only integral
        # floats; never round a genuinely fractional boundary.
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        result.append(_int(item, f"{name}[{index}]", minimum=0, maximum=999))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise AssemblyError(f"{name} must be a non-empty xyxy box")
    return result


def _single_image_trace(trace_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _mapping(_read_json(trace_path), f"trace {trace_path}")
    images = _list(document.get("images"), f"trace {trace_path}.images")
    if len(images) != 1:
        raise AssemblyError(f"{trace_path}: expected exactly one image")
    image = dict(_mapping(images[0], f"trace {trace_path}.images[0]"))
    extended = _mapping(image.get("extended_root_greedy"), f"{trace_path}.extended_root_greedy")
    rows = _list(extended.get("rows"), f"{trace_path}.extended_root_greedy.rows")
    return document, {"image": image, "rows": rows}


def _find_row(rows: Sequence[Any], row_index: int, *, case_id: str) -> dict[str, Any]:
    matches = [item for item in rows if isinstance(item, Mapping) and item.get("row_index") == row_index]
    if len(matches) != 1:
        raise AssemblyError(f"{case_id}: expected one exact row {row_index}, found {len(matches)}")
    return dict(matches[0])


def _validate_row(row: Mapping[str, Any], *, case_id: str) -> None:
    if row.get("accepted_complete_row") is not True or row.get("status") != "success":
        raise AssemblyError(f"{case_id}: row is not an accepted complete successful row")
    parsed = _list(row.get("parsed_predictions"), f"{case_id}.parsed_predictions")
    evidence = _mapping(row.get("parse_evidence"), f"{case_id}.parse_evidence")
    predictions = _list(evidence.get("predictions"), f"{case_id}.parse_evidence.predictions")
    if evidence.get("parse_status") != "accepted" or len(parsed) != 1 or len(predictions) != 1:
        raise AssemblyError(f"{case_id}: coordinate case must have one accepted parsed prediction")
    if parsed[0] != predictions[0]:
        raise AssemblyError(f"{case_id}: parsed prediction disagrees with parse evidence")
    owners = _list(row.get("strict_matched_owner_ids"), f"{case_id}.strict_matched_owner_ids")
    if len(owners) != 1 or not isinstance(owners[0], str) or not owners[0]:
        raise AssemblyError(f"{case_id}: coordinate case must have one strict owner")


def _coord_offsets(token_ids: Sequence[int], *, case_id: str, expected_bins: Sequence[int]) -> list[int]:
    offsets = [index for index, token in enumerate(token_ids) if COORDINATE_TOKEN_START <= token < COORDINATE_TOKEN_END]
    if len(offsets) != 4:
        raise AssemblyError(f"{case_id}: exact candidate must contain four coordinate tokens")
    actual_bins = [token_ids[index] - COORDINATE_TOKEN_START for index in offsets]
    if list(expected_bins) != actual_bins:
        raise AssemblyError(f"{case_id}: parsed coordinate bins disagree with exact token IDs")
    return offsets


def _pad_interval(prompt_ids: Sequence[int], *, image_pad_token_id: int, case_id: str) -> tuple[int, int]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(prompt_ids):
        if token == image_pad_token_id and start is None:
            start = index
        elif token != image_pad_token_id and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(prompt_ids)))
    if len(runs) != 1:
        raise AssemblyError(f"{case_id}: expected one contiguous image-pad run, found {runs}")
    return runs[0]


def _validate_trace_prompt_identity(
    image: Mapping[str, Any], prompt: Mapping[str, Any], prompt_ids: Sequence[int], *, case_id: str
) -> None:
    """Validate the trace's per-image prompt/execution identity without conflating it with bank identity.

    ``prompt_identity_sha256`` in the state-bank manifest identifies the canonical
    training prompt contract.  A trace's ``prompt_token_ids_sha256`` identifies
    this exact executed image prompt and is expected to vary with image media
    expansion.  They are deliberately checked as separate values.
    """

    prompt_hash = _require_hash(
        prompt_ids, prompt.get("prompt_token_ids_sha256"), f"{case_id}.prompt"
    )
    identity = _mapping(image.get("identity_check"), f"{case_id}.identity_check")
    if identity.get("passed") is not True:
        raise AssemblyError(f"{case_id}: exact trace identity_check did not pass")
    checks = _mapping(identity.get("checks"), f"{case_id}.identity_check.checks")
    required_checks = (
        "chat_text_sha256",
        "executed_media_sha256",
        "executed_prompt_token_ids_sha256",
        "height",
        "image_sha256",
        "observed_image_grid_thw",
        "prompt_token_ids_sha256",
        "width",
    )
    if any(checks.get(key) is not True for key in required_checks):
        raise AssemblyError(f"{case_id}: exact trace identity checks are incomplete")
    expected = _mapping(identity.get("expected"), f"{case_id}.identity_check.expected")
    observed = _mapping(identity.get("observed"), f"{case_id}.identity_check.observed")
    if expected != observed:
        raise AssemblyError(f"{case_id}: exact trace identity expected/observed values differ")
    for field in ("prompt_token_ids_sha256", "executed_prompt_token_ids_sha256"):
        if observed.get(field) != prompt_hash:
            raise AssemblyError(f"{case_id}: identity_check {field} disagrees with prompt token hash")
    if prompt.get("chat_text_sha256") != observed.get("chat_text_sha256"):
        raise AssemblyError(f"{case_id}: prompt chat-text hash disagrees with identity_check")
    if prompt.get("image_sha256") != observed.get("image_sha256"):
        raise AssemblyError(f"{case_id}: prompt image hash disagrees with identity_check")
    if prompt.get("width") != observed.get("width") or prompt.get("height") != observed.get("height"):
        raise AssemblyError(f"{case_id}: prompt dimensions disagree with identity_check")


def _load_prompt_contract(
    path: Path, *, expected_prompt_identity_sha256: str
) -> dict[str, Any]:
    """Load and bind the canonical prompt contract used by the bank.

    The contract digest is intentionally checked independently from the
    per-image executed prompt-token digest.  A trace can be internally
    self-consistent while still having been produced with a different
    canonical chat contract; the latter must be rejected by the frozen trace
    binding below.
    """

    payload = _mapping(_read_json(path), "prompt contract")
    if payload.get("schema_version") != PROMPT_CONTRACT_SCHEMA_VERSION:
        raise AssemblyError(
            "unsupported prompt contract schema: "
            f"{payload.get('schema_version')!r}"
        )
    contract = _mapping(payload.get("contract"), "prompt contract.contract")
    declared = _sha256(
        payload.get("prompt_identity_sha256"), "prompt contract.prompt_identity_sha256"
    )
    actual = sha256_json(contract)
    if declared != actual:
        raise AssemblyError(
            "prompt contract identity mismatch: "
            f"expected {actual}, got {declared}"
        )
    if declared != expected_prompt_identity_sha256:
        raise AssemblyError(
            "prompt contract identity differs from the state-bank identity: "
            f"expected {expected_prompt_identity_sha256}, got {declared}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": str(payload["schema_version"]),
        "prompt_identity_sha256": declared,
        "contract": dict(contract),
    }


def _load_frozen_trace_identity(
    path: Path, *, canonical_prompt_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Read expected trace/config identity from one explicitly trusted trace.

    Candidate traces are never used to establish these expectations.  The
    caller chooses this trace before assembly and its file digest is retained
    in the receipt, so a foreign or self-consistent wrong-prompt candidate
    cannot silently redefine the canonical configuration or chat text.
    """

    document, trace_data = _single_image_trace(path)
    image = trace_data["image"]
    prompt = _mapping(image.get("prompt"), "trusted trace.prompt")
    prompt_ids = _list(prompt.get("prompt_token_ids"), "trusted trace.prompt_token_ids")
    prompt_ids = [
        _int(token, f"trusted trace.prompt_token_ids[{index}]")
        for index, token in enumerate(prompt_ids)
    ]
    _validate_trace_prompt_identity(image, prompt, prompt_ids, case_id="trusted-trace")
    config = _mapping(document.get("config"), "trusted trace.config")
    resolved_config_fingerprint = _sha256(
        config.get("resolved_config_fingerprint"),
        "trusted trace.config.resolved_config_fingerprint",
    )
    infer_config_path = Path(
        _string(config.get("infer_config"), "trusted trace.config.infer_config")
    ).expanduser().resolve(strict=True)
    resolved_infer = load_infer_config(infer_config_path)
    if resolved_infer.fingerprint != resolved_config_fingerprint:
        raise AssemblyError(
            "trusted trace resolved-config fingerprint does not match its "
            "recorded inference config"
        )
    template = resolved_infer.config.template
    observed_prompt_contract = {
        "assistant_format": template.assistant_format,
        "object_field_order": template.object_field_order,
        "object_ordering": template.object_ordering,
        "system": template.prompt.system,
        "user": template.prompt.user,
    }
    if observed_prompt_contract != dict(canonical_prompt_contract):
        raise AssemblyError(
            "trusted trace inference config does not instantiate the canonical "
            "prompt contract"
        )
    chat_text_sha256 = _sha256(
        prompt.get("chat_text_sha256"), "trusted trace.prompt.chat_text_sha256"
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "image_id": _image_id(image.get("image_id"), "trusted trace.image_id"),
        "resolved_config_fingerprint": resolved_config_fingerprint,
        "chat_text_sha256": chat_text_sha256,
        "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "infer_config_path": str(infer_config_path),
        "infer_config_sha256": sha256_file(infer_config_path),
        "resolved_prompt_contract_sha256": sha256_json(observed_prompt_contract),
    }


def _validate_source_trace_binding(
    document: Mapping[str, Any], image: Mapping[str, Any],
    frozen_trace_identity: Mapping[str, Any], *, case_id: str,
) -> dict[str, str]:
    """Require a candidate trace to match the trusted config and chat text."""

    config = _mapping(document.get("config"), f"{case_id}.config")
    observed_config = _sha256(
        config.get("resolved_config_fingerprint"),
        f"{case_id}.config.resolved_config_fingerprint",
    )
    expected_config = _sha256(
        frozen_trace_identity.get("resolved_config_fingerprint"),
        "frozen trace.resolved_config_fingerprint",
    )
    if observed_config != expected_config:
        raise AssemblyError(
            f"{case_id}: resolved config fingerprint differs from trusted trace "
            f"(expected {expected_config}, got {observed_config})"
        )
    prompt = _mapping(image.get("prompt"), f"{case_id}.prompt")
    observed_chat = _sha256(
        prompt.get("chat_text_sha256"), f"{case_id}.prompt.chat_text_sha256"
    )
    expected_chat = _sha256(
        frozen_trace_identity.get("chat_text_sha256"),
        "frozen trace.chat_text_sha256",
    )
    if observed_chat != expected_chat:
        raise AssemblyError(
            f"{case_id}: canonical chat-text hash differs from trusted trace "
            f"(expected {expected_chat}, got {observed_chat})"
        )
    return {
        "resolved_config_fingerprint": observed_config,
        "chat_text_sha256": observed_chat,
    }


def _trace_checkpoint_identity(
    document: Mapping[str, Any], expected_checkpoint: CheckpointIdentity, *, case_id: str
) -> dict[str, str]:
    """Verify adapter and embedding payload identities recorded by an exact trace."""

    model_identity = _mapping(document.get("model_identity"), f"{case_id}.model_identity")
    model = _mapping(model_identity.get("model_identity"), f"{case_id}.model_identity.model_identity")
    adapter = _mapping(model.get("adapter"), f"{case_id}.adapter")
    adapter_path = Path(str(adapter.get("adapter_path"))).resolve(strict=True)
    base_model_path = adapter.get("base_model_path")
    observed_adapter = inspect_dora_adapter_payload(
        adapter_path, expected_base_model_path=base_model_path
    )
    if observed_adapter["fingerprint"] != expected_checkpoint.adapter_fingerprint:
        raise AssemblyError(
            f"{case_id}: exact trace adapter fingerprint differs from reference checkpoint"
        )
    embedding = _mapping(model.get("embedding_delta"), f"{case_id}.embedding_delta")
    embedding_identity = _mapping(embedding.get("identity"), f"{case_id}.embedding_delta.identity")
    embedding_path = Path(str(embedding_identity.get("delta_path"))).resolve(strict=True)
    observed_embedding = inspect_special_token_embedding_delta_payload(
        embedding_path,
        expected_base_model_path=base_model_path,
        expected_base_config_sha256=expected_checkpoint.base_config_sha256,
        expected_tokenizer_sha256=expected_checkpoint.tokenizer_sha256,
    )
    if observed_embedding["fingerprint"] != expected_checkpoint.embedding_delta_fingerprint:
        raise AssemblyError(
            f"{case_id}: exact trace embedding fingerprint differs from reference checkpoint"
        )
    frozen = _mapping(document.get("frozen_inputs"), f"{case_id}.frozen_inputs")
    checkpoint_sources = _mapping(
        frozen.get("checkpoint_config_source_identity"),
        f"{case_id}.frozen_inputs.checkpoint_config_source_identity",
    )
    observed_source_files: dict[str, str] = {}
    for key in ("adapter_model", "special_token_embeddings"):
        source = _mapping(checkpoint_sources.get(key), f"{case_id}.{key}")
        source_path = Path(str(source.get("path"))).resolve(strict=True)
        declared_sha = _sha256(source.get("sha256"), f"{case_id}.{key}.sha256")
        actual_sha = sha256_file(source_path)
        if actual_sha != declared_sha:
            raise AssemblyError(f"{case_id}: exact trace source {key} hash mismatch")
        observed_source_files[key] = actual_sha
    return {
        "adapter_fingerprint": observed_adapter["fingerprint"],
        "embedding_delta_fingerprint": observed_embedding["fingerprint"],
        **{f"{key}_sha256": value for key, value in observed_source_files.items()},
    }


def _physical_entities(image: Mapping[str, Any], *, case_id: str) -> list[dict[str, Any]]:
    ledger = _list(image.get("entity_ledger"), f"{case_id}.entity_ledger")
    entities: list[dict[str, Any]] = []
    for index, raw in enumerate(ledger):
        item = _mapping(raw, f"{case_id}.entity_ledger[{index}]")
        entity_id = item.get("entity_id")
        category = item.get("description", item.get("category"))
        bbox = item.get("bbox_norm1000", item.get("reference_bbox"))
        verification = item.get("verification")
        # The exact-trace ledger predates the live state-bank field names.  Its
        # only admissible trust declaration is the literal ``verified`` value;
        # anything else is rejected rather than silently promoted.
        if item.get("entity_trusted") is None and item.get("geometry_trusted") is None:
            if verification != "verified":
                raise AssemblyError(f"{case_id}: entity ledger lacks an explicit trust declaration")
            entity_trusted = geometry_trusted = True
        else:
            entity_trusted = item.get("entity_trusted")
            geometry_trusted = item.get("geometry_trusted")
        required = (entity_id, category, bbox, entity_trusted, geometry_trusted)
        if not isinstance(entity_id, str) or not isinstance(category, str) or any(value is None for value in required[2:]):
            raise AssemblyError(f"{case_id}: entity ledger lacks explicit trust and geometry fields")
        if not isinstance(entity_trusted, bool) or not isinstance(geometry_trusted, bool):
            raise AssemblyError(f"{case_id}: entity trust fields must be explicit booleans")
        entities.append(
            {
                "entity_id": entity_id,
                "category": category,
                "entity_trusted": entity_trusted,
                "geometry_trusted": geometry_trusted,
                "reference_bbox": _validate_bbox(bbox, f"{case_id}.entity_ledger[{index}].bbox"),
                "review_source": str(item.get("review_source", "exact_trace_ledger")),
                "reviewer": str(item.get("reviewer", "exact_trace_ledger")),
                "review_confidence": str(item.get("review_confidence", "trace_bound")),
                "comment": str(item.get("comment", "")),
            }
        )
    if not entities or len({item["entity_id"] for item in entities}) != len(entities):
        raise AssemblyError(f"{case_id}: entity ledger must be non-empty and unique")
    return entities


def _validate_prior_raw_tokens(row: Mapping[str, Any], *, case_id: str) -> list[int]:
    """Validate one prior row's stored token hashes for prefix reconstruction."""

    prefix = _list(row.get("prefix_token_ids"), f"{case_id}.prefix_token_ids")
    raw = _list(row.get("raw_generated_token_ids"), f"{case_id}.raw_generated_token_ids")
    prefix = [_int(token, f"{case_id}.prefix_token_ids[{i}]") for i, token in enumerate(prefix)]
    raw = [_int(token, f"{case_id}.raw_generated_token_ids[{i}]") for i, token in enumerate(raw)]
    _require_hash(prefix, row.get("prefix_token_ids_sha256"), f"{case_id}.prefix")
    _require_hash(raw, row.get("raw_generated_token_ids_sha256"), f"{case_id}.raw")
    return raw


def _prefix_owner_proofs(
    rows: Sequence[Any], row_index: int, entities: Sequence[Mapping[str, Any]], trace_path: Path, *, case_id: str
) -> list[dict[str, Any]]:
    entity_ids = {str(item["entity_id"]) for item in entities}
    proofs: list[dict[str, Any]] = []
    prior_raw: list[int] = []
    for index in range(row_index):
        prior = _find_row(rows, index, case_id=case_id)
        _validate_row(prior, case_id=f"{case_id}.prefix-row-{index}")
        if prior.get("row_index") != index:
            raise AssemblyError(f"{case_id}: prefix row indices are not contiguous")
        prefix = _list(prior.get("prefix_token_ids"), f"{case_id}.prefix-row-{index}.prefix_token_ids")
        raw = _list(prior.get("raw_generated_token_ids"), f"{case_id}.prefix-row-{index}.raw_generated_token_ids")
        _require_hash(prefix, prior.get("prefix_token_ids_sha256"), f"{case_id}.prefix-row-{index}.prefix")
        _require_hash(raw, prior.get("raw_generated_token_ids_sha256"), f"{case_id}.prefix-row-{index}.raw")
        if prefix != prior_raw:
            raise AssemblyError(f"{case_id}: exact prefix is not the concatenation of prior rows")
        prior_raw.extend(raw)
        owners = _list(prior.get("strict_matched_owner_ids"), f"{case_id}.prefix-row-{index}.strict owners")
        if len(owners) != 1 or owners[0] not in entity_ids:
            raise AssemblyError(f"{case_id}: prefix row {index} has no trusted ledger owner")
        coverage = _mapping(prior.get("coverage_receipt"), f"{case_id}.prefix-row-{index}.coverage_receipt")
        if coverage.get("coverage_updated") is not True or coverage.get("owner_id") != owners[0]:
            raise AssemblyError(f"{case_id}: prefix row {index} lacks a committed coverage receipt")
        proofs.append(
            {
                "prefix_object_row_index": index,
                "owner_id": owners[0],
                "review_provenance": {
                    "source": f"{trace_path}#extended_root_greedy.rows[{index}]",
                    "reviewer": "exact_trace_coordinate_assembler",
                    "confidence": "mechanically_verified",
                    "comment": "Owner copied from an accepted exact trace row and bound to the selected physical ledger.",
                },
            }
        )
    return proofs


def _load_reference_identity(
    reference_manifest: Path | None, source_checkpoint_json: Path | None, prompt_identity_sha256: str | None
) -> tuple[CheckpointIdentity, str, list[dict[str, str]], dict[str, Any]]:
    if reference_manifest is not None:
        binding = load_state_bank_manifest_binding(reference_manifest)
        return (
            binding.source_checkpoint,
            binding.prompt_identity_sha256,
            [{"artifact_id": "reference-state-bank-manifest", "sha256": sha256_file(reference_manifest)}],
            {"reference_manifest": str(reference_manifest.resolve()), "source_checkpoint_id": binding.source_checkpoint_id},
        )
    if source_checkpoint_json is None or prompt_identity_sha256 is None:
        raise AssemblyError("provide --reference-bank-manifest or both explicit source identity arguments")
    payload = _read_json(source_checkpoint_json)
    if isinstance(payload, Mapping) and "source_checkpoint" in payload:
        payload = payload["source_checkpoint"]
    checkpoint = CheckpointIdentity.from_mapping(_mapping(payload, "source checkpoint identity"))
    _sha256(prompt_identity_sha256, "prompt_identity_sha256")
    return checkpoint, prompt_identity_sha256, [], {"explicit_source_checkpoint_json": str(source_checkpoint_json.resolve())}


def _case_rows(
    packet_case: Mapping[str, Any], decision: Mapping[str, Any], *, reference_checkpoint_id: str,
    reference_checkpoint: CheckpointIdentity, image_pad_token_id: int, packet_path: Path,
    decisions_path: Path, frozen_trace_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    case_id = str(packet_case.get("case_id"))
    if case_id != str(decision.get("case_id")):
        raise AssemblyError(f"{case_id}: decision case id mismatch")
    trace_path = Path(str(packet_case.get("exact_trace_path"))).resolve(strict=True)
    document, trace_data = _single_image_trace(trace_path)
    image = trace_data["image"]
    rows = trace_data["rows"]
    image_id = _image_id(image.get("image_id"), f"{case_id}.image_id")
    if image_id in BLIND_IMAGE_IDS:
        raise AssemblyError(f"{case_id}: blind image is excluded")
    if image_id != str(packet_case.get("image_id")):
        raise AssemblyError(f"{case_id}: packet image identity differs from exact trace")
    row_index = _int(packet_case.get("row_index"), f"{case_id}.row_index", minimum=0)
    row = _find_row(rows, row_index, case_id=case_id)
    _validate_row(row, case_id=case_id)
    trace_binding = _validate_source_trace_binding(
        document, image, frozen_trace_identity, case_id=case_id
    )
    trace_checkpoint_evidence = _trace_checkpoint_identity(
        document, reference_checkpoint, case_id=case_id
    )
    prompt = _mapping(image.get("prompt"), f"{case_id}.prompt")
    prompt_ids = _list(prompt.get("prompt_token_ids"), f"{case_id}.prompt_token_ids")
    prompt_ids = [_int(token, f"{case_id}.prompt_token_ids[{i}]") for i, token in enumerate(prompt_ids)]
    prompt_hash = _require_hash(prompt_ids, prompt.get("prompt_token_ids_sha256"), f"{case_id}.prompt")
    _validate_trace_prompt_identity(image, prompt, prompt_ids, case_id=case_id)
    image_pad_interval = _pad_interval(prompt_ids, image_pad_token_id=image_pad_token_id, case_id=case_id)
    prefix = _list(row.get("prefix_token_ids"), f"{case_id}.prefix_token_ids")
    raw = _list(row.get("raw_generated_token_ids"), f"{case_id}.raw_generated_token_ids")
    prefix = [_int(token, f"{case_id}.prefix_token_ids[{i}]") for i, token in enumerate(prefix)]
    raw = [_int(token, f"{case_id}.raw_generated_token_ids[{i}]") for i, token in enumerate(raw)]
    prefix_hash = _require_hash(prefix, row.get("prefix_token_ids_sha256"), f"{case_id}.prefix")
    raw_hash = _require_hash(raw, row.get("raw_generated_token_ids_sha256"), f"{case_id}.raw")
    if prefix != packet_case.get("exact_prefix_token_ids") or prefix_hash != packet_case.get("exact_prefix_token_ids_sha256"):
        raise AssemblyError(f"{case_id}: packet exact prefix disagrees with trace")
    if raw != packet_case.get("exact_raw_generated_token_ids") or raw_hash != packet_case.get("exact_raw_generated_token_ids_sha256"):
        raise AssemblyError(f"{case_id}: packet exact candidate tokens disagree with trace")
    parsed = _list(row["parsed_predictions"], f"{case_id}.parsed_predictions")
    prediction = _mapping(parsed[0], f"{case_id}.parsed_predictions[0]")
    bins = _validate_bbox(prediction.get("coord_bins"), f"{case_id}.coord_bins")
    offsets = _coord_offsets(raw, case_id=case_id, expected_bins=bins)
    entities = _physical_entities(image, case_id=case_id)
    owner_id = str(packet_case.get("owner_id"))
    strict_owner = str(_list(row.get("strict_matched_owner_ids"), f"{case_id}.strict owners")[0])
    if owner_id != strict_owner:
        raise AssemblyError(f"{case_id}: packet owner differs from exact strict owner")
    owner = next((item for item in entities if item["entity_id"] == owner_id), None)
    if owner is None or owner["category"] != str(prediction.get("description")):
        raise AssemblyError(f"{case_id}: owner/category binding is not trusted")
    status = decision.get("status")
    if status not in {"accepted_for_coordinate_objective", "rejected"}:
        raise AssemblyError(f"{case_id}: unsupported review status {status!r}")
    event_id = f"coordinate_boundary_{case_id}"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "reviewer": decision.get("reviewer") or _read_json(decisions_path).get("reviewer", "unknown"),
        "review_status": status,
        "review_confidence": decision.get("review_confidence"),
        "comment": decision.get("comment", ""),
        "source_packet": str(packet_path.resolve()),
        "source_packet_sha256": sha256_file(packet_path),
        "source_decisions": str(decisions_path.resolve()),
        "source_decisions_sha256": sha256_file(decisions_path),
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256_file(trace_path),
        "trace_checkpoint_evidence": trace_checkpoint_evidence,
        "trace_binding": trace_binding,
        "trusted_trace": {
            "path": str(frozen_trace_identity["path"]),
            "sha256": str(frozen_trace_identity["sha256"]),
        },
    }
    rollout_candidate = {
        "candidate_id": f"coordinate-diagnostic-greedy-{case_id}",
        "token_ids": raw,
        "token_ids_sha256": raw_hash,
        "generation_provenance": {
            "mode": "greedy",
            "seed": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": float(document.get("config", {}).get("repetition_penalty", 1.0)),
            "checkpoint_id": reference_checkpoint_id,
            "prompt_token_ids_sha256": prompt_hash,
            "prefix_token_ids_sha256": prefix_hash,
        },
        "evidence_text": str(row.get("raw_generated_text", "")),
    }
    rollout = {
        "event_id": event_id,
        "image": {
            "image_id": int(image_id),
            "path": str(prompt.get("image_path")),
            "width": _int(prompt.get("width", image.get("width")), f"{case_id}.width", minimum=1),
            "height": _int(prompt.get("height", image.get("height")), f"{case_id}.height", minimum=1),
            "content_sha256": _sha256(prompt.get("image_sha256"), f"{case_id}.image_sha256"),
        },
        "split": str(decision.get("split_override", packet_case.get("split_role"))),
        "split_group_id": f"image:{image_id}",
        "executed_prompt_token_ids": prompt_ids,
        "executed_prompt_token_ids_sha256": prompt_hash,
        "image_pad_interval": list(image_pad_interval),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": prefix_hash,
        "candidates": [rollout_candidate],
    }
    proofs: list[dict[str, Any]] = []
    prefix_coverage_status = "empty" if row_index == 0 else "unresolved"
    prefix_resolution_error: str | None = None
    if status != "rejected":
        try:
            expected_prefix = [
                token
                for index in range(row_index)
                for token in _validate_prior_raw_tokens(
                    _find_row(rows, index, case_id=case_id),
                    case_id=f"{case_id}.prefix-row-{index}",
                )
            ]
            if prefix != expected_prefix:
                raise AssemblyError(f"{case_id}: prefix reconstruction mismatch")
        except AssemblyError as exc:
            # A malformed token prefix changes the actual replay input and is
            # never admitted.  An otherwise exact prefix with one or more
            # ambiguous owner rows is different: coordinate-only supervision
            # does not use a covered-set objective, so leave the owner proof
            # empty rather than inventing coverage or rejecting the reviewed
            # current-row coordinate boundary.
            rejected_review = {
                "event_id": event_id,
                "admission_status": "rejected",
                "rejection_reason": f"assembler_rejected:{exc}",
                "physical_entities": entities,
                "prefix_covered_owner_proofs": [],
                "prefix_object_row_count": row_index,
                "prefix_coverage_status": prefix_coverage_status,
                "entity_transition_eligible": False,
                "coordinate_boundary_eligible": False,
                "candidates": [],
                "review_provenance": {
                    **provenance,
                    "assembly_rejection_reason": str(exc),
                },
            }
            return rollout, rejected_review, {"artifact_id": f"exact-trace-{image_id}", "sha256": sha256_file(trace_path)}
        if prefix_resolution_error is None and row_index > 0:
            try:
                proofs = _prefix_owner_proofs(rows, row_index, entities, trace_path, case_id=case_id)
                prefix_coverage_status = "resolved"
            except AssemblyError as exc:
                prefix_resolution_error = str(exc)
    provenance["prefix_coverage_status"] = prefix_coverage_status
    if prefix_resolution_error is not None:
        provenance["prefix_coverage_resolution_error"] = prefix_resolution_error
    if status == "accepted_for_coordinate_objective":
        axis = str(decision.get("first_wrong_coordinate"))
        if axis != str(packet_case.get("proposed_review_axis")) or axis not in COORDINATE_ORDER:
            raise AssemblyError(f"{case_id}: first wrong coordinate does not match packet axis")
        ranges = _mapping(decision.get("accepted_coordinate_ranges_inclusive"), f"{case_id}.accepted ranges")
        expected_axes = list(COORDINATE_ORDER[: COORDINATE_ORDER.index(axis) + 1])
        if list(ranges) != expected_axes:
            raise AssemblyError(f"{case_id}: accepted ranges must cover x1..first wrong coordinate")
        observations = []
        for index, coordinate in enumerate(expected_axes):
            bounds = _list(ranges.get(coordinate), f"{case_id}.{coordinate} range")
            if len(bounds) != 2:
                raise AssemblyError(f"{case_id}.{coordinate} range must be inclusive [low, high]")
            low = _int(bounds[0], f"{case_id}.{coordinate}.low", minimum=0, maximum=999)
            high = _int(bounds[1], f"{case_id}.{coordinate}.high", minimum=0, maximum=999)
            if low > high:
                raise AssemblyError(f"{case_id}.{coordinate} range is reversed")
            acceptable = list(range(low, high + 1))
            actual = bins[index]
            if index < len(expected_axes) - 1 and actual not in acceptable:
                raise AssemblyError(f"{case_id}: earlier coordinate {coordinate} is not accepted")
            if index == len(expected_axes) - 1 and actual in acceptable:
                raise AssemblyError(f"{case_id}: final coordinate {coordinate} is not the first wrong boundary")
            observations.append(
                {
                    "coordinate": coordinate,
                    "tolerance_axis": COORDINATE_AXIS[coordinate],
                    "candidate_token_offset": offsets[index],
                    "actual_coordinate_value": actual,
                    "acceptable_coordinate_values": acceptable,
                    "review_provenance": {
                        "source": str(decisions_path.resolve()),
                        "reviewer": provenance["reviewer"],
                        "confidence": str(decision.get("review_confidence", "unspecified")),
                        "comment": str(decision.get("comment", "")),
                    },
                }
            )
        review_candidate = {
            "candidate_id": rollout_candidate["candidate_id"],
            "role": "diagnostic",
            "harmful_kind": None,
            "physical_owner_id": owner_id,
            "coverage_status": "unknown",
            "entity_review_status": "trusted",
            "geometry_review_status": "trusted",
            "entity_eligible": False,
            "geometry_eligible": True,
            "owner_resolution_interval": None,
            "coordinate_decision": {"owner_id": owner_id, "observations": observations},
            "selected_sites": [{"candidate_token_offset": offsets[len(observations) - 1], "intended_token_type": "coordinate"}],
        }
        review = {
            "event_id": event_id,
            "admission_status": "accepted",
            "rejection_reason": None,
            "physical_entities": entities,
            "prefix_covered_owner_proofs": proofs,
            "prefix_object_row_count": row_index,
            "prefix_coverage_status": prefix_coverage_status,
            "entity_transition_eligible": False,
            "coordinate_boundary_eligible": True,
            "candidates": [review_candidate],
            "review_provenance": provenance,
        }
    else:
        review = {
            "event_id": event_id,
            "admission_status": "rejected",
            "rejection_reason": str(decision.get("comment") or "review_rejected"),
            "physical_entities": entities,
            "prefix_covered_owner_proofs": proofs,
            "prefix_object_row_count": row_index,
            "prefix_coverage_status": prefix_coverage_status,
            "entity_transition_eligible": False,
            "coordinate_boundary_eligible": False,
            "candidates": [],
            "review_provenance": provenance,
        }
    artifact = {"artifact_id": f"exact-trace-{image_id}", "sha256": sha256_file(trace_path)}
    return rollout, review, artifact


def assemble_coordinate_state_bank(
    *, packet_path: str | Path, decisions_path: str | Path, output_dir: str | Path,
    reference_bank_manifest: str | Path | None = None, source_checkpoint_json: str | Path | None = None,
    prompt_identity_sha256: str | None = None, prompt_contract_path: str | Path | None = None,
    trusted_trace_path: str | Path | None = None,
    image_pad_token_id: int = DEFAULT_IMAGE_PAD_TOKEN_ID,
) -> dict[str, Any]:
    packet_file = Path(packet_path).expanduser().resolve(strict=True)
    decisions_file = Path(decisions_path).expanduser().resolve(strict=True)
    packet = _mapping(_read_json(packet_file), "review packet")
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        raise AssemblyError(f"unsupported review packet schema: {packet.get('schema_version')!r}")
    decisions = _mapping(_read_json(decisions_file), "review decisions")
    if decisions.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise AssemblyError(f"unsupported review decisions schema: {decisions.get('schema_version')!r}")
    source_packet = _mapping(decisions.get("source_packet"), "review decisions.source_packet")
    if Path(str(source_packet.get("path"))).resolve() != packet_file or source_packet.get("sha256") != sha256_file(packet_file):
        raise AssemblyError("review decisions do not bind the exact review packet")
    cases = _list(packet.get("cases"), "review packet.cases")
    decision_rows = _list(decisions.get("decisions"), "review decisions.decisions")
    case_by_id = {str(item.get("case_id")): item for item in cases}
    decision_by_id = {str(item.get("case_id")): item for item in decision_rows}
    if len(case_by_id) != len(cases) or len(decision_by_id) != len(decision_rows):
        raise AssemblyError("packet and decisions case identifiers must be unique")
    if set(case_by_id) != set(decision_by_id):
        raise AssemblyError("packet and decisions must cover exactly the same cases")
    checkpoint, prompt_hash, source_artifacts, identity_info = _load_reference_identity(
        Path(reference_bank_manifest).expanduser().resolve(strict=True) if reference_bank_manifest else None,
        Path(source_checkpoint_json).expanduser().resolve(strict=True) if source_checkpoint_json else None,
        prompt_identity_sha256,
    )
    if prompt_contract_path is None:
        raise AssemblyError(
            "provide --prompt-contract; a bank must bind the canonical prompt artifact"
        )
    if trusted_trace_path is None:
        raise AssemblyError(
            "provide --trusted-trace; expected config and chat identity must come "
            "from an explicitly frozen trusted trace"
        )
    prompt_contract_file = Path(prompt_contract_path).expanduser().resolve(strict=True)
    trusted_trace_file = Path(trusted_trace_path).expanduser().resolve(strict=True)
    prompt_contract_identity = _load_prompt_contract(
        prompt_contract_file, expected_prompt_identity_sha256=prompt_hash
    )
    frozen_trace_identity = _load_frozen_trace_identity(
        trusted_trace_file,
        canonical_prompt_contract=_mapping(
            prompt_contract_identity.get("contract"),
            "prompt contract.contract",
        ),
    )
    source_artifacts.extend(
        [
            {
                "artifact_id": "prompt-contract",
                "sha256": prompt_contract_identity["sha256"],
            },
            {
                "artifact_id": "trusted-exact-trace",
                "sha256": frozen_trace_identity["sha256"],
            },
            {
                "artifact_id": "trusted-inference-config",
                "sha256": frozen_trace_identity["infer_config_sha256"],
            },
        ]
    )
    checkpoint_id = sha256_json(checkpoint.to_artifact_dict())
    rollout_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    trace_artifacts: dict[str, dict[str, str]] = {}
    for case_id in sorted(case_by_id):
        rollout, review, artifact = _case_rows(
            case_by_id[case_id], decision_by_id[case_id], reference_checkpoint_id=checkpoint_id,
            reference_checkpoint=checkpoint,
            image_pad_token_id=image_pad_token_id,
            packet_path=packet_file, decisions_path=decisions_file,
            frozen_trace_identity=frozen_trace_identity,
        )
        rollout_rows.append(rollout)
        review_rows.append(review)
        trace_artifacts[artifact["artifact_id"]] = artifact
    source_artifacts = [
        *source_artifacts,
        {"artifact_id": "review-packet", "sha256": sha256_file(packet_file)},
        {"artifact_id": "review-decisions", "sha256": sha256_file(decisions_file)},
        *[trace_artifacts[key] for key in sorted(trace_artifacts)],
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rollout_path = destination / "coordinate-rollout-rows.jsonl"
    review_path = destination / "coordinate-review-rows.jsonl"
    _write_once(rollout_path, b"".join(_canonical_bytes(row) + b"\n" for row in rollout_rows))
    _write_once(review_path, b"".join(_canonical_bytes(row) + b"\n" for row in review_rows))
    bank_dir = destination / "state-bank"
    manifest = assemble_state_bank(
        output_dir=bank_dir,
        rollout_rows=rollout_rows,
        review_rows=review_rows,
        source_checkpoint=checkpoint,
        prompt_identity_sha256=prompt_hash,
        source_artifacts=source_artifacts,
    )
    loaded = load_state_bank(
        bank_dir / "manifest.json",
        expected_source_checkpoint=checkpoint,
        expected_prompt_identity_sha256=prompt_hash,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "packet_path": str(packet_file),
        "packet_sha256": sha256_file(packet_file),
        "decisions_path": str(decisions_file),
        "decisions_sha256": sha256_file(decisions_file),
        "rollout_rows_path": str(rollout_path),
        "review_rows_path": str(review_path),
        "accepted_case_ids": [row["event_id"] for row in review_rows if row["admission_status"] == "accepted"],
        "rejected_case_ids": [row["event_id"] for row in review_rows if row["admission_status"] == "rejected"],
        "reference_identity": identity_info,
        "prompt_contract": prompt_contract_identity,
        "trusted_trace": frozen_trace_identity,
        "expected_trace_identity": {
            "resolved_config_fingerprint": frozen_trace_identity[
                "resolved_config_fingerprint"
            ],
            "chat_text_sha256": frozen_trace_identity["chat_text_sha256"],
        },
        "executed_prompt_hash_summary": [
            {
                "event_id": row["event_id"],
                "executed_prompt_token_ids_sha256": row[
                    "executed_prompt_token_ids_sha256"
                ],
                "chat_text_sha256": frozen_trace_identity["chat_text_sha256"],
            }
            for row in sorted(rollout_rows, key=lambda item: str(item["event_id"]))
        ],
        "state_bank": loaded.validation_receipt.to_artifact_dict(),
        "manifest": manifest.to_artifact_dict(),
    }
    receipt_path = destination / "assembly-receipt.json"
    _write_once(receipt_path, json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return {"manifest": manifest.to_artifact_dict(), "receipt": receipt, "output_dir": destination}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-bank-manifest", type=Path)
    parser.add_argument("--source-checkpoint-json", type=Path)
    parser.add_argument("--prompt-identity-sha256")
    parser.add_argument(
        "--prompt-contract",
        type=Path,
        required=True,
        help="canonical prompt-contract JSON artifact bound to the bank identity",
    )
    parser.add_argument(
        "--trusted-trace",
        type=Path,
        required=True,
        help="one frozen exact trace used to establish config and chat identity",
    )
    parser.add_argument("--image-pad-token-id", type=int, default=DEFAULT_IMAGE_PAD_TOKEN_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = assemble_coordinate_state_bank(
        packet_path=args.review_packet,
        decisions_path=args.review_decisions,
        output_dir=args.output_dir,
        reference_bank_manifest=args.reference_bank_manifest,
        source_checkpoint_json=args.source_checkpoint_json,
        prompt_identity_sha256=args.prompt_identity_sha256,
        prompt_contract_path=args.prompt_contract,
        trusted_trace_path=args.trusted_trace,
        image_pad_token_id=args.image_pad_token_id,
    )
    print(json.dumps(result["receipt"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssemblyError",
    "assemble_coordinate_state_bank",
    "main",
    "parse_args",
    "_load_frozen_trace_identity",
    "_load_prompt_contract",
    "_validate_source_trace_binding",
]
