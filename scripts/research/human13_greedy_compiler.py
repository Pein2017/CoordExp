"""Sparse Source-boundary greedy compiler for the Human-13 RP screen.

The compiler is deliberately not a suffix teacher.  It materializes one
first-child comparison per RP-specific Source boundary from the immutable
native alias bank, then consumes only compact raw-logit rows during backward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast
import weakref

import torch

from scripts.research.build_human13_k_union_manifest import (
    Human13KUnionManifest,
    ImageRecord,
)
from scripts.research.build_human13_on_policy_frontier import (
    FrontierCandidateAlias,
    FrontierImage,
    Human13FrontierIteration,
    candidate_aliases_for_owners,
    load_frontier_iteration,
    natural_pre_stop_prefix,
    validate_current_decode_surface,
)
from scripts.research.human13_trajectory_credit import (
    TrajectoryCreditPanelAcquisition,
    TrajectoryCreditLedger,
    _require_scientific_ledger_admission,
)
from src.inference.backend import token_ids_sha256


SCHEMA_VERSION = "human13_source_greedy_compiler.v1"
EXACT_ALIAS_COUNT = 309
KAPPA = 1.0
MARGIN = 1.0e-4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSENT_REASONS = {
    "no_premature_source_boundary",
    "no_trusted_remaining",
    "no_valid_aliases",
}
_SOURCE_PANEL_MARKER = object()
_SOURCE_DECODE_MARKER = object()
_SOURCE_RUNTIME_MARKER = object()
_COMPILER_LEDGER_MARKER = object()
_PACKED_ROW_MARKER = object()
_COMPACT_LOGITS_MARKER = object()
_SOURCE_DECODE_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_SOURCE_RUNTIME_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_SOURCE_PANEL_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_COMPILER_LEDGER_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_PACKED_ROW_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_COMPACT_LOGITS_ADMISSIONS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _token_ids(value: object, *, field: str, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} must be a token-id sequence")
    result = tuple(value)
    if nonempty and not result:
        raise ValueError(f"{field} must not be empty")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in result
    ):
        raise ValueError(f"{field} must contain nonnegative integer token ids")
    return result


def _manifest_sha256(manifest: Human13KUnionManifest) -> str:
    payload = (
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _history_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_bytes({"token_ids": list(token_ids)})).hexdigest()


@dataclass(frozen=True)
class SourceBoundaryInput:
    """One RP-specific sealed Source clean-greedy decode plus its exact prompt."""

    image: FrontierImage
    prompt_token_ids: tuple[int, ...]
    repetition_penalty: float
    image_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.image, FrontierImage):
            raise ValueError("source boundary image must be a FrontierImage")
        object.__setattr__(
            self,
            "prompt_token_ids",
            _token_ids(self.prompt_token_ids, field="prompt_token_ids", nonempty=True),
        )
        repetition_penalty = float(self.repetition_penalty)
        if repetition_penalty not in {1.0, 1.10}:
            raise ValueError("compiler repetition penalty must be exactly 1.0 or 1.10")
        object.__setattr__(self, "repetition_penalty", repetition_penalty)
        object.__setattr__(
            self, "image_sha256", _digest(self.image_sha256, field="image_sha256")
        )

    @property
    def source_decode_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": "human13_source_compiler_boundary.v1",
                "image": asdict(self.image),
                "prompt_token_ids": list(self.prompt_token_ids),
                "repetition_penalty": self.repetition_penalty,
                "image_sha256": self.image_sha256,
            }
        )


@dataclass(frozen=True, init=False)
class AdmittedSourceForwardRuntime:
    """Exact Source runtime identity and common model/tokenizer vocabulary."""

    source_sha256: str
    runtime_artifact_sha256: str
    model_identity_sha256: str
    tokenizer_identity_sha256: str
    vocab_size: int
    admission_sha256: str
    _model: Any = dataclass_field(repr=False, compare=False)
    _model_state_identity: tuple[Any, ...] | None = dataclass_field(
        repr=False, compare=False
    )
    _image_processor: Any = dataclass_field(repr=False, compare=False)
    _factory_marker: object = dataclass_field(repr=False, compare=False)


def _source_runtime_preimage(value: AdmittedSourceForwardRuntime) -> dict[str, Any]:
    return {
        "schema_version": "human13_admitted_source_forward_runtime.v1",
        "source_sha256": value.source_sha256,
        "runtime_artifact_sha256": value.runtime_artifact_sha256,
        "model_identity_sha256": value.model_identity_sha256,
        "tokenizer_identity_sha256": value.tokenizer_identity_sha256,
        "vocab_size": value.vocab_size,
    }


def _require_source_runtime(value: object) -> AdmittedSourceForwardRuntime:
    if type(value) is not AdmittedSourceForwardRuntime:
        raise ValueError("Source forward runtime admission is absent")
    admitted = _SOURCE_RUNTIME_ADMISSIONS.get(id(value))
    expected = _sha256(_source_runtime_preimage(value))
    if (
        value._factory_marker is not _SOURCE_RUNTIME_MARKER
        or admitted is None
        or admitted[0]() is not value
        or admitted[1] != value.admission_sha256
        or value.admission_sha256 != expected
    ):
        raise ValueError("Source forward runtime admission is absent or forged")
    return value


def _construct_source_forward_runtime(
    *,
    source_sha256: str,
    runtime_artifact_sha256: str,
    model_identity_sha256: str,
    tokenizer_identity_sha256: str,
    model_vocab_size: int,
    tokenizer_vocab_size: int,
    model: Any = None,
    image_processor: Any = None,
) -> AdmittedSourceForwardRuntime:
    for field, value in (
        ("source_sha256", source_sha256),
        ("runtime_artifact_sha256", runtime_artifact_sha256),
        ("model_identity_sha256", model_identity_sha256),
        ("tokenizer_identity_sha256", tokenizer_identity_sha256),
    ):
        _digest(value, field=field)
    if (
        isinstance(model_vocab_size, bool)
        or not isinstance(model_vocab_size, int)
        or model_vocab_size <= 0
        or model_vocab_size != tokenizer_vocab_size
    ):
        raise ValueError("Source model/tokenizer vocabulary widths must match exactly")
    result = object.__new__(AdmittedSourceForwardRuntime)
    for field, value in (
        ("source_sha256", source_sha256),
        ("runtime_artifact_sha256", runtime_artifact_sha256),
        ("model_identity_sha256", model_identity_sha256),
        ("tokenizer_identity_sha256", tokenizer_identity_sha256),
        ("vocab_size", model_vocab_size),
        ("admission_sha256", ""),
        ("_model", model),
        ("_image_processor", image_processor),
        (
            "_model_state_identity",
            None if model is None else _model_state_identity(model),
        ),
        ("_factory_marker", _SOURCE_RUNTIME_MARKER),
    ):
        object.__setattr__(result, field, value)
    admission_sha256 = _sha256(_source_runtime_preimage(result))
    object.__setattr__(result, "admission_sha256", admission_sha256)
    _register_admission(_SOURCE_RUNTIME_ADMISSIONS, result, admission_sha256)
    return result


def _model_state_identity(model: Any) -> tuple[Any, ...]:
    """Cheap live seal detecting parameter replacement or in-place mutation."""

    return tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            int(parameter._version),
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype),
        )
        for name, parameter in model.named_parameters()
    )


def _build_source_forward_runtime_for_test(
    *,
    source_sha256: str,
    model_identity_sha256: str,
    tokenizer_identity_sha256: str,
    model_vocab_size: int,
    tokenizer_vocab_size: int,
) -> AdmittedSourceForwardRuntime:
    """Private CPU fixture seam; public runtime admission uses QwenComponents."""

    return _construct_source_forward_runtime(
        source_sha256=source_sha256,
        runtime_artifact_sha256="9" * 64,
        model_identity_sha256=model_identity_sha256,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        model_vocab_size=model_vocab_size,
        tokenizer_vocab_size=tokenizer_vocab_size,
        model=None,
        image_processor=None,
    )


def admit_source_forward_runtime(
    components: Any, *, source_checkpoint_path: str | Path | None = None
) -> AdmittedSourceForwardRuntime:
    """Derive vocabulary and runtime identity from exact loaded Qwen components."""

    from src.qwen.runtime_loading import QwenComponents
    from scripts.research.human13_live_eval import checkpoint_payload_sha256

    if type(components) is not QwenComponents or components.model is None:
        raise ValueError("Source runtime requires exact loaded QwenComponents")
    model_vocab_size = components.model_identity.text_vocab_size
    tokenizer_vocab_size = components.token_identity.tokenizer_vocab_size
    if len(components.tokenizer) != tokenizer_vocab_size:
        raise ValueError("loaded tokenizer length differs from runtime identity")
    output_embeddings = components.model.get_output_embeddings()
    output_weight = getattr(output_embeddings, "weight", None)
    if output_weight is None or int(output_weight.shape[0]) != model_vocab_size:
        raise ValueError("loaded model output vocabulary differs from runtime identity")
    runtime_artifact = components.to_artifact_dict()
    source_path = (
        components.base_model_path
        if source_checkpoint_path is None
        else source_checkpoint_path
    )
    return _construct_source_forward_runtime(
        source_sha256=checkpoint_payload_sha256(source_path),
        runtime_artifact_sha256=_sha256(runtime_artifact),
        model_identity_sha256=_sha256(asdict(components.model_identity)),
        tokenizer_identity_sha256=_sha256(
            {
                "token_identity": asdict(components.token_identity),
                "tokenizer_sha256": components.tokenizer_sha256,
            }
        ),
        model_vocab_size=model_vocab_size,
        tokenizer_vocab_size=tokenizer_vocab_size,
        model=components.model,
        image_processor=components.processor.image_processor,
    )


@dataclass(frozen=True)
class SourceForwardRowReceipt:
    token_index: int
    causal_position: int
    history_token_sha256: str
    chosen_token_id: int
    raw_logit_sha256: str
    runtime_admission_sha256: str
    vocab_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, init=False)
class AdmittedSourceGreedyDecode:
    """Per-step proof that one Source decode was greedy under one exact RP."""

    source_decode_sha256: str
    source_sha256: str
    manifest_sha256: str
    repetition_penalty: float
    prompt_token_sha256: str
    generated_token_ids: tuple[int, ...]
    forward_rows: tuple[SourceForwardRowReceipt, ...]
    admission_sha256: str
    _factory_marker: object = dataclass_field(repr=False, compare=False)


def _source_decode_preimage(value: AdmittedSourceGreedyDecode) -> dict[str, Any]:
    return {
        "schema_version": "human13_admitted_source_rp_greedy_decode.v1",
        "policy": "sign-aware-rp-argmax-no-temperature-top-p-top-k",
        "source_decode_sha256": value.source_decode_sha256,
        "source_sha256": value.source_sha256,
        "manifest_sha256": value.manifest_sha256,
        "repetition_penalty": value.repetition_penalty,
        "prompt_token_sha256": value.prompt_token_sha256,
        "generated_token_ids": list(value.generated_token_ids),
        "forward_rows": [row.to_dict() for row in value.forward_rows],
    }


def _require_source_decode(value: object) -> AdmittedSourceGreedyDecode:
    if type(value) is not AdmittedSourceGreedyDecode:
        raise ValueError("RP-specific Source greedy decode admission is absent")
    admitted = _SOURCE_DECODE_ADMISSIONS.get(id(value))
    expected = _sha256(_source_decode_preimage(value))
    if (
        value._factory_marker is not _SOURCE_DECODE_MARKER
        or admitted is None
        or admitted[0]() is not value
        or admitted[1] != value.admission_sha256
        or value.admission_sha256 != expected
    ):
        raise ValueError(
            "RP-specific Source greedy decode admission is absent or forged"
        )
    return value


def _admit_source_greedy_decode_from_rows(
    boundary: SourceBoundaryInput,
    raw_logits: torch.Tensor,
    *,
    runtime: AdmittedSourceForwardRuntime,
    manifest_sha256: str,
) -> AdmittedSourceGreedyDecode:
    """Verify every realized Source token is the exact RP-processed argmax."""

    runtime = _require_source_runtime(runtime)
    manifest_sha256 = _digest(manifest_sha256, field="manifest_sha256")
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 2:
        raise ValueError(
            "Source greedy evidence logits must be a two-dimensional tensor"
        )
    generated = tuple(boundary.image.generated_token_ids)
    if raw_logits.shape[1] != runtime.vocab_size:
        raise ValueError(
            "Source greedy evidence must use exact runtime vocabulary width"
        )
    if raw_logits.shape[0] != len(generated) or not bool(
        torch.isfinite(raw_logits.detach()).all().item()
    ):
        raise ValueError("Source greedy evidence rows differ from generated tokens")
    if generated and raw_logits.shape[1] <= max(
        {*generated, *boundary.prompt_token_ids}
    ):
        raise ValueError(
            "Source greedy evidence vocabulary omits history or chosen token"
        )
    forward_rows: list[SourceForwardRowReceipt] = []
    for index, chosen_token_id in enumerate(generated):
        history = (*boundary.prompt_token_ids, *generated[:index])
        raw_row = raw_logits[index].detach().to(device="cpu").contiguous()
        row = raw_row.to(dtype=torch.float32)
        processed = row.clone()
        if boundary.repetition_penalty != 1.0 and history:
            repeated = torch.tensor(
                sorted(set(history)), dtype=torch.long, device=processed.device
            )
            selected = processed.index_select(0, repeated)
            processed = processed.scatter(
                0,
                repeated,
                torch.where(
                    selected < 0,
                    selected * boundary.repetition_penalty,
                    selected / boundary.repetition_penalty,
                ),
            )
        if int(processed.argmax().item()) != chosen_token_id:
            raise ValueError(
                "Source chosen token is not the RP-processed greedy argmax"
            )
        causal_position = len(boundary.prompt_token_ids) + index - 1
        history_token_sha256 = _history_sha256(history)
        raw_logit_sha256 = hashlib.sha256(
            _canonical_bytes(
                {
                    "schema_version": "human13_source_forward_row.v1",
                    "runtime_admission_sha256": runtime.admission_sha256,
                    "token_index": index,
                    "causal_position": causal_position,
                    "history_token_sha256": history_token_sha256,
                    "chosen_token_id": chosen_token_id,
                    "dtype": str(raw_row.dtype),
                    "vocab_size": row.numel(),
                }
            )
            + raw_row.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        forward_rows.append(
            SourceForwardRowReceipt(
                token_index=index,
                causal_position=causal_position,
                history_token_sha256=history_token_sha256,
                chosen_token_id=chosen_token_id,
                raw_logit_sha256=raw_logit_sha256,
                runtime_admission_sha256=runtime.admission_sha256,
                vocab_size=runtime.vocab_size,
            )
        )
    result = object.__new__(AdmittedSourceGreedyDecode)
    for field, value in (
        ("source_decode_sha256", boundary.source_decode_sha256),
        ("source_sha256", runtime.source_sha256),
        ("manifest_sha256", manifest_sha256),
        ("repetition_penalty", boundary.repetition_penalty),
        ("prompt_token_sha256", _history_sha256(boundary.prompt_token_ids)),
        ("generated_token_ids", generated),
        ("forward_rows", tuple(forward_rows)),
        ("admission_sha256", ""),
        ("_factory_marker", _SOURCE_DECODE_MARKER),
    ):
        object.__setattr__(result, field, value)
    admission_sha256 = _sha256(_source_decode_preimage(result))
    object.__setattr__(result, "admission_sha256", admission_sha256)
    _register_admission(_SOURCE_DECODE_ADMISSIONS, result, admission_sha256)
    return result


def _admit_source_greedy_decode_for_test(
    boundary: SourceBoundaryInput,
    raw_logits: torch.Tensor,
    *,
    runtime: AdmittedSourceForwardRuntime,
    manifest_sha256: str,
) -> AdmittedSourceGreedyDecode:
    """Private tensor fixture; scientific admission owns the model forward."""

    return _admit_source_greedy_decode_from_rows(
        boundary,
        raw_logits,
        runtime=runtime,
        manifest_sha256=manifest_sha256,
    )


def _derive_source_forward_inputs(
    boundary: SourceBoundaryInput,
    *,
    runtime: AdmittedSourceForwardRuntime,
    prompt_skeleton: Any,
) -> tuple[Any, Any, tuple[int, ...]]:
    """Rederive the one complete Source causal forward surface from evidence.

    This is the only construction seam for the scientific Source path: the
    prompt, generated tokens, MRoPE positions, attention plan, image tensors,
    and kept causal rows all come from the admitted boundary, the runtime's own
    image processor, and the production pack/forward builders.
    """

    from scripts.research.human13_live_census import _clone_skeleton
    from scripts.research.run_human13_k_union_overfit import (
        GLOBAL_MAX_LENGTH,
        LogicalPanelSegment,
        plan_panel_packs,
    )
    from src.qwen.fa2 import build_fa2_varlen_plan
    from src.qwen.forward import build_qwen_forward_inputs
    from src.qwen.images import QwenImageEncoding, materialize_qwen_image_encoding

    if runtime._model is None or runtime._image_processor is None:
        raise ValueError("scientific Source admission requires a loaded Source runtime")
    encoding = getattr(prompt_skeleton, "image_encoding", None)
    if type(encoding) is not QwenImageEncoding:
        raise ValueError("scientific Source admission requires Qwen image encoding")
    if encoding.plan.image_content_sha256 != boundary.image_sha256:
        raise ValueError("Source forward image identity differs from boundary")
    if (
        encoding.pixel_values is not None
        or encoding.image_grid_thw_tensor is not None
        or encoding.image_processor is not runtime._image_processor
    ):
        raise ValueError(
            "Source image encoding must be lazy and owned by admitted runtime"
        )
    prompt_token_count = getattr(prompt_skeleton, "prompt_token_count", None)
    skeleton_token_ids = getattr(prompt_skeleton, "input_ids", None)
    if (
        isinstance(prompt_token_count, bool)
        or not isinstance(prompt_token_count, int)
        or not isinstance(skeleton_token_ids, tuple)
        or prompt_token_count != len(boundary.prompt_token_ids)
        or skeleton_token_ids[:prompt_token_count] != boundary.prompt_token_ids
    ):
        raise ValueError("Source prompt skeleton differs from the admitted boundary")
    generated = tuple(boundary.image.generated_token_ids)
    if not generated:
        raise ValueError("Source boundary carries no generated tokens")
    input_ids = (*boundary.prompt_token_ids, *generated)
    positions = tuple(
        len(boundary.prompt_token_ids) + index - 1 for index in range(len(generated))
    )
    if any(
        position < 0
        or position + 1 >= len(input_ids)
        or input_ids[position + 1] != token_id
        for position, token_id in zip(positions, generated, strict=True)
    ):
        raise ValueError("Source causal positions differ from teacher-forced targets")
    segment_id = f"source-greedy:{boundary.image.image_id}"
    encoded = _clone_skeleton(
        prompt_skeleton,
        segment_id=segment_id,
        image_id=boundary.image.image_id,
        input_ids=input_ids,
    )
    plan = plan_panel_packs(
        [
            LogicalPanelSegment(
                segment_id=segment_id,
                image_id=boundary.image.image_id,
                role="h1_independent",
                encoded_example=encoded,
            )
        ],
        global_max_length=GLOBAL_MAX_LENGTH,
    )
    if len(plan.packs) != 1 or len(plan.packs[0].pack.segments) != 1:
        raise ValueError("Source forward pack plan is not one sealed causal segment")
    packed = plan.packs[0]
    segment = packed.pack.segments[0]
    if segment.start != 0 or tuple(packed.pack.input_ids) != input_ids:
        raise ValueError("Source forward pack differs from the sealed boundary tokens")
    device = next(runtime._model.parameters()).device
    inputs = build_qwen_forward_inputs(
        packed.pack,
        packed.encoded_examples,
        packed.position_inputs,
        fa2_varlen_plan=build_fa2_varlen_plan(packed.pack, device=device),
        logits_to_keep_positions=positions,
        device=device,
        fa2_branch_proof_policy="human13_source_greedy_admission",
    )
    materialized_image = materialize_qwen_image_encoding(encoding)
    expected_pixels = materialized_image.pixel_values
    expected_grid = materialized_image.image_grid_thw_tensor
    if not isinstance(expected_pixels, torch.Tensor) or not isinstance(
        expected_grid, torch.Tensor
    ):
        raise ValueError("Source image encoding did not materialize tensors")
    fa2_plan = inputs.fa2_varlen_plan
    kept = inputs.logits_to_keep
    if (
        not torch.equal(
            inputs.input_ids.detach().cpu(),
            torch.tensor([list(input_ids)], dtype=torch.long),
        )
        or tuple(int(value) for value in inputs.position_ids.shape)
        != (4, 1, len(input_ids))
        or not torch.equal(
            inputs.position_ids.detach().cpu(),
            packed.position_inputs.position_ids.detach().cpu(),
        )
        or not torch.equal(
            inputs.pixel_values.detach().cpu(), expected_pixels.detach().cpu()
        )
        or not torch.equal(
            inputs.image_grid_thw.detach().cpu(), expected_grid.detach().cpu()
        )
        or inputs.logits_position_ids != positions
        or not isinstance(kept, torch.Tensor)
        or tuple(int(value) for value in kept.detach().cpu().reshape(-1)) != positions
        or fa2_plan.attention_mask is not None
        or fa2_plan.segment_boundaries != (0, len(input_ids))
        or (fa2_plan.max_length_q, fa2_plan.max_length_k)
        != (len(input_ids), len(input_ids))
        or tuple(int(value) for value in fa2_plan.cu_seq_lens_q.detach().cpu())
        != (0, len(input_ids))
        or tuple(int(value) for value in fa2_plan.cu_seq_lens_k.detach().cpu())
        != (0, len(input_ids))
    ):
        raise ValueError(
            "derived Source forward surface differs from admitted evidence"
        )
    model_kwargs = inputs.to_model_kwargs()
    receipt = inputs.receipt
    if (
        model_kwargs["use_cache"] is not False
        or model_kwargs["labels"] is not None
        or model_kwargs["attention_mask"] is not None
        or "past_key_values" in model_kwargs
        or "cache_position" in model_kwargs
        or receipt.use_cache
        or receipt.labels_passed
        or receipt.inputs_embeds_used
        or receipt.logits_to_keep != positions
        or receipt.placeholder_token_count != receipt.expected_visual_token_count
    ):
        raise ValueError(
            "derived Source forward must disable cache, labels, and padding"
        )
    return packed, inputs, positions


def admit_source_greedy_decode(
    boundary: SourceBoundaryInput,
    *,
    runtime: AdmittedSourceForwardRuntime,
    manifest_sha256: str,
    prompt_skeleton: Any,
    forward_inputs: Any = None,
) -> AdmittedSourceGreedyDecode:
    """Run the admitted Source model over one internally derived causal surface."""

    from src.qwen.forward import run_qwen_forward

    runtime = _require_source_runtime(runtime)
    if forward_inputs is not None:
        raise ValueError(
            "scientific Source admission derives its own Qwen forward inputs; "
            "caller-supplied forward inputs are rejected"
        )
    _, inputs, positions = _derive_source_forward_inputs(
        boundary,
        runtime=runtime,
        prompt_skeleton=prompt_skeleton,
    )
    if _model_state_identity(runtime._model) != runtime._model_state_identity:
        raise ValueError("admitted Source model parameters changed before forward")
    with torch.no_grad():
        result = run_qwen_forward(
            runtime._model,
            inputs,
            expected_vocab_size=runtime.vocab_size,
        )
    if result.logits_position_ids != positions:
        raise ValueError("Source forward output causal positions differ")
    rows = result.logits.reshape(len(positions), runtime.vocab_size)
    return _admit_source_greedy_decode_from_rows(
        boundary,
        rows,
        runtime=runtime,
        manifest_sha256=manifest_sha256,
    )


@dataclass(frozen=True, init=False)
class AdmittedSourceCompilerPanel:
    """Durable RP-specific Source frontier joined to admitted Task-2 prompts."""

    boundaries: tuple[SourceBoundaryInput, ...]
    source_sha256: str
    manifest_sha256: str
    acquisition_sha256: str
    repetition_penalty: float
    frontier_sha256: str
    source_decode_admission_sha256s: tuple[str, ...]
    admission_sha256: str
    _factory_marker: object = dataclass_field(repr=False, compare=False)


def _register_admission(
    registry: dict[int, tuple[weakref.ReferenceType[Any], str]],
    value: Any,
    digest: str,
) -> None:
    identity = id(value)

    def discard(reference: weakref.ReferenceType[Any]) -> None:
        current = registry.get(identity)
        if current is not None and current[0] is reference:
            registry.pop(identity, None)

    registry[identity] = (weakref.ref(value, discard), digest)


def _source_panel_preimage(panel: AdmittedSourceCompilerPanel) -> dict[str, Any]:
    return {
        "schema_version": "human13_admitted_source_compiler_panel.v1",
        "source_sha256": panel.source_sha256,
        "manifest_sha256": panel.manifest_sha256,
        "acquisition_sha256": panel.acquisition_sha256,
        "repetition_penalty": panel.repetition_penalty,
        "frontier_sha256": panel.frontier_sha256,
        "source_decode_admission_sha256s": list(panel.source_decode_admission_sha256s),
        "boundary_sha256s": [
            boundary.source_decode_sha256 for boundary in panel.boundaries
        ],
    }


def _require_source_panel(value: object) -> AdmittedSourceCompilerPanel:
    if type(value) is not AdmittedSourceCompilerPanel:
        raise ValueError("scientific Source compiler panel admission is absent")
    admitted = _SOURCE_PANEL_ADMISSIONS.get(id(value))
    expected = _sha256(_source_panel_preimage(value))
    if (
        value._factory_marker is not _SOURCE_PANEL_MARKER
        or admitted is None
        or admitted[0]() is not value
        or admitted[1] != value.admission_sha256
        or value.admission_sha256 != expected
    ):
        raise ValueError(
            "scientific Source compiler panel admission is absent or forged"
        )
    return value


def admit_source_compiler_panel(
    manifest: Human13KUnionManifest,
    frontier: Human13FrontierIteration,
    *,
    frontier_path: str | Path,
    acquisition: TrajectoryCreditPanelAcquisition,
    source_decodes: Sequence[AdmittedSourceGreedyDecode],
) -> AdmittedSourceCompilerPanel:
    """Admit only a durable canonical Source frontier and Task-2 prompt lineage."""

    if type(frontier) is not Human13FrontierIteration:
        raise ValueError("Source compiler frontier type differs")
    loaded = load_frontier_iteration(frontier_path)
    if loaded != frontier:
        raise ValueError("Source compiler frontier differs from its durable artifact")
    if type(acquisition) is not TrajectoryCreditPanelAcquisition:
        raise ValueError("Source compiler requires admitted Task-2 panel acquisition")
    # Reconstruction reruns every AdmittedPublication aggregate admission.
    acquisition = TrajectoryCreditPanelAcquisition(acquisition.publications)
    publications = acquisition.publications
    image_ids = tuple(item.execution.plan.image_id for item in publications)
    if image_ids != tuple(
        image.image_id for image in manifest.images
    ) or image_ids != tuple(image.image_id for image in frontier.images):
        raise ValueError("Source frontier/acquisition/manifest image order differs")
    source_hashes = {
        item.replayed_group.identity.source_sha256 for item in publications
    }
    manifest_hashes = {
        item.replayed_group.identity.manifest_sha256 for item in publications
    }
    if len(source_hashes) != 1 or manifest_hashes != {_manifest_sha256(manifest)}:
        raise ValueError("Source frontier acquisition lineage differs")
    source_sha256 = source_hashes.pop()
    if (
        frontier.checkpoint.payload_sha256 != source_sha256
        or frontier.manifest_sha256 != _manifest_sha256(manifest)
    ):
        raise ValueError("Source frontier checkpoint or manifest differs")
    boundaries = tuple(
        SourceBoundaryInput(
            image=frontier_image,
            prompt_token_ids=publication.replayed_group.identity.prompt_token_ids,
            repetition_penalty=acquisition.training_repetition_penalty,
            image_sha256=cast(str, manifest_image.image_sha256),
        )
        for manifest_image, frontier_image, publication in zip(
            manifest.images, frontier.images, publications, strict=True
        )
    )
    decodes = tuple(_require_source_decode(value) for value in source_decodes)
    if len(decodes) != len(boundaries):
        raise ValueError("Source RP decode receipts differ from panel boundaries")
    for boundary, decode in zip(boundaries, decodes, strict=True):
        if (
            decode.source_decode_sha256 != boundary.source_decode_sha256
            or decode.source_sha256 != source_sha256
            or decode.manifest_sha256 != _manifest_sha256(manifest)
            or decode.repetition_penalty != acquisition.training_repetition_penalty
            or decode.prompt_token_sha256 != _history_sha256(boundary.prompt_token_ids)
            or decode.generated_token_ids != boundary.image.generated_token_ids
        ):
            raise ValueError("Source RP decode receipt differs from exact boundary")
    frontier_sha256 = hashlib.sha256(Path(frontier_path).read_bytes()).hexdigest()
    result = object.__new__(AdmittedSourceCompilerPanel)
    for field, value in (
        ("boundaries", boundaries),
        ("source_sha256", source_sha256),
        ("manifest_sha256", _manifest_sha256(manifest)),
        ("acquisition_sha256", acquisition.content_sha256),
        ("repetition_penalty", acquisition.training_repetition_penalty),
        ("frontier_sha256", frontier_sha256),
        (
            "source_decode_admission_sha256s",
            tuple(value.admission_sha256 for value in decodes),
        ),
        ("admission_sha256", ""),
        ("_factory_marker", _SOURCE_PANEL_MARKER),
    ):
        object.__setattr__(result, field, value)
    admission_sha256 = _sha256(_source_panel_preimage(result))
    object.__setattr__(result, "admission_sha256", admission_sha256)
    _register_admission(_SOURCE_PANEL_ADMISSIONS, result, admission_sha256)
    return result


@dataclass(frozen=True)
class AliasChild:
    """The only recursively usable fact: one alias's immediate child token."""

    owner_id: str
    alias_id: str
    token_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id:
            raise ValueError("alias child owner_id must be nonempty")
        if not isinstance(self.alias_id, str) or not self.alias_id:
            raise ValueError("alias child alias_id must be nonempty")
        _token_ids((self.token_id,), field="alias child token_id", nonempty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "alias_id": self.alias_id,
            "token_id": self.token_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> AliasChild:
        if not isinstance(value, Mapping) or set(value) != {
            "owner_id",
            "alias_id",
            "token_id",
        }:
            raise ValueError("alias child fields differ")
        return cls(
            owner_id=value["owner_id"],
            alias_id=value["alias_id"],
            token_id=value["token_id"],
        )


def _normalized_token_weights(
    alias_children: Sequence[AliasChild],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    by_owner: dict[str, list[AliasChild]] = {}
    for child in alias_children:
        by_owner.setdefault(child.owner_id, []).append(child)
    if not by_owner:
        raise ValueError("a compiler site requires valid alias children")
    owner_weight = 1.0 / len(by_owner)
    token_weights: dict[int, float] = {}
    for owner_id in sorted(by_owner):
        aliases = by_owner[owner_id]
        alias_weight = owner_weight / len(aliases)
        for child in aliases:
            token_weights[child.token_id] = (
                token_weights.get(child.token_id, 0.0) + alias_weight
            )
    token_ids = tuple(sorted(token_weights))
    weights = [float(token_weights[token_id]) for token_id in token_ids]
    # Preserve the declared probability simplex exactly after deterministic
    # floating accumulation; this does not change any selector ordering.
    weights[-1] += 1.0 - sum(weights)
    return token_ids, tuple(weights)


@dataclass(frozen=True)
class CompilerSite:
    site_id: str
    packed_segment_id: str
    image_id: int
    source_decode_sha256: str
    alias_bank_sha256: str
    generated_token_index: int
    local_causal_position: int
    bad_token_id: int
    compact_token_ids: tuple[int, ...]
    repeated_token_ids: tuple[int, ...]
    prompt_token_count: int
    prompt_token_sha256: str
    source_prefix_token_count: int
    source_prefix_token_sha256: str
    source_history_token_count: int
    source_history_sha256: str
    alias_children: tuple[AliasChild, ...]
    valid_token_ids: tuple[int, ...]
    valid_token_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, str) or not self.site_id:
            raise ValueError("compiler site_id must be nonempty")
        if not isinstance(self.packed_segment_id, str) or not self.packed_segment_id:
            raise ValueError("compiler packed_segment_id must be nonempty")
        for field in (
            "source_decode_sha256",
            "alias_bank_sha256",
            "prompt_token_sha256",
            "source_prefix_token_sha256",
            "source_history_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        for field in (
            "image_id",
            "generated_token_index",
            "local_causal_position",
            "prompt_token_count",
            "source_prefix_token_count",
            "source_history_token_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        _token_ids((self.bad_token_id,), field="bad_token_id", nonempty=True)
        compact = _token_ids(
            self.compact_token_ids, field="compact_token_ids", nonempty=True
        )
        repeated = _token_ids(self.repeated_token_ids, field="repeated_token_ids")
        if tuple(sorted(set(compact))) != compact:
            raise ValueError("compact token ids must be sorted and unique")
        if tuple(sorted(set(repeated))) != repeated or not set(repeated) <= set(
            compact
        ):
            raise ValueError("repeated token ids must be a compact-token subset")
        children = tuple(self.alias_children)
        if not children or any(not isinstance(child, AliasChild) for child in children):
            raise ValueError("compiler alias children differ")
        identities = [(child.owner_id, child.alias_id) for child in children]
        if len(set(identities)) != len(identities):
            raise ValueError("compiler alias identities must be unique")
        expected_ids, expected_weights = _normalized_token_weights(children)
        valid_ids = _token_ids(
            self.valid_token_ids, field="valid_token_ids", nonempty=True
        )
        weights = tuple(float(value) for value in self.valid_token_weights)
        if valid_ids != expected_ids or weights != expected_weights:
            raise ValueError(
                "valid token weights do not rederive owner-then-alias normalization"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("valid token weights must be finite and positive")
        if sum(weights) != 1.0:
            raise ValueError("valid token weights must sum exactly to one")
        if self.bad_token_id in set(valid_ids):
            raise ValueError("realized bad child cannot also be a frozen valid child")
        if compact != tuple(sorted({*valid_ids, self.bad_token_id})):
            raise ValueError(
                "compact token ids differ from valid and realized bad children"
            )
        object.__setattr__(self, "compact_token_ids", compact)
        object.__setattr__(self, "repeated_token_ids", repeated)
        object.__setattr__(self, "alias_children", children)
        object.__setattr__(self, "valid_token_ids", valid_ids)
        object.__setattr__(self, "valid_token_weights", weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "packed_segment_id": self.packed_segment_id,
            "image_id": self.image_id,
            "source_decode_sha256": self.source_decode_sha256,
            "alias_bank_sha256": self.alias_bank_sha256,
            "generated_token_index": self.generated_token_index,
            "local_causal_position": self.local_causal_position,
            "bad_token_id": self.bad_token_id,
            "compact_token_ids": list(self.compact_token_ids),
            "repeated_token_ids": list(self.repeated_token_ids),
            "prompt_token_count": self.prompt_token_count,
            "prompt_token_sha256": self.prompt_token_sha256,
            "source_prefix_token_count": self.source_prefix_token_count,
            "source_prefix_token_sha256": self.source_prefix_token_sha256,
            "source_history_token_count": self.source_history_token_count,
            "source_history_sha256": self.source_history_sha256,
            "alias_children": [child.to_dict() for child in self.alias_children],
            "valid_token_ids": list(self.valid_token_ids),
            "valid_token_weights": list(self.valid_token_weights),
        }

    @classmethod
    def from_dict(cls, value: object) -> CompilerSite:
        if not isinstance(value, Mapping):
            raise ValueError("compiler site must be a mapping")
        required = {
            "site_id",
            "packed_segment_id",
            "image_id",
            "source_decode_sha256",
            "alias_bank_sha256",
            "generated_token_index",
            "local_causal_position",
            "bad_token_id",
            "compact_token_ids",
            "repeated_token_ids",
            "prompt_token_count",
            "prompt_token_sha256",
            "source_prefix_token_count",
            "source_prefix_token_sha256",
            "source_history_token_count",
            "source_history_sha256",
            "alias_children",
            "valid_token_ids",
            "valid_token_weights",
        }
        if set(value) != required:
            raise ValueError("compiler site fields differ")
        return cls(
            site_id=value["site_id"],
            packed_segment_id=value["packed_segment_id"],
            image_id=value["image_id"],
            source_decode_sha256=value["source_decode_sha256"],
            alias_bank_sha256=value["alias_bank_sha256"],
            generated_token_index=value["generated_token_index"],
            local_causal_position=value["local_causal_position"],
            bad_token_id=value["bad_token_id"],
            compact_token_ids=tuple(value["compact_token_ids"]),
            repeated_token_ids=tuple(value["repeated_token_ids"]),
            prompt_token_count=value["prompt_token_count"],
            prompt_token_sha256=value["prompt_token_sha256"],
            source_prefix_token_count=value["source_prefix_token_count"],
            source_prefix_token_sha256=value["source_prefix_token_sha256"],
            source_history_token_count=value["source_history_token_count"],
            source_history_sha256=value["source_history_sha256"],
            alias_children=tuple(
                AliasChild.from_dict(child) for child in value["alias_children"]
            ),
            valid_token_ids=tuple(value["valid_token_ids"]),
            valid_token_weights=tuple(value["valid_token_weights"]),
        )


@dataclass(frozen=True)
class CompilerImageLedger:
    image_id: int
    source_decode_sha256: str
    site: CompilerSite | None
    absent_reason: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.image_id, bool)
            or not isinstance(self.image_id, int)
            or self.image_id < 0
        ):
            raise ValueError("compiler image_id must be nonnegative")
        object.__setattr__(
            self,
            "source_decode_sha256",
            _digest(self.source_decode_sha256, field="source_decode_sha256"),
        )
        if self.site is None:
            if self.absent_reason not in _ABSENT_REASONS:
                raise ValueError("absent compiler image requires a declared reason")
        elif (
            not isinstance(self.site, CompilerSite)
            or self.site.image_id != self.image_id
            or self.site.source_decode_sha256 != self.source_decode_sha256
            or self.absent_reason is not None
        ):
            raise ValueError("present compiler site/image evidence differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_decode_sha256": self.source_decode_sha256,
            "site": None if self.site is None else self.site.to_dict(),
            "absent_reason": self.absent_reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> CompilerImageLedger:
        if not isinstance(value, Mapping) or set(value) != {
            "image_id",
            "source_decode_sha256",
            "site",
            "absent_reason",
        }:
            raise ValueError("compiler image ledger fields differ")
        return cls(
            image_id=value["image_id"],
            source_decode_sha256=value["source_decode_sha256"],
            site=None
            if value["site"] is None
            else CompilerSite.from_dict(value["site"]),
            absent_reason=value["absent_reason"],
        )


@dataclass(frozen=True)
class CompilerLedger:
    source_sha256: str
    manifest_sha256: str
    acquisition_sha256: str
    trajectory_credit_sha256: str
    repetition_penalty: float
    logical_image_count: int
    frozen_alias_count: int
    alias_bank_sha256: str
    source_panel_sha256: str
    images: tuple[CompilerImageLedger, ...]
    admission_sha256: str | None = None
    _factory_marker: object | None = dataclass_field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for field in (
            "source_sha256",
            "manifest_sha256",
            "acquisition_sha256",
            "trajectory_credit_sha256",
            "alias_bank_sha256",
            "source_panel_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        repetition_penalty = float(self.repetition_penalty)
        if repetition_penalty not in {1.0, 1.10}:
            raise ValueError("compiler ledger RP must be exactly 1.0 or 1.10")
        object.__setattr__(self, "repetition_penalty", repetition_penalty)
        if (
            isinstance(self.logical_image_count, bool)
            or not isinstance(self.logical_image_count, int)
            or self.logical_image_count <= 0
        ):
            raise ValueError("logical_image_count must be positive")
        if self.frozen_alias_count != EXACT_ALIAS_COUNT:
            raise ValueError("compiler requires the exact frozen 309-alias bank")
        images = tuple(self.images)
        if len(images) != self.logical_image_count or len(
            {image.image_id for image in images}
        ) != len(images):
            raise ValueError("compiler images differ from logical image count")
        if any(
            image.site is not None
            and image.site.alias_bank_sha256 != self.alias_bank_sha256
            for image in images
        ):
            raise ValueError("compiler site alias-bank lineage differs")
        object.__setattr__(self, "images", images)
        if self.admission_sha256 is not None:
            object.__setattr__(
                self,
                "admission_sha256",
                _digest(self.admission_sha256, field="admission_sha256"),
            )

    def _preimage(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_sha256": self.acquisition_sha256,
            "trajectory_credit_sha256": self.trajectory_credit_sha256,
            "repetition_penalty": self.repetition_penalty,
            "logical_image_count": self.logical_image_count,
            "frozen_alias_count": self.frozen_alias_count,
            "alias_bank_sha256": self.alias_bank_sha256,
            "source_panel_sha256": self.source_panel_sha256,
            "admission_sha256": self.admission_sha256,
            "images": [image.to_dict() for image in self.images],
        }

    @property
    def content_sha256(self) -> str:
        return _sha256(self._preimage())

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        preimage = self._preimage()
        return {**preimage, "content_sha256": _sha256(preimage)}

    @classmethod
    def from_dict(cls, value: object) -> CompilerLedger:
        if not isinstance(value, Mapping):
            raise ValueError("compiler ledger must be a mapping")
        required = {
            "schema_version",
            "source_sha256",
            "manifest_sha256",
            "acquisition_sha256",
            "trajectory_credit_sha256",
            "repetition_penalty",
            "logical_image_count",
            "frozen_alias_count",
            "alias_bank_sha256",
            "source_panel_sha256",
            "admission_sha256",
            "images",
            "content_sha256",
        }
        if set(value) != required or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("compiler ledger schema fields differ")
        result = cls(
            source_sha256=value["source_sha256"],
            manifest_sha256=value["manifest_sha256"],
            acquisition_sha256=value["acquisition_sha256"],
            trajectory_credit_sha256=value["trajectory_credit_sha256"],
            repetition_penalty=value["repetition_penalty"],
            logical_image_count=value["logical_image_count"],
            frozen_alias_count=value["frozen_alias_count"],
            alias_bank_sha256=value["alias_bank_sha256"],
            source_panel_sha256=value["source_panel_sha256"],
            admission_sha256=value["admission_sha256"],
            images=tuple(
                CompilerImageLedger.from_dict(image) for image in value["images"]
            ),
        )
        if value["content_sha256"] != result.content_sha256:
            raise ValueError("compiler ledger content SHA-256 differs")
        return result


def _alias_payload(alias: FrontierCandidateAlias, *, image_id: int) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "owner_id": alias.owner_id,
        "row_id": alias.row_id,
        "trajectory_id": alias.trajectory_id,
        "seed": alias.seed,
        "owner_iou": alias.owner_iou,
        "token_ids": list(alias.token_ids),
    }


def _frozen_alias_bank(
    manifest: Human13KUnionManifest,
) -> tuple[tuple[FrontierCandidateAlias, ...], str]:
    aliases: list[FrontierCandidateAlias] = []
    payload: list[dict[str, Any]] = []
    for image in manifest.images:
        current = candidate_aliases_for_owners(image, owner_ids=set(image.h_owner_ids))
        aliases.extend(current)
        payload.extend(
            _alias_payload(alias, image_id=image.image_id) for alias in current
        )
    if len(aliases) != EXACT_ALIAS_COUNT:
        raise ValueError(
            "compiler requires exact 309 frozen metric-valid native aliases"
        )
    return tuple(aliases), _sha256(
        {"schema_version": "human13_frozen_alias_bank.v1", "aliases": payload}
    )


def _build_image_ledger(
    manifest_image: ImageRecord,
    boundary: SourceBoundaryInput,
    *,
    alias_bank_sha256: str,
) -> CompilerImageLedger:
    image = boundary.image
    if image.image_id != manifest_image.image_id:
        raise ValueError("Source boundary image differs from manifest image")
    # FrontierImage carries the same decode-surface fields consumed by this
    # validator; its narrower annotation names only CurrentDecode.
    validate_current_decode_surface(cast(Any, image))
    h_owners = set(manifest_image.h_owner_ids)
    uncovered = set(image.uncovered_h_owner_ids)
    covered = set(image.covered_h_owner_ids)
    if (
        not uncovered <= h_owners
        or not covered <= h_owners
        or uncovered & covered
        or uncovered | covered != h_owners
    ):
        raise ValueError("Source covered/uncovered H owners must be an exact partition")
    expected_aliases = candidate_aliases_for_owners(manifest_image, owner_ids=uncovered)
    if tuple(image.candidate_aliases) != expected_aliases:
        raise ValueError("Source boundary aliases differ from frozen manifest aliases")
    decode_sha256 = boundary.source_decode_sha256
    if image.terminal_token_index is None:
        return CompilerImageLedger(
            image.image_id,
            decode_sha256,
            None,
            "no_premature_source_boundary",
        )
    if not uncovered:
        return CompilerImageLedger(
            image.image_id, decode_sha256, None, "no_trusted_remaining"
        )
    if not expected_aliases:
        return CompilerImageLedger(
            image.image_id, decode_sha256, None, "no_valid_aliases"
        )
    terminal_index = image.terminal_token_index
    if terminal_index != len(image.generated_token_ids) - 1:
        raise ValueError("Source bad child must be the realized terminal child")
    prefix = natural_pre_stop_prefix(image)
    history = (*boundary.prompt_token_ids, *prefix)
    bad_token_id = image.generated_token_ids[terminal_index]
    children = tuple(
        AliasChild(alias.owner_id, alias.row_id, alias.token_ids[0])
        for alias in expected_aliases
    )
    valid_token_ids, valid_token_weights = _normalized_token_weights(children)
    compact_token_ids = tuple(sorted({*valid_token_ids, bad_token_id}))
    repeated_token_ids = tuple(sorted(set(compact_token_ids) & set(history)))
    site = CompilerSite(
        site_id=f"source-compiler:rp{boundary.repetition_penalty:.2f}:{image.image_id}",
        packed_segment_id=(
            f"on-policy-score:{image.image_id}:"
            f"{expected_aliases[0].owner_id}:{expected_aliases[0].row_id}"
        ),
        image_id=image.image_id,
        source_decode_sha256=decode_sha256,
        alias_bank_sha256=alias_bank_sha256,
        generated_token_index=terminal_index,
        local_causal_position=len(boundary.prompt_token_ids) + terminal_index - 1,
        bad_token_id=bad_token_id,
        compact_token_ids=compact_token_ids,
        repeated_token_ids=repeated_token_ids,
        prompt_token_count=len(boundary.prompt_token_ids),
        prompt_token_sha256=token_ids_sha256(boundary.prompt_token_ids),
        source_prefix_token_count=len(prefix),
        source_prefix_token_sha256=token_ids_sha256(prefix),
        source_history_token_count=len(history),
        source_history_sha256=_history_sha256(history),
        alias_children=children,
        valid_token_ids=valid_token_ids,
        valid_token_weights=valid_token_weights,
    )
    return CompilerImageLedger(image.image_id, decode_sha256, site, None)


def _construct_compiler_ledger(
    manifest: Human13KUnionManifest,
    boundaries: Sequence[SourceBoundaryInput],
    *,
    source_sha256: str,
    acquisition_sha256: str,
    trajectory_credit_sha256: str,
    source_panel_sha256: str | None = None,
    admit_scientific: bool = False,
) -> CompilerLedger:
    if not isinstance(manifest, Human13KUnionManifest):
        raise ValueError("compiler manifest must be a Human13KUnionManifest")
    source_sha256 = _digest(source_sha256, field="source_sha256")
    acquisition_sha256 = _digest(acquisition_sha256, field="acquisition_sha256")
    trajectory_credit_sha256 = _digest(
        trajectory_credit_sha256, field="trajectory_credit_sha256"
    )
    boundaries = tuple(boundaries)
    if not boundaries or any(
        not isinstance(item, SourceBoundaryInput) for item in boundaries
    ):
        raise ValueError("compiler boundaries must be SourceBoundaryInput records")
    if tuple(item.image.image_id for item in boundaries) != tuple(
        image.image_id for image in manifest.images
    ):
        raise ValueError("compiler boundaries must cover manifest images in order")
    repetition_penalties = {item.repetition_penalty for item in boundaries}
    if len(repetition_penalties) != 1:
        raise ValueError("compiler ledger must contain one RP-specific Source surface")
    aliases, alias_bank_sha256 = _frozen_alias_bank(manifest)
    if source_panel_sha256 is None:
        source_panel_sha256 = _sha256(
            {
                "schema_version": "human13_test_source_compiler_panel.v1",
                "boundary_sha256s": [
                    boundary.source_decode_sha256 for boundary in boundaries
                ],
            }
        )
    else:
        source_panel_sha256 = _digest(source_panel_sha256, field="source_panel_sha256")
    images = tuple(
        _build_image_ledger(
            manifest_image,
            boundary,
            alias_bank_sha256=alias_bank_sha256,
        )
        for manifest_image, boundary in zip(manifest.images, boundaries, strict=True)
    )
    ledger = CompilerLedger(
        source_sha256=source_sha256,
        manifest_sha256=_manifest_sha256(manifest),
        acquisition_sha256=acquisition_sha256,
        trajectory_credit_sha256=trajectory_credit_sha256,
        repetition_penalty=repetition_penalties.pop(),
        logical_image_count=len(images),
        frozen_alias_count=len(aliases),
        alias_bank_sha256=alias_bank_sha256,
        source_panel_sha256=source_panel_sha256,
        images=images,
    )
    if not admit_scientific:
        return ledger
    admission_sha256 = _sha256(
        {
            **ledger._preimage(),
            "admission_sha256": None,
            "admission_kind": "live-source-and-task3",
        }
    )
    object.__setattr__(ledger, "admission_sha256", admission_sha256)
    object.__setattr__(ledger, "_factory_marker", _COMPILER_LEDGER_MARKER)
    _register_admission(_COMPILER_LEDGER_ADMISSIONS, ledger, admission_sha256)
    return ledger


def _build_compiler_ledger_for_test(
    manifest: Human13KUnionManifest,
    boundaries: Sequence[SourceBoundaryInput],
    *,
    source_sha256: str,
    acquisition_sha256: str,
    trajectory_credit_sha256: str,
) -> CompilerLedger:
    """Formula fixture seam; scientific admission remains the public builder."""

    return _construct_compiler_ledger(
        manifest,
        boundaries,
        source_sha256=source_sha256,
        acquisition_sha256=acquisition_sha256,
        trajectory_credit_sha256=trajectory_credit_sha256,
    )


def _admit_compiler_ledger_for_test(ledger: CompilerLedger) -> CompilerLedger:
    """Private registry seam for exercising public receipt checks on CPU."""

    if type(ledger) is not CompilerLedger or ledger.admission_sha256 is not None:
        raise ValueError("test compiler admission requires a fresh private ledger")
    admission_sha256 = _sha256(
        {
            **ledger._preimage(),
            "admission_sha256": None,
            "admission_kind": "live-source-and-task3",
        }
    )
    object.__setattr__(ledger, "admission_sha256", admission_sha256)
    object.__setattr__(ledger, "_factory_marker", _COMPILER_LEDGER_MARKER)
    _register_admission(_COMPILER_LEDGER_ADMISSIONS, ledger, admission_sha256)
    return ledger


def build_compiler_ledger(
    manifest: Human13KUnionManifest,
    source_panel: AdmittedSourceCompilerPanel,
    trajectory_credit_ledger: TrajectoryCreditLedger,
) -> CompilerLedger:
    """Build from the admitted Task-3 ledger and exact Source/alias evidence."""

    source_panel = _require_source_panel(source_panel)
    admitted = _require_scientific_ledger_admission(trajectory_credit_ledger)
    if admitted.manifest_sha256 != _manifest_sha256(manifest):
        raise ValueError("Task-3 ledger manifest differs from compiler manifest")
    if (
        source_panel.source_sha256 != admitted.source_sha256
        or source_panel.manifest_sha256 != admitted.manifest_sha256
        or source_panel.acquisition_sha256 != admitted.acquisition_sha256
    ):
        raise ValueError("Source panel differs from Task-3 acquisition lineage")
    if source_panel.repetition_penalty != admitted.training_repetition_penalty:
        raise ValueError("Task-3 ledger RP differs from compiler Source boundary")
    if admitted.logical_image_count != len(manifest.images):
        raise ValueError("Task-3 ledger image count differs from compiler manifest")
    return _construct_compiler_ledger(
        manifest,
        source_panel.boundaries,
        source_sha256=admitted.source_sha256,
        acquisition_sha256=admitted.acquisition_sha256,
        trajectory_credit_sha256=admitted.content_sha256,
        source_panel_sha256=source_panel.admission_sha256,
        admit_scientific=True,
    )


def construct_hf_native_one_image_compiler_ledger(
    manifest: Human13KUnionManifest,
    manifest_image: ImageRecord,
    boundary: SourceBoundaryInput,
    trajectory_credit_ledger: TrajectoryCreditLedger,
) -> CompilerLedger:
    """Admit the image-1584 compiler without forging full-panel coverage.

    The frozen alias bank and manifest hash still come from the full manifest;
    only the logical update surface is narrowed to the selected image.
    """

    admitted = _require_scientific_ledger_admission(trajectory_credit_ledger)
    if (
        not isinstance(manifest, Human13KUnionManifest)
        or not isinstance(manifest_image, ImageRecord)
        or not isinstance(boundary, SourceBoundaryInput)
    ):
        raise ValueError("HF-native compiler requires canonical manifest records")
    selected = tuple(image for image in manifest.images if image.image_id == 1584)
    if len(selected) != 1 or selected[0] != manifest_image:
        raise ValueError("HF-native compiler requires the exact manifest image-1584")
    manifest_sha256 = _manifest_sha256(manifest)
    if (
        admitted.logical_image_count != 1
        or tuple(image.image_id for image in admitted.images) != (1584,)
        or admitted.manifest_sha256 != manifest_sha256
        or boundary.image.image_id != 1584
        or boundary.repetition_penalty != admitted.training_repetition_penalty
    ):
        raise ValueError("HF-native compiler trajectory/Source lineage differs")
    aliases, alias_bank_sha256 = _frozen_alias_bank(manifest)
    image_ledger = _build_image_ledger(
        manifest_image,
        boundary,
        alias_bank_sha256=alias_bank_sha256,
    )
    source_panel_sha256 = _sha256(
        {
            "schema_version": "human13_hf_native_one_image_source_panel.v1",
            "manifest_sha256": manifest_sha256,
            "image_id": 1584,
            "source_decode_sha256": boundary.source_decode_sha256,
            "trajectory_credit_sha256": admitted.content_sha256,
        }
    )
    ledger = CompilerLedger(
        source_sha256=admitted.source_sha256,
        manifest_sha256=manifest_sha256,
        acquisition_sha256=admitted.acquisition_sha256,
        trajectory_credit_sha256=admitted.content_sha256,
        repetition_penalty=cast(float, admitted.training_repetition_penalty),
        logical_image_count=1,
        frozen_alias_count=len(aliases),
        alias_bank_sha256=alias_bank_sha256,
        source_panel_sha256=source_panel_sha256,
        images=(image_ledger,),
    )
    admission_sha256 = _sha256(
        {
            **ledger._preimage(),
            "admission_sha256": None,
            "admission_kind": "live-source-and-task3",
        }
    )
    object.__setattr__(ledger, "admission_sha256", admission_sha256)
    object.__setattr__(ledger, "_factory_marker", _COMPILER_LEDGER_MARKER)
    _register_admission(_COMPILER_LEDGER_ADMISSIONS, ledger, admission_sha256)
    return _require_compiler_admission(ledger)


def _require_compiler_admission(ledger: object) -> CompilerLedger:
    if type(ledger) is not CompilerLedger:
        raise ValueError("scientific compiler admission requires exact ledger type")
    admitted = _COMPILER_LEDGER_ADMISSIONS.get(id(ledger))
    expected = _sha256(
        {
            **ledger._preimage(),
            "admission_sha256": None,
            "admission_kind": "live-source-and-task3",
        }
    )
    if (
        ledger._factory_marker is not _COMPILER_LEDGER_MARKER
        or admitted is None
        or admitted[0]() is not ledger
        or admitted[1] != ledger.admission_sha256
        or ledger.admission_sha256 != expected
    ):
        raise ValueError("scientific compiler admission is absent or forged")
    return ledger


def load_compiler_ledger(
    value: object,
    manifest: Human13KUnionManifest,
    boundaries: Sequence[SourceBoundaryInput],
    *,
    source_sha256: str,
    acquisition_sha256: str,
    trajectory_credit_sha256: str,
) -> CompilerLedger:
    """Reload only if canonical bytes exactly match a complete deterministic rerun."""

    loaded = CompilerLedger.from_dict(value)
    expected = _construct_compiler_ledger(
        manifest,
        boundaries,
        source_sha256=source_sha256,
        acquisition_sha256=acquisition_sha256,
        trajectory_credit_sha256=trajectory_credit_sha256,
    )
    if loaded != expected or loaded.to_dict() != expected.to_dict():
        raise ValueError("stored compiler ledger differs from deterministic rerun")
    return expected


@dataclass(frozen=True)
class CompilerSiteScore:
    valid_score: torch.Tensor
    max_valid_score: torch.Tensor
    bad_score: torch.Tensor
    hinge: torch.Tensor


@dataclass(frozen=True)
class PackedCompilerLineage:
    """Semantic lineage for rows emitted by the external live pack planner."""

    acquisition_sha256: str
    trajectory_credit_sha256: str
    repetition_penalty: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acquisition_sha256",
            _digest(self.acquisition_sha256, field="acquisition_sha256"),
        )
        object.__setattr__(
            self,
            "trajectory_credit_sha256",
            _digest(
                self.trajectory_credit_sha256,
                field="trajectory_credit_sha256",
            ),
        )
        repetition_penalty = float(self.repetition_penalty)
        if repetition_penalty not in {1.0, 1.10}:
            raise ValueError("packed compiler lineage RP must be exactly 1.0 or 1.10")
        object.__setattr__(self, "repetition_penalty", repetition_penalty)


@dataclass(frozen=True, init=False)
class PackedLogitRows:
    """Real pack-plan remapping joined to one injected packed-forward tensor."""

    site_id: str
    segment_id: str
    pack_index: int
    packed_causal_position: int
    mapping_sha256: str
    compiler_ledger_sha256: str | None
    compiler_admission_sha256: str | None
    admission_sha256: str
    raw_logits: torch.Tensor
    _factory_marker: object = dataclass_field(repr=False, compare=False)


def _bind_packed_compiler_logits(
    prepared: Any,
    site: CompilerSite,
    *,
    compiler_ledger_sha256: str | None,
    compiler_admission_sha256: str | None,
    pack_index: int,
    logits_position_ids: Sequence[int],
    raw_logits: torch.Tensor,
) -> PackedLogitRows:
    """Bind actual packed positions through PreparedCandidateScoring's plan."""

    from scripts.research.human13_on_policy_scoring import PreparedCandidateScoring

    if not isinstance(prepared, PreparedCandidateScoring):
        raise ValueError("packed compiler binding requires PreparedCandidateScoring")
    bindings = {binding.segment_id: binding for binding in prepared.bindings}
    binding = bindings.get(site.packed_segment_id)
    if (
        binding is None
        or binding.path.image_id != site.image_id
        or not binding.local_causal_positions
        or binding.local_causal_positions[0] != site.local_causal_position
    ):
        raise ValueError("compiler site differs from prepared Source-prefix segment")
    if binding.prompt_token_sha256 != site.prompt_token_sha256:
        raise ValueError("prepared prompt token digest differs from compiler Source")
    if binding.natural_pre_stop_prefix_token_sha256 != site.source_prefix_token_sha256:
        raise ValueError(
            "prepared Source prefix token digest differs from compiler Source"
        )
    logical_segments = {
        segment.segment_id: segment for segment in prepared.packed_plan.logical_segments
    }
    logical = logical_segments.get(site.packed_segment_id)
    if logical is None:
        raise ValueError("compiler logical segment is absent from prepared scoring")
    encoded_ids = tuple(logical.encoded_example.input_ids)
    prompt = encoded_ids[: site.prompt_token_count]
    prefix_start = site.prompt_token_count
    prefix_end = prefix_start + site.source_prefix_token_count
    prefix = encoded_ids[prefix_start:prefix_end]
    if token_ids_sha256(prompt) != site.prompt_token_sha256:
        raise ValueError("rederived prompt token digest differs from compiler Source")
    if token_ids_sha256(prefix) != site.source_prefix_token_sha256:
        raise ValueError(
            "rederived Source prefix token digest differs from compiler Source"
        )
    packs = {pack.pack.pack_index: pack for pack in prepared.packed_plan.packs}
    pack = packs.get(pack_index)
    if pack is None:
        raise ValueError("packed compiler pack index differs from prepared plan")
    packed_segments = {segment.example_id: segment for segment in pack.pack.segments}
    segment = packed_segments.get(site.packed_segment_id)
    if segment is None:
        raise ValueError("compiler segment is absent from selected packed plan")
    if tuple(pack.pack.input_ids[segment.start : segment.end]) != encoded_ids:
        raise ValueError("packed compiler tokens differ from prepared logical segment")
    expected_position = segment.start + site.local_causal_position
    positions = tuple(logits_position_ids)
    if positions != (expected_position,):
        raise ValueError("packed compiler causal position differs from remapped plan")
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 2:
        raise ValueError("packed compiler raw logits must be a two-dimensional tensor")
    if raw_logits.shape[0] != 1 or not bool(
        torch.isfinite(raw_logits.detach()).all().item()
    ):
        raise ValueError("packed compiler output differs from requested causal row")
    mapping = {
        "schema_version": "human13_prepared_compiler_row.v1",
        "site_id": site.site_id,
        "segment_id": site.packed_segment_id,
        "source_decode_sha256": site.source_decode_sha256,
        "alias_bank_sha256": site.alias_bank_sha256,
        "prompt_token_sha256": site.prompt_token_sha256,
        "source_prefix_token_sha256": site.source_prefix_token_sha256,
        "compiler_ledger_sha256": compiler_ledger_sha256,
        "compiler_admission_sha256": compiler_admission_sha256,
        "prepared_segment_token_sha256": token_ids_sha256(encoded_ids),
        "pack_index": pack_index,
        "segment_start": segment.start,
        "local_causal_position": site.local_causal_position,
        "packed_causal_position": expected_position,
    }
    result = object.__new__(PackedLogitRows)
    for field, value in (
        ("site_id", site.site_id),
        ("segment_id", site.packed_segment_id),
        ("pack_index", pack_index),
        ("packed_causal_position", expected_position),
        ("mapping_sha256", _sha256(mapping)),
        ("compiler_ledger_sha256", compiler_ledger_sha256),
        ("compiler_admission_sha256", compiler_admission_sha256),
        ("admission_sha256", ""),
        ("raw_logits", raw_logits),
        ("_factory_marker", _PACKED_ROW_MARKER),
    ):
        object.__setattr__(result, field, value)
    admission_sha256 = _sha256(_packed_row_admission_preimage(result))
    object.__setattr__(result, "admission_sha256", admission_sha256)
    _register_admission(_PACKED_ROW_ADMISSIONS, result, admission_sha256)
    return result


def _bind_packed_compiler_logits_for_test(
    prepared: Any,
    site: CompilerSite,
    *,
    pack_index: int,
    logits_position_ids: Sequence[int],
    raw_logits: torch.Tensor,
) -> PackedLogitRows:
    """Private CPU fixture seam without scientific ledger admission."""

    return _bind_packed_compiler_logits(
        prepared,
        site,
        compiler_ledger_sha256=None,
        compiler_admission_sha256=None,
        pack_index=pack_index,
        logits_position_ids=logits_position_ids,
        raw_logits=raw_logits,
    )


def bind_packed_compiler_logits(
    prepared: Any,
    ledger: CompilerLedger,
    *,
    site_id: str,
    pack_index: int,
    logits_position_ids: Sequence[int],
    raw_logits: torch.Tensor,
) -> PackedLogitRows:
    """Bind a prepared row to the exact site owned by an admitted ledger."""

    ledger = _require_compiler_admission(ledger)
    sites = {
        image.site.site_id: image.site
        for image in ledger.images
        if image.site is not None
    }
    site = sites.get(site_id)
    if site is None:
        raise ValueError("compiler site is absent from admitted ledger")
    return _bind_packed_compiler_logits(
        prepared,
        site,
        compiler_ledger_sha256=ledger.content_sha256,
        compiler_admission_sha256=ledger.admission_sha256,
        pack_index=pack_index,
        logits_position_ids=logits_position_ids,
        raw_logits=raw_logits,
    )


def _packed_panel_plan_sha256(packed_plan: Any) -> str:
    """Content identity for the exact external logical and physical plan."""

    return _sha256(
        {
            "global_max_length": packed_plan.global_max_length,
            "logical_segments": [
                {
                    "segment_id": segment.segment_id,
                    "image_id": segment.image_id,
                    "role": segment.role,
                    "input_ids": list(segment.encoded_example.input_ids),
                }
                for segment in packed_plan.logical_segments
            ],
            "packs": [
                {
                    "pack_index": packed.pack.pack_index,
                    "input_ids": list(packed.pack.input_ids),
                    "segments": [
                        {
                            "example_id": segment.example_id,
                            "start": segment.start,
                            "end": segment.end,
                        }
                        for segment in packed.pack.segments
                    ],
                }
                for packed in packed_plan.packs
            ],
        }
    )


def admit_compiler_compact_logits_from_packed_plan(
    packed_plan: Any,
    ledger: CompilerLedger,
    *,
    rows: Sequence[Any],
    lineage: PackedCompilerLineage,
) -> AdmittedCompilerCompactLogits:
    """Admit live compact rows against their exact externally built pack plan.

    This is the Task-4/materializer join.  It deliberately constructs no
    ``PreparedCandidateScoring`` and no second packing plan: prompt, Source
    prefix, site, physical position, RP, and acquisition lineage are all
    rederived from the supplied ``PackedPanelPlan`` and admitted ledger.
    """

    from scripts.research.human13_rp_crossover_live_packs import CompilerPackedRow
    from scripts.research.run_human13_k_union_overfit import PackedPanelPlan

    if type(packed_plan) is not PackedPanelPlan:
        raise ValueError("external compiler binding requires an exact PackedPanelPlan")
    admitted = _require_compiler_admission(ledger)
    if not isinstance(lineage, PackedCompilerLineage) or (
        lineage.acquisition_sha256,
        lineage.trajectory_credit_sha256,
        lineage.repetition_penalty,
    ) != (
        admitted.acquisition_sha256,
        admitted.trajectory_credit_sha256,
        admitted.repetition_penalty,
    ):
        raise ValueError("external packed compiler lineage differs from its ledger")

    compact_rows = tuple(rows)
    if not compact_rows or any(
        type(row) is not CompilerPackedRow for row in compact_rows
    ):
        raise ValueError("external compiler rows must be typed materializer rows")
    if len({row.site_id for row in compact_rows}) != len(compact_rows):
        raise ValueError("external compiler rows duplicate a site")

    sites = {
        image.site.site_id: image.site
        for image in admitted.images
        if image.site is not None
    }
    logical_segments = {
        segment.segment_id: segment for segment in packed_plan.logical_segments
    }
    packs = {packed.pack.pack_index: packed for packed in packed_plan.packs}
    plan_sha256 = _packed_panel_plan_sha256(packed_plan)
    bound_rows: list[PackedLogitRows] = []
    for row in compact_rows:
        site = sites.get(row.site_id)
        if site is None:
            raise ValueError("external compiler site is absent from admitted ledger")
        logical = logical_segments.get(site.packed_segment_id)
        if logical is None or logical.image_id != site.image_id:
            raise ValueError("compiler logical segment differs from admitted site")
        encoded_ids = tuple(logical.encoded_example.input_ids)
        prompt = encoded_ids[: site.prompt_token_count]
        prefix_end = site.prompt_token_count + site.source_prefix_token_count
        prefix = encoded_ids[site.prompt_token_count : prefix_end]
        history = encoded_ids[:prefix_end]
        if (
            token_ids_sha256(prompt) != site.prompt_token_sha256
            or token_ids_sha256(prefix) != site.source_prefix_token_sha256
            or len(history) != site.source_history_token_count
            or _history_sha256(history) != site.source_history_sha256
        ):
            raise ValueError("external plan prompt or Source prefix lineage differs")

        packed = packs.get(row.pack_index)
        if packed is None:
            raise ValueError("external compiler pack index is absent from packed plan")
        physical = {
            segment.example_id: segment for segment in packed.pack.segments
        }.get(site.packed_segment_id)
        if physical is None:
            raise ValueError("compiler segment is absent from selected packed plan")
        if tuple(packed.pack.input_ids[physical.start : physical.end]) != encoded_ids:
            raise ValueError(
                "external packed compiler tokens differ from logical segment"
            )
        expected_position = physical.start + site.local_causal_position
        if tuple(row.logits_position_ids) != (expected_position,):
            raise ValueError("external packed compiler causal position differs")
        raw_logits = row.raw_logits
        if (
            not isinstance(raw_logits, torch.Tensor)
            or raw_logits.ndim != 2
            or raw_logits.shape[0] != 1
            or not bool(torch.isfinite(raw_logits.detach()).all().item())
        ):
            raise ValueError(
                "external packed compiler output differs from requested row"
            )

        mapping = {
            "schema_version": "human13_external_packed_compiler_row.v1",
            "packed_panel_plan_sha256": plan_sha256,
            "site_id": site.site_id,
            "segment_id": site.packed_segment_id,
            "pack_index": row.pack_index,
            "packed_causal_position": expected_position,
            "prompt_token_sha256": site.prompt_token_sha256,
            "source_prefix_token_sha256": site.source_prefix_token_sha256,
            "source_history_sha256": site.source_history_sha256,
            "compiler_ledger_sha256": admitted.content_sha256,
            "compiler_admission_sha256": admitted.admission_sha256,
            "acquisition_sha256": lineage.acquisition_sha256,
            "trajectory_credit_sha256": lineage.trajectory_credit_sha256,
            "repetition_penalty": lineage.repetition_penalty,
        }
        bound = object.__new__(PackedLogitRows)
        for field, value in (
            ("site_id", site.site_id),
            ("segment_id", site.packed_segment_id),
            ("pack_index", row.pack_index),
            ("packed_causal_position", expected_position),
            ("mapping_sha256", _sha256(mapping)),
            ("compiler_ledger_sha256", admitted.content_sha256),
            ("compiler_admission_sha256", cast(str, admitted.admission_sha256)),
            ("admission_sha256", ""),
            ("raw_logits", raw_logits),
            ("_factory_marker", _PACKED_ROW_MARKER),
        ):
            object.__setattr__(bound, field, value)
        admission_sha256 = _sha256(_packed_row_admission_preimage(bound))
        object.__setattr__(bound, "admission_sha256", admission_sha256)
        _register_admission(_PACKED_ROW_ADMISSIONS, bound, admission_sha256)
        bound_rows.append(bound)
    return admit_compiler_compact_logits(bound_rows, admitted)


def _packed_row_admission_preimage(value: PackedLogitRows) -> dict[str, Any]:
    return {
        "schema_version": "human13_admitted_prepared_compiler_row.v1",
        "site_id": value.site_id,
        "segment_id": value.segment_id,
        "pack_index": value.pack_index,
        "packed_causal_position": value.packed_causal_position,
        "mapping_sha256": value.mapping_sha256,
        "compiler_ledger_sha256": value.compiler_ledger_sha256,
        "compiler_admission_sha256": value.compiler_admission_sha256,
        "raw_logit_sha256": _compact_tensor_sha256(value.raw_logits),
    }


def _require_packed_row(value: object) -> PackedLogitRows:
    if type(value) is not PackedLogitRows:
        raise ValueError("packed compiler row admission is absent")
    admitted = _PACKED_ROW_ADMISSIONS.get(id(value))
    expected = _sha256(_packed_row_admission_preimage(value))
    if (
        value._factory_marker is not _PACKED_ROW_MARKER
        or admitted is None
        or admitted[0]() is not value
        or admitted[1] != value.admission_sha256
        or value.admission_sha256 != expected
    ):
        raise ValueError("packed compiler row admission is absent or forged")
    return value


def _gather_compiler_compact_logits(
    packed_rows: Sequence[PackedLogitRows],
    ledger: CompilerLedger,
    *,
    require_all: bool,
) -> dict[str, torch.Tensor]:
    if not isinstance(ledger, CompilerLedger):
        raise ValueError("packed compiler gather requires a CompilerLedger")
    sites = {
        image.site.site_id: image.site
        for image in ledger.images
        if image.site is not None
    }
    rows = tuple(packed_rows)
    try:
        rows = tuple(_require_packed_row(row) for row in rows)
    except ValueError as error:
        raise ValueError("packed compiler rows differ or are forged") from error
    row_ids = tuple(row.site_id for row in rows)
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("packed compiler rows must cover sites exactly once")
    if not set(row_ids) <= set(sites):
        raise ValueError("packed compiler rows differ from ledger sites")
    if require_all and set(row_ids) != set(sites):
        raise ValueError("packed compiler rows must cover sites exactly once")
    compact: dict[str, torch.Tensor] = {}
    for row in rows:
        site = sites[row.site_id]
        if ledger.admission_sha256 is not None and (
            row.compiler_ledger_sha256 != ledger.content_sha256
            or row.compiler_admission_sha256 != ledger.admission_sha256
        ):
            raise ValueError("packed compiler row differs from admitted ledger")
        if row.segment_id != site.packed_segment_id:
            raise ValueError("packed compiler segment lineage differs")
        if row.raw_logits.shape[1] <= max(site.compact_token_ids):
            raise ValueError("packed compiler vocabulary omits a required token")
        indexes = torch.tensor(
            site.compact_token_ids,
            dtype=torch.long,
            device=row.raw_logits.device,
        )
        compact[row.site_id] = row.raw_logits[0].index_select(0, indexes)
    return compact


def gather_compiler_compact_logits(
    packed_rows: Sequence[PackedLogitRows], ledger: CompilerLedger
) -> dict[str, torch.Tensor]:
    """Private-shape gather; public training consumes an admitted receipt."""

    return _gather_compiler_compact_logits(packed_rows, ledger, require_all=True)


def _compact_tensor_sha256(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema_version": "human13_compact_logit_tensor.v1",
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
            }
        )
        + detached.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True, init=False)
class AdmittedCompilerCompactLogits:
    """Factory-bound compact rows joined to one admitted compiler ledger."""

    compiler_ledger_sha256: str
    compiler_admission_sha256: str
    site_ids: tuple[str, ...]
    packed_mapping_sha256s: tuple[str, ...]
    compact_tensor_sha256s: tuple[str, ...]
    admission_sha256: str
    _raw_logits: Mapping[str, torch.Tensor] = dataclass_field(repr=False, compare=False)
    _factory_marker: object = dataclass_field(repr=False, compare=False)


def _compact_logits_preimage(value: AdmittedCompilerCompactLogits) -> dict[str, Any]:
    return {
        "schema_version": "human13_admitted_compiler_compact_logits.v1",
        "compiler_ledger_sha256": value.compiler_ledger_sha256,
        "compiler_admission_sha256": value.compiler_admission_sha256,
        "site_ids": list(value.site_ids),
        "packed_mapping_sha256s": list(value.packed_mapping_sha256s),
        "compact_tensor_sha256s": list(value.compact_tensor_sha256s),
    }


def admit_compiler_compact_logits(
    packed_rows: Sequence[PackedLogitRows], ledger: CompilerLedger
) -> AdmittedCompilerCompactLogits:
    """Admit only rows produced by the prepared-pack binding factory."""

    ledger = _require_compiler_admission(ledger)
    rows = tuple(packed_rows)
    compact = _gather_compiler_compact_logits(rows, ledger, require_all=False)
    if not rows:
        raise ValueError("compact-logit receipt requires at least one compiler site")
    result = object.__new__(AdmittedCompilerCompactLogits)
    for field, value in (
        ("compiler_ledger_sha256", ledger.content_sha256),
        ("compiler_admission_sha256", cast(str, ledger.admission_sha256)),
        ("site_ids", tuple(row.site_id for row in rows)),
        ("packed_mapping_sha256s", tuple(row.mapping_sha256 for row in rows)),
        (
            "compact_tensor_sha256s",
            tuple(_compact_tensor_sha256(compact[row.site_id]) for row in rows),
        ),
        ("admission_sha256", ""),
        ("_raw_logits", compact),
        ("_factory_marker", _COMPACT_LOGITS_MARKER),
    ):
        object.__setattr__(result, field, value)
    admission_sha256 = _sha256(_compact_logits_preimage(result))
    object.__setattr__(result, "admission_sha256", admission_sha256)
    _register_admission(_COMPACT_LOGITS_ADMISSIONS, result, admission_sha256)
    return result


def admit_hf_native_compiler_compact_logits(
    ledger: CompilerLedger,
    boundary: SourceBoundaryInput,
    *,
    raw_logits: torch.Tensor,
) -> AdmittedCompilerCompactLogits:
    """Bind one same-session Source row without an old packed-plan surrogate."""

    admitted = _require_compiler_admission(ledger)
    if not isinstance(boundary, SourceBoundaryInput):
        raise ValueError("HF-native compact logits require a Source boundary")
    if admitted.logical_image_count != 1 or len(admitted.images) != 1:
        raise ValueError("HF-native compact logits require one logical image")
    image = admitted.images[0]
    site = image.site
    if (
        image.image_id != 1584
        or image.source_decode_sha256 != boundary.source_decode_sha256
        or site is None
    ):
        raise ValueError("HF-native compact logits lack the required Source site")
    if (
        not isinstance(raw_logits, torch.Tensor)
        or raw_logits.ndim != 2
        or raw_logits.shape[0] != 1
        or not raw_logits.requires_grad
        or not bool(torch.isfinite(raw_logits.detach()).all().item())
    ):
        raise ValueError("HF-native compact logits require one finite graph row")
    if raw_logits.shape[1] <= max(site.compact_token_ids):
        raise ValueError("HF-native compact logits omit a required vocabulary token")
    mapping = {
        "schema_version": "human13_hf_native_compiler_row.v1",
        "site_id": site.site_id,
        "source_decode_sha256": site.source_decode_sha256,
        "prompt_token_sha256": site.prompt_token_sha256,
        "source_prefix_token_sha256": site.source_prefix_token_sha256,
        "source_history_sha256": site.source_history_sha256,
        "generated_token_index": site.generated_token_index,
        "local_causal_position": site.local_causal_position,
        "compiler_ledger_sha256": admitted.content_sha256,
        "compiler_admission_sha256": admitted.admission_sha256,
    }
    row = object.__new__(PackedLogitRows)
    for field, value in (
        ("site_id", site.site_id),
        ("segment_id", site.packed_segment_id),
        ("pack_index", 0),
        ("packed_causal_position", site.local_causal_position),
        ("mapping_sha256", _sha256(mapping)),
        ("compiler_ledger_sha256", admitted.content_sha256),
        ("compiler_admission_sha256", admitted.admission_sha256),
        ("admission_sha256", ""),
        ("raw_logits", raw_logits),
        ("_factory_marker", _PACKED_ROW_MARKER),
    ):
        object.__setattr__(row, field, value)
    row_admission = _sha256(_packed_row_admission_preimage(row))
    object.__setattr__(row, "admission_sha256", row_admission)
    _register_admission(_PACKED_ROW_ADMISSIONS, row, row_admission)
    return admit_compiler_compact_logits((row,), admitted)


def _require_compact_logits(
    value: object, ledger: CompilerLedger
) -> AdmittedCompilerCompactLogits:
    if type(value) is not AdmittedCompilerCompactLogits:
        raise ValueError("scientific compact-logit receipt is required")
    admitted = _COMPACT_LOGITS_ADMISSIONS.get(id(value))
    expected = _sha256(_compact_logits_preimage(value))
    tensor_hashes = tuple(
        _compact_tensor_sha256(value._raw_logits[site_id])
        for site_id in value.site_ids
        if site_id in value._raw_logits
    )
    if (
        value._factory_marker is not _COMPACT_LOGITS_MARKER
        or admitted is None
        or admitted[0]() is not value
        or admitted[1] != value.admission_sha256
        or value.admission_sha256 != expected
        or value.compiler_ledger_sha256 != ledger.content_sha256
        or value.compiler_admission_sha256 != ledger.admission_sha256
        or set(value._raw_logits) != set(value.site_ids)
        or tensor_hashes != value.compact_tensor_sha256s
    ):
        raise ValueError("scientific compact-logit receipt is absent or forged")
    return value


def _processed_compact_logits(
    raw_logits: torch.Tensor,
    site: CompilerSite,
    *,
    repetition_penalty: float,
) -> torch.Tensor:
    if not isinstance(raw_logits, torch.Tensor) or raw_logits.ndim != 1:
        raise ValueError("compact raw logits must be a one-dimensional tensor")
    if raw_logits.numel() != len(site.compact_token_ids):
        raise ValueError("compact raw logits differ from site token ids")
    if not bool(torch.isfinite(raw_logits.detach()).all().item()):
        raise ValueError("compact raw logits must be finite")
    if repetition_penalty not in {1.0, 1.10}:
        raise ValueError("compiler RP must be exactly 1.0 or 1.10")
    logits = raw_logits.to(dtype=torch.float32)
    if repetition_penalty == 1.0 or not site.repeated_token_ids:
        return logits
    repeated = set(site.repeated_token_ids)
    mask = torch.tensor(
        [token_id in repeated for token_id in site.compact_token_ids],
        dtype=torch.bool,
        device=logits.device,
    )
    penalized = torch.where(
        logits < 0, logits * repetition_penalty, logits / repetition_penalty
    )
    return torch.where(mask, penalized, logits)


def greedy_compiler_site_score(
    raw_logits: torch.Tensor,
    site: CompilerSite,
    *,
    repetition_penalty: float,
) -> CompilerSiteScore:
    """Compute the detached-selector valid score and realized-child hinge."""

    if not isinstance(site, CompilerSite):
        raise ValueError("site must be CompilerSite evidence")
    z = _processed_compact_logits(
        raw_logits, site, repetition_penalty=repetition_penalty
    )
    compact_index = {
        token_id: index for index, token_id in enumerate(site.compact_token_ids)
    }
    valid_indices = torch.tensor(
        [compact_index[token_id] for token_id in site.valid_token_ids],
        dtype=torch.long,
        device=z.device,
    )
    valid_logits = z.index_select(0, valid_indices)
    weights = torch.tensor(
        site.valid_token_weights,
        dtype=z.dtype,
        device=z.device,
    )
    valid_score = KAPPA * torch.logsumexp(
        torch.log(weights) + valid_logits / KAPPA, dim=0
    )
    max_valid = valid_logits.max()
    tolerance = (
        torch.finfo(z.dtype).eps * 8.0 * max(1.0, abs(float(max_valid.detach().item())))
    )
    if (
        float(valid_score.detach().item())
        > float(max_valid.detach().item()) + tolerance
    ):
        raise ValueError("normalized valid log-mean-exp exceeds max valid logit")
    bad_score = z[compact_index[site.bad_token_id]]
    hinge = torch.relu(
        torch.as_tensor(MARGIN, dtype=z.dtype, device=z.device)
        + bad_score
        - valid_score
    )
    return CompilerSiteScore(valid_score, max_valid, bad_score, hinge)


def _zero_from_mapping(raw_logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if raw_logits:
        first = next(iter(raw_logits.values()))
        if not isinstance(first, torch.Tensor):
            raise ValueError("compiler logit values must be tensors")
        return first.sum() * 0.0
    return torch.zeros((), dtype=torch.float32)


def _greedy_compiler_numerator_impl(
    raw_logits: Mapping[str, torch.Tensor],
    ledger: CompilerLedger,
    *,
    site_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return an unnormalized pack numerator; absent images contribute zero."""

    if not isinstance(raw_logits, Mapping) or not isinstance(ledger, CompilerLedger):
        raise ValueError("compiler numerator requires a logit mapping and ledger")
    sites = {
        image.site.site_id: image.site
        for image in ledger.images
        if image.site is not None
    }
    selected = tuple(sites) if site_ids is None else tuple(site_ids)
    if len(set(selected)) != len(selected) or not set(selected) <= set(sites):
        raise ValueError("compiler numerator site selection differs from ledger")
    if set(raw_logits) != set(selected):
        raise ValueError(
            "compact logits must cover exactly the selected compiler sites"
        )
    if not selected:
        return _zero_from_mapping(raw_logits)
    terms = [
        greedy_compiler_site_score(
            raw_logits[site_id],
            sites[site_id],
            repetition_penalty=ledger.repetition_penalty,
        ).hinge
        for site_id in selected
    ]
    return torch.stack(terms).sum()


def _greedy_compiler_numerator_for_test(
    raw_logits: Mapping[str, torch.Tensor],
    ledger: CompilerLedger,
    *,
    site_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Private tensor-formula seam for small CPU fixtures."""

    return _greedy_compiler_numerator_impl(raw_logits, ledger, site_ids=site_ids)


def greedy_compiler_numerator(
    compact_logits: AdmittedCompilerCompactLogits,
    ledger: CompilerLedger,
) -> torch.Tensor:
    """Public pack numerator accepts only live admitted compiler evidence."""

    if type(compact_logits) is not AdmittedCompilerCompactLogits:
        raise ValueError("scientific compact-logit receipt is required")
    admitted = _require_compiler_admission(ledger)
    receipt = _require_compact_logits(compact_logits, admitted)
    return _greedy_compiler_numerator_impl(
        receipt._raw_logits, admitted, site_ids=receipt.site_ids
    )


def _greedy_compiler_loss_for_test(
    raw_logits: Mapping[str, torch.Tensor], ledger: CompilerLedger
) -> torch.Tensor:
    """Private exact loss for explicitly small CPU fixtures."""

    return (
        _greedy_compiler_numerator_impl(raw_logits, ledger) / ledger.logical_image_count
    )


def greedy_compiler_loss(
    compact_logits: AdmittedCompilerCompactLogits, ledger: CompilerLedger
) -> torch.Tensor:
    """Apply one global denominator to live admitted compiler evidence."""

    if type(compact_logits) is not AdmittedCompilerCompactLogits:
        raise ValueError("scientific compact-logit receipt is required")
    admitted = _require_compiler_admission(ledger)
    receipt = _require_compact_logits(compact_logits, admitted)
    expected_site_ids = tuple(
        image.site.site_id for image in admitted.images if image.site is not None
    )
    if set(receipt.site_ids) != set(expected_site_ids):
        raise ValueError("compact-logit receipt must cover all compiler sites")
    return (
        _greedy_compiler_numerator_impl(receipt._raw_logits, admitted)
        / admitted.logical_image_count
    )


def combined_loss(
    trajectory_loss: torch.Tensor, compiler_loss: torch.Tensor
) -> torch.Tensor:
    """Arm B's fixed coefficient is exactly one, with no observed-norm tuning."""

    if not isinstance(trajectory_loss, torch.Tensor) or not isinstance(
        compiler_loss, torch.Tensor
    ):
        raise ValueError("combined losses must be tensors")
    return trajectory_loss + compiler_loss


@dataclass(frozen=True, init=False)
class NestedArmArtifacts:
    """In-memory assembly proving B reuses A's exact acquisition/credit bytes."""

    arm_a_shared_artifacts: tuple[bytes, bytes]
    arm_b_shared_artifacts: tuple[bytes, bytes]
    arm_a_compiler_artifact: None
    arm_b_compiler_artifact: bytes
    compiler_ledger_sha256: str
    _marker: object = dataclass_field(repr=False, compare=False)


_NESTED_ARM_MARKER = object()


def _build_nested_arm_artifacts_for_test(
    *,
    acquisition_artifact: bytes,
    trajectory_credit_artifact: bytes,
    compiler_ledger: CompilerLedger,
) -> NestedArmArtifacts:
    if (
        not isinstance(acquisition_artifact, bytes)
        or not acquisition_artifact
        or not isinstance(trajectory_credit_artifact, bytes)
        or not trajectory_credit_artifact
    ):
        raise ValueError("nested arms require nonempty immutable shared artifact bytes")
    if not isinstance(compiler_ledger, CompilerLedger):
        raise ValueError("nested arm B requires a CompilerLedger")
    shared = (acquisition_artifact, trajectory_credit_artifact)
    result = object.__new__(NestedArmArtifacts)
    object.__setattr__(result, "arm_a_shared_artifacts", shared)
    object.__setattr__(result, "arm_b_shared_artifacts", shared)
    object.__setattr__(result, "arm_a_compiler_artifact", None)
    object.__setattr__(
        result, "arm_b_compiler_artifact", compiler_ledger.canonical_bytes
    )
    object.__setattr__(result, "compiler_ledger_sha256", compiler_ledger.content_sha256)
    object.__setattr__(result, "_marker", _NESTED_ARM_MARKER)
    return result


def build_nested_arm_artifacts(
    *,
    acquisition: TrajectoryCreditPanelAcquisition,
    trajectory_credit_ledger: TrajectoryCreditLedger,
    compiler_ledger: CompilerLedger,
) -> NestedArmArtifacts:
    """Assemble A/B only from their admitted, mutually bound artifacts."""

    if type(acquisition) is not TrajectoryCreditPanelAcquisition:
        raise ValueError("nested arms require exact Task-2 panel acquisition")
    acquisition = TrajectoryCreditPanelAcquisition(acquisition.publications)
    credit = _require_scientific_ledger_admission(trajectory_credit_ledger)
    compiler = _require_compiler_admission(compiler_ledger)
    if (
        acquisition.content_sha256 != credit.acquisition_sha256
        or compiler.acquisition_sha256 != credit.acquisition_sha256
        or compiler.trajectory_credit_sha256 != credit.content_sha256
        or compiler.source_sha256 != credit.source_sha256
        or compiler.manifest_sha256 != credit.manifest_sha256
    ):
        raise ValueError("nested A/B acquisition or credit lineage differs")
    return _build_nested_arm_artifacts_for_test(
        acquisition_artifact=_canonical_bytes(acquisition._preimage()),
        trajectory_credit_artifact=_canonical_bytes(credit.to_dict()),
        compiler_ledger=compiler,
    )


__all__ = [
    "EXACT_ALIAS_COUNT",
    "KAPPA",
    "MARGIN",
    "AdmittedSourceCompilerPanel",
    "AdmittedSourceGreedyDecode",
    "AliasChild",
    "CompilerImageLedger",
    "CompilerLedger",
    "CompilerSite",
    "CompilerSiteScore",
    "NestedArmArtifacts",
    "PackedCompilerLineage",
    "PackedLogitRows",
    "SourceBoundaryInput",
    "admit_source_compiler_panel",
    "admit_hf_native_compiler_compact_logits",
    "admit_compiler_compact_logits_from_packed_plan",
    "admit_source_greedy_decode",
    "bind_packed_compiler_logits",
    "build_compiler_ledger",
    "construct_hf_native_one_image_compiler_ledger",
    "build_nested_arm_artifacts",
    "combined_loss",
    "gather_compiler_compact_logits",
    "greedy_compiler_loss",
    "greedy_compiler_numerator",
    "greedy_compiler_site_score",
    "load_compiler_ledger",
]
