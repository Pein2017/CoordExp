"""GPU-facing no-update census capture over injected Human-13 runtimes.

The helper owns causal-site and packed-position mapping only.  Processor-built
prompt skeletons, an already loaded packed model/runtime/tokenizer, and an
already opened exact HF fp32/SDPA scorer are injected by the caller.  Importing
this module performs no model load, forward, or GPU action.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch

from scripts.research.materialize_human13_no_update_census import (
    CoherentSitePlan,
    ExactLogitEvidence,
    Human13CensusPlan,
    PlanSite,
)
from scripts.research.run_human13_k_union_overfit import (
    GLOBAL_MAX_LENGTH,
    LogicalPanelSegment,
    PackedPanelMicroStep,
    PackedPanelPlan,
    plan_panel_packs,
)
from src.inference.backend import token_ids_sha256
from src.qwen.fa2 import build_fa2_varlen_plan
from src.qwen.forward import build_qwen_forward_inputs, run_qwen_forward


PackedForward = Callable[[Any, Any, Any, PackedPanelMicroStep, tuple[int, ...]], Any]


@dataclass(frozen=True)
class CaptureSiteBinding:
    site_id: str
    surface: str
    segment_id: str
    local_causal_position: int


@dataclass(frozen=True)
class PreparedCensusCapture:
    packed_plan: PackedPanelPlan
    bindings: tuple[CaptureSiteBinding, ...]
    coherent_segment_ids: tuple[str, ...]
    isolated_trie_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class Human13LiveCensusResult:
    evidence: tuple[ExactLogitEvidence, ...]
    receipt: Mapping[str, int]


def _prompt_ids(
    skeleton: Any, expected_sha256: str, *, image_id: int
) -> tuple[int, ...]:
    count = getattr(skeleton, "prompt_token_count", None)
    input_ids = getattr(skeleton, "input_ids", None)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(input_ids, tuple)
        or len(input_ids) < count
    ):
        raise ValueError(f"image {image_id} has an invalid processor prompt skeleton")
    prompt = tuple(int(token_id) for token_id in input_ids[:count])
    if token_ids_sha256(prompt) != expected_sha256:
        raise ValueError(f"image {image_id} prompt token identity differs from census")
    return prompt


def _clone_skeleton(
    skeleton: Any,
    *,
    segment_id: str,
    image_id: int,
    input_ids: tuple[int, ...],
) -> Any:
    if is_dataclass(skeleton):
        encoded = replace(skeleton, example_id=segment_id, input_ids=input_ids)
        object.__setattr__(encoded, "human13_image_id", image_id)
        return encoded
    values = dict(vars(skeleton))
    values.update(
        example_id=segment_id,
        human13_image_id=image_id,
        input_ids=input_ids,
    )
    return SimpleNamespace(**values)


def _logical_segment(
    *,
    skeleton: Any,
    segment_id: str,
    image_id: int,
    role: str,
    input_ids: tuple[int, ...],
) -> LogicalPanelSegment:
    return LogicalPanelSegment(
        segment_id=segment_id,
        image_id=image_id,
        role=role,  # type: ignore[arg-type]
        encoded_example=_clone_skeleton(
            skeleton,
            segment_id=segment_id,
            image_id=image_id,
            input_ids=input_ids,
        ),
    )


def prepare_census_capture(
    *,
    plan: Human13CensusPlan,
    prompt_skeletons: Mapping[int, Any],
    global_max_length: int = GLOBAL_MAX_LENGTH,
) -> PreparedCensusCapture:
    """Build the minimum distinct packed contexts for trie and A1 sites."""

    if plan.schema_version != "human13_no_update_census_plan.v1":
        raise ValueError("Human-13 census plan schema differs")
    segment_plan_by_id = {segment.segment_id: segment for segment in plan.segments}
    if len(segment_plan_by_id) != len(plan.segments):
        raise ValueError("A1 census segment identities are not unique")
    coherent_by_segment: dict[str, list[CoherentSitePlan]] = {}
    for site in plan.coherent_sites:
        coherent_by_segment.setdefault(site.segment_id, []).append(site)
    if set(coherent_by_segment) != set(segment_plan_by_id):
        raise ValueError("coherent sites do not cover every A1 census segment")

    logical: list[LogicalPanelSegment] = []
    bindings: list[CaptureSiteBinding] = []
    context_binding: dict[tuple[int, tuple[int, ...]], tuple[str, int]] = {}
    clean_prefix_by_image: dict[int, tuple[int, ...]] = {}

    for segment_id, segment in segment_plan_by_id.items():
        skeleton = prompt_skeletons.get(segment.image_id)
        if skeleton is None:
            raise ValueError(
                f"missing processor prompt skeleton for image {segment.image_id}"
            )
        prompt = _prompt_ids(
            skeleton,
            segment.prompt_token_ids_sha256,
            image_id=segment.image_id,
        )
        continuation = tuple(segment.token_ids)
        if continuation[: len(segment.fixed_prefix_token_ids)] != tuple(
            segment.fixed_prefix_token_ids
        ):
            raise ValueError("A1 segment does not begin with its frozen P_clean")
        clean_prefix_by_image[segment.image_id] = tuple(segment.fixed_prefix_token_ids)
        encoded_ids = (*prompt, *continuation)
        logical.append(
            _logical_segment(
                skeleton=skeleton,
                segment_id=segment_id,
                image_id=segment.image_id,
                role="a1_full_h",
                input_ids=encoded_ids,
            )
        )
        ordered_sites = tuple(
            sorted(
                coherent_by_segment[segment_id],
                key=lambda site: site.segment_token_offset,
            )
        )
        if tuple(site.segment_token_offset for site in ordered_sites) != tuple(
            range(len(segment.fixed_prefix_token_ids), len(continuation))
        ):
            raise ValueError("coherent A1 target offsets are not contiguous")
        for site in ordered_sites:
            offset = site.segment_token_offset
            if (
                offset < 0
                or offset >= len(continuation)
                or continuation[:offset] != tuple(site.segment_prefix_token_ids)
                or continuation[offset] != site.target_token_id
            ):
                raise ValueError("coherent A1 site prefix/target binding differs")
            local_position = len(prompt) + offset - 1
            bindings.append(
                CaptureSiteBinding(
                    site_id=site.site_id,
                    surface="packed",
                    segment_id=segment_id,
                    local_causal_position=local_position,
                )
            )
            context_binding[(site.image_id, tuple(site.segment_prefix_token_ids))] = (
                segment_id,
                local_position,
            )

    rows_by_image: dict[int, list[Any]] = {}
    for row in plan.selected_rows:
        rows_by_image.setdefault(row.image_id, []).append(row)
    isolated_by_leaf: dict[tuple[int, tuple[int, ...]], str] = {}
    for site in plan.trie_sites:
        skeleton = prompt_skeletons.get(site.image_id)
        if skeleton is None:
            raise ValueError(
                f"missing processor prompt skeleton for image {site.image_id}"
            )
        prompt = _prompt_ids(
            skeleton,
            site.prompt_token_ids_sha256,
            image_id=site.image_id,
        )
        clean_prefix = clean_prefix_by_image.get(site.image_id)
        if clean_prefix is None or tuple(site.model_prefix_token_ids) != (
            *clean_prefix,
            *site.prefix_token_ids,
        ):
            raise ValueError(
                "trie model prefix differs from frozen P_clean plus row prefix"
            )
        context = (site.image_id, tuple(site.model_prefix_token_ids))
        resolved = context_binding.get(context)
        if resolved is None:
            prefix = tuple(site.prefix_token_ids)
            viable = set(site.viable_child_token_ids)
            candidates = tuple(
                row
                for row in rows_by_image.get(site.image_id, ())
                if len(row.token_ids) > len(prefix)
                and tuple(row.token_ids[: len(prefix)]) == prefix
                and row.token_ids[len(prefix)] in viable
            )
            if not candidates:
                raise ValueError("trie site is not covered by any selected native row")
            row = min(candidates, key=lambda item: (item.token_ids, item.row_id))
            leaf_key = (site.image_id, tuple(row.token_ids))
            segment_id = isolated_by_leaf.get(leaf_key)
            if segment_id is None:
                segment_id = f"census:trie-row:{site.image_id}:{row.row_id}"
                isolated_by_leaf[leaf_key] = segment_id
                logical.append(
                    _logical_segment(
                        skeleton=skeleton,
                        segment_id=segment_id,
                        image_id=site.image_id,
                        role="h1_independent",
                        input_ids=(*prompt, *clean_prefix, *row.token_ids),
                    )
                )
            resolved = (
                segment_id,
                len(prompt) + len(clean_prefix) + len(prefix) - 1,
            )
        bindings.append(
            CaptureSiteBinding(
                site_id=site.site_id,
                surface="trie",
                segment_id=resolved[0],
                local_causal_position=resolved[1],
            )
        )

    expected = {
        *((site.site_id, "trie") for site in plan.trie_sites),
        *((site.site_id, "packed") for site in plan.coherent_sites),
    }
    observed = {(binding.site_id, binding.surface) for binding in bindings}
    if observed != expected or len(bindings) != len(expected):
        raise ValueError("prepared packed census site coverage differs")
    packed_plan = plan_panel_packs(logical, global_max_length=global_max_length)
    return PreparedCensusCapture(
        packed_plan=packed_plan,
        bindings=tuple(bindings),
        coherent_segment_ids=tuple(segment_plan_by_id),
        isolated_trie_segment_ids=tuple(isolated_by_leaf.values()),
    )


def _vocab_size(tokenizer: Any) -> int:
    try:
        value = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise ValueError(
            "loaded tokenizer does not expose its full vocabulary"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("loaded tokenizer has an invalid full vocabulary")
    return value


def _position_rows(
    result: Any,
    *,
    expected_positions: tuple[int, ...],
    expected_vocab_size: int,
    label: str,
    require_fp32: bool = False,
) -> dict[int, torch.Tensor]:
    logits = getattr(result, "logits", None)
    positions = getattr(result, "logits_position_ids", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"{label} must return logits with shape [1,N,V]")
    if int(logits.shape[2]) != expected_vocab_size:
        raise ValueError(f"{label} did not return the full vocabulary")
    if require_fp32 and logits.dtype != torch.float32:
        raise ValueError("HF census logits must be fp32")
    if positions is None:
        raise ValueError(f"{label} logits position coverage is missing")
    position_ids = tuple(int(position) for position in positions)
    if (
        len(position_ids) != int(logits.shape[1])
        or len(set(position_ids)) != len(position_ids)
        or set(position_ids) != set(expected_positions)
    ):
        raise ValueError(f"{label} logits position coverage differs")
    return {
        position: logits[0, row].detach().cpu().contiguous()
        for row, position in enumerate(position_ids)
    }


def _default_packed_forward(
    model: Any,
    runtime: Any,
    tokenizer: Any,
    packed: PackedPanelMicroStep,
    positions: tuple[int, ...],
) -> Any:
    accelerator = getattr(runtime, "accelerator", None)
    device = getattr(accelerator, "device", None)
    if device is None:
        raise ValueError("packed runtime does not expose accelerator.device")
    inputs = build_qwen_forward_inputs(
        packed.pack,
        packed.encoded_examples,
        packed.position_inputs,
        fa2_varlen_plan=build_fa2_varlen_plan(packed.pack, device=device),
        logits_to_keep_positions=positions,
        device=device,
        fa2_branch_proof_policy="human13_no_update_census",
    )
    with torch.inference_mode():
        return run_qwen_forward(
            model,
            inputs,
            expected_vocab_size=_vocab_size(tokenizer),
            capture_fa2_branch=True,
            require_fa2_branch_proof=True,
        )


def _site_by_id(plan: Human13CensusPlan) -> dict[str, PlanSite]:
    result: dict[str, PlanSite] = {}
    for site in (*plan.trie_sites, *plan.coherent_sites):
        if site.site_id in result:
            raise ValueError("census plan site identities are not unique")
        result[site.site_id] = site
    return result


def _capture_packed(
    *,
    plan: Human13CensusPlan,
    prepared: PreparedCensusCapture,
    packed_model: Any,
    packed_runtime: Any,
    tokenizer: Any,
    packed_forward: PackedForward,
) -> tuple[ExactLogitEvidence, ...]:
    bindings_by_segment: dict[str, list[CaptureSiteBinding]] = {}
    for binding in prepared.bindings:
        bindings_by_segment.setdefault(binding.segment_id, []).append(binding)
    sites = _site_by_id(plan)
    evidence: list[ExactLogitEvidence] = []
    for packed in prepared.packed_plan.packs:
        segment_ranges = {
            segment.example_id: (segment.start, segment.end)
            for segment in packed.pack.segments
        }
        resolved: list[tuple[CaptureSiteBinding, int]] = []
        for segment_id, (start, end) in segment_ranges.items():
            for binding in bindings_by_segment.get(segment_id, ()):
                position = start + binding.local_causal_position
                if not start <= position < end:
                    raise ValueError(
                        "packed census causal position escapes its segment"
                    )
                resolved.append((binding, position))
        positions = tuple(sorted({position for _, position in resolved}))
        if not positions:
            raise ValueError(
                "packed census micro-step has no requested logit positions"
            )
        output = packed_forward(
            packed_model,
            packed_runtime,
            tokenizer,
            packed,
            positions,
        )
        rows = _position_rows(
            output,
            expected_positions=positions,
            expected_vocab_size=_vocab_size(tokenizer),
            label="packed",
        )
        for binding, position in resolved:
            evidence.append(
                ExactLogitEvidence.observed(
                    plan=plan,
                    site=sites[binding.site_id],
                    surface=binding.surface,  # type: ignore[arg-type]
                    logits=rows[position],
                )
            )
    expected = {
        *((site.site_id, "trie") for site in plan.trie_sites),
        *((site.site_id, "packed") for site in plan.coherent_sites),
    }
    if {(item.site_id, item.surface) for item in evidence} != expected or len(
        evidence
    ) != len(expected):
        raise ValueError("packed exact logit evidence coverage differs")
    return tuple(evidence)


def _capture_hf(
    *,
    plan: Human13CensusPlan,
    prepared: PreparedCensusCapture,
    hf_scorer: Any,
    tokenizer: Any,
) -> tuple[ExactLogitEvidence, ...]:
    if getattr(hf_scorer, "model_dtype", None) != "torch.float32":
        raise ValueError("HF census scorer must be exact fp32")
    if getattr(hf_scorer, "attention_implementation", None) != "sdpa":
        raise ValueError("HF census scorer must use SDPA")
    score = getattr(hf_scorer, "score_causal_logits", None)
    if not callable(score):
        raise ValueError("HF census scorer lacks score_causal_logits")
    coherent_bindings = tuple(
        binding for binding in prepared.bindings if binding.surface == "packed"
    )
    by_segment: dict[str, list[CaptureSiteBinding]] = {}
    for binding in coherent_bindings:
        by_segment.setdefault(binding.segment_id, []).append(binding)
    encoded_by_id = {
        segment.segment_id: segment.encoded_example
        for segment in prepared.packed_plan.logical_segments
    }
    sites = _site_by_id(plan)
    evidence: list[ExactLogitEvidence] = []
    for segment_id in prepared.coherent_segment_ids:
        bindings = by_segment.get(segment_id, ())
        positions = tuple(
            sorted({binding.local_causal_position for binding in bindings})
        )
        if not positions:
            raise ValueError("coherent HF segment has no requested logit positions")
        output = score(encoded_by_id[segment_id], positions)
        rows = _position_rows(
            output,
            expected_positions=positions,
            expected_vocab_size=_vocab_size(tokenizer),
            label="HF",
            require_fp32=True,
        )
        for binding in bindings:
            evidence.append(
                ExactLogitEvidence.observed(
                    plan=plan,
                    site=sites[binding.site_id],
                    surface="hf",
                    logits=rows[binding.local_causal_position],
                )
            )
    expected = {(site.site_id, "hf") for site in plan.coherent_sites}
    if {(item.site_id, item.surface) for item in evidence} != expected or len(
        evidence
    ) != len(expected):
        raise ValueError("HF exact logit evidence coverage differs")
    return tuple(evidence)


def capture_human13_live_census(
    *,
    plan: Human13CensusPlan,
    prompt_skeletons: Mapping[int, Any],
    packed_model: Any,
    packed_runtime: Any,
    tokenizer: Any,
    hf_scorer: Any,
    packed_forward: PackedForward | None = None,
    global_max_length: int = GLOBAL_MAX_LENGTH,
) -> Human13LiveCensusResult:
    """Capture complete trie/packed/HF evidence without loading any runtime."""

    if getattr(hf_scorer, "model_dtype", None) != "torch.float32":
        raise ValueError("HF census scorer must be exact fp32")
    if getattr(hf_scorer, "attention_implementation", None) != "sdpa":
        raise ValueError("HF census scorer must use SDPA")
    prepared = prepare_census_capture(
        plan=plan,
        prompt_skeletons=prompt_skeletons,
        global_max_length=global_max_length,
    )
    packed_evidence = _capture_packed(
        plan=plan,
        prepared=prepared,
        packed_model=packed_model,
        packed_runtime=packed_runtime,
        tokenizer=tokenizer,
        packed_forward=packed_forward or _default_packed_forward,
    )
    hf_evidence = _capture_hf(
        plan=plan,
        prepared=prepared,
        hf_scorer=hf_scorer,
        tokenizer=tokenizer,
    )
    evidence = (*packed_evidence, *hf_evidence)
    expected_count = len(plan.trie_sites) + 2 * len(plan.coherent_sites)
    if len(evidence) != expected_count:
        raise ValueError("live census exact logit evidence coverage differs")
    return Human13LiveCensusResult(
        evidence=evidence,
        receipt={
            "logical_segment_count": len(prepared.packed_plan.logical_segments),
            "coherent_segment_count": len(prepared.coherent_segment_ids),
            "isolated_trie_segment_count": len(prepared.isolated_trie_segment_ids),
            "physical_pack_count": len(prepared.packed_plan.packs),
            "packed_forward_count": len(prepared.packed_plan.packs),
            "hf_forward_count": len(prepared.coherent_segment_ids),
            "trie_site_count": len(plan.trie_sites),
            "coherent_site_count": len(plan.coherent_sites),
            "evidence_count": len(evidence),
        },
    )


__all__ = [
    "CaptureSiteBinding",
    "Human13LiveCensusResult",
    "PreparedCensusCapture",
    "capture_human13_live_census",
    "prepare_census_capture",
]
