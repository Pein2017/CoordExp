# Stage-1 Static Packing Exact Atom-Position Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review status: REVIEW-HOLD RESOLUTION DRAFT as of 2026-06-25. Do not
> implement until the user gives final approval.**
> A previous review round incorrectly interpreted "single image input" as "one
> image per packed forward." The corrected contract is: each original training
> sample has exactly one image plus one GT assistant response, while Stage-1
> static packing may concatenate multiple such samples into one long sequence
> and therefore one model forward may contain multiple images. Packed loss
> semantics should be batch-equivalent: each packed segment contributes its own
> teacher-forced CE/objective/ledger terms, with explicit segment offsets and no
> cross-segment object or metric leakage.
>
> This plan is still not implementation-approved. The reviewed roadmap now
> specifies the batch-like packed-multi-sample contract: one image per original
> sample, no videos, no multi-image samples, exact text atom offsets,
> segment-isolated attention boundaries, bridge-local visual-token offsets,
> production `packing_enabled` wiring from packed batch extras, and per-forward
> metric aggregation that matches the normalized loss contribution actually
> added to the forward.

**Goal:** Enable Stage-1 latest teacher-forcing detection configs, including coverage-ledger auxiliary loss, to use deterministic static packing with exact atom-position and sidecar offset mapping while keeping Stage-2 self-rollout packing rejected.

**Architecture:** Reuse the standard static-packing forward semantics only if
they are proven to provide segment-aware attention isolation; a packed row must
not behave like one natural sequence with cross-sample attention. Add one
explicit packed-segment map that is produced by the batch-extras collator after
static-packed rows are flattened. Rewrite teacher-forcing IR atoms and
coverage-ledger sidecars from segment-local token positions to packed-row token
positions, and pass those rewritten sidecars through existing non-model
batch-extras plumbing. Coverage-ledger sidecars remain sample-local for
geometry: every sidecar describes one image and its objects. The bridge validates
packed segment offsets, sidecar order, and Qwen image-grid row order before it
maps each sidecar's bbox regions against that sidecar-local `image_grid_thw` and
applies the packed segment's cumulative visual-token start before pooling from
the concatenated Qwen image embeddings.

**Tech Stack:** Python dataclasses, PyTorch tensors, CoordExp static packing dataset wrapper, ms-swift Qwen3-VL collator/position metadata, pytest.

---

## Scope And Decisions

- Apply to Stage-1 `objective.id=research_teacher_forcing` with `training.packing=true` and `packing.static_packing=true`.
- Keep the explicit config knob `packing.padding_free_packed=true` rejected for latest teacher-forcing and coverage-ledger until a separate FlashAttention varlen runtime gate is specified. This does not remove the need to validate the existing static-packing template path: `StaticPackedCaptionDataset` sets `template.packing=True` and `template.padding_free=True`, so implementation must prove the collated Qwen batch carries the required 4-row `position_ids`/`text_position_ids` metadata and segment order expected by `prepare_forward_inputs`.
- V1 supports `training.eval_packing=true` for the standard SFT evaluation
  step. HF/Swift evaluation calls the trainer `compute_loss` path when labels
  are present, so eval packing should use the same packed collator extras,
  Qwen packed forward metadata, and `TrainerLossBridge` path as training.
  Add a regression test that `prediction_step`/eval loss reaches the packed
  teacher-forcing bridge with `packing_enabled=True`.
- Keep Stage-2 self-rollout and rollout-correction ownership paths rejected from this feature. Existing Stage-2 trainer-owned post-rollout packing remains separate and is not reused.
- Interpret `training.effective_batch_size: 32` as 32 packed sequence rows per optimizer step. With `per_device_train_batch_size: 1`, this yields grad accumulation 8 on 4 GPUs and 4 on 8 GPUs.
- Keep `training.encoded_sample_cache.enabled=false` for packed latest teacher-forcing in this feature. Static packing length/plan cache is allowed.
- Reject authored `training.encoded_sample_cache.enabled=true` for this feature even when another runtime path would bypass an ineligible cache. The packed contract should be explicit, not metadata-bypass-dependent.
- Preserve exactly one image per original sample. A packed row may contain multiple images only because it contains multiple original samples.
- Reject videos, multi-frame samples, and samples that contain more than one image.
- Treat packed training as batch-like accumulation over packed segments with
  no cross-segment attention leakage. Ledger coverage and row-object binding
  negatives are local to each segment/sidecar, not cross-image or cross-sample
  negatives.
- Before any static-packed teacher-forcing/ledger loss is trusted, run a real
  two-segment packed-forward leakage test: perturb segment A tokens and assert
  segment B logits/final hidden states are unchanged, and vice versa. If this
  fails, stop implementation and redesign the runtime attention path instead of
  continuing with sidecar remapping.
- Normalize packed coverage-ledger auxiliary loss by the same semantic unit as
  the unpacked objective, not by raw segment count accidentally induced by pack
  fill. The metric contribution must equal the normalized scalar added to the
  forward loss; do not sum per-segment mean losses unless the unpacked
  equivalence test proves that is the intended training scale.
- Treat "batch-like" as an implementation contract to verify, not a slogan:
  labels, image placeholders, `image_grid_thw` rows, Qwen 4-row
  `position_ids`, FlashAttention varlen boundaries (`cu_seq_lens_q`,
  `cu_seq_lens_k`, `max_length_q`, `max_length_k`) when present,
  `text_position_ids` reset points, packed segment offsets, and sidecar ranges
  must share one physical flattening order.
- Do not rely on a plain 2D `attention_mask` to isolate packed samples. Prefer
  explicit varlen boundary metadata from the collator/batch builder. If the
  upstream static-packing path infers boundaries from Qwen packed
  `position_ids`, tests and preflight must prove those inferred boundaries
  exactly match `packed_segment_offsets`; otherwise fail fast.
- Keep `PackedSegmentOffset` text/sample-local. Do not store `visual_token_start` there; compute cumulative visual-token starts in the coverage-ledger bridge from the validated sidecar and `image_grid_thw` order.
- Keep `CoverageLedgerObjectEntry.image_index == 0` as a sidecar-local invariant. Do not encode packed global image order by mutating object entries; carry packed media order through segment metadata and visual-token offsets.
- Require globally unique `sample_id` values across every flattened segment in a
  trainer batch. The current mapper is keyed by sample id; duplicate ids across
  packs must fail fast instead of overwriting silently.

## File Map

- Create: `src/training/teacher_forcing/packing_offsets.py`
  - Owns packed-segment metadata, token-position shifting for `TeacherForcingTargetIR`, and validation helpers.
- Modify: `src/training/coverage_ledger/sidecars.py`
  - Add pure functions that return shifted coverage-ledger sidecars without changing row-local builder behavior.
- Modify: `src/training/coverage_ledger/visual_regions.py`
  - Add a helper that offsets region flattened indices by a cumulative visual-token start.
- Modify: `src/training/coverage_ledger/qwen_capture.py`
  - Support packed multi-sample image forwards by accepting `image_grid_thw.shape == (num_segments, 3)` when every row has `T == 1`, while still rejecting videos, multi-frame rows, and missing image placeholders.
- Modify: `src/data_collators/enrichers.py`
  - Replace packed fail-fast branches for teacher-forcing IR and coverage-ledger sidecars with exact packed rewrite logic.
- Modify: `src/trainers/batch_extras.py`
  - Add `packed_segment_offsets` as a non-model batch extra.
- Modify: `src/detection/dataset.py`
  - Register `packed_segment_offsets` as a trainer batch extra so detection-side
    stripping and direct callers agree on ownership.
- Modify: `src/training/encoding/model_inputs.py`
  - Register `packed_segment_offsets` as a runner/sidecar-owned key so direct
    bridge callers and future trainer surfaces have explicit key ownership.
- Modify: `src/trainers/metrics/teacher_forcing.py`
  - Build packed-aware supervision without relying on `sample_id_to_batch_index` for shifted packed atoms.
- Modify: `src/training/bridge/loss_bridge.py`
  - Accept multiple shifted coverage-ledger sidecars only for validated packed
    rows and aggregate their normalized losses/metrics.
- Modify: `src/config/schema.py`
  - Loosen both generic latest teacher-forcing packing guards and the
    coverage-ledger-specific guard only for Stage-1 `training.packing=true`
    plus `packing.static_packing=true`.
  - Keep Stage-2/self-rollout and `padding_free_packed` rejection.
- Modify: `src/detection/runtime.py`
  - Mirror schema changes in runtime support checks.
- Modify: `src/sft.py`
  - Ensure Stage-1 static packing policy allows latest teacher-forcing after the new guards are present.
- Modify: `src/training/coverage_ledger/preflight.py`
  - Build or fixture a deterministic two-segment static pack and persist packed
    alignment evidence for the no-training gate.
- Modify: `scripts/training/coverage_ledger_preflight.py`
  - Expose the packed preflight evidence path if needed by the script entrypoint.
- Modify: `docs/data/PACKING.md`
  - Update the Stage-1 compact detection row in the matrix from disabled to static dataset packing for latest teacher-forcing only.
- Test: `tests/test_teacher_forcing_packed_offsets.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`
- Test: `tests/test_batch_extras_contract.py`
- Test: `tests/test_teacher_forcing_config_contract.py`
- Test: `tests/test_detection_training_config_contract.py`
- Test: `tests/test_stage1_static_packing_runtime_config.py`
- Test: `tests/test_coverage_ledger_bridge_integration.py`
- Test: `tests/test_coverage_ledger_qwen_capture.py`
- Test: `tests/test_coverage_ledger_loss.py`
- Test: `tests/test_coverage_ledger_preflight_artifacts.py`

---

### Task 0: Prove Packed Attention Isolation Before Sidecar Work

**Files:**
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 1: Write the real static-pack leakage test**

Add a test that builds a real two-segment static-packed Qwen-style detection
batch through the same template/collator path used by Stage-1 static packing.
Run the model or a minimal Qwen-compatible attention probe twice:

- baseline packed row `[segment_a, segment_b]`;
- the same row with only segment A text tokens perturbed outside segment B.

Assert segment B logits and final hidden states at supervised positions are
unchanged within deterministic tolerance. Repeat in the opposite direction for
segment A. The test must also assert that `cu_seq_lens_q/cu_seq_lens_k` and
`max_length_q/max_length_k`, when present, match `packed_segment_offsets`; when
the upstream path relies on Qwen packed `position_ids` inference, assert
`text_position_ids` reset points match the same offsets.

- [ ] **Step 2: Run and interpret the leakage test**

Run:

```bash
pytest tests/test_teacher_forcing_sidecar_bridge.py::test_static_packed_forward_has_no_cross_segment_attention_leakage -q
```

Expected outcomes:

- If the test passes, continue to Task 1.
- If it fails, stop this implementation plan. Do not implement only sidecar
  remapping on top of a leaking forward path. Draft a smaller runtime-attention
  redesign plan, likely around explicit FlashAttention varlen boundaries, and
  ask for user approval before continuing.

---

### Task 1: Add Packed Segment Metadata And Teacher-Forcing IR Rewriter

**Files:**
- Create: `src/training/teacher_forcing/packing_offsets.py`
- Test: `tests/test_teacher_forcing_packed_offsets.py`

- [ ] **Step 1: Write failing tests for segment offsets and atom shifting**

Create `tests/test_teacher_forcing_packed_offsets.py`:

```python
from __future__ import annotations

import torch
import pytest

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.packing_offsets import (
    PackedSegmentOffset,
    build_packed_segment_offsets,
    shift_teacher_forcing_target_ir,
)
from src.training.teacher_forcing.roles import TokenRole


def _atom(*, batch_index: int = 0, logit_position: int, target_position: int):
    return SupervisionAtom(
        batch_index=batch_index,
        logit_position=logit_position,
        target_position=target_position,
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({100 + target_position}),
        selected_token_id=100 + target_position,
        latent_valid_token_ids=frozenset(),
        coverage_target_weights=None,
        loss_tags=frozenset({"coord"}),
        loss_weight=1.0,
        coord_role="x1",
        provenance={
            "continuation_boundary": True,
            "bbox_positive_area": True,
            "coord_label_positions": (2, 3, 4, 5),
            "position_list": [6, 7],
            "nested": {"box_end_position": 8},
            "bbox_positive_area_valid_token_ids": (11, 12),
            "bbox_positive_area_invalid_token_ids": (13, 14),
        },
    )


def test_build_packed_segment_offsets_uses_row_local_lengths_and_pack_rows():
    raw_batch = [
        [
            {"sample_id": "a", "input_ids": [1, 2, 3]},
            {"sample_id": "b", "length": 5},
        ]
    ]
    collated = {
        "input_ids": torch.zeros((1, 8), dtype=torch.long),
        "labels": torch.zeros((1, 8), dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }

    offsets = build_packed_segment_offsets(raw_batch=raw_batch, collated=collated)

    assert offsets == (
        PackedSegmentOffset(
            pack_index=0,
            segment_index=0,
            sample_id="a",
            token_start=0,
            token_end=3,
        ),
        PackedSegmentOffset(
            pack_index=0,
            segment_index=1,
            sample_id="b",
            token_start=3,
            token_end=8,
        ),
    )


def test_build_packed_segment_offsets_rejects_length_mismatch():
    raw_batch = [[{"sample_id": "a", "input_ids": [1, 2, 3]}]]
    collated = {
        "input_ids": torch.zeros((1, 8), dtype=torch.long),
        "labels": torch.zeros((1, 8), dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }

    with pytest.raises(ValueError, match=r"non-padding token count"):
        build_packed_segment_offsets(raw_batch=raw_batch, collated=collated)


def test_build_packed_segment_offsets_rejects_duplicate_sample_ids_across_packs():
    raw_batch = [
        [{"sample_id": "dup", "input_ids": [1, 2]}],
        [{"sample_id": "dup", "input_ids": [3, 4]}],
    ]
    collated = {
        "input_ids": torch.zeros((2, 2), dtype=torch.long),
        "labels": torch.zeros((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }

    with pytest.raises(ValueError, match=r"duplicate packed sample_id"):
        build_packed_segment_offsets(raw_batch=raw_batch, collated=collated)


def test_shift_teacher_forcing_target_ir_offsets_atom_positions_and_provenance():
    ir = TeacherForcingTargetIR(
        schema_version=1,
        atoms=(_atom(logit_position=1, target_position=2),),
        metadata={"sample_id": "a", "prompt_end_position": 10},
    )
    offset = PackedSegmentOffset(
        pack_index=0,
        segment_index=1,
        sample_id="a",
        token_start=30,
        token_end=44,
    )

    shifted = shift_teacher_forcing_target_ir(ir, offset)

    assert shifted.atoms[0].batch_index == 0
    assert shifted.atoms[0].logit_position == 31
    assert shifted.atoms[0].target_position == 32
    assert shifted.atoms[0].provenance["coord_label_positions"] == (32, 33, 34, 35)
    assert shifted.atoms[0].provenance["position_list"] == (36, 37)
    assert shifted.atoms[0].provenance["nested"]["box_end_position"] == 38
    assert shifted.atoms[0].provenance["bbox_positive_area_valid_token_ids"] == (11, 12)
    assert shifted.atoms[0].provenance["bbox_positive_area_invalid_token_ids"] == (13, 14)
    assert shifted.metadata["packed"] is True
    assert shifted.metadata["pack_index"] == 0
    assert shifted.metadata["segment_index"] == 1
    assert shifted.metadata["token_start"] == 30
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_teacher_forcing_packed_offsets.py -q
```

Expected: import failure for `src.training.teacher_forcing.packing_offsets`.

- [ ] **Step 3: Implement packed segment offset helpers**

Create `src/training/teacher_forcing/packing_offsets.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR


_POSITION_KEY_SUFFIXES = (
    "_position",
    "_positions",
    "_position_ids",
    "_token_positions",
)
_POSITION_KEYS = {
    "logit_position",
    "target_position",
    "prompt_end_position",
    "box_start_position",
    "box_end_position",
    "object_ref_end_position",
    "coord_label_positions",
}


@dataclass(frozen=True, slots=True)
class PackedSegmentOffset:
    pack_index: int
    segment_index: int
    sample_id: str
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        for field_name in (
            "pack_index",
            "segment_index",
            "token_start",
            "token_end",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")


def build_packed_segment_offsets(
    *,
    raw_batch: Sequence[Any],
    collated: Mapping[str, Any],
) -> tuple[PackedSegmentOffset, ...]:
    if not raw_batch:
        return ()
    if not isinstance(raw_batch[0], (list, tuple)):
        return ()
    input_ids = collated.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("packed offset mapping requires collated input_ids shape [packs, seq]")
    labels = collated.get("labels")
    if isinstance(labels, torch.Tensor) and labels.shape[:2] != input_ids.shape[:2]:
        raise ValueError("packed offset mapping requires labels and input_ids row/seq agreement")
    if len(raw_batch) != int(input_ids.shape[0]):
        raise ValueError(
            "packed raw batch row count must match collated input_ids rows: "
            f"raw={len(raw_batch)} collated={int(input_ids.shape[0])}"
        )
    attention_mask = collated.get("attention_mask")
    offsets: list[PackedSegmentOffset] = []
    seen_sample_ids: set[str] = set()
    for pack_index, pack in enumerate(raw_batch):
        if not isinstance(pack, (list, tuple)):
            raise TypeError("packed raw batch must contain pack sequences")
        token_cursor = 0
        for segment_index, sample in enumerate(pack):
            if not isinstance(sample, Mapping):
                raise TypeError("packed samples must be mappings")
            length = _sample_token_length(sample)
            sample_id = _sample_id(sample, pack_index=pack_index, segment_index=segment_index)
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate packed sample_id: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            offsets.append(
                PackedSegmentOffset(
                    pack_index=pack_index,
                    segment_index=segment_index,
                    sample_id=sample_id,
                    token_start=token_cursor,
                    token_end=token_cursor + length,
                )
            )
            token_cursor += length
        if token_cursor > int(input_ids.shape[1]):
            raise ValueError(
                "packed segment token offsets exceed collated sequence length: "
                f"pack_index={pack_index} token_end={token_cursor} seq_len={int(input_ids.shape[1])}"
            )
        if isinstance(attention_mask, torch.Tensor):
            non_padding = int(attention_mask[pack_index].long().sum().item())
            if token_cursor != non_padding:
                raise ValueError(
                    "packed segment lengths must equal attention_mask non-padding token count: "
                    f"pack_index={pack_index} token_end={token_cursor} non_padding={non_padding}"
                )
        elif token_cursor != int(input_ids.shape[1]):
            raise ValueError(
                "packed segment lengths must equal collated sequence length when no attention_mask is present: "
                f"pack_index={pack_index} token_end={token_cursor} seq_len={int(input_ids.shape[1])}"
            )
    return tuple(offsets)


def shift_teacher_forcing_target_ir(
    ir: TeacherForcingTargetIR,
    offset: PackedSegmentOffset,
) -> TeacherForcingTargetIR:
    if type(ir) is not TeacherForcingTargetIR:
        raise TypeError("ir must be a TeacherForcingTargetIR")
    shifted_atoms = tuple(_shift_atom(atom, offset) for atom in ir.atoms)
    metadata = {
        **dict(ir.metadata),
        "packed": True,
        "pack_index": offset.pack_index,
        "segment_index": offset.segment_index,
        "token_start": offset.token_start,
        "token_end": offset.token_end,
        "sample_id": offset.sample_id,
    }
    return TeacherForcingTargetIR(
        schema_version=ir.schema_version,
        atoms=shifted_atoms,
        metadata=metadata,
    )


def _shift_atom(atom: SupervisionAtom, offset: PackedSegmentOffset) -> SupervisionAtom:
    return SupervisionAtom(
        batch_index=offset.pack_index,
        logit_position=atom.logit_position + offset.token_start,
        target_position=atom.target_position + offset.token_start,
        allowed_token_roles=atom.allowed_token_roles,
        selected_token_role=atom.selected_token_role,
        valid_token_ids=atom.valid_token_ids,
        selected_token_id=atom.selected_token_id,
        latent_valid_token_ids=atom.latent_valid_token_ids,
        coverage_target_weights=atom.coverage_target_weights,
        loss_tags=atom.loss_tags,
        loss_weight=atom.loss_weight,
        coord_role=atom.coord_role,
        provenance=_shift_provenance(atom.provenance, offset.token_start),
    )


def _shift_provenance(value: Any, delta: int, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _shift_provenance(v, delta, key=str(k))
            for k, v in value.items()
        }
    if _is_position_key(key):
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value) + delta
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(
                int(v) + delta if isinstance(v, int) and not isinstance(v, bool) else v
                for v in value
            )
    return value


def _is_position_key(key: str | None) -> bool:
    if key is None:
        return False
    if key.endswith("_token_ids") or key in {"stop_token_id", "continuation_token_ids"}:
        return False
    return key in _POSITION_KEYS or key.endswith(_POSITION_KEY_SUFFIXES)


def _sample_token_length(sample: Mapping[str, Any]) -> int:
    if "length" in sample:
        length = int(sample["length"])
    else:
        input_ids = sample.get("input_ids")
        if input_ids is None:
            raise ValueError("packed sample must provide length or input_ids")
        length = int(len(input_ids))
    if length <= 0:
        raise ValueError(f"packed sample length must be positive, got {length}")
    return length


def _sample_id(sample: Mapping[str, Any], *, pack_index: int, segment_index: int) -> str:
    raw = sample.get("sample_id")
    if raw is None:
        raw = f"pack{pack_index}:segment{segment_index}"
    if type(raw) is not str or not raw:
        raise ValueError("packed sample_id must be a non-empty string when provided")
    return raw


__all__ = [
    "PackedSegmentOffset",
    "build_packed_segment_offsets",
    "shift_teacher_forcing_target_ir",
]
```

Only sequence positions are shifted. Vocabulary-token payloads such as
`bbox_positive_area_valid_token_ids`, `bbox_positive_area_invalid_token_ids`,
`continuation_token_ids`, `stop_token_id`, and any other `*_token_ids` values
must remain unchanged.

- [ ] **Step 4: Run task tests**

Run:

```bash
pytest tests/test_teacher_forcing_packed_offsets.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/teacher_forcing/packing_offsets.py tests/test_teacher_forcing_packed_offsets.py
git commit -m "feat: add packed teacher forcing offset helpers"
```

---

### Task 2: Rewrite Teacher-Forcing IR And Coverage Sidecars In The Collator

> **Ordering guard:** do not add module-level imports of
> `shift_coverage_ledger_sidecar` or `offset_visual_token_region` until Task 3's
> helpers exist. Execute Task 3 before Task 2 Step 6, or treat Tasks 2 and 3 as
> one atomic implementation slice with a single green commit. A commit after
> Task 2 that imports not-yet-created helpers is forbidden.

**Files:**
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/trainers/batch_extras.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/training/encoding/model_inputs.py`
- Modify: `src/training/coverage_ledger/sidecars.py`
- Modify: `src/training/coverage_ledger/visual_regions.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`
- Test: `tests/test_batch_extras_contract.py`
- Test: `tests/test_model_input_bundle_contract.py`
- Test: `tests/test_coverage_ledger_sidecar_builder.py`
- Test: `tests/test_coverage_ledger_loss.py`

- [ ] **Step 1: Write failing collator tests for packed teacher-forcing IR**

Add to `tests/test_teacher_forcing_sidecar_bridge.py`:

```python
def _target_ir_with_atom(*, logit_position: int, target_position: int, sample_id: str):
    from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
    from src.training.teacher_forcing.roles import TokenRole

    return TeacherForcingTargetIR(
        schema_version=1,
        atoms=(
            SupervisionAtom(
                batch_index=0,
                logit_position=logit_position,
                target_position=target_position,
                allowed_token_roles=frozenset({TokenRole.COORD}),
                selected_token_role=TokenRole.COORD,
                valid_token_ids=frozenset({11}),
                selected_token_id=11,
                latent_valid_token_ids=frozenset(),
                coverage_target_weights=None,
                loss_tags=frozenset({"coord"}),
                loss_weight=1.0,
                coord_role="x1",
                provenance={"sample_id": sample_id},
            ),
        ),
        metadata={"sample_id": sample_id},
    )


def test_teacher_forcing_target_ir_is_shifted_for_static_packed_batch() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        lambda batch: {
            "input_ids": torch.zeros((1, 6), dtype=torch.long),
            "labels": torch.zeros((1, 6), dtype=torch.long),
        },
    )
    first = _target_ir_with_atom(logit_position=0, target_position=1, sample_id="a")
    second = _target_ir_with_atom(logit_position=0, target_position=1, sample_id="b")

    out = collator(
        [
            [
                {"sample_id": "a", "input_ids": [1, 2], TEACHER_FORCING_TARGET_IR_KEY: first},
                {"sample_id": "b", "input_ids": [3, 4, 5, 6], TEACHER_FORCING_TARGET_IR_KEY: second},
            ]
        ]
    )

    shifted = out[TEACHER_FORCING_TARGET_IR_KEY]
    assert len(shifted) == 2
    assert shifted[0].atoms[0].batch_index == 0
    assert shifted[0].atoms[0].target_position == 1
    assert shifted[1].atoms[0].batch_index == 0
    assert shifted[1].atoms[0].target_position == 3
    assert out["packed_segment_offsets"][1].token_start == 2
```

- [ ] **Step 2: Write failing collator tests for packed coverage-ledger sidecars**

Add to `tests/test_teacher_forcing_sidecar_bridge.py`:

```python
def test_coverage_ledger_sidecars_are_shifted_for_static_packed_batch() -> None:
    collator = build_dataset_metrics_collator(
        _DummyTemplate(),
        lambda batch: {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "labels": torch.zeros((1, 8), dtype=torch.long),
        },
        coverage_ledger_cfg={"enabled": True},
    )
    first = _coverage_ledger_sidecar(sample_id="a")
    second = _coverage_ledger_sidecar(sample_id="b")

    out = collator(
        [
            [
                {
                    "sample_id": "a",
                    "input_ids": [1, 2, 3],
                    "training_sidecars": TrainingSidecars(
                        supervision=SupervisionSidecars(payloads=(first,))
                    ),
                },
                {
                    "sample_id": "b",
                    "input_ids": [4, 5, 6, 7, 8],
                    "training_sidecars": TrainingSidecars(
                        supervision=SupervisionSidecars(payloads=(second,))
                    ),
                },
            ]
        ]
    )

    payloads = out["training_sidecars"].supervision.payloads
    assert len(payloads) == 2
    assert payloads[0].prompt_end_position == first.prompt_end_position
    assert payloads[1].prompt_end_position == second.prompt_end_position + 3
    assert payloads[1].object_entries[0].box_end_position == second.object_entries[0].box_end_position + 3
```

- [ ] **Step 3: Write failing collator/forward-prep alignment test**

Add a test to `tests/test_teacher_forcing_sidecar_bridge.py` or
`tests/test_trainer_loss_bridge_qwen3vl_contract.py` that uses the real static
packed collator path for two one-image detection samples in one pack and asserts:

- `packed_segment_offsets` has two entries in the same order as the raw packed
  samples.
- `training_sidecars.supervision.payloads` has two shifted
  `CoverageLedgerSidecar` payloads in the same order.
- `image_grid_thw.shape == (2, 3)` and both rows have `T == 1`.
- image-token placeholder counts equal the total post-merge token count implied
  by both image-grid rows.
- Qwen packed forward metadata includes the 4-row `position_ids` form accepted
  by `prepare_forward_inputs(..., packing_enabled=True)`.
- `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`, when
  emitted by the upstream collator, define the same segment boundaries as
  `packed_segment_offsets`; when they are not emitted, the Qwen packed
  `position_ids`/`text_position_ids` boundary inference path is explicitly
  validated against the same offsets.
- segment text offsets, image-grid row order, sidecar order, and label positions
  all agree.
- a packed segment with unsupported row sidecar fields such as
  `diagnostics.metadata`, `dataset.sample_id`, or `stage2.rollout_payload`
  fails with the same contract as the unpacked coverage-ledger collator.
- a packed segment with non-ledger `supervision.payloads` fails fast for this
  feature; do not silently preserve or drop arbitrary payloads in the first
  static-packing implementation.

- [ ] **Step 4: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_teacher_forcing_sidecar_bridge.py::test_teacher_forcing_target_ir_is_shifted_for_static_packed_batch tests/test_teacher_forcing_sidecar_bridge.py::test_coverage_ledger_sidecars_are_shifted_for_static_packed_batch tests/test_teacher_forcing_sidecar_bridge.py::test_static_packed_collator_preserves_qwen_position_and_media_order -q
```

Expected: current fail-fast errors mention packing incompatibility.

- [ ] **Step 5: Add packed extras to batch-extras plumbing**

Modify `src/trainers/batch_extras.py`:

```python
PACKED_SEGMENT_OFFSETS_KEY = "packed_segment_offsets"
```

Add it to `BATCH_EXTRAS_KEYS`, add the dataclass field:

```python
packed_segment_offsets: Any = None
```

and pop it:

```python
packed_segment_offsets=inputs.pop(PACKED_SEGMENT_OFFSETS_KEY, None),
```

Also register the key at the stricter boundaries:

- `src/detection/dataset.py::TRAINER_BATCH_EXTRA_KEYS`;
- `src/training/encoding/model_inputs.py` as a sidecar/runner-owned key.

Add tests to `tests/test_batch_extras_contract.py` and
`tests/test_model_input_bundle_contract.py` so future direct bridge callers do
not accidentally forward this key into `model(**inputs)`.

- [ ] **Step 6: Implement collator packed rewrite**

In `src/data_collators/enrichers.py`, import:

```python
from src.training.teacher_forcing.packing_offsets import (
    build_packed_segment_offsets,
    shift_teacher_forcing_target_ir,
)
from src.training.coverage_ledger.sidecars import shift_coverage_ledger_sidecar
```

In `TeacherForcingTargetIREnricher.__call__`, replace the packed fail-fast branch with:

```python
if packed:
    offsets = build_packed_segment_offsets(raw_batch=raw_batch, collated=collated)
    shifted: list[Any] = []
    for offset, sample in zip(offsets, _iter_packed_samples(raw_batch), strict=True):
        if not isinstance(sample, Mapping) or self.out_field not in sample:
            raise ValueError(
                "teacher_forcing_target_ir sidecar must be present for every packed segment"
            )
        shifted.append(shift_teacher_forcing_target_ir(sample[self.out_field], offset))
    collated[self.out_field] = tuple(shifted)
    collated["packed_segment_offsets"] = offsets
    collated["sample_id"] = tuple(offset.sample_id for offset in offsets)
    return
```

In `CoverageLedgerSidecarEnricher.__call__`, replace the packed fail-fast branch with:

```python
if packed:
    offsets = build_packed_segment_offsets(raw_batch=raw_batch, collated=collated)
    batch_payloads: list[Any] = []
    for offset, sample in zip(offsets, _iter_packed_samples(raw_batch), strict=True):
        sidecars = self._training_sidecars(sample)
        self._validate_aggregatable_row_sidecars(
            sidecars,
            sample_index=offset.segment_index,
        )
        payloads = self._coverage_payloads_from_sidecars(sidecars)
        if len(payloads) != 1:
            raise ValueError(
                "CoverageLedgerSidecar payload must be present exactly once for every packed segment"
            )
        if len(sidecars.supervision.payloads) != 1:
            raise ValueError(
                "packed coverage-ledger rows support exactly one CoverageLedgerSidecar "
                "payload per segment and no additional supervision payloads"
            )
        batch_payloads.append(shift_coverage_ledger_sidecar(payloads[0], offset))
    collated["packed_segment_offsets"] = offsets
    collated["training_sidecars"] = TrainingSidecars(
        supervision=SupervisionSidecars(payloads=tuple(batch_payloads))
    )
    return
```

Add this helper at module scope:

```python
def _iter_packed_samples(raw_batch: Sequence[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for pack in raw_batch:
        if not isinstance(pack, (list, tuple)):
            raise TypeError("packed raw batch must contain pack sequences")
        out.extend(pack)
    return tuple(out)
```

- [ ] **Step 7: Run collator and batch extras tests**

Run:

```bash
pytest tests/test_teacher_forcing_sidecar_bridge.py tests/test_batch_extras_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/data_collators/enrichers.py src/trainers/batch_extras.py src/detection/dataset.py src/training/encoding/model_inputs.py src/training/coverage_ledger/sidecars.py src/training/coverage_ledger/visual_regions.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_coverage_ledger_sidecar_builder.py tests/test_coverage_ledger_loss.py
git commit -m "feat: rewrite teacher forcing sidecars for packed rows"
```

---

### Task 3: Shift Coverage-Ledger Positions And Visual Regions

**Files:**
- Modify: `src/training/coverage_ledger/sidecars.py`
- Modify: `src/training/coverage_ledger/visual_regions.py`
- Test: `tests/test_coverage_ledger_sidecar_builder.py`
- Test: `tests/test_coverage_ledger_loss.py`

- [ ] **Step 1: Write failing sidecar shift test**

Add to `tests/test_coverage_ledger_sidecar_builder.py`:

```python
def test_shift_coverage_ledger_sidecar_offsets_all_token_positions() -> None:
    from src.training.coverage_ledger.sidecars import shift_coverage_ledger_sidecar
    from src.training.teacher_forcing.packing_offsets import PackedSegmentOffset

    sidecar = CoverageLedgerSidecar(
        sample_id="a",
        prompt_end_position=9,
        object_entries=(
            CoverageLedgerObjectEntry(
                object_instance_id="a:object-0",
                source_object_index=0,
                emitted_order_index=0,
                image_index=0,
                bbox_norm1000_xyxy=(10, 20, 300, 400),
                box_start_position=11,
                coord_label_positions=(12, 13, 14, 15),
                object_ref_end_position=10,
                box_end_position=16,
            ),
        ),
        image_grid_thw=(1, 16, 16),
        processed_width=640,
        processed_height=640,
        image_identity="a.jpg",
    )
    offset = PackedSegmentOffset(
        pack_index=0,
        segment_index=2,
        sample_id="a",
        token_start=100,
        token_end=200,
    )

    shifted = shift_coverage_ledger_sidecar(sidecar, offset)

    assert shifted.sample_id == "a"
    assert shifted.prompt_end_position == sidecar.prompt_end_position + 100
    assert shifted.object_entries[0].box_start_position == sidecar.object_entries[0].box_start_position + 100
    assert shifted.object_entries[0].coord_label_positions == tuple(
        pos + 100 for pos in sidecar.object_entries[0].coord_label_positions
    )
    assert shifted.object_entries[0].object_ref_end_position == sidecar.object_entries[0].object_ref_end_position + 100
    assert shifted.object_entries[0].box_end_position == sidecar.object_entries[0].box_end_position + 100
```

- [ ] **Step 2: Write failing visual region offset test**

Add to `tests/test_coverage_ledger_loss.py`:

```python
def test_offset_visual_token_region_adds_visual_token_start() -> None:
    from src.training.coverage_ledger.visual_regions import (
        VisualTokenRegion,
        offset_visual_token_region,
    )

    region = VisualTokenRegion(
        row_start=0,
        row_end=1,
        col_start=0,
        col_end=2,
        flattened_indices=(0, 1),
    )

    shifted = offset_visual_token_region(region, visual_token_start=10)

    assert shifted.flattened_indices == (10, 11)
    assert shifted.row_start == 0
    assert shifted.row_end == 1
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
pytest tests/test_coverage_ledger_sidecar_builder.py::test_shift_coverage_ledger_sidecar_offsets_all_token_positions tests/test_coverage_ledger_loss.py::test_offset_visual_token_region_adds_visual_token_start -q
```

Expected: import failures for `shift_coverage_ledger_sidecar` and `offset_visual_token_region`.

- [ ] **Step 4: Implement coverage sidecar shift helper**

Modify `src/training/coverage_ledger/sidecars.py`:

```python
from dataclasses import replace


def shift_coverage_ledger_sidecar(
    sidecar: CoverageLedgerSidecar,
    offset: object,
) -> CoverageLedgerSidecar:
    token_start = int(getattr(offset, "token_start"))
    sample_id = str(getattr(offset, "sample_id"))
    if sample_id != sidecar.sample_id:
        raise ValueError(
            "packed coverage-ledger offset sample_id mismatch: "
            f"offset={sample_id!r} sidecar={sidecar.sample_id!r}"
        )
    return replace(
        sidecar,
        prompt_end_position=sidecar.prompt_end_position + token_start,
        object_entries=tuple(
            replace(
                entry,
                box_start_position=entry.box_start_position + token_start,
                coord_label_positions=tuple(pos + token_start for pos in entry.coord_label_positions),
                object_ref_end_position=entry.object_ref_end_position + token_start,
                box_end_position=entry.box_end_position + token_start,
            )
            for entry in sidecar.object_entries
        ),
    )
```

Add it to `__all__`.

- [ ] **Step 5: Implement visual region offset helper**

Modify `src/training/coverage_ledger/visual_regions.py`:

```python
from dataclasses import replace


def offset_visual_token_region(
    region: VisualTokenRegion,
    *,
    visual_token_start: int,
) -> VisualTokenRegion:
    if type(region) is not VisualTokenRegion:
        raise TypeError("region must be a VisualTokenRegion")
    if not isinstance(visual_token_start, int) or isinstance(visual_token_start, bool):
        raise TypeError("visual_token_start must be an integer")
    if visual_token_start < 0:
        raise ValueError("visual_token_start must be non-negative")
    return replace(
        region,
        flattened_indices=tuple(
            int(index) + visual_token_start for index in region.flattened_indices
        ),
    )
```

Add it to `__all__`.

- [ ] **Step 6: Run coverage-ledger focused tests**

Run:

```bash
pytest tests/test_coverage_ledger_sidecar_builder.py tests/test_coverage_ledger_loss.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Continue to Task 2 Step 6 before committing**

These helpers are consumed by the collator rewrite. Keep the tree uncommitted
until Task 2's collator and batch-extra key ownership tests are green, then use
the atomic Task 2 commit.

---

### Task 4: Make The Teacher-Forcing Runner Consume Shifted Packed IRs

**Files:**
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Test: `tests/test_teacher_forcing_objective_runner.py`
- Test: `tests/test_teacher_forcing_metric_contract.py`

- [ ] **Step 1: Write failing packed runner test**

Add to `tests/test_teacher_forcing_objective_runner.py`:

```python
from dataclasses import replace


def _target_ir_with_atom(*, batch_index: int, logit_position: int, target_position: int, sample_id: str):
    from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
    from src.training.teacher_forcing.roles import TokenRole

    return TeacherForcingTargetIR(
        schema_version=1,
        atoms=(
            SupervisionAtom(
                batch_index=batch_index,
                logit_position=logit_position,
                target_position=target_position,
                allowed_token_roles=frozenset({TokenRole.COORD}),
                selected_token_role=TokenRole.COORD,
                valid_token_ids=frozenset({11}),
                selected_token_id=11,
                latent_valid_token_ids=frozenset(),
                coverage_target_weights=None,
                loss_tags=frozenset({"coord"}),
                loss_weight=1.0,
                coord_role="x1",
                provenance={"sample_id": sample_id},
            ),
        ),
        metadata={"sample_id": sample_id},
    )


def test_teacher_forcing_runner_accepts_shifted_packed_atoms_same_batch_row():
    first = replace(
        _target_ir_with_atom(batch_index=0, logit_position=1, target_position=2, sample_id="a"),
        metadata={"sample_id": "a", "packed": True},
    )
    second = replace(
        _target_ir_with_atom(batch_index=0, logit_position=5, target_position=6, sample_id="b"),
        metadata={"sample_id": "b", "packed": True},
    )

    supervision, sample_id_to_batch_index = _build_teacher_forcing_supervision(
        target_irs=(first, second),
        sample_ids=("a", "b"),
    )

    assert len(supervision.spans) == 2
    assert [span.label_positions for span in supervision.spans] == [(2,), (6,)]
    assert [
        span.distribution.target_ir.atoms[0].batch_index
        for span in supervision.spans
    ] == [0, 0]
    assert sample_id_to_batch_index == {"a": 0, "b": 0}


def test_teacher_forcing_runner_preserves_unpacked_multi_sample_rows():
    first = _target_ir_with_atom(batch_index=0, logit_position=1, target_position=2, sample_id="a")
    second = _target_ir_with_atom(batch_index=0, logit_position=1, target_position=2, sample_id="b")

    supervision, sample_id_to_batch_index = _build_teacher_forcing_supervision(
        target_irs=(first, second),
        sample_ids=("a", "b"),
    )

    assert [
        span.distribution.target_ir.atoms[0].batch_index
        for span in supervision.spans
    ] == [0, 1]
    assert sample_id_to_batch_index == {"a": 0, "b": 1}
```

- [ ] **Step 2: Run test and verify the current behavior**

Run:

```bash
pytest tests/test_teacher_forcing_objective_runner.py::test_teacher_forcing_runner_accepts_shifted_packed_atoms_same_batch_row tests/test_teacher_forcing_objective_runner.py::test_teacher_forcing_runner_preserves_unpacked_multi_sample_rows -q
```

Expected: the packed test fails because current helper rewrites the second IR to
batch row 1; the unpacked regression test should pass before and after the
change.

- [ ] **Step 3: Update sample-id mapping to honor shifted atom batch indices**

In `src/trainers/metrics/teacher_forcing.py`, modify
`_build_teacher_forcing_supervision` so packed shifted IRs contribute their own
atom batch index while unpacked IRs continue to use `_with_batch_index`:

```python
for batch_index, (sample_id, ir) in enumerate(zip(sample_ids, target_irs, strict=True)):
    packed = bool(ir.metadata.get("packed", False))
    if packed:
        atom_batch_indices = {int(atom.batch_index) for atom in ir.atoms}
        if len(atom_batch_indices) != 1:
            raise ValueError(
                "packed teacher_forcing_target_ir atoms for one sample must reference one batch row"
            )
        physical_batch_index = next(iter(atom_batch_indices))
        shifted_ir = ir
    else:
        physical_batch_index = int(batch_index)
        shifted_ir = _with_batch_index(ir, batch_index=physical_batch_index)
    if str(sample_id) in sample_id_to_batch_index:
        raise ValueError(
            f"duplicate teacher_forcing sample_id in batch: {str(sample_id)!r}"
        )
    sample_id_to_batch_index[str(sample_id)] = physical_batch_index
    spans.append(
        SupervisionSpan(
            sample_id=str(sample_id),
            role="schema",
            label_positions=tuple(atom.target_position for atom in shifted_ir.atoms),
            distribution=TeacherForcingTargetDistribution(target_ir=shifted_ir),
            provenance="teacher_forcing_target_ir",
        )
    )
```

Do not call `_with_batch_index` for packed shifted IRs; the collator has already
assigned the physical packed row. Keep `_with_batch_index` for unpacked
multi-sample batches, and keep the existing validation that atom positions are
non-negative and target ids are valid.

- [ ] **Step 4: Wire production bridge packing state from packed batch extras**

In `src/trainers/metrics/teacher_forcing.py`, derive the bridge packing state
from the collator-produced packed offset map:

```python
packing_enabled = getattr(extras, "packed_segment_offsets", None) is not None
bridge = TrainerLossBridge(
    settings=TrainerLossBridgeSettings(
        packing_enabled=packing_enabled,
        coverage_ledger=_coverage_ledger_config(objective_cfg),
    )
)
```

Do not derive this from YAML alone. `packed_segment_offsets` is the hard signal
that the actual batch was statically packed and sidecars were rewritten. The
bridge must consume the actual `batch_extras.packed_segment_offsets`, not only
the boolean, when validating coverage-ledger sidecar order and visual-token
starts.

Add tests proving:

- a packed teacher-forcing/coverage-ledger batch through
  `TeacherForcingObjectiveMixin.compute_loss` passes `packing_enabled=True` into
  `TrainerLossBridgeSettings`;
- a packed Qwen-style batch without 4-row `position_ids` fails through the
  production mixin path, not only through direct `prepare_forward_inputs` tests;
- a valid packed Qwen-style batch with 4-row `position_ids` is accepted by the
  production mixin path;
- a packed coverage-ledger batch with swapped sidecar order, missing offsets,
  offset/sidecar count mismatch, duplicated offset sample ids, or
  `image_grid_thw` row-count mismatch fails before computing ledger loss.

- [ ] **Step 5: Run runner tests**

Run:

```bash
pytest tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_training_runtime_sft_integration.py::test_teacher_forcing_objective_mixin_passes_packing_enabled_for_static_packed_ir tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/trainers/metrics/teacher_forcing.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_training_runtime_sft_integration.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "feat: support packed teacher forcing atom rows"
```

---

### Task 5: Aggregate Coverage-Ledger Loss Across Packed Segment Sidecars

**Files:**
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `src/training/coverage_ledger/visual_regions.py`
- Modify: `src/training/coverage_ledger/qwen_capture.py`
- Test: `tests/test_coverage_ledger_bridge_integration.py`
- Test: `tests/test_coverage_ledger_qwen_capture.py`
- Test: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 1: Write failing capture and bridge tests for packed one-image samples**

Add tests to `tests/test_coverage_ledger_qwen_capture.py` using the existing
fake-Qwen capture fixture pattern:

- `test_capture_accepts_packed_multi_image_grid_when_each_segment_is_one_frame`
  must build `image_grid_thw` with shape `(2, 3)`, every row `T == 1`, and
  total image-token placeholders equal to the sum of per-row post-merge visual
  tokens.
- `test_capture_rejects_video_multiframe_or_missing_packed_image_rows` must
  reject `pixel_values_videos`, `video_grid_thw`, any row with `T != 1`, and a
  missing or malformed `image_grid_thw`.

Add to `tests/test_coverage_ledger_bridge_integration.py`:

```python
def _shifted_sidecar(*, sample_id: str, token_delta: int) -> CoverageLedgerSidecar:
    base = _sidecar(sample_id=sample_id)
    entry = base.object_entries[0]
    return CoverageLedgerSidecar(
        sample_id=base.sample_id,
        prompt_end_position=base.prompt_end_position + token_delta,
        object_entries=(
            CoverageLedgerObjectEntry(
                object_instance_id=entry.object_instance_id,
                source_object_index=entry.source_object_index,
                emitted_order_index=entry.emitted_order_index,
                image_index=entry.image_index,
                bbox_norm1000_xyxy=entry.bbox_norm1000_xyxy,
                box_start_position=entry.box_start_position + token_delta,
                coord_label_positions=tuple(pos + token_delta for pos in entry.coord_label_positions),
                object_ref_end_position=entry.object_ref_end_position + token_delta,
                box_end_position=entry.box_end_position + token_delta,
            ),
        ),
        image_grid_thw=base.image_grid_thw,
        processed_width=base.processed_width,
        processed_height=base.processed_height,
        image_identity=base.image_identity,
    )


def _fake_model_with_head() -> _FakeQwenModel:
    model = _FakeQwenModel(logits=torch.zeros((1, 16, 32), dtype=torch.float32))
    model.coverage_ledger_head = CoverageLedgerHead(
        hidden_size=8,
        visual_dim=8,
        ledger_projection_dim=4,
        normalize_eps=1.0e-6,
    )
    return model


def _fake_coverage_ledger_result(*, weighted_loss: float):
    from src.training.coverage_ledger.loss import CoverageLedgerDebugRows, CoverageLedgerLossResult

    scalar = torch.tensor(float(weighted_loss), dtype=torch.float32)
    return CoverageLedgerLossResult(
        total_loss=scalar,
        coverage_loss=scalar,
        region_anchor_loss=scalar,
        weighted_loss=scalar,
        coverage_weight=1.0,
        region_anchor_weight=1.0,
        metric_events=(),
        debug_rows=CoverageLedgerDebugRows(
            coverage_state_positions=(1,),
            region_anchor_positions=(2,),
            region_anchor_object_indices=(0,),
            coverage_targets=torch.ones((1, 1), dtype=torch.float32),
            coverage_logits=torch.zeros((1, 1), dtype=torch.float32),
            region_anchor_targets=torch.ones((1, 1), dtype=torch.float32),
            region_anchor_logits=torch.zeros((1, 1), dtype=torch.float32),
            object_count=1,
            coverage_state_count=1,
            coverage_pair_count=1,
            region_anchor_pair_count=1,
        ),
    )


def test_bridge_accepts_multiple_packed_coverage_ledger_sidecars(monkeypatch):
    first = _sidecar(sample_id="a")
    second = _shifted_sidecar(sample_id="b", token_delta=5)
    sidecars = _enabled_sidecars(first, second)
    calls = []

    def fake_compute_coverage_ledger_loss(**kwargs):
        calls.append(kwargs)
        return _fake_coverage_ledger_result(weighted_loss=0.25)

    monkeypatch.setattr(
        "src.training.bridge.loss_bridge.compute_coverage_ledger_loss",
        fake_compute_coverage_ledger_loss,
    )

    result = TrainerLossBridge(
        settings=TrainerLossBridgeSettings(
            packing_enabled=True,
            coverage_ledger={"enabled": True},
        )
    ).compute_loss(
        model=_fake_model_with_head(),
        raw_batch=_raw_batch(time_steps=5),
        batch_extras=BatchExtras(
            packed_segment_offsets=(
                PackedSegmentOffset(pack_index=0, segment_index=0, sample_id="a", token_start=0, token_end=5),
                PackedSegmentOffset(pack_index=0, segment_index=1, sample_id="b", token_start=5, token_end=10),
            )
        ),
        training_sidecars=sidecars,
        supervision=_supervision("a"),
        objectives=(),
        sample_id_to_batch_index={"a": 0, "b": 0},
    )

    assert len(calls) == 2
    assert result.loss.item() == pytest.approx(0.25)
```

Also add `test_packed_coverage_ledger_metric_events_report_forward_total`:

- create two fake segment results with `weighted_loss=0.25`;
- include real coverage-ledger metric events, not `metric_events=()`;
- assert the bridge returns the normalized packed loss contribution, not the raw
  sum of segment-local means;
- flatten or reduce `result.metric_events` and assert
  `teacher_forcing/loss/coverage_ledger_auxiliary/contribution` equals the
  exact normalized scalar added to `result.loss`;
- assert diagnostic counts sum and diagnostic weighted means/ratios use summed
  pair/count denominators.
- add an equivalence test comparing two sidecars in one packed row with the same
  two examples evaluated as separate unpacked microbatches under the same
  trainer loss scale; the packed auxiliary contribution must not grow merely
  because average pack fill increased.

Also add `test_packed_coverage_ledger_visual_offsets_pool_from_the_correct_image`:

- build two one-image sidecars in one packed row with identical local bboxes;
- use captured image embeddings where segment A's visual tokens are all `1`
  and segment B's visual tokens are all `7`;
- assert sidecar A's pooled visual object embeddings come from the first slice
  and sidecar B's pooled visual object embeddings come from the second slice;
- fail if all visual starts are accidentally zero or if
  `offset_visual_token_region` is not applied.

- [ ] **Step 2: Run bridge test and verify it fails**

Run:

```bash
pytest tests/test_coverage_ledger_qwen_capture.py::test_capture_accepts_packed_multi_image_grid_when_each_segment_is_one_frame -q
pytest tests/test_coverage_ledger_bridge_integration.py::test_bridge_accepts_multiple_packed_coverage_ledger_sidecars -q
```

Expected: capture failure from `image_grid_thw must have shape (1, 3)` and bridge
failure from "exactly one CoverageLedgerSidecar".

- [ ] **Step 3: Replace single-sidecar requirement with packed segment sequence requirement**

In `src/training/bridge/loss_bridge.py`, replace `_require_coverage_ledger_sidecar`
with a packed-aware resolver. The resolver must keep the old exactly-one-sidecar
rule when `batch_extras.packed_segment_offsets` is absent:

```python
def _require_coverage_ledger_sidecars(
    self,
    sidecars: TrainingSidecars,
    *,
    packed_segment_offsets: tuple[object, ...] | None,
) -> tuple[CoverageLedgerSidecar, ...]:
    payloads = tuple(
        payload
        for payload in sidecars.supervision.payloads
        if type(payload) is CoverageLedgerSidecar
    )
    if packed_segment_offsets is None:
        if len(payloads) != 1:
            raise ValueError(
                "coverage_ledger.enabled=true requires exactly one CoverageLedgerSidecar "
                "when packed_segment_offsets are absent"
            )
        return payloads
    if not payloads:
        raise ValueError(
            "coverage_ledger.enabled=true requires at least one CoverageLedgerSidecar"
        )
    if len(payloads) != len(packed_segment_offsets):
        raise ValueError(
            "packed coverage-ledger sidecar count must match packed_segment_offsets"
        )
    return payloads
```

Before computing per-sidecar losses, validate the packed media contract:

- `packing_enabled=True` and `batch_extras.packed_segment_offsets` is present
  before accepting more than one sidecar or more than one image-grid row.
- `len(coverage_ledger_sidecars) == len(packed_segment_offsets)`.
- `len(coverage_ledger_sidecars) == image_grid_thw.shape[0]`.
- For every segment, `sidecar.sample_id == offset.sample_id`,
  `offset.pack_index` matches the physical packed row being forwarded, and
  `offset.segment_index` matches the sidecar/image-grid row order.
- Each sidecar's local `image_grid_thw` equals the corresponding model
  `image_grid_thw[segment_index]`.
- Every sidecar has exactly one local image and `image_grid_thw[0] == 1`.
- Every sidecar object entry keeps `image_index == 0`.
- The packed forward may contain multiple images only because it contains
  multiple one-image samples.
- The sidecar order must match the flattened Qwen `image_grid_thw` row order and
  packed segment order.

In `src/training/coverage_ledger/qwen_capture.py`, replace the hard
`image_grid_thw.shape == (1, 3)` validator with a packed-aware validator that
accepts `(num_segments, 3)` when `num_segments >= 1` and every row has `T == 1`.
Keep `(1, 3)` as the only accepted shape when `packing_enabled=False`; accept
multi-row grids only when `packing_enabled=True` and the bridge has validated
matching packed offsets. Keep `pixel_values_videos`, `video_grid_thw`, and any
multi-frame row rejected.

Update the call site to loop:

```python
coverage_ledger_results = []
for segment_index, sidecar in enumerate(coverage_ledger_sidecars):
    visual_regions = tuple(
        map_norm1000_bbox_to_visual_token_region(
            entry.bbox_norm1000_xyxy,
            image_grid_thw=sidecar.image_grid_thw,
            processed_width=sidecar.processed_width,
            processed_height=sidecar.processed_height,
            patch_size=visual_config["patch_size"],
            spatial_merge_size=visual_config["spatial_merge_size"],
        )
        for entry in sidecar.object_entries
    )
    pooled = pool_object_visual_embeddings(
        captured.image_embeds,
        tuple(
            offset_visual_token_region(region, visual_token_start=visual_token_start)
            for region in visual_regions
        ),
    )
    coverage_ledger_results.append(
        compute_coverage_ledger_loss(
            head=coverage_ledger_head,
            final_hidden_states=captured.final_hidden_states,
            pooled_visual_object_embeddings=pooled,
            sidecar=sidecar,
            config=self._coverage_ledger_loss_config(),
            sample_id_to_batch_index=sample_id_to_batch_index,
        )
    )
```

Compute `visual_token_start` cumulatively from prior segment sidecars:

```python
visual_token_start = sum(
    _coverage_ledger_visual_token_count(prev, visual_config)
    for prev in coverage_ledger_sidecars[:segment_index]
)
```

Add helper:

```python
def _coverage_ledger_visual_token_count(
    sidecar: CoverageLedgerSidecar,
    visual_config: Mapping[str, int],
) -> int:
    t, h, w = sidecar.image_grid_thw
    merge = int(visual_config["spatial_merge_size"])
    if h % merge != 0 or w % merge != 0:
        raise ValueError("coverage ledger packed image grid must be divisible by spatial_merge_size")
    return int(t) * (int(h) // merge) * (int(w) // merge)
```

Add the normalized packed ledger contribution to the main loss. Do not
concatenate raw segment metric events when their reducer is `last`. Emit or
reduce a per-forward coverage-ledger metric set where
`teacher_forcing/loss/coverage_ledger_auxiliary/contribution` equals the exact
normalized ledger scalar added to the forward loss, and diagnostic weighted
means/ratios/sums aggregate by their documented `MetricEvent` reducer semantics.

- [ ] **Step 4: Run bridge and Qwen contract tests**

Run:

```bash
pytest tests/test_coverage_ledger_bridge_integration.py::test_bridge_accepts_multiple_packed_coverage_ledger_sidecars tests/test_coverage_ledger_bridge_integration.py::test_packed_coverage_ledger_metric_events_report_forward_total tests/test_coverage_ledger_bridge_integration.py::test_packed_coverage_ledger_visual_offsets_pool_from_the_correct_image tests/test_coverage_ledger_qwen_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/bridge/loss_bridge.py src/training/coverage_ledger/qwen_capture.py tests/test_coverage_ledger_bridge_integration.py tests/test_coverage_ledger_qwen_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "feat: aggregate coverage ledger loss for packed segments"
```

---

### Task 6: Loosen Stage-1 Static Packing Guards And Keep Stage-2 Rejected

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/sft.py`
- Test: `tests/test_teacher_forcing_config_contract.py`
- Test: `tests/test_detection_training_config_contract.py`
- Test: `tests/test_stage1_static_packing_runtime_config.py`
- Test: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 1: Write failing config tests for Stage-1 latest teacher-forcing static packing**

Add to `tests/test_teacher_forcing_config_contract.py`:

```python
def test_latest_teacher_forcing_allows_stage1_static_packing_after_exact_mapping() -> None:
    payload = _latest_teacher_payload()
    payload["training"]["packing"] = True
    payload["training"]["eval_packing"] = True
    payload["training"]["per_device_train_batch_size"] = 1
    payload["training"]["effective_batch_size"] = 32
    payload["packing"]["static_packing"] = True
    payload["packing"]["padding_free_packed"] = False

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.training["packing"] is True
    assert cfg.training["eval_packing"] is True
    assert cfg.packing.static_packing is True
```

Add a coverage-ledger-specific positive test using a payload with
`objective.terms.coverage_ledger.enabled=true`. This must pass the separate
`_detection_validate_teacher_forcing_coverage_ledger_contract` guard with
`training.packing=true`, `training.eval_packing=true`, and
`packing.static_packing=true`.

Add an eval-path regression test proving standard SFT evaluation uses the same
packed loss bridge:

```python
def test_teacher_forcing_eval_step_accepts_static_packed_batch() -> None:
    # Build the composed trainer/mixin or a minimal Trainer-compatible harness.
    # Its eval/prediction step must call compute_loss with the packed collator
    # extras and reach TrainerLossBridgeSettings(packing_enabled=True).
    ...
```

Add to `tests/test_teacher_forcing_config_contract.py`:

```python
def test_latest_teacher_forcing_still_rejects_padding_free_packed() -> None:
    payload = _latest_teacher_payload()
    payload["training"]["packing"] = True
    payload["packing"]["static_packing"] = True
    payload["packing"]["padding_free_packed"] = True

    with pytest.raises(ValueError, match=r"padding_free_packed=false"):
        DetectionTrainingConfig.from_mapping(payload)
```

Add an encoded-cache strictness test:

```python
def test_latest_teacher_forcing_static_packing_rejects_encoded_cache_bypass() -> None:
    payload = _latest_teacher_payload()
    payload["training"]["packing"] = True
    payload["packing"]["static_packing"] = True
    payload["training"]["encoded_sample_cache"] = {
        "enabled": True,
        "ineligible_policy": "bypass",
    }

    with pytest.raises(ValueError, match=r"encoded_sample_cache.enabled=false"):
        DetectionTrainingConfig.from_mapping(payload)
```

- [ ] **Step 2: Write failing Stage-2 boundary tests with current APIs**

Add to `tests/test_training_runtime_sft_integration.py`:

```python
def test_stage2_self_rollout_rejects_static_packing_for_teacher_forcing_mapping() -> None:
    cfg = _stage2_rollout_correction_training_config(
        training={"packing": True, "per_device_train_batch_size": 1, "effective_batch_size": 32},
        packing={"static_packing": True, "padding_free_packed": False},
    )

    with pytest.raises(ValueError, match=r"rejects training\.packing=true"):
        validate_training_runtime_preflight(cfg)
```

Keep or add a separate positive test proving normal Stage-2 rollout-correction
trainer-owned post-rollout packing remains supported. Do not route that positive
case through `_validate_stage1_static_packing_policy`; that helper intentionally
returns for trainer-owned Stage-2 packing.

- [ ] **Step 3: Run config tests and verify they fail**

Run:

```bash
pytest tests/test_teacher_forcing_config_contract.py::test_latest_teacher_forcing_allows_stage1_static_packing_after_exact_mapping tests/test_teacher_forcing_config_contract.py::test_latest_teacher_forcing_still_rejects_padding_free_packed -q
```

Expected: first test fails with current exact atom-position packing guard, second test still fails with the desired rejection message after implementation if not yet adjusted.

- [ ] **Step 4: Update schema guard**

In `src/config/schema.py`, do not replace the generic teacher-forcing packing
guard with a bare return. Loosen only the exact Stage-1 detection
`objective.id=research_teacher_forcing` static-packing route covered by this
spec, and update both guard functions:

- `_detection_validate_packing_runtime_contract`;
- `_detection_validate_teacher_forcing_coverage_ledger_contract`.

```python
if training_packing:
    if not packing.static_packing:
        raise ValueError(
            "objective.id=research_teacher_forcing requires "
            "packing.static_packing=true when training.packing=true"
        )
    if packing.padding_free_packed:
        raise ValueError(
            "objective.id=research_teacher_forcing requires "
            "packing.padding_free_packed=false"
        )
if training_eval_packing:
    if not training_packing or not packing.static_packing:
        raise ValueError(
            "objective.id=research_teacher_forcing requires training.packing=true "
            "and packing.static_packing=true when training.eval_packing=true"
        )
```

For `_detection_validate_teacher_forcing_coverage_ledger_contract`, relax the
`training.packing` and `packing.static_packing` branches only under the same
exact Stage-1 static contract. Allow `training.eval_packing=true` only under
that same static contract. Keep `packing.padding_free_packed=true` rejected.

Preserve rejection for unsupported generic teacher-forcing configs, Stage-2
self-rollout paths, and padding-free packed mode. Add an explicit check near the
detection-specific packing validation:

```python
if objective is not None and objective.id == TEACHER_FORCING_OBJECTIVE_ID:
    if packing.padding_free_packed:
        raise ValueError(
            "objective.id=research_teacher_forcing requires packing.padding_free_packed=false"
        )
```

- [ ] **Step 5: Update runtime guard**

In `src/detection/runtime.py`, inside `assert_detection_runtime_supported`, replace latest teacher-forcing packing rejection with:

```python
if bool(training_config.training.get("packing", False)) and not training_config.packing.static_packing:
    raise ValueError(
        "latest research_teacher_forcing target IR requires packing.static_packing=true when training.packing=true"
    )
if training_config.packing.padding_free_packed:
    raise ValueError(
        "latest research_teacher_forcing target IR requires packing.padding_free_packed=false"
    )
if bool(training_config.training.get("eval_packing", False)) and (
    not bool(training_config.training.get("packing", False))
    or not training_config.packing.static_packing
):
    raise ValueError(
        "latest research_teacher_forcing target IR requires training.packing=true "
        "and packing.static_packing=true when training.eval_packing=true"
    )
```

Keep encoded sample cache disabled:

```python
if getattr(encoded_sample_cache_cfg, "enabled", False):
    raise ValueError(
        "latest research_teacher_forcing target IR requires training.encoded_sample_cache.enabled=false with static packing"
    )
```

- [ ] **Step 6: Update SFT static packing policy**

In `src/sft.py`, update `_validate_stage1_static_packing_policy` so Stage-1 `research_teacher_forcing` can pass through dataset static packing, but Stage-2 rollout-owned runtime plans still bypass or reject according to the existing Stage-2 policy. The branch must explicitly check the pipeline/objective id and not infer from config shape alone.
Do not feed `objective.id="research_teacher_forcing"` into
`resolve_static_sft_training_mode`; that helper accepts detection training
modes, not objective ids. Add a positive `_validate_stage1_static_packing_policy`
test for Stage-1 teacher forcing.

- [ ] **Step 7: Run guard tests**

Run:

```bash
pytest tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_sft_integration.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/config/schema.py src/detection/runtime.py src/sft.py tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_sft_integration.py
git commit -m "feat: allow stage1 teacher forcing static packing"
```

---

### Task 7: Add Packed Configs And Materialization Probes

**Files:**
- Modify: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`
- Modify: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml`
- Modify: `configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml`
- Test: `tests/test_coverage_ledger_smoke_configs.py`

- [ ] **Step 1: Write failing smoke config assertions**

Add to `tests/test_coverage_ledger_smoke_configs.py`:

```python
def test_coverage_ledger_smoke_configs_use_stage1_static_packing() -> None:
    ledger = _load_resolved(LEDGER_CONFIG)
    baseline = _load_resolved(BASELINE_CONFIG)

    for cfg in (ledger, baseline):
        assert cfg["global_max_length"] == 12000
        assert cfg["training"]["packing"] is True
        assert cfg["training"]["eval_packing"] is True
        assert cfg["training"]["per_device_train_batch_size"] == 1
        assert cfg["training"]["effective_batch_size"] == 32
        assert cfg["packing"]["static_packing"] is True
        assert cfg["packing"]["padding_free_packed"] is False
        assert cfg["training"]["encoded_sample_cache"]["enabled"] is False
```

- [ ] **Step 2: Run config test and verify it fails**

Run:

```bash
pytest tests/test_coverage_ledger_smoke_configs.py::test_coverage_ledger_smoke_configs_use_stage1_static_packing -q
```

Expected: current configs use no packing.

- [ ] **Step 3: Update smoke and production YAML**

Set in all three configs:

```yaml
global_max_length: 12000
training:
  packing: true
  eval_packing: true
  packing_mode: static
  per_device_train_batch_size: 1
  effective_batch_size: 32
  encoded_sample_cache:
    enabled: false
packing:
  static_packing: true
  padding_free_packed: false
```

Keep the current ledger/baseline objective deltas unchanged.

- [ ] **Step 4: Run config materialization tests**

Run:

```bash
pytest tests/test_coverage_ledger_smoke_configs.py -q
```

Expected: all tests pass, and the loader derives grad accumulation from
`effective_batch_size` in topology-specific tests. Document that
`effective_batch_size: 32` requires GPU world sizes that divide 32; the intended
smoke/prod topologies are 4 GPUs and 8 GPUs.

- [ ] **Step 5: Commit**

```bash
git add configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml tests/test_coverage_ledger_smoke_configs.py
git commit -m "config: enable static packing for ledger stage1 configs"
```

---

### Task 8: Full Targeted Verification And Documentation

**Files:**
- Modify: `src/training/coverage_ledger/preflight.py`
- Modify: `scripts/training/coverage_ledger_preflight.py`
- Modify: `docs/data/PACKING.md`
- Modify: `docs/superpowers/plans/2026-06-23-coverage-ledger-auxiliary-loss.md`
- Test: `tests/test_artifact_contract_docs.py`
- Test: `tests/test_coverage_ledger_preflight_artifacts.py`
- Test: `tests/test_coverage_ledger_smoke_configs.py`

- [ ] **Step 1: Update packing documentation**

In `docs/data/PACKING.md`, update the matrix row:

```markdown
| Stage-1 compact latest teacher forcing | `12000` | config-defined | static dataset packing | Requires packed segment offsets for teacher-forcing IR and coverage-ledger sidecars; Stage-2 self-rollout remains out of scope. |
```

Add a short contract paragraph:

```markdown
Latest Stage-1 teacher forcing may enable static dataset packing only when the collator emits packed segment offsets and shifted supervision sidecars. Each original sample remains a segment with one image. The packed row may contain multiple images; coverage-ledger visual pooling uses cumulative visual-token offsets rather than merging image identities.
```

- [ ] **Step 2: Update coverage-ledger launch plan notes**

In `docs/superpowers/plans/2026-06-23-coverage-ledger-auxiliary-loss.md`, add:

```markdown
Static packing launch contract: `training.effective_batch_size: 32` counts packed sequence rows, not original source samples. For 4 GPUs the expected gradient accumulation is 8; for 8 GPUs it is 4. Raw source-sample throughput must be interpreted from `pack_num_samples` and packing fill telemetry.
```

- [ ] **Step 3: Extend coverage-ledger preflight to materialize packed evidence**

Update `src/training/coverage_ledger/preflight.py` and, if needed,
`scripts/training/coverage_ledger_preflight.py` so the no-training preflight
builds or fixtures at least one deterministic two-segment static pack from the
same smoke selection/runtime configuration. It must persist enough evidence to
prove the packed contract:

- `packed_segment_offsets` with sample ids, pack index, segment index, token
  start, and token end;
- shifted label positions for each segment;
- sidecar order and `image_grid_thw` row order;
- image-token placeholder counts versus post-merge visual-token counts;
- Qwen 4-row `position_ids`/`text_position_ids` validation;
- attention boundary evidence: explicit `cu_seq_lens_*`/max lengths if present,
  otherwise documented Qwen packed position reset inference.

Add or extend tests in `tests/test_coverage_ledger_preflight_artifacts.py` and
`tests/test_coverage_ledger_smoke_configs.py` to prove the preflight dataset
uses the same sorted roll-in/object ordering as training and that the artifact
contains the packed alignment evidence above. The existing unpacked per-row
sidecar loop is not sufficient.

- [ ] **Step 4: Run targeted full suite**

Run:

```bash
pytest \
  tests/test_teacher_forcing_packed_offsets.py \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_batch_extras_contract.py \
  tests/test_coverage_ledger_sidecar_builder.py \
  tests/test_coverage_ledger_loss.py \
  tests/test_coverage_ledger_bridge_integration.py \
  tests/test_coverage_ledger_qwen_capture.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  tests/test_teacher_forcing_objective_runner.py \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_detection_training_config_contract.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_coverage_ledger_smoke_configs.py \
  tests/test_coverage_ledger_preflight_artifacts.py \
  tests/test_artifact_contract_docs.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Run strict residue checks**

Run:

```bash
rg -n "coverage_ledger.*packing=false|research_teacher_forcing.*training\\.packing=false|exact atom-position packing mapping is not implemented" src tests docs configs
git diff --check
```

Expected: remaining hits only refer to rejected Stage-2, rejected padding-free packed mode, encoded-sample-cache rejection, or historical documentation. `git diff --check` exits 0.

- [ ] **Step 6: Run no-training materialization and preflight gate**

First run the config topology probe:

```bash
python - <<'PY'
from pathlib import Path
from src.config.loader import ConfigLoader

paths = [
    Path("configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml"),
    Path("configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml"),
    Path("configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml"),
]
for path in paths:
    _train_args, cfg = ConfigLoader.load_training_config(path)
    print(path)
    print("  packing", cfg.training["packing"])
    print("  eval_packing", cfg.training["eval_packing"])
    print("  per_device", cfg.training["per_device_train_batch_size"])
    print("  effective_batch_size", cfg.training["effective_batch_size"])
    print("  static_packing", cfg.packing.static_packing)
    print("  padding_free_packed", cfg.packing.padding_free_packed)
PY
```

Expected: each config prints `packing True`, `eval_packing True`,
`per_device 1`, `effective_batch_size 32`, `static_packing True`, and
`padding_free_packed False`.

Then run a no-training packed coverage-ledger preflight:

```bash
python scripts/training/coverage_ledger_preflight.py \
  --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  --output-root temp/coverage_ledger_preflight_smoke_static_packed
```

Expected preflight checks:

- materializes at least one deterministic two-segment static pack from the smoke
  selection or a dedicated fixture;
- records packed segment offsets, shifted label positions, `image_grid_thw`
  row order, image-token placeholder counts, and Qwen 4-row `position_ids`
  validation;
- writes/validates `ledger/selected_samples.json`,
  `ledger/alignment_debug.jsonl`, and `ledger/overlays/index.json`;
- emits 16 overlay images for the smoke overfit review path;
- confirms baseline and ledger configs differ only by the intended ledger term
  and static-packing-safe objective deltas.

Do not launch smoke training or production training from this task. Training
launch remains a separate user approval gate after this plan is implemented and
the no-training preflight passes.

- [ ] **Step 7: Commit**

```bash
git add src/training/coverage_ledger/preflight.py scripts/training/coverage_ledger_preflight.py docs/data/PACKING.md docs/superpowers/plans/2026-06-23-coverage-ledger-auxiliary-loss.md tests/test_artifact_contract_docs.py tests/test_coverage_ledger_preflight_artifacts.py tests/test_coverage_ledger_smoke_configs.py
git commit -m "feat: add packed coverage ledger preflight gate"
```

---

## Self-Review

- Spec coverage: The plan covers packed segment map creation, teacher-forcing atom offsets, coverage-ledger token offsets, visual-token offsets, bridge aggregation, config/runtime guard changes, YAML materialization, docs, and verification.
- Stage boundary: Stage-1 training and standard SFT eval-step static packing are
  enabled after the attention-isolation gate; Stage-2 self-rollout remains
  rejected; existing Stage-2 trainer-owned post-rollout packing is untouched.
- Batch semantics: `effective_batch_size: 32` is explicitly packed sequence rows per optimizer step.
- Placeholder scan: the only unresolved runtime dependency is deliberate and
  front-loaded: real static-packed attention isolation must be proven before
  sidecar remapping proceeds.
- Risk focus: The highest-risk areas are attention isolation, visual-token
  offset agreement with Qwen image embedding order, metric interpretation of
  packed rows versus raw samples, and preflight proving the same packed
  contract. Each now has an explicit test or preflight gate.
