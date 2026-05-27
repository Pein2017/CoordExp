# Unified Inference Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one shared `src/infer` prompt/decode/backend/parse/provenance foundation for offline inference and online Stage-2 rollout, then delete overlapping legacy decode layouts.

**Architecture:** `src/infer` becomes the shared runtime root while public authored config namespaces stay stable: offline uses `infer.*`, Stage-2 uses `rollout_matching.*`, and Stage-2 objectives stay under `stage2_rollout_correction.*`. Stage-2 target construction, residual slicing, duplicate control, IoU matching, DDP coordination, and loss execution remain trainer-owned.

**Tech Stack:** Python, pytest, Transformers/Qwen3-VL, vLLM, ms-swift server rollout, CoordExp strict YAML config schemas, OpenSpec-governed artifact contracts.

---

## Approval Boundary

Do not implement production code until the user gives final implementation approval. This plan is the execution handoff after the OpenSpec review loop.

If the worktree already contains dirty production-code scaffolding from another
agent or an earlier interrupted attempt, treat that scaffold as unapproved
external work: inspect it, preserve it, and audit it against this plan, but do
not extend, stage, or describe it as completed implementation until final
approval reopens the production-code phase.

Current pre-approval audit note (2026-05-26): this worktree may already contain
unapproved scaffold under `src/infer/__init__.py`, `src/infer/artifacts.py`,
`src/infer/prompt.py`, `src/infer/runtime.py`, `tests/test_decode_*`,
`tests/test_detection_prompt_input_codec.py`, and
`tests/test_stage2_rollout_runtime.py`. Before the implementation phase adopts
any of it, reconcile it against these known risks:

- `src.infer.backend` is referenced by a trace test but may not exist yet;
- prompt-token parity must use the real chat/template/processor tokenization
  path, not a simplified text string;
- comparable provenance must not accept best-effort or transitional
  fingerprints synthesized from partial owner fields;
- decode fingerprints must exclude operational details such as batch size,
  rank, device, and server URL;
- `src/infer/__init__.py` must not keep legacy `InferenceEngine`,
  `GenerationConfig`, or `InferenceConfig` as final public aliases.

## Governing Artifacts

- Design spec: `docs/superpowers/specs/2026-05-26-unified-inference-runtime-refactor-design.md`
- OpenSpec change: `openspec/changes/unify-inference-runtime/`
- Superseded vLLM full-sync note: `openspec/changes/materialize-vllm-full-sync-adapter-rows/`

## Implementation Kickoff Checklist

Run this only after explicit final implementation approval:

- Re-read the OpenSpec change and this plan before editing production code.
- Inspect `git status --short --branch` and preserve unrelated user work.
- Classify existing dirty scaffold as one of: adopt after audit, replace, or
  leave untouched. Do not assume scaffold under `src/infer` or `tests` is
  correct because it already exists.
- Run or prepare the narrow contract-test slice first:
  `tests/test_detection_prompt_input_codec.py`,
  `tests/test_decode_backend_trace_contract.py`,
  `tests/test_decode_provenance_contract.py`, and the touched Stage-2 eval
  artifact checks in `tests/test_stage2_rollout_runtime.py`.
- Before adopting any scaffold, prove it satisfies the stricter contracts for
  real prompt-token parity, one-image visual loading with `do_resize=false`,
  canonical trace shape, non-transitional comparable provenance, and final
  removal of legacy public aliases.
- Keep import-ban/search-gate tests phase-scoped until the deletion phase so
  early slices can remain runnable without weakening the final no-compatibility
  target.

## Pre-Approval State Snapshot

As of 2026-05-26, the durable planning state is:

- OpenSpec review and Superpowers plan review are complete in
  `openspec/changes/unify-inference-runtime/tasks.md` items 1.1 through 1.6.
- Item 1.7 remains intentionally unchecked: production code work waits for
  explicit final user approval.
- Existing dirty files under `src/infer` and `tests` are scaffold to audit,
  not implementation evidence.
- The older `materialize-vllm-full-sync-adapter-rows` change is historical or
  deferred for active unified Stage-2 server training; active server sync is
  adapter sync plus CoordExp coord-row updates under the unified runtime plan.
- `openspec` and `conda` may be unavailable in lightweight shells. If so,
  record that validation was skipped for environment reasons and rerun strict
  OpenSpec validation plus `conda run -n ms ...` tests in a full CoordExp
  environment before claiming implementation readiness.

## File Structure

Create or replace shared runtime files:

- Create: `src/infer/runtime.py`
- Create: `src/infer/prompt.py`
- Create: `src/infer/backend.py`
- Create: `src/infer/backend_sync.py`
- Create: `src/infer/parsing.py`
- Create: `src/infer/constraints.py`
- Modify: `src/infer/artifacts.py`
- Modify: `src/infer/checkpoints.py`
- Modify: `src/infer/pipeline.py`
- Modify: `src/infer/__init__.py`

Delete or make inactive by final phase:

- Delete: `src/infer/engine.py`
- Done: `src/infer/backends.py` folded into `src/infer/backend.py` and deleted
- Done: `src/infer/compact_grammar.py` and `src/infer/stop_pressure.py`
  folded into private `src/infer/_constraints_impl.py` behind
  `src/infer/constraints.py` and deleted
- Delete: `src/trainers/rollout_runtime/`
- Delete or reduce to non-decode facade, then preferably delete: `src/trainers/stage2_rollout_runtime.py`

Modify Stage-2 and training integration:

- Modify: `src/common/detection_chat.py`
- Modify: `src/datasets/builders/jsonlines.py`
- Modify: `src/datasets/dense_caption.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/trainers/stage2_rollout_correction_impl.py`
- Modify: `src/trainers/rollout_correction/target_builder.py`
- Modify: `src/trainers/rollout_correction/teacher_forcing_adapter.py`
- Modify: `src/trainers/rollout_matching/preflight.py`
- Modify: `src/config/rollout_matching_schema.py`
- Modify: `src/config/schema.py`
- Modify: `src/bootstrap/trainer_setup.py`
- Modify: `src/training_runtime/plan.py`
- Modify: `src/training_runtime/profile.py`
- Modify: `src/sft.py`

Modify entrypoints, docs, and scripts:

- Modify: `scripts/run_infer.py`
- Modify: `scripts/train_stage2.sh`
- Create: `scripts/stamp_inference_provenance.py`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/AGENT_INDEX.md`
- Modify: `docs/catalog.yaml`
- Modify: `docs/ARTIFACTS.md`
- Modify: `docs/eval/WORKFLOW.md`
- Modify: `docs/training/STAGE2_RUNBOOK.md`
- Modify: `docs/training/METRICS.md`

Primary tests to create or update:

- Create: `tests/test_detection_prompt_input_codec.py`
- Create: `tests/test_decode_backend_trace_contract.py`
- Create: `tests/test_vllm_logprob_trace_contract.py`
- Create: `tests/test_decode_provenance_contract.py`
- Create: `tests/test_infer_layout_import_gates.py`
- Modify: `tests/test_infer_batch_decoding.py`
- Modify: `tests/test_unified_infer_pipeline.py`
- Modify: `tests/test_qwen_generation_contract.py`
- Modify: `tests/test_stage2_rollout_runtime.py`
- Modify: `tests/test_vllm_server_rollout_contract.py`
- Modify: `tests/test_vllm_server_adapter_payload.py`
- Modify: `tests/test_swift_rollout_endpoints_contract.py`
- Modify: `tests/test_training_config_strict_unknown_keys.py`
- Modify: `tests/test_training_runtime_plan.py`
- Modify: `tests/test_training_runtime_profile.py`
- Modify: `tests/test_training_runtime_sft_integration.py`

## Task 1: Golden Contract Tests Before Movement

**Files:**

- Create: `tests/test_detection_prompt_input_codec.py`
- Create: `tests/test_decode_backend_trace_contract.py`
- Create: `tests/test_decode_provenance_contract.py`
- Modify: `tests/test_stage2_rollout_runtime.py`

- [ ] **Step 1: Add prompt/input parity fixtures**

Add tiny fixture helpers inside `tests/test_detection_prompt_input_codec.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


def _one_image_sample(tmp_path: Path) -> dict:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT"
        b"\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return {
        "image": str(image_path),
        "images": [str(image_path)],
        "width": 640,
        "height": 480,
        "objects": [{"bbox_2d": [10, 20, 110, 220], "desc": "cat"}],
    }
```

- [ ] **Step 2: Add expected failing tests for one-image and prompt policy**

Add tests that import the future API and fail until `src/infer/prompt.py` exists:

```python
def test_prompt_codec_rejects_multiple_images(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    sample["images"] = [sample["images"][0], sample["images"][0]]

    with pytest.raises(ValueError, match="exactly one image"):
        build_prompt_bundle(sample, DetectionPromptPolicy.default_for_detection())


def test_prompt_policy_fingerprint_is_stable(tmp_path: Path) -> None:
    from src.infer.prompt import DetectionPromptPolicy, build_prompt_bundle

    sample = _one_image_sample(tmp_path)
    policy = DetectionPromptPolicy.default_for_detection()
    first = build_prompt_bundle(sample, policy)
    second = build_prompt_bundle(sample, policy)

    assert first.prompt_policy_fingerprint == second.prompt_policy_fingerprint
    assert first.visual_metadata["image_count"] == 1
    assert first.visual_metadata["do_resize"] is False
```

- [ ] **Step 3: Add trace shape tests**

Add future API tests to `tests/test_decode_backend_trace_contract.py`:

```python
import pytest


def test_decode_result_rejects_missing_logprob() -> None:
    from src.infer.backend import DetectionDecodeResult, validate_decode_trace

    result = DetectionDecodeResult(
        text="abc",
        generated_token_ids=[1, 2, 3],
        generated_tokens=["a", "b", "c"],
        generated_logprobs=[-0.1, -0.2],
        stop_reason="length",
        backend="fake",
    )

    with pytest.raises(ValueError, match="trace shape"):
        validate_decode_trace(result, trace_logprobs=True)
```

Also cover the missing-trace cases that the canonical contract forbids:

- `trace_logprobs=true` with nonempty generated text and empty token IDs;
- generated token IDs present but generated token text missing;
- generated token IDs/text present but generated logprobs missing;
- non-finite generated logprobs;
- backend responses that try to pass by padding, clipping, or defaulting trace
  arrays instead of returning a complete generated-token trace.

- [x] **Step 4: Add provenance carrier tests**

Add future API tests to `tests/test_decode_provenance_contract.py`:

```python
from pathlib import Path

import pytest


def test_moved_jsonl_without_provenance_is_not_comparable(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    artifact = tmp_path / "gt_vs_pred_scored.jsonl"
    artifact.write_text('{"gt":[],"pred":[],"width":1,"height":1}\\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing_provenance"):
        load_comparable_artifact(artifact)
```

- [ ] **Step 5: Add Stage-2 eval artifact preservation assertions**

Extend the existing Stage-2 eval artifact assertions in
`tests/test_stage2_rollout_runtime.py` with two fixture-level checks:

- no `pred_token_trace.jsonl` is required when no trace metadata exists;
- `pred_token_trace.jsonl` is required when trace metadata exists.

The trace-enabled artifact set includes:

```python
EXPECTED_STAGE2_EVAL_FILES = {
    "gt_vs_pred.jsonl",
    "gt_vs_pred_scored.jsonl",
    "infer_summary.json",
    "metrics.json",
    "per_image.json",
    "raw_rollouts.jsonl",
    "pred_token_trace.jsonl",
}
```

- [ ] **Step 6: Run expected failing tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_detection_prompt_input_codec.py tests/test_decode_backend_trace_contract.py tests/test_decode_provenance_contract.py tests/test_stage2_rollout_runtime.py -q
```

Expected: new tests fail because the shared runtime APIs do not exist yet. Do
not leave these tests as unmarked broad-suite failures across a commit boundary:
either implement the paired task in the same slice or temporarily mark them with
a strict `future_contract` marker/xfail that is removed when the paired
implementation lands.

## Task 2: Shared Runtime Types And Prompt Codec

**Files:**

- Create: `src/infer/runtime.py`
- Create: `src/infer/prompt.py`
- Modify: `src/infer/__init__.py`

- [ ] **Step 1: Add shared runtime dataclasses**

Create `src/infer/runtime.py` with typed dataclasses for:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict[str, Any]]
    prompt_text: str
    prompt_policy_fingerprint: str
    visual_metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None


@dataclass(frozen=True)
class DetectionDecodeRequest:
    backend: Literal["hf", "vllm"]
    backend_mode: str
    decode_mode: Literal["greedy", "sampling", "beam"]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    num_beams: int = 1
    repetition_penalty: float | None = None
    seed: int | None = None
    stop_tokens: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    trace_logprobs: bool = False
    trace_prompt_logprobs: bool = False
    decode_policy_fingerprint: str = ""
```

- [ ] **Step 2: Add prompt policy and one-image validation**

Create `src/infer/prompt.py` with `DetectionPromptPolicy.default_for_detection()` and `build_prompt_bundle(...)`. The first implementation can build a deterministic text-only prompt bundle for tests, but it must validate exactly one image and record visual metadata.

- [ ] **Step 3: Keep target construction out of prompt.py**

Do not import these names in `src/infer/prompt.py` or `src/infer/runtime.py`:

```text
ResidualBoundaryAdapter
TeacherForcingTargetIR
build_residual_set_target_ir
target_builder
greedy_match_iou
associate_one_to_one_greedy_iou
duplicate_control
```

- [x] **Step 4: Export only stable new surfaces**

Update `src/infer/__init__.py` to export `PromptBundle`, `DetectionDecodeRequest`, `DetectionPromptPolicy`, and `build_prompt_bundle`.

Status update: the package root now exports only the shared runtime/prompt/backend
surface and does not expose legacy `GenerationConfig`, `InferenceConfig`, or
`InferenceEngine` aliases. Covered by `tests/test_infer_layout_import_gates.py`.

- [ ] **Step 5: Run prompt tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_detection_prompt_input_codec.py -q
```

Expected: prompt tests pass.

## Task 2A: Teacher-Forced Prompt-Prefix Migration

**Files:**

- Modify: `src/common/detection_chat.py`
- Modify: `src/datasets/builders/jsonlines.py`
- Modify: `src/datasets/dense_caption.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/trainers/rollout_correction/teacher_forcing_adapter.py`
- Modify: `tests/test_detection_prompt_input_codec.py`

- [ ] **Step 1: Add teacher-forced parity test**

Add a focused test to `tests/test_detection_prompt_input_codec.py` named
`test_teacher_forced_offline_rollout_prompt_token_and_visual_parity`. It should
build the same one-image sample through offline prompt preparation, online
rollout prompt preparation, and teacher-forced prompt-prefix preparation, then
assert matching prompt policy fingerprint, prompt text, prompt token IDs when
available, and visual metadata.

- [ ] **Step 2: Route dataset prompt-prefix rendering through the shared codec**

Update `src/common/detection_chat.py`, `src/datasets/builders/jsonlines.py`,
`src/datasets/dense_caption.py`, and `src/detection/dataset.py` so they call the
shared prompt codec for prompt-prefix/message construction instead of
maintaining independent prompt templates.

- [ ] **Step 3: Keep assistant targets trainer-owned**

Do not move assistant-target rendering, residual slicing, target IR
construction, duplicate-control target semantics, loss masks, or
teacher-forcing target spans into `src/infer`. Those remain under training or
trainer modules.

- [ ] **Step 4: Run prompt-prefix parity tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_detection_prompt_input_codec.py::test_teacher_forced_offline_rollout_prompt_token_and_visual_parity -q
```

Expected: offline, rollout, and teacher-forced prompt-prefix paths use one
prompt codec and parity metadata; no target IR code is imported by `src/infer`.

## Task 3: Backend Trace Contract And Adapter Interface

**Files:**

- Create: `src/infer/backend.py`
- Create: `src/infer/backend_sync.py`
- Modify: `tests/test_decode_backend_trace_contract.py`
- Create: `tests/test_vllm_logprob_trace_contract.py`
- Modify: `tests/test_vllm_server_rollout_contract.py`

- [ ] **Step 1: Add `DetectionDecodeResult` and trace validator**

Create `src/infer/backend.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class DetectionDecodeResult:
    text: str
    generated_token_ids: list[int] = field(default_factory=list)
    generated_tokens: list[str] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    backend: str = ""
    prompt_token_ids: list[int] | None = None
    visual_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


def validate_decode_trace(result: DetectionDecodeResult, *, trace_logprobs: bool) -> None:
    if not trace_logprobs:
        return
    lengths = {
        "generated_token_ids": len(result.generated_token_ids),
        "generated_tokens": len(result.generated_tokens),
        "generated_logprobs": len(result.generated_logprobs),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"trace shape mismatch: {lengths}")
    if any(not isfinite(float(value)) for value in result.generated_logprobs):
        raise ValueError("trace logprobs must be finite")
```

- [ ] **Step 2: Add backend adapter protocol**

Add `BackendAdapter.prepare()`, `BackendAdapter.generate_many(...)`, and `BackendAdapter.close()` protocol or base class in `src/infer/backend.py`. Keep HF/vLLM concrete adapters minimal until callers migrate.

- [x] **Step 3: Add vLLM trace source tests**

Create `tests/test_vllm_logprob_trace_contract.py` with fake vLLM responses that prove:

```text
OpenAI-style token strings without token IDs fail trace-required decode.
ms-swift return_details payload with prompt_token_ids, token_ids, and logprobs passes.
short traces fail.
long traces fail except for one verified non-emitted stop token.
```

Also invert existing permissive trace expectations in
`tests/test_vllm_server_rollout_contract.py` so short traces and unverified
long traces fail. The only accepted long-trace exception is exactly one verified
non-emitted stop/special token by token ID and text policy.

Status update: vLLM trace hardening is covered by
`tests/test_decode_backend_trace_contract.py`,
`tests/test_vllm_server_rollout_contract.py`, and focused
`tests/test_stage2_rollout_runtime.py` trace cases. ms-swift choices responses
with aligned `prompt_token_ids`, `token_ids`, `logprobs.content[*].token`, and
`logprobs.content[*].logprob` now normalize through
`src.infer.backend.normalize_vllm_trace_response`. Short traces, arbitrary
long traces, and old stop-tail clipping behavior fail fast through the shared
trace validator instead of being accepted by trainer-local clamping.

- [ ] **Step 4: Add backend sync provenance skeleton**

Create `src/infer/backend_sync.py` with a constrained `BackendSyncStatus`
dataclass containing:

```python
from typing import Literal


SyncStatus = Literal["requested", "failed", "worker_verified"]


mode: str
sync_policy: str
lora_tensors_digest: str | None
coord_offset_digest: str | None
coord_ids_digest: str | None
synced_step: int | None
server_ids: tuple[str, ...]
status: SyncStatus
```

- [ ] **Step 5: Run trace tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_decode_backend_trace_contract.py tests/test_vllm_logprob_trace_contract.py tests/test_vllm_server_rollout_contract.py -q
```

Expected: trace tests pass without touching Stage-2 rollout code yet.

## Task 4: Parsing, Provenance, And Artifact Guards

**Files:**

- Create: `src/infer/parsing.py`
- Modify: `src/infer/artifacts.py`
- Modify: `src/eval/detection_orchestrator.py`
- Modify: `src/eval/detection_records.py`
- Modify: `src/eval/detection_coco.py`
- Modify: `scripts/export_coco_submission.py`
- Create: `scripts/stamp_inference_provenance.py`
- Modify: `tests/test_decode_provenance_contract.py`
- Modify: `tests/test_detection_eval_output_parity.py`
- Create or modify: `tests/test_parser_policy_parity.py`

- [ ] **Step 1: Add parser result separation**

Create `src/infer/parsing.py` with separate strict and diagnostic result objects. Strict results can be metric-bearing. Diagnostic salvage results must carry `metric_bearing=False`.

- [x] **Step 2: Add comparable artifact loader**

Update `src/infer/artifacts.py` with `load_comparable_artifact(path: Path)` that resolves provenance from:

```text
path.parent / "resolved_config.json"
path.parent / "summary.json"
path.parent / "infer_summary.json"
path.parent / "resolved_config.path"
path.with_suffix(path.suffix + ".provenance.json")
```

If none provide required provenance, raise `ValueError("missing_provenance: ...")`.
For `resolved_config.path`, follow the pointer to the parent training run
metadata without moving or renaming the frozen Stage-2 eval artifacts.

Do not treat best-effort or transitional fingerprints synthesized from partial
owner fields as comparable provenance. Comparable loading requires canonical
first-class prompt/decode/model fingerprints that match the approved
fingerprint vocabulary. Migration tools may emit inspection-only
`comparable:false` records when exact reconstruction is impossible, but they
must not invent fingerprints to promote legacy artifacts into official eval or
comparison paths.

- [x] **Step 3: Add score policy fingerprint helpers**

Add helper functions in `src/infer/artifacts.py` that compute score fingerprints from policy name, score source, aggregation rule, token span rule, constant-score value, source raw artifact identity, parser policy, and `metric_bearing`.

Status update: `load_comparable_artifact(..., require_score=True)` now gates
official scored-artifact consumption, rejects transitional or unbound run-level
carriers, requires score sidecars to bind to the exact scored artifact they
validate, and requires raw artifacts to remain `score_policy: none`. Pipeline
materialized scored artifacts now get colocated `.provenance.json` sidecars
with prompt/decode/model fingerprints plus `score_policy_fingerprint` when raw
provenance is exact; cache hits without valid score provenance are recomputed
rather than stamped. Raw artifact paths are kept as lineage in the sidecar, but
the score-policy fingerprint uses the raw content hash rather than the absolute
path so equivalent copied artifacts do not get artificial score-policy drift.

- [x] **Step 4: Add historical stamping script**

Create `scripts/stamp_inference_provenance.py`. It must:

```text
Accept --run-dir.
Read summary.json and resolved_config.json.
Write a sidecar only when exact provenance can be reconstructed.
Otherwise print a comparable=false reason and exit nonzero unless --allow-inspection-stamp is set.
Never invent prompt/decode/model/score fingerprints.
```

- [x] **Step 5: Wire official eval and export through provenance gates**

Update COCO/LVIS/both official eval paths and COCO submission export to call
the comparable-artifact/provenance gate before metric computation or submission
export. Inspection/debug/visualization loaders may still read historical JSONL
as `comparable: false`, but official metric/report/export paths must reject
missing fingerprints with `missing_provenance`.

Status update: official metric/export reducers now gate score provenance at
the lower-level boundary. `evaluate_and_save()` calls
`load_comparable_artifact(..., require_score=True)` for COCO/LVIS/both metrics,
and `export_coco_submission()` gates score provenance before export. The
offline pipeline, `scripts/evaluate_detection.py`, and
`scripts/export_coco_submission.py` therefore fail fast before official metrics
or submission export. Proxy eval bundles now require source score provenance
before official per-view evaluation and write bound sidecars for derived views.
Stage-1 and Stage-2 training-time materialized official eval artifacts now
write bound score-provenance sidecars before `evaluate_and_save()` runs.
Stage-2 official eval also fails fast when `materialize_artifacts: false` or
`training.output_dir` would force in-memory AP computation without a
score-provenanced `gt_vs_pred_scored.jsonl`.

- [ ] **Step 6: Add Stage-2 eval provenance carrier test**

Add a test using a Stage-2 `eval_detection/step_<global_step>/` style fixture
where provenance resolves through `infer_summary.json` and/or
`resolved_config.path`. The comparable loader must not require a colocated
`summary.json` or `resolved_config.json` in that directory.

- [x] **Step 7: Run parser/provenance tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_decode_provenance_contract.py tests/test_parser_policy_parity.py tests/test_detection_eval_output_parity.py -q
```

Expected: provenance and parser policy tests pass.

Status update: ran the broader provenance/eval/inference bundle under the
`ms` environment:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_decode_request_mapping.py tests/test_infer_pipeline_shared_decode_request.py tests/test_detection_prompt_input_codec.py tests/test_inference_runtime_backend_facade.py tests/test_decode_provenance_contract.py tests/test_infer_batch_decoding.py tests/test_infer_artifact_metadata.py tests/test_infer_stop_pressure.py tests/test_run_infer_legacy_shared_runtime.py tests/test_unified_infer_pipeline.py tests/test_qwen_generation_contract.py tests/test_stage2_rollout_runtime.py tests/test_infer_layout_import_gates.py tests/test_coco_submission_export_provenance.py tests/test_evaluate_detection_provenance.py tests/test_detection_eval_output_parity.py tests/test_stamp_inference_provenance.py tests/test_score_policy_fingerprint.py tests/test_proxy_eval_bundle.py tests/test_proxy_eval_views.py tests/test_stage1_detection_eval.py -q
```

Result: `299 passed, 2 warnings`.

## Task 5: Constraint Facade And Old Constraint Import Migration

**Files:**

- Create: `src/infer/constraints.py`
- Done: `src/infer/compact_grammar.py` and `src/infer/stop_pressure.py`
  folded into private `src/infer/_constraints_impl.py`
- Modify: `tests/test_infer_batch_decoding.py`
- Modify: Stage-2 tests that monkeypatch old constraint modules

- [x] **Step 1: Add public constraint facade**

Create `src/infer/constraints.py` and re-export compact grammar and stop-pressure builders through stable facade names.

- [x] **Step 2: Update callers**

Replace caller imports of:

```text
src.infer.compact_grammar
src.infer.stop_pressure
```

with:

```text
src.infer.constraints
```

- [x] **Step 3: Privatize or delete old modules**

Either move implementation into `constraints.py` or rename helpers to private names such as `src/infer/_compact_constraints.py` and `src/infer/_stop_pressure_constraints.py`.

- [x] **Step 4: Run constraint tests and search gate**

Run:

```bash
conda run -n ms python -m pytest tests/test_infer_batch_decoding.py tests/test_qwen_generation_contract.py -q
rg -n "src\\.infer\\.(compact_grammar|stop_pressure)|from src\\.infer\\.(compact_grammar|stop_pressure)" src tests scripts docs openspec configs
```

Expected: tests pass; search gate returns no active caller-facing imports.

Status update: compact grammar and stop-pressure implementations now live
behind the public `src.infer.constraints` facade in private
`src.infer._constraints_impl`. The old public modules are deleted and
`tests/test_infer_constraints_facade.py` asserts they cannot be imported.
Stop-pressure policy constants also moved out of legacy `src.infer.engine` into
the same shared facade, so `src.infer.pipeline` no longer imports old engine
names for constraint policy validation.
Focused constraint/decode verification passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_constraints_facade.py tests/test_infer_compact_grammar.py tests/test_infer_stop_pressure.py tests/test_infer_batch_decoding.py tests/test_qwen_generation_contract.py tests/test_stage2_rollout_runtime.py::test_hf_rollout_logits_processor_wires_compact_grammar -q
```

Result: `54 passed, 2 warnings`. The broader focused bundle also passed with
`313 passed, 2 warnings`.

Follow-up status update: after moving stop-pressure policy constants, the
focused pipeline/stop-pressure/layout band passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_pipeline_shared_decode_request.py tests/test_infer_stop_pressure.py tests/test_infer_constraints_facade.py tests/test_infer_layout_import_gates.py -q
```

Result: `35 passed`.

## Task 6: Offline Inference Integration

**Files:**

- Modify: `scripts/run_infer.py`
- Modify: `src/infer/pipeline.py`
- Modify: `src/infer/checkpoints.py`
- Modify: `src/infer/artifacts.py`
- Modify: `tests/test_unified_infer_pipeline.py`
- Modify: `tests/test_infer_batch_decoding.py`
- Modify: `tests/test_qwen_generation_contract.py`

- [x] **Step 1: Map `infer.*` config into shared request objects**

Add conversion helpers in `src/infer/pipeline.py` or `src/infer/runtime.py` that map current `infer.generation.*`, backend settings, and checkpoint settings into `DetectionPromptPolicy`, `DetectionDecodeRequest`, and model identity.

Status update: offline prompt, decode, and model identity now map through shared
runtime helpers. Decode requests emit canonical `decode:*` fingerprints, fail
fast for unsupported or misleading bridge policies (`top_k`, beam mode,
contradictory decode mode/temperature, and unknown backend modes), and
propagate into `InferenceConfig` plus `resolved_config.json`. Model identity
emits canonical `model:*` handle-provenance fingerprints, includes
identity-bearing vLLM `backend.model`, and excludes operational server
placement. Prompt policy emits canonical `prompt_policy:*` fingerprints from
resolved template prompts, one-image, and `do_resize=false`; `infer.mode: auto`
is resolved once at the pipeline boundary and the concrete runtime mode is
passed to `InferenceConfig`.

- [ ] **Step 2: Route `scripts/run_infer.py` through shared runtime**

Replace direct construction of old `InferenceEngine`, `GenerationConfig`, and `InferenceConfig` with the shared runtime entrypoint.

- [x] **Step 3: Preserve legacy flag-only entrypoint behavior**

Add a focused script-level test such as
`tests/test_unified_infer_pipeline.py::test_run_infer_legacy_flags_route_shared_runtime`
that exercises documented flag-only `scripts/run_infer.py` usage through shared
runtime conversion without loading a real model.

Status update: `scripts/run_infer.py` legacy flag-only mode now builds a
temporary pipeline config and routes through `run_pipeline`; coverage lives in
`tests/test_run_infer_legacy_shared_runtime.py`.

- [ ] **Step 4: Preserve artifact schemas**

Ensure `gt_vs_pred.jsonl`, `summary.json`, `resolved_config.json`, token traces, `raw_output_json`, `raw_special_tokens`, and error rows remain schema-compatible.

Status update: raw `gt_vs_pred.jsonl` remains unscored, infer-only `xyxy`
runs no longer auto-create scored artifacts unless `confidence:` or official
eval requests them, and scored artifacts use sidecar provenance instead of
mutating raw JSONL carriers.

- [x] **Step 5: Run offline tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_unified_infer_pipeline.py tests/test_unified_infer_pipeline.py::test_run_infer_legacy_flags_route_shared_runtime tests/test_infer_batch_decoding.py tests/test_qwen_generation_contract.py -q
```

Expected: offline inference tests pass with the shared runtime.

Status update: targeted offline/provenance/runtime bands passed under
`/root/miniconda3/bin/conda run -n ms`, including
`tests/test_unified_infer_pipeline.py`, `tests/test_qwen_generation_contract.py`,
`tests/test_infer_batch_decoding.py`, `tests/test_infer_artifact_metadata.py`,
`tests/test_infer_stop_pressure.py`, `tests/test_decode_provenance_contract.py`,
and the new shared-runtime contract tests.

## Task 7: Stage-2 Rollout Integration

**Files:**

- Modify: `src/trainers/stage2_rollout_correction_impl.py`
- Modify: `src/trainers/stage2_rollout_runtime.py`
- Modify: `src/trainers/rollout_correction/target_builder.py`
- Modify: `src/trainers/rollout_correction/teacher_forcing_adapter.py`
- Modify: `src/trainers/rollout_matching/preflight.py`
- Modify: `tests/test_stage2_rollout_runtime.py`

Final owner map for responsibilities currently mixed into
`src/trainers/stage2_rollout_runtime.py`:

- Decode prompt/backend/trace/parser behavior moves to `src/infer`.
- Rollout attempt scheduling and execution coordination stays in
  `src/trainers/rollout_correction/executors.py` and
  `src/trainers/rollout_correction/coordination.py`.
- Dynamic post-rollout packing stays in
  `src/trainers/rollout_correction/pack_schedule.py` or a trainer-owned
  sibling if that file would become too large.
- Eval materialization stays in `src/trainers/rollout_aligned_evaluator.py`.
- DDP metric reduction and rank coordination stay in trainer-owned coordination
  helpers, not `src/infer`.
- Dataloader wrapping and trainer lifecycle glue stay in
  `src/trainers/stage2_rollout_correction_impl.py`.

If `src/trainers/stage2_rollout_runtime.py` remains temporarily, it may only be
a trainer-owned migration facade. It must not define or import decode-owned
surfaces such as `_prepare_samples_for_rollout`, `_rollout_many*`,
`_parse_vllm_server_output*`, `src.trainers.rollout_runtime.*`, or direct
backend adapters by the deletion phase.

- [x] **Step 1: Map `rollout_matching.*` into shared decode request**

Add a Stage-2 conversion function that maps rollout backend, max-new-tokens, sampling settings, stop policy, trace flags, and vLLM server settings into `DetectionDecodeRequest`.

Status update: `src.infer.runtime.build_decode_request_from_rollout_matching_config`
maps `rollout_matching.*` into shared `DetectionDecodeRequest` objects, including
backend/mode, decoding fields, overrides, and canonical `decode:*` fingerprints.
Implicit positive-temperature per-call overrides normalize to `sampling`, while
explicit contradictory decode modes fail fast.

- [ ] **Step 2: Call shared runtime for HF/local rollout generation**

Replace Stage-2 local HF/local `_rollout_many_*`,
`_prepare_samples_for_rollout`, and backend dispatch logic with
`InferenceRuntime.generate_many(...)`. Server-mode vLLM rollout is not claimed
green until Task 8 wires server adapter sync and verifies the server path in the
same slice.

- [ ] **Step 3: Preserve trainable prompt parity gate**

Keep the existing behavior where rollout prompt token IDs are compared with the local teacher-forced prefix. If IDs or visual metadata mismatch, drop the sample or fail before residual target construction.

- [ ] **Step 4: Keep target construction trainer-owned**

Do not move residual correction event construction, greedy IoU assignment, unmatched-GT recovery, target IR creation, duplicate filtering, or objective execution into `src/infer`.

- [ ] **Step 5: Preserve Stage-2 eval materialization**

When `rollout_matching.eval_detection.materialize_artifacts=true`, assert the output directory contains `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `infer_summary.json`, `metrics.json`, `per_image.json`, `raw_rollouts.jsonl`, and `pred_token_trace.jsonl` when traces are available.

- [x] **Step 6: Run Stage-2 rollout tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py -q
```

Expected: Stage-2 rollout tests pass.

Status update: `tests/test_stage2_rollout_runtime.py` passed under
`/root/miniconda3/bin/conda run -n ms` after aligning server-mode vLLM adapter
sync validation with colocate lifecycle behavior.

Follow-up status update: `src.trainers.rollout_runtime.vllm_compat` was a pure
vLLM EngineArgs compatibility helper and has been moved into
`src.infer.backend`. The old trainer-local module is deleted and guarded by
`tests/test_infer_layout_import_gates.py`. Stage-2 rollout metrics now derive
`rollout/do_sample`, `rollout/temperature`, `rollout/top_p`, and
`rollout/top_k` from the canonical rollout `DetectionDecodeRequest` instead
of the duplicate `_decoding_cfg()` / `_decoding_params()` path. ms-swift/vLLM
`RequestConfig` base kwargs now come from `src.infer.backend` and consume the
canonical decode request, including `stop_strings`; per-call seed and
`logprobs=True` overlays stay in runtime-owned rollout backend modules. Stage-2 eval
score-provenance decode fingerprints are now derived from the canonical
rollout decode request, with only eval backend/mode and trace-logprob policy
overridden. Current vLLM rollout helpers were subsequently moved from
`src.trainers.rollout_runtime` into `src.infer.backend_vllm_*`,
`src.infer.rollout_dispatch`, and `src.infer.backend_sync` without changing the
ms-swift synchronization semantics. The Stage-2 rollout/preflight bundle passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_rollout_matching_decoding_cfg.py tests/test_vllm_server_rollout_contract.py -q
```

Result: `111 passed, 2 warnings`. Focused eval provenance coverage also passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py::test_evaluate_emits_rollout_metrics_and_runs_callback tests/test_stage2_rollout_runtime.py::test_evaluate_emits_coco_map_metrics_when_eval_detection_enabled tests/test_stage2_rollout_runtime.py::test_evaluate_emits_coco_map_metrics_with_confidence_postop tests/test_stage2_rollout_runtime.py::test_evaluate_emits_coco_map_metrics_with_confidence_postop_vllm tests/test_decode_provenance_contract.py tests/test_score_policy_fingerprint.py -q
```

Result: `32 passed, 2 warnings`.

Second follow-up status update: HF `GenerationConfig` projection now lives in
`src.infer.backend.apply_hf_generation_config_from_decode_request`, and both
Stage-2 HF rollout paths consume it from the canonical
`DetectionDecodeRequest`. ms-swift/vLLM traced server and colocate outputs now
route through `src.infer.backend.normalize_vllm_trace_response`; trainer code
only adapts the validated shared result back to the existing tuple shape. The
focused vLLM/HF decode band passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_vllm_server_rollout_contract.py tests/test_stage2_rollout_runtime.py tests/test_rollout_matching_decoding_cfg.py tests/test_infer_decode_request_mapping.py tests/test_decode_backend_trace_contract.py -q
```

Result: `158 passed, 2 warnings`.

Third follow-up status update: the trainer-local
`src.trainers.rollout_runtime.swift_infer_compat` shim was folded into
`src.infer.backend` and deleted. The shared backend now owns ms-swift
`RequestConfig`, `InferRequest`, and `to_device` compatibility imports, while
Stage-2 callers keep only orchestration and tuple adaptation. The focused
layout/vLLM/HF band passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_layout_import_gates.py tests/test_vllm_server_rollout_contract.py tests/test_stage2_rollout_runtime.py::test_rollout_many_hf_training_rollout_does_not_force_optimizer_offload tests/test_stage2_rollout_runtime.py::test_hf_rollout_logits_processor_wires_compact_grammar tests/test_stage2_rollout_runtime.py::test_vllm_colocate_rollout_sets_seed_from_request_offset tests/test_stage2_rollout_runtime.py::test_vllm_server_rollout_uses_decode_override_request_config -q
```

Result: `19 passed, 2 warnings`.

Reviewer follow-up status update: confidence-postop Stage-2 eval now hard-fails
on generated-token trace violations before materializing metric-bearing
artifacts. The vLLM trace-required eval path no longer retries/skips malformed
trace samples, and malformed/missing confidence traces no longer downgrade to
constant-score eval artifacts. The focused Stage-2/vLLM trace band passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_vllm_server_rollout_contract.py tests/test_rollout_matching_decoding_cfg.py tests/test_infer_layout_import_gates.py tests/test_decode_backend_trace_contract.py -q
```

Result: `135 passed, 2 warnings`.

Fourth follow-up status update: ms-swift `RequestConfig` object construction
now lives in `src.infer.backend.build_swift_request_config_from_decode_request`.
Trainer rollout modules pass only caller-owned overlays such as deterministic
seed and trace-logprob policy. The refreshed Stage-2/vLLM decode band passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_vllm_server_rollout_contract.py tests/test_rollout_matching_decoding_cfg.py tests/test_infer_layout_import_gates.py tests/test_decode_backend_trace_contract.py tests/test_infer_decode_request_mapping.py -q
```

Result: `165 passed, 2 warnings`.

Code-quality follow-up status update: stale confidence trace fallback counters,
unreachable constant-score fallback plumbing, and the old
`trace_fallback_count` metric lookup were removed after metric-bearing
confidence eval became fail-fast. Non-traced colocate vLLM outputs now raise on
malformed token metadata instead of being converted to empty rollouts. Current
canonical routing docs now point inference backend ownership at
`src.infer.backend` / `src.infer.runtime` instead of deleted
`src.infer.backends.py`. Focused verification passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py::test_evaluate_vllm_confidence_trace_violation_fails_fast tests/test_stage2_rollout_runtime.py::test_evaluate_vllm_confidence_trace_exception_is_fatal tests/test_stage2_rollout_runtime.py::test_evaluate_emits_coco_map_metrics_with_confidence_postop tests/test_stage2_rollout_runtime.py::test_evaluate_emits_coco_map_metrics_with_confidence_postop_vllm tests/test_stage2_rollout_runtime.py::test_vllm_colocate_rollout_rejects_malformed_token_metadata tests/test_stage2_rollout_runtime.py::test_vllm_colocate_rollout_sets_seed_from_request_offset -q
```

Result: `6 passed, 2 warnings`.

Consolidated checkpoint after the code-quality follow-up:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_decode_request_mapping.py tests/test_infer_pipeline_shared_decode_request.py tests/test_detection_prompt_input_codec.py tests/test_inference_runtime_backend_facade.py tests/test_decode_provenance_contract.py tests/test_infer_batch_decoding.py tests/test_infer_artifact_metadata.py tests/test_infer_stop_pressure.py tests/test_run_infer_legacy_shared_runtime.py tests/test_unified_infer_pipeline.py tests/test_qwen_generation_contract.py tests/test_stage2_rollout_runtime.py tests/test_infer_layout_import_gates.py tests/test_infer_constraints_facade.py tests/test_infer_compact_grammar.py tests/test_coco_submission_export_provenance.py tests/test_evaluate_detection_provenance.py tests/test_detection_eval_output_parity.py tests/test_stamp_inference_provenance.py tests/test_score_policy_fingerprint.py tests/test_proxy_eval_bundle.py tests/test_proxy_eval_views.py tests/test_stage1_detection_eval.py tests/test_stage2_launcher_preflight_contract.py tests/test_stage2_preflight_server_knob_plumbing.py tests/test_stage2_vllm_server_launcher.py tests/test_rollout_matching_decoding_cfg.py tests/test_vllm_server_rollout_contract.py -q
```

Result: `380 passed, 2 warnings`.

Follow-up status update: server-mode vLLM response parsing now routes directly
through `src.infer.backend.normalize_vllm_trace_response`; Stage-2
`_parse_vllm_server_output*` wrapper methods were removed. Focused parser and
server override verification passed:

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_vllm_server_rollout_contract.py tests/test_stage2_rollout_runtime.py::test_parse_vllm_server_traced_single_trailing_stop_fails_fast tests/test_stage2_rollout_runtime.py::test_parse_vllm_server_traced_large_trailing_trace_fails_fast tests/test_stage2_rollout_runtime.py::test_parse_vllm_server_traced_strips_left_padded_prompt_token_ids tests/test_stage2_rollout_runtime.py::test_vllm_server_rollout_uses_decode_override_request_config tests/test_stage2_ab_vllm_server_mode_smoke.py::test_vllm_server_prompt_tokenization_parity_smoke -q
```

Result: `14 passed, 1 skipped, 2 warnings`.

## Task 8: vLLM Server Rollout And Sync Migration

**Files:**

- Modify: `src/infer/backend.py`
- Modify: `src/infer/backend_sync.py`
- Done: `src/trainers/rollout_runtime/vllm_server.py` moved to
  `src/infer/backend_vllm_server.py`
- Done: `src/trainers/rollout_runtime/vllm_config.py` moved to
  `src/infer/backend_vllm_config.py`
- Done: `src/trainers/rollout_runtime/vllm_engine.py` moved to
  `src/infer/backend_vllm_engine.py`
- Done: `src/trainers/rollout_runtime/vllm_infer.py` moved to
  `src/infer/backend_vllm_infer.py`
- Done: `src/trainers/rollout_runtime/dispatch.py` moved to
  `src/infer/rollout_dispatch.py`
- Done: `src/trainers/rollout_runtime/swift_coord_row_patch.py` moved to
  `src/infer/backend_sync.py`
- Modify: `tests/test_vllm_server_adapter_payload.py`
- Modify: `tests/test_swift_rollout_endpoints_contract.py`
- Modify: `tests/test_ddp_vllm_sync_failure_propagation.py`
- Modify: `tests/test_training_config_strict_unknown_keys.py`

- [ ] **Step 1: Preserve official adapter sync settings**

Ensure active Stage-2 vLLM server rollout requires:

```yaml
rollout_matching:
  vllm:
    mode: server
    enable_lora: true
    sync:
      mode: adapter
```

- [ ] **Step 2: Migrate LoRA filtering and coord-row sync**

Move LoRA-compatible tensor filtering and patched coord-row update client behavior into the shared backend sync layer. Keep functional behavior equivalent.

- [ ] **Step 3: Wire server rollout through shared runtime in the same slice**

Route Stage-2 server-mode vLLM rollout generation through the shared backend
adapter only after adapter sync, coord-row sync, request adaptation, and trace
normalization are available in `src/infer`.

- [ ] **Step 4: Preserve DDP visible-failure propagation**

Rank-0 sync exceptions visible before rollout must abort all ranks. Fire-and-forget server acknowledgement must be recorded as requested/learner-side unless worker verification exists.

- [ ] **Step 5: Add explicit sync-status tests**

Add tests that simulate the current `/update_token_row_offsets/` request-ack
behavior and assert `BackendSyncStatus.status == "requested"` rather than
`"worker_verified"`. Add a separate positive test where a fake worker
verification signal is required before `worker_verified` is allowed.

- [ ] **Step 6: Run vLLM server rollout and sync tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_vllm_server_adapter_payload.py tests/test_swift_rollout_endpoints_contract.py tests/test_ddp_vllm_sync_failure_propagation.py tests/test_training_config_strict_unknown_keys.py tests/test_stage2_rollout_runtime.py -q
```

Expected: vLLM server sync tests pass and full-sync defaults are not accepted as the active Stage-2 server path.

## Task 9: Delete Overlapping Layout

**Files:**

- Delete: `src/trainers/rollout_runtime/`
- Delete: `src/infer/engine.py`
- Done: `src/infer/backends.py` folded into `src/infer/backend.py` and deleted
- Delete or privatize: `src/infer/compact_grammar.py`
- Delete or privatize: `src/infer/stop_pressure.py`
- Delete or reduce: `src/trainers/stage2_rollout_runtime.py`
- Create: `tests/test_infer_layout_import_gates.py`

- [ ] **Step 1: Add final import-ban test**

Create `tests/test_infer_layout_import_gates.py` that scans active source text and fails on:

```text
src.trainers.rollout_runtime
src.trainers.stage2_rollout_runtime defining/importing decode-owned surfaces
src.infer.engine
src.infer.compact_grammar
src.infer.stop_pressure
direct src.infer.backend_sync imports outside src/infer owner modules
direct src.infer._*constraint* helper imports outside src/infer owner modules
```

Status update: the `src.infer.backends` module has been removed. HF/vLLM
batch generation now lives under `src.infer.backend`, and
`tests/test_infer_layout_import_gates.py` asserts the removed module cannot be
imported.
`src.infer.compact_grammar` and `src.infer.stop_pressure` are also removed;
the only public constraint import surface is now `src.infer.constraints`.
Trainer-local HF generation-config projection and ms-swift/vLLM trace clipping
were also removed from active Stage-2 rollout helpers. `src.trainers` still
owns rollout orchestration, server lifecycle, and tuple adaptation, so the
final import ban remains intentionally open.
The trainer-local ms-swift infer compatibility shim was removed too; import
compatibility for `RequestConfig`, `InferRequest`, and `to_device` now lives in
`src.infer.backend`.

- [ ] **Step 2: Add Stage-2 target-boundary gate**

The target-boundary gate must scan `src/infer` only, not all tests. Tests may
continue to import trainer-owned target builders when explicitly testing target
construction.

Forbidden under `src/infer`:

```text
ResidualBoundaryAdapter
TeacherForcingTargetIR
build_residual_set_target_ir
rollout_correction.target_builder
greedy_match_iou
associate_one_to_one_greedy_iou
src.common.duplicate_control
```

- [ ] **Step 3: Delete old files after callers are migrated**

Delete only after targeted tests from prior tasks pass.

- [ ] **Step 4: Run active code import gates**

Run:

```bash
conda run -n ms python -m pytest tests/test_infer_layout_import_gates.py -q
rg -n "src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)|from src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)|src\\.trainers\\.rollout_runtime|from \\.rollout_runtime" src tests scripts
rg -n "ResidualBoundaryAdapter|TeacherForcingTargetIR|build_residual_set_target_ir|rollout_correction\\.target_builder|greedy_match_iou|associate_one_to_one_greedy_iou|src\\.common\\.duplicate_control" src/infer
```

Expected: active code gates pass. Docs/spec/config route gates run after Task
10 updates those surfaces.

## Task 10: Docs, Stable Specs, And Config Routes

**Files:**

- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/AGENT_INDEX.md`
- Modify: `docs/catalog.yaml`
- Modify: `docs/ARTIFACTS.md`
- Modify: `docs/eval/WORKFLOW.md`
- Modify: `docs/training/STAGE2_RUNBOOK.md`
- Modify: `docs/training/METRICS.md`
- Modify: stable specs under `openspec/specs/`
- Modify: configs that mention old inference/rollout import surfaces

- [ ] **Step 1: Update docs routes**

Document `src/infer` as shared implementation root while preserving `infer.*` and `rollout_matching.*` authored namespaces.

- [ ] **Step 2: Update artifact docs**

Document prompt/decode/model/score fingerprints, provenance carrier resolution, `missing_provenance`, raw/scored split, and Stage-2 `eval_detection/step_*` preservation.

- [ ] **Step 3: Apply OpenSpec deltas to stable specs**

Only after implementation and tests pass, apply the new requirements from `openspec/changes/unify-inference-runtime/specs/**` into `openspec/specs/**`.

- [ ] **Step 4: Run docs/spec route gates**

Run after docs/spec/config updates, not before:

```bash
rg -n "src/infer/(engine|backends|compact_grammar|stop_pressure)\\.py|src/trainers/rollout_runtime/|src\\.infer\\.(engine|backends)|src\\.trainers\\.rollout_runtime" docs openspec/specs configs
```

Expected: no stale active docs/spec/config routes to removed surfaces.

- [ ] **Step 5: Run docs/spec tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_artifact_contract_docs.py tests/test_experiment_manifest_file.py tests/test_run_manifest_files.py tests/test_run_metadata_file.py -q
```

Expected: docs/artifact contract tests pass.

## Task 11: Final Verification

**Files:**

- No new code files unless failures require fixes.

- [ ] **Step 1: Run inference/config/runtime tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_detection_prompt_input_codec.py tests/test_decode_backend_trace_contract.py tests/test_vllm_logprob_trace_contract.py tests/test_decode_provenance_contract.py tests/test_parser_policy_parity.py tests/test_unified_infer_pipeline.py tests/test_infer_batch_decoding.py tests/test_qwen_generation_contract.py -q
```

- [ ] **Step 2: Run Stage-2 and vLLM tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_vllm_server_rollout_contract.py tests/test_vllm_server_adapter_payload.py tests/test_swift_rollout_endpoints_contract.py tests/test_ddp_vllm_sync_failure_propagation.py -q
```

- [ ] **Step 3: Run config/runtime integration tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_training_config_strict_unknown_keys.py tests/test_training_runtime_plan.py tests/test_training_runtime_profile.py tests/test_training_runtime_sft_integration.py -q
```

- [ ] **Step 4: Run final search gates**

Run:

```bash
rg -n "custom\\.trainer_variant: stage2_two_channel|configs/stage2_two_channel|src/trainers/stage2_two_channel|stage2_ab/schedule|stage2_ab\\.schedule|stage2_ab\\.pipeline|stage2_ab/b_ratio_realized|stage2/channel_[ab]|_stage2_ab_channel" src scripts configs
rg -n "stage2_ab|stage2_two_channel|Channel-A|Channel-B|channel_a|channel_b|b_ratio" docs openspec/specs configs | rg -v "retired|historical|history|hard-cut|reject|removed|forbidden|migration|archive|search gate|compatibility pointer"
rg -n "src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)|from src\\.infer\\.(engine|backends|compact_grammar|stop_pressure)|src\\.trainers\\.rollout_runtime|from \\.rollout_runtime" src tests scripts
rg -n "src/infer/(engine|backends|compact_grammar|stop_pressure)\\.py|src/trainers/rollout_runtime/|src\\.infer\\.(engine|backends)|src\\.trainers\\.rollout_runtime" docs openspec/specs configs
rg -n "ResidualBoundaryAdapter|TeacherForcingTargetIR|build_residual_set_target_ir|rollout_correction\\.target_builder|greedy_match_iou|associate_one_to_one_greedy_iou|src\\.common\\.duplicate_control" src/infer
```

Expected: no live config directories, trainer modules, public config handles, or
active docs route users to the removed Stage-2 AB/two-channel surfaces. Stable
specs and tests that prove hard-cut rejection are allowed when they are clearly
removal contracts rather than active routes. No Stage-2 target-construction
imports may appear under `src/infer`.

- [ ] **Step 5: Report skipped expensive checks**

If no production-scale smoke is run, report that explicitly and name the narrowest recommended smoke:

```bash
conda run -n ms python -m pytest tests/test_stage2_rollout_runtime.py tests/test_vllm_server_rollout_contract.py -q
```

## Self-Review Notes

Current verification snapshot (2026-05-26):

```bash
/root/miniconda3/bin/conda run -n ms python -m pytest tests/test_infer_decode_request_mapping.py tests/test_infer_pipeline_shared_decode_request.py tests/test_detection_prompt_input_codec.py tests/test_inference_runtime_backend_facade.py tests/test_decode_provenance_contract.py tests/test_infer_batch_decoding.py tests/test_infer_artifact_metadata.py tests/test_infer_stop_pressure.py tests/test_run_infer_legacy_shared_runtime.py tests/test_unified_infer_pipeline.py tests/test_qwen_generation_contract.py tests/test_stage2_rollout_runtime.py tests/test_infer_layout_import_gates.py tests/test_infer_constraints_facade.py tests/test_infer_compact_grammar.py tests/test_coco_submission_export_provenance.py tests/test_evaluate_detection_provenance.py tests/test_detection_eval_output_parity.py tests/test_stamp_inference_provenance.py tests/test_score_policy_fingerprint.py tests/test_proxy_eval_bundle.py tests/test_proxy_eval_views.py tests/test_stage1_detection_eval.py tests/test_stage2_launcher_preflight_contract.py tests/test_stage2_preflight_server_knob_plumbing.py tests/test_stage2_vllm_server_launcher.py tests/test_rollout_matching_decoding_cfg.py tests/test_vllm_server_rollout_contract.py -q
```

Result: `375 passed, 2 warnings`.

Spec coverage:

- Shared prompt/decode/backend/parse/provenance contracts are covered by Tasks 1 through 4.
- Offline inference integration is covered by Task 6.
- Stage-2 rollout integration and target-boundary preservation are covered by Task 7.
- vLLM logprob and adapter sync boundaries are covered by Tasks 3 and 8.
- Deletion of overlapping layouts is covered by Tasks 5 and 9.
- Docs, stable specs, and final verification are covered by Tasks 10 and 11.

Completeness scan:

- The plan avoids deferred-action markers.
- Each task names concrete files and verification commands.

Execution recommendation:

- Use subagent-driven development after final approval.
- Assign independent workers by phase only after the shared dataclasses and prompt/trace tests are stable.
- Do not run final import-ban tests as required pass gates until the deletion phase.
