---
doc_id: docs.research-probe-infra-base
layer: docs
doc_type: workflow
status: canonical
domain: research
summary: Selects the smallest existing mechanics owner for a research-probe producer.
tags: [research, probes, artifacts, admission, inference]
updated: 2026-08-30
---

# Research Probe Infrastructure Base

This page routes each producer to existing mechanics. It does not define a
runner, lifecycle, scientific plan, or result schema. The fixed integration
route is `research-probe-infras -> research-probes -> probe/<direction>`; the
authoritative rules and authorization gates are in
[`BRANCH_AND_WORKTREE_POLICY.md`](BRANCH_AND_WORKTREE_POLICY.md).

## Select The Smallest Profile

Select capabilities independently for each operating-system producer. Adding a
journal does not imply admission; adding admission does not imply the inference
session. A custom differentiable runtime does not fit the deterministic decode
session and remains direction-local.

| Producer need | Public owner | Caller still owns | Failure meaning | Cheapest acceptance check |
| --- | --- | --- | --- | --- |
| One strict immutable result | `src.artifacts`: `validate_json_value`, `canonical_json_bytes`, `json_sha256`, `publish_json_exclusive`, `load_canonical_json` | Payload schema and meaning, output path, whether write-once publication fits | Invalid strict JSON or an occupied path fails closed; existing bytes are not replaced | Publish and reload one temporary CPU result, then verify a second publication is rejected without changing its hash |
| Several durable work items or process continuation | `src.artifacts.ExecutionEvidenceJournal` | Plan meaning, producer scheduling, retry authorization, terminal interpretation | Accepted records remain durable; an incomplete or failed attempt is not execution completion and does not authorize retry | Complete a temporary CPU journal, reopen or inspect it, and verify its exact identity and record set |
| CPU evidence before a bounded model launch | `src.artifacts`: binding request types, capture/revalidation functions, and `ResearchProbeAdmission` | Scientific plan, target worktree choice, launcher, output meaning, launch authorization, continuation and stop rule | A typed binding, target, stage, or durability failure closes the mechanics gate; `mechanically_admitted` is non-authorizing | Run the owner-specific CPU preflight and inspect the resulting mechanics-only dossier |
| Production-aligned deterministic decode | `src.inference.runtime` frontend/launch and `src.inference.backend` request/result/session contracts | Requested contrast, model suitability, parsing, evaluation, metrics, and claims | Contract or session failure is decode-mechanics evidence only; explicit cleanup still applies | Exercise an injectable CPU session contract, or the smallest separately authorized production-shaped smoke |

The normative owners remain the
[journal contract](../openspec/specs/coordexp-swift-execution-evidence-journal/spec.md),
[admission contract](../openspec/specs/coordexp-swift-research-probe-admission/spec.md),
and existing [inference runtime](../openspec/specs/coordexp-swift-infer-config-runtime/spec.md)
and [pipeline](../openspec/specs/coordexp-swift-infer-pipeline/spec.md) contracts.

## Public Imports

The one-shot profile needs no journal, admission root, or generated runner:

```python
from src.artifacts import (
    canonical_json_bytes,
    json_sha256,
    load_canonical_json,
    publish_json_exclusive,
    validate_json_value,
)

publish_json_exclusive(result_path, caller_owned_payload)
assert load_canonical_json(result_path) == caller_owned_payload
```

`publish_json_exclusive` never offers overwrite fallback. Use it only for a
final path that must be absent. Overwrite-permitted intermediates remain
caller-owned.

The journaled profile adds one owner directly:

```python
from src.artifacts import ExecutionEvidenceJournal

journal = ExecutionEvidenceJournal.create(
    root=producer_root,
    execution_id=producer_id,
    execution_identity=caller_owned_identity,
    expected_work_item_ids=caller_owned_work_items,
    context=caller_owned_strict_context,
)
```

Each independently scheduled producer uses its own output or journal root and
its own local record sequence. A common immutable input may be digest-bound in
each execution identity. This does not create an experiment-global journal,
lock, append order, stage order, worker registry, DAG, or terminal barrier.

The admitted-launch profile uses the public artifact facade while keeping all
behavior in the admission owner:

```python
from src.artifacts import (
    AbsoluteExecutableBinding,
    AdmissionInspection,
    BindingManifest,
    DirectoryTreeBinding,
    RegularFileBinding,
    ResearchProbeAdmission,
    ResearchProbeAdmissionError,
    ReservedOutputPath,
    ResolvedDataFileBinding,
    StageEvidence,
    StrictValueBinding,
    TargetTreeBinding,
    TargetTreeIdentity,
    capture_binding_manifest,
    capture_target_tree_binding,
    revalidate_binding_manifest,
    revalidate_target_tree_binding,
)
```

The caller must revalidate the complete target-tree identity immediately before
any launcher, model load, GPU allocation, or vertical artifact creation.

Inference keeps its own public module paths rather than being re-exported by
`src.artifacts`:

```python
from src.inference.backend import (
    BackendLaunch,
    BackendSession,
    BackendSessionReceipt,
    DecodeRequest,
    DecodeResult,
    open_backend_session,
)
from src.inference.runtime import (
    InferenceFrontend,
    assemble_frontend,
    prepare_backend_launch,
)
```

Use the context-managed session so cleanup remains explicit. Do not wrap direct
differentiable model mutation as ordinary inference.

## Evidence Boundary

Shared validation may report strict serialization, exact identity, exclusive
durability, plan completeness, attempt history, or launch-gate closure. Payload
fields such as cohort, intervention, objective, optimizer, metric, threshold,
outcome, claim, continuation decision, and stop rule remain opaque and
caller-owned. `completed` and `mechanically_admitted` are mechanical statuses,
not scientific success, promotion, retry, launch, or publication decisions.

The base does not own worktree lifecycle, producer scheduling, distributed rank
groups or barriers, training mutation, optimizer or RNG state, rollback,
checkpoint semantics, monitoring, metrics, claims, continuation, or stop rules.

## Specimen Boundary

Image2299 is the live reference specimen. Its immutable leaves can reuse strict
identity and exclusive publication, but its distributed stages, barriers,
direct differentiable HF runtime, updates, checkpoints, metrics, and stop rules
remain direction-local.

Human13 is a historical specimen, not a live consumer or resumed route. Its
independently scheduled acquisition, rebase, and cell producers demonstrate
per-producer identity and durability. Its optimizer/RNG transaction, rollback,
restoration, unfinished all-HF route, metrics, and stop rules are not part of
the shared base.

The specimens therefore establish only common identity and artifact durability;
they do not establish common runtime, scheduling, rollback, checkpoint, or
scientific semantics.

## Promotion Gate

Before proposing another shared execution layer, all of the following must be
true:

- At least two live cross-direction consumers use the same caller-visible
  contract; repeated use in one direction plus a historical specimen does not
  qualify.
- Their semantics, scheduling topology, artifacts, and failure behavior agree.
- The duplicated mechanics are identified and direct composition of current
  owners is shown to be insufficient.
- The smallest proposed seam preserves each producer's topology.
- One cheapest bounded comparison is named that would falsify the common seam.

A generic coordinator, phase DSL, trainable-HF session, optimizer/RNG
transaction, checkpoint API, monitor, and lifecycle automation are deferred,
not planned features.
