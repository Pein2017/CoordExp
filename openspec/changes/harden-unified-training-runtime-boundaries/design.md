## Context

The current branch has already archived a unified inference runtime refactor.
Stable specs now carry much of that contract, but the architecture audit found
that several ownership seams remain shallow. The docs and specs say "unified
runtime," while live code still lets important behavior travel through trainer
attributes, broad `owner` objects, and compatibility names from older Stage-2
branches.

The desired endpoint is not a new framework. It is a set of deeper, smaller
modules whose responsibilities survive the deletion test:

```text
authored config
  -> typed config
  -> resolved runtime projection
  -> narrow caller adapter
  -> shared or trainer-owned concern module
  -> artifact/provenance/metric surface
```

Deleting any of these modules should remove a real responsibility, not merely a
label around logic still owned by `sft.py`, `Stage2RolloutRuntime`, or broad
owner-shaped helpers.

## Key Decisions

### Treat Archived OpenSpec Changes As History

The archived `unify-inference-runtime` change is useful provenance, but it is
not an active implementation target. This change references stable specs under
`openspec/specs/` as current authority and uses archived specs only to identify
promotion gaps or historical intent.

### Keep Public Namespaces Stable

This change hardens internal module boundaries while preserving authored
workflow names:

- offline inference continues to author `infer.*`;
- Stage-2 runtime/backend/decode/eval continues to author `rollout_matching.*`;
- Stage-2 objectives continue to author `stage2_rollout_correction.*`;
- Stage-2 remains `custom.trainer_variant: stage2_rollout_correction`.

Any eventual migration away from these public names requires a separate
compatibility-sensitive OpenSpec change.

### Resolve Runtime Meaning Before Trainer Wiring

`src/sft.py` currently performs too much interpretation: it merges rollout
matching config, Stage-2 objective config, packing, prompt fallback, geometry
format, object ordering, and manifests, then pushes those decisions into the
trainer through attributes.

The target architecture keeps `sft.py` as a launcher over already-resolved
runtime state. A resolved Stage-2 runtime projection owns:

- which public namespace supplied each policy;
- the effective rollout prompt/decode/backend/eval policies;
- packing ownership and post-rollout packing settings;
- object ordering, object field order, bbox format, and detection sequence
  format needed by rollout and training;
- pipeline manifest and policy provenance payloads.

Compatibility bridges may remain during migration, but they must be explicit
and provenance-bearing. Silent fallbacks such as deriving eval prompt behavior
from `custom.extra.prompt_variant` must not become hidden stable defaults.

### Make Stage-2 Target Construction A Real Boundary

Stage-2 target construction should be owned by trainer-side correction modules,
but it should not require vLLM lifecycle, DDP phase orchestration, or full
trainer state to test residual-set target behavior.

The target boundary is responsible for:

- parsed rollout objects and GT state;
- duplicate filtering and triage decisions;
- false-negative insertion policy;
- residual event construction;
- supervision metadata needed by the loss path;
- metric/diagnostic ingredients that describe target construction.

The trainer remains responsible for:

- calling rollout generation;
- DDP/packing coordination;
- model forward and loss execution;
- optimizer step behavior;
- training metric projection and logging.

### Narrow Shared Inference Runtime Inputs

`src/infer` may own prompt rendering, visual input normalization, decode request
mapping, backend lifecycle/adaptation, parser policy, constraints, artifacts,
checkpoint resolution, visualization helpers, and offline pipeline
orchestration.

It must not discover policy by calling arbitrary trainer-private methods or by
probing a broad offline-engine object. Owner-like compatibility is allowed only
at the edge adapter that translates an existing caller into resolved prompt,
decode, model, backend, parser, and artifact facts.

Backend-heavy imports should stay behind the backend path that actually needs
them so config, prompt, parser, and artifact helpers remain cheap to import.

### Single-Own Score And Metric-Bearing Provenance

Offline inference, unified infer/eval pipeline stages, Stage-2 eval materialize
paths, and official eval comparability all need the same answer to:

- what raw artifact was scored;
- what score policy made a prediction metric-bearing;
- what parser policy was used;
- whether diagnostic salvage was excluded from official metric inputs;
- how prompt/decode/model/score fingerprints are recorded.

The final architecture should have one writer/validator for those semantics.
Pipelines decide when to materialize artifacts; they should not each re-create
the score/provenance schema.

### Treat Metric-Bearing Stage-2 Eval Validity As A First Slice

Review found that Stage-2 eval artifact validity is the highest-risk shared
inference/provenance issue. Metric-bearing Stage-2 eval must fail before
official artifact materialization if it lacks exact source image identity,
width, height, parser strictness, or real prompt/decode/model/score
provenance.

The design does not allow:

- fabricated image names such as `image_<idx>.jpg` in official eval artifacts;
- default dimensions such as `1000x1000` for metric-bearing rows;
- best-effort post-hoc rescaling that silently rewrites already-materialized
  official rows;
- salvage parser recoveries in `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, guarded companions, Stage-2 official eval,
  confidence post-op, COCO/LVIS/mAP, or comparable reports;
- synthetic prompt fingerprints that do not reflect the actual prompt bundle,
  visual metadata, detection sequence format, bbox format, template family, and
  prompt-token/visual parity status used for generation.

Diagnostic artifacts may record incomplete geometry or salvage recoveries only
when they declare `metric_bearing: false` and cannot be loaded as comparable or
official eval inputs.

User decision on 2026-05-27: this P0 eval-validity hardening is approved as the
first implementation slice. It should land before broader training runtime
projection, Stage-2 target-construction, DDP/packing, or A/B deletion work.

### Keep Shadow And Historical Surfaces From Becoming Authority

Training surface descriptors, progress notes, retired rollout-matching specs,
and A/B or Channel-B compatibility tests must be classified precisely:

- active runtime contract;
- temporary migration adapter;
- audit-only or shadow descriptor;
- historical evidence.

Docs and catalog entries must not route future agents to retired or shadow
surfaces as if they are current authority.

## Sequencing

1. Finish proposal review and subagent convergence before production edits.
2. Implement the approved P0 metric-bearing Stage-2 eval validity slice:
   exact visual source identity/geometry, strict metric parser outputs, and
   real prompt/decode/model/score provenance before official artifact
   materialization.
3. Add characterization and deletion tests for the current contracts.
4. Move Stage-2 runtime projection out of `sft.py`.
5. Deepen Stage-2 target construction and packing/DDP boundaries.
6. Narrow `src/infer` caller adapters and artifact/provenance writers.
7. Delete or quarantine retired A/B, Channel-B, dead manifest, and shadow
   runtime authority paths.
8. Tighten architecture gates only after the relevant boundary has been moved.

## Review Requirements

Before implementation starts:

- OpenSpec deltas must be reviewed for compatibility scope.
- Superpowers docs must name approval boundaries and implementation order.
- Subagents should converge that this is one coherent architecture-hardening
  change and not an unbounded rewrite.
- The user must explicitly approve implementation.

## Open Questions For Convergence

- Should target construction and shared-inference owner narrowing be implemented
  in one branch after approval, or split into two sequential implementation
  branches under this same OpenSpec?
- Should unknown non-empty `custom.trainer_variant` rejection happen in the
  first implementation slice, or wait until any extension-use cases are ruled
  out?
- Should retired spec/catalog cleanup be a prerequisite before code movement,
  or land with the deletion-gate slice?
- Should the stable `shared-inference-runtime` Purpose placeholder be repaired
  in this proposal change, or handled as a separate spec hygiene pass before
  implementation?
