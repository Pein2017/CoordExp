---
title: PVCI Causal Proposal Bridge - Bounded Implementation Plan
description: File-level plan and verification gates for the non-promoted A/B/C split-layer proposal experiment.
type: idea
role: research-plan
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-causal-proposal-bridge
status: active
updated: 2026-07-11
---

# PVCI Causal Proposal Bridge - Bounded Implementation Plan

## Objective and Stop Rule

Implement only the mechanism needed to distinguish proposal correlation from
causal proposal consumption in the parent [research unit](unit.md). Stop before
training if any authored/runtime/artifact contract cannot be attested. Stop
after one ordered terminal label; do not automatically add coverage contrast,
slots, STOP supervision, rollout training, or a second bridge surface.

## Reuse Boundary

Reuse the current owners for raw examples, rendering spans, Qwen image plans,
packing, causal positions, loss normalization, optimizer groups, Accelerate,
artifacts, adapters, special-token deltas, inference parsing, and scoring. New
code owns only row-state views, proposal/bridge tensors, their receipts, and
their checkpoint payload. Historical painter/proposal branches are idea donors,
not code donors.

## Phase 0 - Real Runtime Attestation

Add one bounded probe under `scripts/probes/coordexp_swift/` that loads the
local Qwen3-VL-2B runtime and a real fixture before modifying training.

It must record and assert:

1. installed package/version and exact model/module paths;
2. main/deepstack merged shapes, order, dtype, device, grad status, and call
   counts;
3. physical image-token intervals for one image and a two-image unequal-grid
   packed input;
4. cloned layer-0 visual/boundary states before deepstack and the intended
   layer-1 injection positions;
5. exact zero/no-op parity and one localized synthetic delta under both SDPA
   and FA2, including logical application counts;
6. one exact 2B+FA2 packed-isolation pass as a hard launch gate;
7. cached prefill plus row-token steps under both SDPA and FA2 proving proposal
   recompute/reset at the prompt tail and `<|box_end|>`;
8. structural parity of the installed public return type and tuple/dict
   fields, `past_key_values`, `cache_position`, and cache lengths, in addition
   to logits/loss/generated-token parity. Installed Transformers 4.57.1
   accepts `output_hidden_states` / `output_attentions` through `**kwargs` but
   does not guarantee both tuples for every attention backend; the probe must
   receipt the actual installed exposure identically on native and zero paths
   and use explicit layer-boundary captures for seam evidence rather than
   fabricate a new public return contract. The first real CUDA execution
   exposed 29 hidden-state entries but no attention tuple under either SDPA or
   FA2;
9. a future-token perturbation test: change current-row phrase/coordinate
   tokens strictly after `s_t-1` and require boundary `q_t/p_t/delta_t` to be
   invariant within the predeclared tolerance;
10. float32 proposal/norm islands and unchanged BF16 model path outside them.

The probe directly loads the installed Qwen module and explicitly receipts its
bypass of `src/qwen/forward.py`, whose production V1 contract intentionally
forces `use_cache=False` and rejects hidden/cache overrides. This bypass is
probe-only, not silent production behavior. Before execution, the probe writes
its deterministic-mode, backend, dtype, and per-tensor absolute/relative
tolerance policy into the artifact request; tolerances cannot be loosened after
observing results.

This probe may contain temporary instrumentation but no trainable mechanism.
Its artifact root and raw receipt are required by VF0/VF1.

## Phase 1 - Data and Causal-Schedule Contracts

### Row-state views

Add a dedicated explicit-order row-state renderer adjacent to the canonical
template owner. Do not change canonical `render_example` semantics.

- canonical views: existing full response and EOS supervision;
- jitter/duplicate views: historical rows are serialized but ignored, current
  row only is supervised, terminal EOS/newline ignored;
- target row and canonical realized order are fixed before perturbation;
- duplicate occurrences use view occurrence identity without weakening
  `RawExample` object-ID uniqueness;
- every view has a unique example/view ID and immutable receipt.

Extend encoding metadata only as needed to carry the view receipt and explicit
row target interval. Packing and dense-label construction remain canonical.
Static pack-cache identity must include the view-manifest hash and renderer
policy/code identity.

### Spatial bags and row plans

Materialize typed records for:

- image/segment-local visual physical interval and grid/merge metadata;
- every current row target interval `[s_t,e_t)` and causal bridge interval
  `[s_t-1,e_t-1)`;
- frozen IoM positive-bag indices and covered-region monitoring bags;
- eligibility, zero-positive, overlap/shared-token, size, count, class, and
  prefix-condition strata.

Records must survive packing without recomputing from model scores. A two-image
fixture must prove no interval, bag, proposal denominator, or delta crosses a
segment.

## Phase 2 - Functional Qwen Split Bridge

Add a research-owned proposal module with registered float32 parameters:

```text
parameter-free FP32 layer_norm(2048, eps=1e-5)
W_q: 2048 -> 256
W_k: 2048 -> 256
U:   2048 -> 2048, zero initialized
```

`W_q/W_k/U` are bias-free float32 parameters; normalization has no affine
parameters. The module consumes cloned layer-0 outputs, computes per-row proposal scores
and pooled states in float32, applies the frozen relative norm cap, and returns
an out-of-place layer-1 input plus typed auxiliary outputs.

Integrate it through a functional variant of the installed Qwen text-forward
loop. Do not use layer hooks for the train path. Preserve every upstream
position, attention-mask, cache, hidden-state, deepstack, gradient-checkpoint,
and return-value behavior. The bridge operation occurs outside checkpointed
decoder-layer calls so logical metrics are not duplicated by recomputation.

Arm policy:

- A: module registered, fixed initialization, frozen, no proposal loss/delta;
- B: train `W_q/W_k` plus authorized existing trainables with `L_prop`; `U`
  frozen at zero; proposal never alters hidden states;
- C: train `W_q/W_k/U`; predicted proposal alters only declared causal row
  positions;
- C-off: same C checkpoint, functional loop, proposal, row schedule, and call
  count; delta replaced by exact zero.

## Phase 3 - Loss, Optimizer, Metrics, and Checkpoint Ownership

Extend existing typed owners rather than creating a parallel trainer.

- loss runner: one FP32 positive-bag term with raw/weighted numerator,
  denominator, eligible/skipped counts, finite status, and DDP reduction;
- optimizer planner: explicit proposal-query/key and bridge-projection groups;
  unmatched trainables still fail;
- trainable-surface receipt: parameter name, shape, dtype, arm, optimizer group,
  requires-grad, and initialization hash;
- checkpoint writer: proposal/bridge safetensors plus JSON identity/semantics;
- checkpoint handoff: optional proposal-bridge payload identity;
- reload: byte/hash and output parity before any metric-bearing inference;
- metrics: proposal, coverage-monitoring, prefix-condition, gradient, norm,
  saturation, and row-binding events with distributed-safe denominators.

Proposal metrics must never label unmatched visual regions as background.
Diagnostic checkpoint payloads are evaluation-only. The 512-step screen is
uninterrupted/non-resumable under the current optimizer-state contract; an
interrupted arm restarts from the frozen initialization rather than silently
resetting AdamW or scheduler state.

Post-training representation production must stream complete, identical row
cohorts for A/B/C and emit the versioned `pvci-representation-controls-v1`
panel frozen in `unit.md`.  Native, raster-position, row-depth, image-mean,
and content-shuffle records share the same row/grid identity and differ only
through the receipted control transform.  All gate-input artifacts require a
validated producer sidecar binding checkpoint, resolved config, cache/view
manifest, cohort, control recipe, code/runtime fingerprint, row IDs, and
support/grid identity.  Raw metric JSONL without that sidecar is not a valid
gate input.

Representation evaluation has an ordered two-stage scope: first the matched
native A/B/C rank-0 monitor shard under the frozen eight-rank schedule as an
in-sample learnability sample, then—only after it passes—the complete
five-control predeclared val200-plus-120 held-out panel with canonical, jitter,
and duplicate-history views.  Full-vector control production over all roughly
93k train rows is not required and must not be mistaken for held-out evidence.
A pilot subset is mechanics-only and cannot satisfy either gate.

## Phase 4 - Inference Controller

Use the existing model, adapter, special-token, image, prompt, parser, and
scorer owners. A research inference controller may wrap generation only where
the production backend lacks the split-layer seam.

For every sequence in the batch it must:

1. capture layer-0 visual states during multimodal prefill;
2. compute the first proposal at the prompt-tail boundary;
3. apply the pulse while each current-row causal token is processed;
4. on input `<|box_end|>`, close the prior row, recompute the next proposal,
   and replace the active pulse;
5. on terminal output, clear state;
6. emit per-row proposal, delta, application/reset, cache-position, parser, and
   condition receipts;
7. support C-on, C-off, another-image, within-image permutation,
   position-only, and norm-matched-random controls without changing decode
   policy.

Metric-bearing inference must support batch-safe independent controller state
and exact RP1.10/RP1.00 config receipts.

The controller enforces `max_row_tokens=32`. Nested row starts, terminal before
box end, repeated box end, timeout, parser-invalid state, and max-generation
cutoff trigger an explicit fail-closed zero/reset receipt; no proposal pulse
may persist across the invalid boundary.

## Verification Matrix

### Unit/contract tests

- exact canonical/jitter/duplicate row-state bytes and spans;
- no historical or EOS atoms in row-state supervision;
- target digest unchanged and realized order preserved;
- deterministic condition/cohort manifests and cache-fingerprint sensitivity;
- IoM bag geometry, small boxes, zero-positive, edge cells, unequal grids;
- row target/causal interval off-by-one checks through `<|box_end|>`;
- two-image/two-segment isolation;
- proposal FP32 math, saturation, zero-positive, finite/nonfinite behavior;
- A/B/C parameter and optimizer-group receipts;
- C-off exact zero path and bitwise no-op logits/loss/gradient parity under
  FA2 with `FLASH_ATTENTION_DETERMINISTIC=1`, preceded by a bitwise native
  backward repeat; deterministic SDPA remains an additional companion check;
- checkpoint save/reload identity.

### VF gates

Execute VF0-VF5 in the order declared by the parent unit. Run VF6 only after
all pass. VF6 is a two-step A/B/C real smoke with checkpoint reload and one
C-on/C-off generation panel; it is wiring evidence only.

### Launch gate

Before the 512-step screen, obtain an independent innovation-risk audit over:

- live diff and resolved configs;
- real-runtime receipts;
- tiny loss/gradient and packed-isolation evidence;
- optimizer/checkpoint/train-infer parity;
- exact cohort/view manifests;
- rollback and artifact roots.

Only `LAUNCH` permits GPU training. `AMEND` returns to the affected phase.
`HOLD` records `contract_fail` if the same bounded handle cannot be made
trustworthy without changing the scientific question.

## Work Ownership

Implementation is split into non-overlapping lanes:

1. row-state rendering, view manifests, row intervals, and IoM bags;
2. functional Qwen proposal/bridge and real-runtime probes;
3. loss/optimizer/metrics/checkpoint ownership;
4. inference controller and evaluation artifact integration.

The primary agent owns schema integration, cross-lane semantics, tests, smoke,
launch decisions, result interpretation, and preservation of unrelated work.
