# Add Stage-1 Set-Continuation Training

## Why

Stage-1 detection SFT currently trains a single serialized object order as though the annotated next object were the only correct continuation. That is a poor fit for object detection as set prediction. When several ground-truth objects remain after a prefix, ordinary fixed-order SFT rewards one object and implicitly treats the other observed objects as negatives.

Recent Stage-1 diagnostics point to symptoms that this objective can plausibly amplify:

- prefix sensitivity and poor prefix basins;
- early coordinate-vertex conditioning failures around object starts;
- duplicate generation and same-class competition;
- premature global list closure or stop behavior;
- fixed-order dependence that is orthogonal to image evidence.

Sparse or incomplete labels make the stop problem more delicate. Observed ground truth should provide positive evidence, not necessarily complete evidence that the image contains no other objects. Stage-1 needs an experiment surface that can test whether subset-conditioned full-entry multi-positive supervision improves continuation behavior without changing the lm-head-only grounding paradigm.

The source research direction is tracked at `progress/directions/full_idea_v5.md`.

## What Changes

Add an off-by-default Stage-1 trainer variant for subset-conditioned set-continuation training:

- Introduce `custom.trainer_variant: stage1_set_continuation`.
- Introduce a strict config block, tentatively `custom.stage1_set_continuation`.
- Introduce a typed top-level `benchmark` config block for A-F group identity
  and comparator metadata.
- Sample an already-emitted object subset `S` per training example.
- Score remaining observed objects `R = O - S` as full serialized object entries.
- Optimize the full-entry multi-positive loss:

  ```text
  score(o) = log P(entry(o) | image, prompt, prefix)
  loss/mp = -logsumexp(score(o) for o in candidates)
  ```

- Keep object-entry end tokens supervised as part of `entry(o)`.
- Treat global detection-list structural closure separately from object-entry end tokens.
- Add optional close-start suppression when observed remaining objects exist.
- Add weak or masked structural-close supervision when no observed GT remains.
- Support fixed-rho positive-evidence margin (PEM) in v1, configurable and disabled by default.
- Support exact all-remaining candidate scoring and configurable candidate subsampling.
- Implement the first version with repeated independent branch semantics, but
  refine the train-step forward runtime so production can trade more compute
  time for lower peak memory pressure.
- Add a config-first train-forward runtime policy with:
  - a legacy retained-graph mode for parity/debugging;
  - exact supervised-suffix logits so branch forwards do not materialize
    full-prefix logits that are never supervised;
  - an explicit DDP candidate-padding policy so production can avoid
    zero-loss padding forwards while retaining the max-count policy as rollback;
    no-padding DDP requires `training.ddp_broadcast_buffers=false`;
  - a checkpoint/recompute exact mode as an optional memory fallback, not the
    immediate production default;
  - an automatic approximate fallback that uses explicit uniform subsampling
    when candidate/token/memory budgets would be exceeded;
  - disabled prefix reuse in the immediate bridge; CPU/tokenization prefix reuse,
    GPU KV prefix caching, and branch attention masks remain follow-on runtime
    modes.
- Preserve existing verified behavior when `train_forward` is omitted:
  retained-graph candidate scoring, fallback disabled, prefix encoding cache
  disabled, GPU KV cache disabled, and branch attention masks disabled.
- Record exact versus approximate objective fidelity, fallback reason,
  branch-token budgets, and prefix-reuse mode in metrics and runtime artifacts.
- Support only coord-token object coordinates (`<|coord_*|>`) in this trainer variant.
- Reject raw-text integer coordinate training for this paradigm in v1.
- Fail fast for dataset packing, including train/eval static packing, in the first version.
- Keep coord and geometry auxiliary losses toggleable only through branch-local set-continuation adapters; do not reuse ordinary one-sequence SFT mixins blindly.

The first implementation prioritizes correctness, branch-isolation clarity,
metrics, and benchmark comparability. The train-forward refinement keeps those
semantics but makes the production path reduce wasted branch work: exact MP/PEM
scores can request only the supervised suffix logits, DDP can skip candidate
padding forwards, and over-budget samples can use an explicit cap-8
uniform-importance fallback. Any approximate fallback must be visible in config,
logs, and artifacts. Prefix KV/cache reuse is deliberately out of scope for this
bridge.

## Capabilities

This change introduces a new capability:

- `stage1-set-continuation-training`

This change also modifies existing capabilities:

- `packing-dataset`: carve out a v1 fail-fast exception for this Stage-1 trainer variant.
- `coord-token-mode`: define repeated-forward coord-token scoring for set-continuation instead of ordinary single-forward SFT composition.
- `coord-aux-loss`: define branch-local `coord_soft_ce_w1` adapter semantics.
- `bbox-size-aux-loss`: define branch-local bbox geometry/size adapter semantics.
- `trainer-metrics-components`: classify new MP, structural-close, candidate, budget, and aux metrics.
- `encoded-training-cache`: define cache bypass/eligibility for trainer-side branch construction.
- `stage1-set-continuation-training`: refine the train-step forward runtime
  contract for branch checkpointing, budgeted approximate fallback, prefix
  encoding reuse, objective-fidelity labels, and future prefix-cache/branch-mask
  compatibility.

It will also require updates to existing documentation and metric surfaces:

- Stage-1 objective documentation;
- Stage-1 metrics documentation;
- training config strictness tests;
- static packing documentation and validation;
- encoded-cache eligibility documentation and validation;
- benchmark profile documentation.

## Impact

Expected code touch points:

- `src/config/schema.py` for strict config dataclasses.
- `src/sft.py` and trainer setup code for variant routing.
- A new Stage-1 set-continuation trainer module.
- A candidate fragment serializer that preserves the current Qwen3-VL JSON/chat-template contract.
- A subset sampler and candidate sampler.
- A raw-sample-preserving collator path that exposes `assistant_payload.objects`, messages/image identity, sample ids, and metadata to the trainer.
- A branch encoder that owns template state, multimodal/image alignment, branch tokenization, label masks, and structural-close spans.
- A train-forward runtime layer that owns candidate budget decisions, branch
  execution mode, exact/approx objective-fidelity labels, and prefix encoding
  reuse.
- Branch-local auxiliary loss adapters for compatible coord-token losses.
- Metrics logging for MP, PEM, structural-close, candidate, auxiliary-loss, and budget diagnostics.
- Benchmark configs for ordinary SFT, weak structural-close SFT, MP, MP plus close-start suppression, fixed-rho PEM threshold loss, and leave-one-out emphasis.
- Focused tests for serialization, sampler behavior, loss math, stop-token handling, packing rejection, and metric key parity.

Correctness and reproducibility impact:

- This changes Stage-1 loss semantics and therefore must be config-first and OpenSpec-governed.
- Supervised-suffix logits are exact for the existing MP/PEM and structural-close
  labels because they crop only unsupervised prefix logits. No-padding DDP is a
  runtime synchronization policy. Automatic fallback to subsampled MP changes
  estimator fidelity and must never be reported as exact.
- It does not change the offline JSONL data contract.
- It does not change image geometry, resizing, bbox surfaces, or inference artifact contracts.
- It must record enough resolved config, effective runtime, experiment manifest, benchmark identity, and metric evidence to distinguish ordinary SFT, weak structural-close SFT, and MP variants.

Evaluation-validity impact:

- MP runs may predict more objects and may increase apparent sparse-label false positives.
- Benchmark reports must state scope exactly, for example `val200`, `limit=200`, proxy, or full-val.
- Same-budget comparisons should log the number of prefix tokens, candidate tokens, candidates scored, and repeated-forward budget ratios.
