---
doc_id: docs.training.permanent-owner-bridge-stage1
layer: docs
doc_type: operator-runbook
status: canonical
domain: training
summary: Architecture and guarded operator route for the optional permanent-owner-bridge Stage1 composition.
tags: [training, stage1, owner-bridge, qwen3-vl, operator]
updated: 2026-08-09
---

# Permanent Owner Bridge Stage1

This is the current operator note for the optional
`qwen3_vl_2b_owner_bridge_v1` composition. It describes mechanical Stage1
behavior and the guarded launch path; it is not evidence that production is
ready, that a launch occurred, or that model quality improved.

## Architecture and supervision

The profile keeps the native Qwen3-VL-2B image path, residual path, object-row
opener, and stop decision. After zero-based block 20 it derives four static,
locally anchored owner atoms per merged visual carrier. Each shared-latent atom
has objectness, a complete box, a normalized 128-wide key, and a normalized
512-wide value. Stage1 assigns labeled owners once per image with a global
injective Hungarian match; it never creates a persistent covered-set ledger or
a hard without-replacement mask.

At each boundary, the post-final-RMSNorm query routes over atoms plus a learned
null. Aggregate atom-versus-null availability supplies only a capped 3% RMS
soft admission residual, so native decoding still chooses whether to open a row
or stop. Teacher forcing latches the current gold owner's matched value;
dynamic HF decoding latches the highest-scoring non-null atom after the native
opener. The value is written after block 20 from opener through box end,
inclusive, with a zero-initialized output and a cap ramping to 10% RMS. Blocks
21-27 form owner-conditioned row KV state before the next final-normalized
boundary read. The latch is row- and sequence-local and clears after the
row-closing boundary is evaluated.

Stage1 is teacher-forced only and is one continuous optimizer/scheduler run over
four complete presentations:

1. authored `geo_sorted`;
2. deterministic `random-1`;
3. the same validated `geo_sorted` cache;
4. deterministic, distinct `random-2`.

The production source is `ordinary_partial`. Every unmatched atom is therefore
unknown-neutral: it is neither an objectness positive nor negative and is
excluded from route normalization. Final-null route supervision is also masked.
The native terminal-token autoregressive CE remains grammar supervision, not
proof that unmatched visual content is background. Trusted-exhaustive branches
exist as mechanically tested behavior but receive no examples in this first
production leaf.

The resolved loss is

```text
L = L_AR + L_atom + 0.5 * L_route + 0.1 * L_use
```

- `L_AR` is full-wrapper, token-normalized teacher-forced CE.
- `L_atom` supervises matched objectness and complete boxes with
  `5 * L1 + 2 * (1 - GIoU)` geometry terms.
- `L_route` rewards normalized mass on the known-uncovered matched-owner set,
  without prescribing which uncovered owner must be next.
- `L_use` is the balanced same-image four-branch correct-versus-swapped owner
  comparison over separate description and geometry likelihoods.

`L_route` reaches its configured weight over the first 5% and `L_use` over the
first 20% of the resolved planned-step budget; only completed safe optimizer
updates advance those ramps. The bridge, existing language DoRA, and selected
token embeddings train jointly under fresh optimizer state; the vision tower
and merger remain frozen.

## Exact configs

- Production:
  `configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1.yaml`
- Bounded eight-rank smoke:
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_train256_val64_4epoch_warmup0p1.yaml`
- Smallest single-rank smoke:
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_single_gpu_ebs24_train1_val1_4epoch_warmup0p1.yaml`

The production leaf binds the step-2444 S-lineage adapter and selected-token
payloads, original COCO-80 train/eval JSONL, no-resize 1024 limits,
`global_max_length=12000`, global EBS 24, and `ordinary_partial`. Smoke configs
are mechanical fixtures and cannot authorize or substitute for production.

## Prepare caches

Run from the repository root. Choose one explicit absolute cache root and use
that same root for preparation and the guard:

```bash
OWNER_BRIDGE_CONFIG=configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1.yaml
OWNER_BRIDGE_CACHE_ROOT="$PWD/.cache/coordexp_swift/packing"

COORDEXP_SWIFT_PACK_CACHE_ROOT="$OWNER_BRIDGE_CACHE_ROOT" \
  conda run -n ms python -m src.prepare_train_cache \
  --config "$OWNER_BRIDGE_CONFIG"
```

This single-process step loads tokenizer/processor components without loading
the model and builds or validates four semantic caches: three train identities
(`geo_sorted`, `random-1`, `random-2`) plus one fixed `geo_sorted` eval cache.
The sorted train cache is consumed twice. Cache identity includes trust,
presentation/order, owner records, bridge profile, renderer/tokenizer/processor,
and forward semantics; incompatible older caches are rebuilt, not upgraded.

## Guarded production launch

The default command is a non-activating dry preflight:

```bash
conda run -n ms python -m scripts.coordexp_swift.launch_owner_bridge_stage1 \
  --config "$OWNER_BRIDGE_CONFIG" \
  --cache-root "$OWNER_BRIDGE_CACHE_ROOT"
```

It revalidates the exact config and source lineage, requires all four caches to
be existing nonempty hits under that cache root, resolves the four-presentation
schedule and eight midpoint/end teacher-forced eval landmarks, checks repository
identity and eight-GPU headroom, and writes a preflight receipt. It does not
build caches, reserve the production claim, or launch training.

Only after current mechanical readiness and launch authorization are established,
activate the same inspected intent once:

```bash
conda run -n ms python -m scripts.coordexp_swift.launch_owner_bridge_stage1 \
  --config "$OWNER_BRIDGE_CONFIG" \
  --cache-root "$OWNER_BRIDGE_CACHE_ROOT" \
  --execute
```

The ledger root is fixed at
`/data/CoordExp/outputs/prod/coordexp_swift/.owner_bridge_stage1_launch_ledger`;
the CLI does not let an operator redirect it. Execute exclusively publishes the
host-global `stage1-production-launch-claim.json` before activation, then binds
one eight-rank Accelerate process to its nonce and exact Linux process lineage.
Before constructing Accelerator, every worker validates the launcher boot/PID
identity, claims one immutable rank slot, and waits for the complete rank-0
through rank-7 quorum. A naked process replaying the persisted nonce is outside
that lineage and fails closed. The singleton is shared across worktrees and run
directories, not scoped to one shell invocation. A claim survives an activation
or binding failure, and a later execute is rejected.
Treat an uncertain execute as at-most-once: inspect the ledger, logs, process,
and run binding; do not delete the claim or blindly retry.

The original production claim was consumed by an attested pre-run
infrastructure failure. It remains immutable. Following the user's explicit
one-time recovery authorization, the only permitted successor path is:

```bash
conda run -n ms python -m scripts.coordexp_swift.launch_owner_bridge_stage1 \
  --config "$OWNER_BRIDGE_CONFIG" \
  --cache-root "$OWNER_BRIDGE_CACHE_ROOT" \
  --recovery-attempt 1
```

This is still a non-activating dry preflight. After the repaired tree is
committed, its fixed-target lifecycle audit passes, and a fresh preflight is
current, the lead-only executor may add `--execute` exactly once. The guard
then reserves the fixed
`stage1-production-recovery-attempt-1-claim.json` with durable no-replace
creation before deriving its fresh nonce or payload. The successor is bound to
the fixed hashes of the original claim, intent, activation, preflight, eight
worker admissions, failed binding, and logs. Workers select exactly one claim
by nonce and seal the selected path and raw SHA-256 into admission and pre-model
receipts before distributed initialization.

Any empty, partial, malformed, failed, or uncertain attempt-1 successor is
terminally consumed. Preserve all evidence and stop. There is no attempt 2, no
third activation, and no authorized deletion, rotation, overwrite, alternate
ledger root, or reuse of either claim.

## Checkpoints, inference, and evidence boundaries

A bridge checkpoint is inference-complete only when its concrete `step-*`
directory atomically contains `adapter/`, `special_token_embeddings/`, and
`owner_bridge/` with mutually validated manifests and payload identities.
`checkpoints/final.json` is a training convenience pointer. Checkpoints do not
contain base weights and do not promise exact optimizer, scheduler, RNG,
dataloader, or presentation resume.

For a completed run, materialize one canonical inference leaf from the concrete
final composition:

```bash
conda run -n ms python -m scripts.coordexp_swift.materialize_owner_bridge_infer_config \
  --training-run-dir /absolute/path/to/completed-run \
  --input-jsonl /absolute/path/to/inference-input.jsonl \
  --artifact-root /absolute/path/to/inference-artifacts \
  --name owner-bridge-stage1-final \
  --mode production
```

The materializer validates the completed `run.json`, resolved training identity,
final checkpoint event, all three learned payloads, and bridge lineage. It then
publishes a no-replace YAML plus receipt under
`configs/coordexp_swift/infer/materialized_owner_bridge/`, with explicit concrete
payload paths. Decode that generated leaf through dynamic HF:

```bash
conda run -n ms python -m src.infer \
  --config configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-final.yaml
```

Bridge inference is dynamic-HF-only in this stage; bridge-plus-vLLM fails before
model loading. The training run's `resolved_config.json`, `run.json`,
`logging.jsonl`, `owner_bridge_events/`, `owner_bridge_checkpoint_events/`, and
atomic checkpoint payloads are execution evidence. The eight scheduled
in-training evaluations are fixed-cache teacher-forced forward/loss evidence,
not natural decode evidence. A dry preflight is not launch evidence; a run
binding is not a completed safe step; a smoke is not a quality result; and an
inference materialization receipt is not a recall/duplication conclusion.

Production readiness remains conditional on the current targeted suites,
strict OpenSpec validation, algorithm-reference checks, single- and eight-rank
mechanical smokes, and fixed-tree audits having no unresolved P0/P1. The first
natural greedy HF evaluation is a post-checkpoint observation and has no
favorable-quality threshold for saving the Stage1 checkpoint. Stage2
self-prefix/rollout learning, model-quality interpretation, vLLM bridge support,
and any promotion decision remain pending separate evidence and authority.
