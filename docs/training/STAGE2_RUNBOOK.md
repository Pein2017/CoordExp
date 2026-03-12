---
doc_id: docs.training.stage2-runbook
layer: docs
doc_type: runbook
status: canonical
domain: training
summary: Operational Stage-2 workflow and YAML-first run guidance.
updated: 2026-03-09
---

# Stage-2 Training Runbook (Rollout-Aligned + Two-Channel)

This document is the consolidated runbook for Stage-2 training workflows in CoordExp:

- **Stage-2 Rollout-Aligned Teacher Forcing**: `custom.trainer_variant: stage2_rollout_aligned`
- **Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout)** (scheduler over channels A/B): `custom.trainer_variant: stage2_two_channel`

Stage-2 aims to align the model with its own decoded outputs while recovering missing GT objects.

---

## Quickstart Commands

From repo root:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--verbose]
```

Recommended wrapper (config-driven; handles `PYTHONPATH`, `torchrun`, proxy hygiene):

```bash
# Single GPU
config=<yaml> gpus=0 bash scripts/train.sh

# Multi-GPU
config=<yaml> gpus=0,1,2,3 bash scripts/train.sh
```

Multi-GPU:

```bash
PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>] [--verbose]
```

Multi-GPU + vLLM server mode (recommended topology for long rollouts): this is supported, but requires
the default Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-B step-budgeted pathway.

Requirements:
- `rollout_matching.rollout_backend: vllm`
- `rollout_matching.vllm.mode: server`
- Under multi-process learners (`torchrun`, `world_size > 1`), `rollout_matching.vllm.sync.mode` must resolve to `full`
  (DDP-safe rank0-only full-weight sync with strict barriers).

Where this lives in code:
- Stage-2 Rollout-Aligned Teacher Forcing trainer: `src/trainers/stage2_rollout_aligned.py`
- Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) trainer: `src/trainers/stage2_two_channel.py`
- Channel-B rollout parsing + matching helpers: `src/trainers/rollout_matching/parsing.py`, `src/trainers/rollout_matching/matching.py`
- Explicit dead-anchor suppression module: `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py`
- Training entrypoint (YAML loader + wiring): `src/sft.py`
- Import note: `src/trainers/stage2_two_channel/__init__.py` intentionally uses a proxy-style dynamic loader
  to preserve monkeypatch/import compatibility with the historical single-file module; avoid "simplifying"
  this wrapper without updating the associated tests.

---

## Core Objective (What Stage-2 Trains)

Stage-2 performs:

rollout (no grad) -> bounded container salvage + strict record acceptance -> teacher-forced target construction -> masked losses

For `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B assistant target is:

`Y_train = Y_anchor_edited_clean + SerializeAppend(FN_gt_objects) + EOS`

Key policies:
- Rollout parsing uses bounded container salvage plus strict record acceptance. Invalid predicted objects are dropped deterministically and never repaired into positives.
- Missing GT objects (FN) are always appended in the tail (recall recovery stays mandatory).
- Bbox geometry loss uses **SmoothL1 + CIoU** on expectation-decoded coords (no GIoU; boxes are canonicalized for CIoU stability).
- Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-B now rebuilds its positive prefix from an anchor-edited clean accepted sequence, not from raw rollout token ids.
- Text/structure CE is supervised with explicit masking/weights (Channel-A CE@A1; Channel-B uses clean-prefix CE plus dead-anchor duplicate UL).

---

## Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (Scheduler + Channels)

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) composes two channels:

- **Channel-A** (Expectation Loop): builds teacher-forced targets from GT (no rollouts), then runs packed SFT forward/backward.
- **Channel-B** (Rollout Matching): generates rollouts (no grad), parses + matches, builds `Y_train`, then runs packed SFT forward/backward.

Scheduler:
- Config: `stage2_ab.schedule.b_ratio: float` in `[0,1]` (0.0=A-only, 1.0=B-only, 0.05≈5% Channel-B).
- Runtime: deterministic Bresenham-style schedule from `TrainerState.global_step`:
  - Channel-B iff `floor((s+1)*b_ratio) > floor(s*b_ratio)`, else Channel-A.

### Channel-B Execution (Single Step-Budgeted Path)

Channel-B is standardized to a single step-budgeted pathway (legacy `micro/async` modes removed).

Key semantics:
- `training.effective_batch_size` is REQUIRED for `custom.trainer_variant: stage2_two_channel`.
- One optimizer step consumes exactly `effective_batch_size` raw rollout samples globally (across all learner ranks).
- Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) requires `effective_batch_size` to be divisible by `learner_world_size`, so each learner rank receives exactly `effective_batch_size / learner_world_size` raw samples.
- Raw rollouts are dynamically post-packed into a variable number of packed sequences (<= `global_max_length`), and the trainer runs
  multiple forward/backward passes inside the optimizer step (using `no_sync` for intermediate packs under DDP).

Rollout decode batching:
- Canonical knobs:
  - `rollout_matching.channel_b_decode_batch_size`: per-rollout-GPU cap for Channel-B train-time generation.
  - `rollout_matching.eval_decode_batch_size`: per-rollout-GPU cap for eval-step generation.
- HF + vLLM colocate:
  - Each learner rank chunks its local requests to the active context knob (`channel_b_decode_batch_size` for train, `eval_decode_batch_size` for eval).
- vLLM server mode:
  - The learner queries each rollout server DP `world_size` via `/get_world_size/` and derives a per-rank request chunk size so the
    per-GPU cap holds when all learner ranks generate concurrently.
  - Pipeline overlap (produce segments while consuming packs) is enabled in server mode only for single-process learners; under DDP it is disabled intentionally to prevent cross-rank sync deadlocks.

Worked example (default launcher):
- 6 rollout GPUs (server DP world size = 6), 2 learner GPUs (DDP world size = 2), `channel_b_decode_batch_size=4`:
  - per-rank chunk size = `floor(4 * 6 / 2) = 12` requests per call.
  - Across 2 ranks: 24 requests per synchronized round; average 4 per rollout GPU.

### Channel-A Contract (Expectation Loop)

- Default grad semantics: `stage2_ab.softctx_grad_mode: unroll` (no detach anywhere in the soft self-context loop). Use `em_detach` only for explicit ablations.
- CE anchor split: Channel-A computes CE on the **A1** teacher-forced logits and computes geometry (bbox loss + coord regularizers) from the **final** softctx iteration logits.

### Channel-B Contract (K=2 Triage + Dead-Anchor UL)

- Clean-prefix Channel-B is the only supported `stage2_two_channel` contract.
- This is the chosen v3 training contract for CoordExp Stage-2 Channel-B; it is not a claim that other mathematical formulations are invalid, only that this is the canonical repo contract.
- Canonical Channel-B flow:
  - anchor rollout (greedy) + explorer rollout (stochastic)
  - per-run bounded container salvage + strict record acceptance
  - per-run bbox-valid filtering
  - per-run sequential dedup
  - per-run `accepted_objects_clean` + Hungarian matching
  - deterministic one-to-one anchor/explorer association
  - triage into `anchor_gt_backed`, `recovered_fn`, `shielded_anchor`, `dead_anchor`, `dead_explorer`
  - anchor-edited clean-prefix teacher forcing + dead-anchor `loss_dead_anchor_suppression`
- Per-run sequential dedup is bbox-only in v1:
  - compare each candidate against previously accepted clean bbox objects only,
  - duplicate iff `normalize_desc(desc)` matches exactly and `IoU >= stage2_ab.channel_b.duplicate_iou_threshold`,
  - default `duplicate_iou_threshold: 0.90`.
- Matching / FN detection / matched geometry all run on per-run `accepted_objects_clean`, not on the raw duplicate-heavy parsed list.
- Positive teacher-forced prefix:
  - `Y_anchor_edited_clean` is the canonical assistant serialization of the kept anchor objects,
  - anchor order is preserved for retained objects,
  - `anchor_gt_backed` objects stay positive,
  - `shielded_anchor` objects stay in-prefix but neutral,
  - `dead_anchor` objects are removed from the positive prefix,
  - raw rollout token spans are diagnostic-only; they are not the positive-prefix source of truth.
- CE masking policy (clean-prefix Channel-B):
  - matched clean prefix objects: structure CE ON, desc CE OFF, coord CE OFF;
  - shielded anchor objects: structure/desc/coord CE all OFF (neutral context only);
  - FN-injected tail objects: structure CE ON, desc CE ON, coord CE OFF, with recovered GTs receiving higher per-object desc+geo+coord weight.
- FP-neutral geometry: Channel-B geometry loss includes matched clean prefix objects and FN-injected objects; shielded anchor objects contribute no geometry loss.
- Duplicate UL:
  - `loss_dead_anchor_suppression` is an explicit Channel-B-only objective module in `stage2_ab.pipeline.objective`,
  - `loss_dead_anchor_suppression.config` must be `{}` in v1 and the module `weight` is the only scaling surface,
  - dead anchor continuations are removed from the positive teacher-forced prefix and reintroduced only as boundary-local UL targets,
  - the bad token is the first true LCP-divergence token of the dead-anchor continuation relative to the canonical clean continuation,
  - same-boundary dead-anchor continuations sharing the same divergence token collapse to one UL term.
- Deterministic FN injection:
  - retain an append-ready canonical clean prefix inside `{"objects": [ ...`,
  - append unmatched GT records as extra `objects[]` elements,
  - insert a leading comma iff the retained clean prefix body already has object entries.
- Closure supervision: keep CE ON for the same outermost `}` used as FN injection anchor, and keep CE ON for `<|im_end|>` (no stop-neutral masking).
- Closure bookkeeping fallback: if explicit closure-marker bookkeeping becomes ambiguous after the clean target is built, keep the sample on the deterministic FN-tail supervision path and increment `stage2_ab/channel_b/closure_supervision/N_drop`.
- Strict-drop diagnostics: invalid predicted objects are dropped deterministically (no repair) but counted in metrics:
  - `stage2_ab/channel_b/strict_drop/N_valid_pred`
  - `stage2_ab/channel_b/strict_drop/N_drop_invalid`
  - `stage2_ab/channel_b/strict_drop/reason/<bucket>`
- Duplicate-collapse diagnostics are also emitted:
  - `dup/max_desc_count`
  - `dup/saturation_rate`
  - `dup/near_iou90_pairs_same_desc_count`
  - `dup/near_iou90_pairs_any_desc_count`
  - `stage2_ab/channel_b/dup/N_{raw_bbox_valid,clean_accepted,duplicates,duplicate_bursts,ul_boundaries,ul_skipped_no_divergence}`
- Triage diagnostics are also emitted:
  - `train/triage/{gt_backed_count,unlabeled_consistent_count,dead_anchor_count,explorer_only_dead_count,recovered_ground_truth_count}`
  - `train/triage/{recovered_ground_truth_rate_num,recovered_ground_truth_rate_den,dead_anchor_rate_num,dead_anchor_rate_den}`
- Stage-2 Channel-B config block:
  - `stage2_ab.channel_b.triage_posterior.explorer_temperature`
  - `stage2_ab.channel_b.triage_posterior.explorer_top_p`
  - `stage2_ab.channel_b.triage_posterior.explorer_top_k`
  - `stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold`
  - `stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier`
- Optional Channel-B runtime timeouts:
  - `stage2_ab.channel_b.ddp_phase_timeout_s` (seconds): Channel-B DDP phase-barrier watchdog (monitored barrier timeout). Must be `> 0` under DDP; `120` is the default/recommended fail-fast setting for faster error exposure.
  - `stage2_ab.channel_b.producer_wait_timeout_s` (seconds): rollout-producer queue wait timeout (`0` = auto).
- vLLM server request timeouts:
  - `rollout_matching.vllm.server.timeout_s` stays the finite connection/control-plane timeout.
  - `rollout_matching.vllm.server.infer_timeout_s` now defaults to that same finite timeout when unset.
  - Infinite `/infer/` waits require explicit opt-in via `rollout_matching.vllm.server.allow_infinite_infer_timeout: true`.

---

## Post-Rollout Packing (Important Gotchas)

Packing is supported post-rollout only:
- Enable with `training.packing: true`.
- Rollout generation remains un-packed (padded batch). The trainer temporarily disables padding-free / packing during rollouts.
- Stage-2 uses dynamic post-rollout packing inside the trainer (dataset-level packing wrappers are not used).
- Post-rollout packing here means **post-rollout, pre-forward**: the trainer builds per-sample teacher-forced segments (`Y_train`)
  first, then packs those segments into packed sequences under the `packing_length` cap before any forward/backward.
- Selection uses a deterministic constant-volume binpacking heuristic; the `binpacking` dependency is required when packing is enabled.
  - `stage2_rollout_aligned`: spec-compliant selection that never returns a pack shorter than the FIFO-greedy baseline for the same buffer state.
  - `stage2_two_channel` (step-budgeted): pool-aware selection that prioritizes minimizing the total number of packed sequences per optimizer step
    (fewer forward/backward calls), and secondarily avoids tiny remainder packs when feasible.
- Stage-2 uses micro-scope dynamic post-rollout packing only (window lookahead removed).
- `training.packing_drop_last: true` is required (the trainer does not run flush steps at the end).
  - `stage2_rollout_aligned`: carries leftover segments across optimizer steps (carry-only buffer).
  - `stage2_two_channel`: the per-step pool is fully consumed by contract (no cross-step carry), but `packing_drop_last` remains required for stable semantics.

The raw rollout prefix is no longer the positive teacher-forced prefix for `stage2_two_channel` Channel-B:
- raw rollout token space is still parsed/trimmed conservatively for diagnostics and strict acceptance,
- the positive Channel-B prefix is canonically reserialized from `accepted_objects_clean`.

---

## Rollout Parsing Policy (Current)

The current rollout behavior commonly includes a trailing `<|im_end|>` token and can sometimes truncate mid-object.

Parsing policy:
- Treat `<|im_end|>` as a hard stop (strip it, even when fused into the final token).
- If the rollout is truncated mid-object, suffix-trim to the last complete object boundary.
- Keep the prefix append-ready inside the top-level `objects` array (`{"objects": [` or `{"objects": [{...}`).
- Failure fallback:
  - If no valid `{"objects": [...]}` container exists, or no append-ready prefix can be produced, use `Y_rollout_prefix = "{\"objects\": ["`
    (empty predicted set; all GT are appended as FN).

---

## Matching Knobs (Starting Point)

Reasonable smoke defaults:
- `candidate_top_k: 5`
- `maskiou_gate: 0.3`

Interpretation:
- `candidate_top_k` prunes GT candidates per predicted object before expensive geometry.
- `maskiou_gate` rejects low-quality matches early; rejected GT remain FN and are appended.

---

## Config Checklist

Start from a template config and fill in dataset + rollout knobs:

- Stage-2 Rollout-Aligned Teacher Forcing base: `configs/stage2_rollout_aligned/base.yaml`
- Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) base: `configs/stage2_two_channel/base.yaml`
- Examples:
  - `configs/stage2_rollout_aligned/`
  - `configs/stage2_two_channel/`

Minimum required edits:
- Set `custom.train_jsonl` / `custom.val_jsonl`.
- Set `custom.offline_max_pixels` to the offline resize budget enforced by your prepared JSONLs.
  - Example: `786432` for the standard `32*32*768` pipeline.
  - Example: `1048576` for the `32*32*1024` pipeline.
- Set top-level `rollout_matching.*` (including `rollout_matching.decoding.*` + matching knobs).
- If using Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`), provide a top-level `stage2_ab` section (typed) including:
  - `stage2_ab.schedule.b_ratio`
- Stage-2 pipelines are **required** (no implicit defaults):
  - For `custom.trainer_variant: stage2_two_channel`, `stage2_ab.pipeline` MUST be present.
  - For `custom.trainer_variant: stage2_rollout_aligned`, `rollout_matching.pipeline` MUST be present.
- `template.max_pixels` and `custom.offline_max_pixels` now serve different purposes:
  - `template.max_pixels`: runtime processor/server setting (often kept large to disable HF auto-resize)
  - `custom.offline_max_pixels`: offline dataset contract enforced by launcher prechecks and dataset runtime
- For Stage-2 server-mode rollout configs, `rollout_matching.vllm.mm_processor_kwargs.do_resize: false` is required.
  - Treat this as a fail-fast invariant, not an optional tuning knob.
- Golden rule:
  - preprocessed images are the source of truth for both training and evaluation,
  - do not let the runtime vision processor resize them,
  - if a path cannot honor `do_resize=false`, that path is misconfigured.
- Set `training.packing: true` if you want post-rollout packing for the teacher-forced forward pass.

Objective pipeline declaration (required, ordered):
- `stage2_ab.pipeline.objective` declares loss-changing modules in execution order.
- `stage2_ab.pipeline.diagnostics` declares metrics-only modules in execution order.
- Canonical module names:
  - objective: `token_ce`, `loss_dead_anchor_suppression`, `bbox_geo`, `coord_reg`
  - diagnostics: `coord_diag`
- **Order matters**:
  - `stage2_two_channel`: `token_ce -> loss_dead_anchor_suppression -> bbox_geo -> coord_reg`
  - `stage2_rollout_aligned`: `bbox_geo` MUST run before `coord_reg` when both are enabled (coord_reg consumes bbox_geo state).
- Pipeline module specs are strict and explicit (no silent defaults):
  - each module spec MUST include `enabled`, `weight`, `channels`, `config`;
  - each module config MUST include exactly the allowlisted keys (missing/unknown fail fast).
- Flat objective knobs are intentionally removed; author all loss weights in `*.pipeline.<objective|diagnostics>[i].config`.

Optional Channel-A A1 anchor losses (ablation knobs):
- `bbox_geo.config.{a1_smoothl1_weight,a1_ciou_weight}` add bbox-geometry supervision on the A1 (GT-anchor) forward.
- `coord_reg.config.{a1_soft_ce_weight,a1_w1_weight}` add coord distribution supervision on the A1 (GT-anchor) forward (SoftCE/W1 only; no hard CE or gates in this path).
- When enabled (non-zero), these atoms are logged under `loss/A1_coord/*` and contribute to the total loss.

### Stage-2 Two-Channel examples (A-only / B-only / AB-mixed)

AB-mixed (recommended A-hot / B-cold production starting point):

```yaml
stage2_ab:
  schedule: {b_ratio: 0.05}
  n_softctx_iter: 2
  channel_b:
    duplicate_iou_threshold: 0.90
  pipeline:
    objective:
      - name: token_ce
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          desc_ce_weight: 1.0
          self_context_struct_ce_weight: 0.1
          rollout_fn_desc_weight: 1.0
          rollout_matched_prefix_struct_weight: 1.0
      - name: loss_dead_anchor_suppression
        enabled: true
        weight: 1.0
        channels: [B]
        config: {}
      - name: bbox_geo
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          smoothl1_weight: 2.0
          ciou_weight: 0.2
          a1_smoothl1_weight: 0.0
          a1_ciou_weight: 0.0
      - name: coord_reg
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          coord_ce_weight: 0.02
          coord_el1_weight: 0.0
          coord_ehuber_weight: 0.0
          coord_huber_delta: 0.001
          coord_entropy_weight: 0.0
          coord_gate_weight: 1.0
          text_gate_weight: 0.1
          soft_ce_weight: 0.1
          self_context_soft_ce_weight: 0.1
          w1_weight: 0.1
          a1_soft_ce_weight: 0.0
          a1_w1_weight: 0.0
          temperature: 1.0
          target_sigma: 2.0
          target_truncate: 8
    diagnostics:
      - name: coord_diag
        enabled: true
        weight: 1.0
        channels: [A, B]
        config: {}
```

A-only (disable rollouts; keep Channel-A expectation loop + self-context objectives):

```yaml
stage2_ab:
  schedule: {b_ratio: 0.0}
  n_softctx_iter: 2
  channel_b:
    duplicate_iou_threshold: 0.90
  pipeline:
    objective:
      - name: token_ce
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          desc_ce_weight: 1.0
          self_context_struct_ce_weight: 0.1
          rollout_fn_desc_weight: 1.0
          rollout_matched_prefix_struct_weight: 1.0
      - name: loss_dead_anchor_suppression
        enabled: true
        weight: 1.0
        channels: [B]
        config: {}
      - name: bbox_geo
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          smoothl1_weight: 2.0
          ciou_weight: 0.2
          a1_smoothl1_weight: 0.0
          a1_ciou_weight: 0.0
      - name: coord_reg
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          coord_ce_weight: 0.02
          coord_el1_weight: 0.0
          coord_ehuber_weight: 0.0
          coord_huber_delta: 0.001
          coord_entropy_weight: 0.0
          coord_gate_weight: 1.0
          text_gate_weight: 0.1
          soft_ce_weight: 0.1
          self_context_soft_ce_weight: 0.1
          w1_weight: 0.1
          a1_soft_ce_weight: 0.0
          a1_w1_weight: 0.0
          temperature: 1.0
          target_sigma: 2.0
          target_truncate: 8
    diagnostics:
      - name: coord_diag
        enabled: true
        weight: 1.0
        channels: [A, B]
        config: {}
```

B-only (always rollouts; skip Channel-A steps via scheduler):

```yaml
stage2_ab:
  schedule: {b_ratio: 1.0}
  n_softctx_iter: 2
  channel_b:
    duplicate_iou_threshold: 0.90
  pipeline:
    objective:
      - name: token_ce
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          desc_ce_weight: 1.0
          self_context_struct_ce_weight: 0.1
          rollout_fn_desc_weight: 1.0
          rollout_matched_prefix_struct_weight: 1.0
      - name: loss_dead_anchor_suppression
        enabled: true
        weight: 1.0
        channels: [B]
        config: {}
      - name: bbox_geo
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          smoothl1_weight: 2.0
          ciou_weight: 0.2
          a1_smoothl1_weight: 0.0
          a1_ciou_weight: 0.0
      - name: coord_reg
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          coord_ce_weight: 0.02
          coord_el1_weight: 0.0
          coord_ehuber_weight: 0.0
          coord_huber_delta: 0.001
          coord_entropy_weight: 0.0
          coord_gate_weight: 1.0
          text_gate_weight: 0.1
          soft_ce_weight: 0.1
          self_context_soft_ce_weight: 0.1
          w1_weight: 0.1
          a1_soft_ce_weight: 0.0
          a1_w1_weight: 0.0
          temperature: 1.0
          target_sigma: 2.0
          target_truncate: 8
    diagnostics:
      - name: coord_diag
        enabled: true
        weight: 1.0
        channels: [A, B]
        config: {}
```

### Rollout-aligned Stage-2 example (standalone rollout-matching SFT)

Rollout-aligned Stage-2 uses the same objective modules, but the pipeline lives under `rollout_matching.pipeline`:

```yaml
custom:
  trainer_variant: stage2_rollout_aligned

rollout_matching:
  rollout_backend: vllm
  channel_b_decode_batch_size: 4
  eval_decode_batch_size: 4
  pipeline:
    objective:
      - name: token_ce
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          desc_ce_weight: 1.0
          self_context_struct_ce_weight: 0.0
          rollout_fn_desc_weight: 1.0
          rollout_matched_prefix_struct_weight: 1.0
      - name: bbox_geo
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          smoothl1_weight: 1.0
          ciou_weight: 1.0
          a1_smoothl1_weight: 0.0
          a1_ciou_weight: 0.0
      - name: coord_reg
        enabled: true
        weight: 1.0
        channels: [A, B]
        config:
          coord_ce_weight: 0.0
          coord_el1_weight: 0.0
          coord_ehuber_weight: 0.0
          coord_huber_delta: 0.001
          coord_entropy_weight: 0.0
          coord_gate_weight: 1.0
          text_gate_weight: 0.0
          soft_ce_weight: 1.0
          self_context_soft_ce_weight: 0.0
          w1_weight: 1.0
          a1_soft_ce_weight: 0.0
          a1_w1_weight: 0.0
          temperature: 1.0
          target_sigma: 2.0
          target_truncate: 16
    diagnostics:
      - name: coord_diag
        enabled: true
        weight: 1.0
        channels: [A, B]
        config: {}
```

Geometry objective expectation:
- `bbox_geo` is the canonical bbox geometry objective in both trainer variants
  (`stage2_rollout_aligned` and `stage2_two_channel`).
- Geometry uses expectation/ST decode with bbox canonicalization and the
  `SmoothL1 + CIoU` composition.

Breaking config migrations (no backward compatibility):
- Rollout sampling knobs are configured under `rollout_matching.decoding.*`:
  - `rollout_matching.decoding.temperature`, `rollout_matching.decoding.top_p`, `rollout_matching.decoding.top_k`
- Rollout-matching settings must be authored under top-level `rollout_matching.*`:
  - `custom.extra.rollout_matching.*` is removed and MUST fail fast if present.
- Legacy keys are removed and MUST fail fast if present:
  - `rollout_matching.temperature`, `rollout_matching.top_p`, `rollout_matching.top_k`
  - `rollout_matching.rollout_buffer` (buffered reuse is removed; use vLLM server mode + derived chunking + context-specific decode batch sizes to scale throughput)
  - `stage2_ab.channel_b.reordered_gt_sft` (removed; unified Channel-B only)
  - `stage2_ab.channel_b.desc_ce_weight_matched` (removed; no matched-desc CE knob)
  - `stage2_ab.channel_b.semantic_desc_gate` (removed; no training-time semantic gating)

Logging tip:
- Stage-2 metrics are logged once per optimizer step (aggregated across gradient accumulation).
- If you reuse the same `training.run_name` and `training.logging_dir`, multiple `events.out.tfevents.*` files can accumulate.
  Prefer unique run names, or leave `training.logging_dir` unset (default unique per run).

---

## Colocate Offload (Peak Memory Relief During Rollouts)

Stage-2 uses a strict offload+drain handoff window for **eval_step vLLM colocate** to keep teardown safe and avoid allocator lifecycle issues.

Policy:
- User-configurable rollout offload (`rollout_matching.offload.*`) is **disabled/ignored** in this stack to avoid severe performance regressions and cross-rank stragglers in DDP.
- Offload is only performed in internal forced windows (currently: eval-time vLLM colocate handoff), not during train_step rollouts.

Notes:
- Offload is not supported with DeepSpeed/ZeRO in this trainer; if you need ZeRO, use vLLM `server` mode for eval or disable vLLM colocate eval.
- For sleep-mode guidance, see **Sleep-Mode Policy (colocate)** below.

---

## Rollout Backend Options

### Fixed Stage-2 pipeline (no legacy fallback)

Stage-2 supports configuring rollout backends separately for training and eval in YAML:

- `rollout_matching.rollout_backend`: training-time rollout backend (`hf` or `vllm`).
- `rollout_matching.eval_rollout_backend`: optional eval override (`null`/missing, `hf`, or `vllm`).
  - When `null`/missing, eval rollouts inherit `rollout_backend`.

Combined server-mode launcher contract (`scripts/train_stage2.sh`):
- `rollout_matching.rollout_backend: vllm`
- `rollout_matching.vllm.mode: server`

### Sleep-Mode Policy (colocate)

- Sleep mode is removed from Stage-2 runtime lifecycle.
- Do not set `rollout_matching.vllm.enable_sleep_mode`, `rollout_matching.vllm.reinit_each_eval`, or `rollout_matching.vllm.sleep_level`.
- If you need memory headroom, prefer vLLM `server` mode (process isolation) or reduce rollout lengths/throughput knobs.

### vLLM (colocate vs server)

- `colocate` (default): learner instantiates a local vLLM engine on the same GPU(s) as training for eval_step.
  - Requires `rollout_matching.vllm.max_model_len`.
  - Weight sync:
    - Supported (required in this stack): `rollout_matching.vllm.enable_lora: false`
      - The trainer merges adapters into full weights and loads merged full weights into vLLM on rollout steps.
      - Adapter-only sync (`enable_lora: true` / `add_lora`) is unsupported and will fail fast.

- `server`: learner connects to a pre-launched `swift rollout` server and generates eval_step rollouts on dedicated GPUs.
  - Supports multi-process learner (`torchrun`, `world_size > 1`).
  - Under `world_size > 1`, the trainer performs rank0-only weight sync with strict barriers and requires `rollout_matching.vllm.sync.mode: full`.
  - Connectivity is configured in YAML under `rollout_matching.vllm.server`.
  - Weight sync is configured under `rollout_matching.vllm.sync`:
    - Only `sync.mode: full` is supported in this stack (full merged-weight sync; robust for multimodal + DoRA).
    - Adapter-only sync options are unsupported and will fail fast.
  - Deploy-readiness gates (enforced by `scripts/train_stage2.sh`):
    - `rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`
    - `rollout_matching.vllm.server.servers[0].base_url` must be `http://<host>:<port>` (YAML-only; no runtime overrides)
    - `model.model` must point to a local model directory (avoid accidental Hub-ID resolution)
    - `server_gpus` and `train_gpus` must be disjoint device sets
    - no external repeat-terminate plugin is required

---

## Decoding Tips

- Start with deterministic non-beam decoding for the anchor policy: `rollout_matching.decode_mode: greedy`, `rollout_matching.decoding.temperature: 0.0`.
- For v3 Channel-B, keep the explorer policy under `stage2_ab.channel_b.triage_posterior` and start with `explorer_temperature: 0.7`.
- `rollout_matching.decode_mode` is a **beam vs non-beam selector** in Stage-2 configs; sampling is controlled by `rollout_matching.decoding.temperature/top_p/top_k`.
  - `rollout_matching.decode_mode: greedy` can still produce **sampling** rollouts when `rollout_matching.decoding.temperature > 0.0`.
  - Metrics tip: use `rollout/do_sample` plus the anchor/explorer rollout tags (`rollout/anchor_*`, `rollout/explorer_*`) to disambiguate policies, not `rollout/decode_non_beam_count`.
- vLLM rollout backends support non-beam dual-policy Channel-B requests (greedy anchor + stochastic explorer), but still reject `decode_mode=beam`; use `rollout_matching.rollout_backend: hf` if you need beam search.
- For long dense JSON generations, set a mild `repetition_penalty` (e.g. `1.05`) to reduce loop-y rollouts.
- Ensure `max_new_tokens` is large enough to avoid systematic truncation (dense detection outputs can be very long).

---

## Optional: Description Monitor (Metrics Only)

Stage-2 can optionally monitor `desc` quality on matched pairs. This does not affect training loss.

Enable under `rollout_matching.desc_monitor`:

```yaml
rollout_matching:
  desc_monitor:
    enabled: true
    # 'exact'|'semantic'|'both'
    mode: semantic
    every_steps: 20
    semantic_model: sentence-transformers/all-MiniLM-L6-v2
    semantic_threshold: 0.6
    semantic_device: cpu
    semantic_batch_size: 32
    max_pairs: 64
```

---

## GPU Topology (Server Mode)

Server mode (recommended for long rollouts) runs rollouts on dedicated GPUs and supports a multi-GPU learner:

- Constraint: `server_gpus` and `train_gpus` must be disjoint.
- vLLM parallelism constraint: `len(server_gpus) == server_dp * server_tp`.

Recommended starting points (when each GPU can fit the full model):

- 8 GPUs (balanced; **server data-parallel**, learner DDP):
  - `server_gpus=0,1,2,3 train_gpus=4,5,6,7`
  - Default config: `rollout_matching.vllm.tensor_parallel_size: 1` (so `server_dp=4`).
  - Why: maximizes rollout throughput while keeping learner throughput high.

- 4 GPUs (minimal server, multi-GPU learner):
  - `server_gpus=0 train_gpus=1,2,3`
  - Why: keeps rollouts on a dedicated GPU while preserving a multi-GPU learner.

If the model / long-context KV cache does **not** fit as a single replica, use tensor-parallel server mode:
- Set `rollout_matching.vllm.tensor_parallel_size` in YAML (e.g., `4`) and allocate matching `server_gpus` (e.g., `0,1,2,3`).
  - Derived: `server_dp = len(server_gpus) / tensor_parallel_size` (e.g., `1` sharded server engine).

Notes:
- Server and learner GPU sets must be disjoint.
- Server GPUs will be idle on steps that do not call the rollout backend (e.g., Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) Channel-A steps). This is expected.
  - In this idle state, vLLM may still reserve a large amount of VRAM for weights/KV cache, so `nvidia-smi` can show high memory usage with near-zero utilization.

---

## Launch (vLLM Server Mode)

Recommended launcher (starts server + learner, sets `NO_PROXY` for the rollout server host, waits for `/health/`, cleans up on exit):

```bash
bash scripts/train_stage2.sh \
  server_gpus=0,1,2,3 train_gpus=4,5,6,7 \
  config=configs/stage2_two_channel/prod/ab_mixed.yaml
```

Launcher knobs (runtime-only; YAML remains the source of truth for rollout hyperparameters):
- `server_gpus=<csv>` / `SERVER_GPUS=<csv>`: rollout server GPUs (required).
- `train_gpus=<csv>` / `TRAIN_GPUS=<csv>`: learner GPUs (required).
- `wait_timeout=<seconds>` / `WAIT_TIMEOUT=<seconds>`: server readiness timeout (default: 900).
- `wait_interval=<seconds>` / `WAIT_INTERVAL=<seconds>`: readiness poll interval (default: 2).

Image root contract (`ROOT_IMAGE_DIR`):
- The learner and rollout server must resolve image paths identically, otherwise multimodal tokens will not correspond to the same pixels the labels were built from.
- When using `scripts/train_stage2.sh`, `ROOT_IMAGE_DIR` is resolved from `custom.train_jsonl` (the JSONL parent directory) during preflight and exported to both the server and learner processes.
  - You normally do not need to set `ROOT_IMAGE_DIR` manually in this launcher flow.
- If you run `python -m src.sft --config ...` directly (no combined launcher), you must export `ROOT_IMAGE_DIR` yourself unless your JSONL uses absolute image paths.

Multi-server note:
- Multi-server rollout is unsupported in this stack.
  - `rollout_matching.vllm.server.servers` MUST contain exactly 1 entry.

Operational tip:
- Run the launcher inside `tmux` so a single `Ctrl-C` cleanly terminates both learner and server and frees GPU memory quickly.

---

## Evaluation (Production-Style)

Stage-2 evaluation runs a production-style pipeline on `custom.val_jsonl`:

rollout (no grad) -> strict parse -> Hungarian match -> report metrics

Important:
- Eval intentionally skips teacher-forced encoding and loss computation to keep eval fast and reflective of real rollout performance.
- As a result, Stage-2 eval does not report `eval_loss`.

Eval metrics include:
- `eval/detection/precision`, `eval/detection/recall`, `eval/detection/f1`
- Counters: `eval/detection/pred_objects`, `eval/detection/gt_objects_total`, `eval/detection/matched`, `eval/detection/fp_total`, `eval/detection/fn_total` (aliases: `eval/detection/fp`, `eval/detection/fn`)
- Parse health: `eval/parsing/parse_truncated_rate`, `eval/parsing/parse_dropped_invalid`, `eval/parsing/parse_dropped_ambiguous`
- Sample health: `eval/parsing/sample_valid_pred_rate`, `eval/detection/sample_any_match_rate`
- Geometry quality: `eval/detection/matched_maskiou_mean`
- COCO detection (when `rollout_matching.eval_detection.enabled: true`): `eval/detection/mAP`
  - Eval-step COCO summary keys are intentionally compact: `eval/detection/mAP` plus a small set of `eval/runtime/coco_counter_*` counters (no `eval/detection/bbox_*` or `eval/detection/segm_*` keys).
  - On COCO-eval failure, training/evaluation continues and `eval/detection/mAP` is set to `0.0`; status is surfaced via `eval/runtime/coco_eval_ok`.

Best-checkpoint selection:
- Prefer `training.metric_for_best_model: rollout/f1` and `training.greater_is_better: true`.

---

## Monitoring & Metrics

### Key Health Metrics (Most Load-Bearing)

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout):
- `stage2/channel_a` (1 on A steps)
- `stage2/channel_b` (1 on B steps)
- `stage2/raw_rollouts` (0 on A steps; >0 on B steps)

Rollout health (Stage-2 Rollout-Aligned Teacher Forcing):
- `rollout/parse_truncated_rate`
- `rollout/sample_valid_pred_rate`
- `rollout/f1`

Throughput (both variants; Channel-B steps only for two-channel):
- `time/rollout_generate_s`
- `rollout/gen_tokens_per_s`

Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) extras (Channel-B steps):
- `stage2_ab/channel_b/strict_drop/*` (strict-drop diagnostics; see Channel-B contract above)
- `stage2_ab/channel_b/closure_supervision/N_drop` (legacy-named counter for closure-resolution fallback activations; should stay near 0)
- `stage2/invalid_rollout` (alias: `stage2_ab/channel_b/invalid_rollout`; parser-level invalid rollout fallback count; should stay near 0)
- `stage2/drop_poly`, `stage2/drop_unknown`, `stage2/drop_bbox_invalid` (strict-drop breakdown; should stay near 0 in bbox-only rollouts)
- `rollout/backend_{hf,vllm}` and `rollout/decode_mode_{greedy,beam}` (backend/decode provenance tags)

### Qualitative Monitoring Dumps

`train_monitor_dump` and `eval_monitor_dump` use the shared dump writer under
`<training.output_dir>/monitor_dumps/` and write both `.json` and `.md` files
(rank 0 only).

Behavior depends on the execution path:

- `eval_step` uses `rollout_matching.eval_monitor_dump.every_evals`.
  `every_evals: N` means “dump every Nth evaluation window”, so dumps align to
  `N * training.eval_steps` in step-space.
- `stage2_two_channel` Channel-B `train_step` uses
  `rollout_matching.train_monitor_dump.every_channel_b_steps` when set, otherwise
  it falls back to `rollout_matching.train_monitor_dump.every_steps`. It does
  **not** dump every suspicious step by default. It first applies the train
  cadence, then buffers the current optimizer step, selects suspicious
  duplicate-heavy rollouts only, and writes the top `max_samples` candidates for
  that dumped step.
- `every_channel_b_steps: N` counts realized Channel-B rollout steps rather than
  raw `global_step`, so the dump cadence does not alias against the AB schedule
  or `stage2_ab.schedule.b_ratio`.
- Suspicious Channel-B train dumps keep the same top-level dump hierarchy as eval
  (`kind`, `global_step`, `epoch`, `time`, `meta`, `metrics`, `samples`) and each
  sample includes:
  - image handle / `image_id`
  - `gt`
  - `pred`
  - full rollout / prefix / train-target text
  - duplicate diagnostics (`duplicates`, duplicate bursts, near-IoU duplicate counts)
- For those suspicious Channel-B train dumps, text is written in full even if
  `max_text_chars` is configured. `max_text_chars` still applies to the periodic
  eval-side monitor dumps.

Example config:

```yaml
rollout_matching:
  train_monitor_dump:
    enabled: true
    every_channel_b_steps: 4
    max_events: 50
    max_samples: 1
    max_text_chars: 4000
    async_write: true
    max_pending_writes: 2
    min_free_gb: 2.0
  eval_monitor_dump:
    enabled: true
    # Dump every eval window; set 2 for every other eval, etc.
    every_evals: 1
    max_events: 50
    max_samples: 1
    max_text_chars: 4000
    async_write: true
    max_pending_writes: 2
    min_free_gb: 2.0
```

Legacy note:
- `rollout_matching.monitor_dump` is still accepted as a backward-compatible
  fallback, but new configs should use the split `train_monitor_dump` and
  `eval_monitor_dump` namespaces.

---

## Troubleshooting

### Stuck at First Rollout Step (Server Mode)

Symptom:
- `/health/` returns 200, but training hangs at 0% GPU util.

Interpretation:
- `/health/` validates the HTTP server, but `group_port` is a separate TCP port used to initialize a communicator for weight sync.
- If `group_port` is not listening/reachable, the learner can block before the first rollout step.

Checks:
1) Confirm HTTP health:
   - `curl --noproxy '*' -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health/`
2) Confirm `group_port` is open:
   - `conda run -n ms python - <<'PY'\nimport socket\nhost='127.0.0.1'; port=51216\ns=socket.socket(); s.settimeout(2)\ntry:\n    s.connect((host, port))\n    print('group_port connect: ok')\nexcept Exception as e:\n    print('group_port connect: failed', e)\nfinally:\n    s.close()\nPY`

Mitigations:
- Change `rollout_matching.vllm.server.servers[].group_port` to an unused port and restart server and learner.
- Ensure localhost connections are not routed through proxies (prefer the helper launcher, or unset proxies + set `NO_PROXY`).

### Hard Abort at Process Exit (Colocate + Sleep Mode)

Symptom:
- Process aborts during shutdown (often after eval windows).
- Logs include allocator signatures like `CUDAPluggableAllocator::raw_delete` or "Trying to free a pointer not allocated here".

Mitigations:
- Apply **Sleep-Mode Policy (colocate)** above.
- For memory relief, use vLLM `server` mode for process isolation (or reduce rollout lengths/throughput knobs).

### Channel-B never executes

Symptom:
- `stage2/channel_b` stays 0.0 across training (Channel-A runs instead).

Interpretation:
- In the standardized pathway, Channel-B execution is deterministic and schedule-driven. This typically indicates:
  - `stage2_ab.schedule.b_ratio` is 0.0 (A-only), or
  - you are not actually running `custom.trainer_variant: stage2_two_channel`.

Checks:
1) Confirm schedule:
   - `stage2_ab.schedule.b_ratio` (1.0 for B-only; 0.0 for A-only).
2) Confirm trainer variant:
   - `custom.trainer_variant: stage2_two_channel`

Mitigations (smoke runs):
- Reduce rollout length (`rollout_matching.max_new_tokens`) for faster steps.
- Reduce rollout decode batching by setting `rollout_matching.channel_b_decode_batch_size: 1` and, if eval is involved, `rollout_matching.eval_decode_batch_size: 1`.
- Lower `rollout_matching.vllm.gpu_memory_utilization` or increase server GPUs if the server is capacity-bound.

### Length Constraints ("Long rollout" failures)

Two separate limits interact:
- `rollout_matching.max_new_tokens`: rollout generation budget.
- `global_max_length`: model/server max length (also passed as vLLM `--vllm_max_model_len` in server mode).

Rule of thumb:
- Ensure `global_max_length >= prompt_len + max_new_tokens`.
- If `max_new_tokens` is ~3k, `global_max_length: 2048` is too small; use `global_max_length: 4096` (or higher).

---

## Preflight Validation (Suggested)

- Unit tests (Stage-2):
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_offload_context.py`

---

## Rollout Backend Benchmarking (HF vs vLLM)

If you want to benchmark rollout generation throughput (analysis only; not an official launch script):

- Runner: `scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py`
- Example config: `configs/bench/rollout_backend_bench.yaml`

General expectations:
- vLLM is typically substantially faster for long rollouts but can reserve a large KV cache.
- `gpu_memory_utilization` primarily controls how much VRAM vLLM can reserve (mostly KV cache); lowering it can reduce peak VRAM.

---

## See Also

- **Metrics Guide**: [`METRICS.md`](METRICS.md)
- **Packing Guide**: [`../data/PACKING.md`](../data/PACKING.md)
- **Stage-1 SFT**: [`../data/README.md`](../data/README.md)
