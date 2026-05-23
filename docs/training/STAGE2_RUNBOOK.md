---
doc_id: docs.training.stage2-runbook
layer: docs
doc_type: runbook
status: canonical
domain: training
summary: YAML-first runbook for active Stage-2 training, including direct learner runs and vLLM server-mode launches.
updated: 2026-05-16
---

# Stage-2 Training Runbook

Use this page for the active Stage-2 path.

The current contract is:

- `custom.trainer_variant: stage2_two_channel`
- Channel-A runs one GT-anchored teacher-forced forward
- Channel-B keeps the rollout-aligned clean-prefix supervision path
- Stage-2 remains YAML-first; no new CLI flags are required

The older rollout-matching public trainer variants have been removed.
`src/trainers/stage2_rollout_runtime.py` remains as an internal runtime base for
rollout prompt preparation, HF/vLLM/server dispatch, eval artifacts, and
post-rollout packing.

## Normative References

- [`openspec/specs/stage2-ab-training/spec.md`](../../openspec/specs/stage2-ab-training/spec.md)
- [`openspec/specs/rollout-matching-sft/spec.md`](../../openspec/specs/rollout-matching-sft/spec.md)
- [`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`](../../openspec/specs/teacher-forcing-unified-loss-registry/spec.md)
- [`openspec/specs/trainer-metrics-components/spec.md`](../../openspec/specs/trainer-metrics-components/spec.md)
- [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md)

## Runtime Ownership

Stable public entrypoints:

- direct training entrypoint:
  - `src/sft.py`
- operator-facing server-mode wrapper:
  - `scripts/train_stage2.sh`

Current internal ownership seams:

- bootstrap/provenance:
  - `src/bootstrap/pipeline_manifest.py`
  - `src/bootstrap/trainer_setup.py`
  - `src/bootstrap/run_metadata.py`
- Stage-2 trainer/runtime:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/`
  - `src/trainers/stage2_rollout_runtime.py`
  - `src/trainers/rollout_aligned_targets.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/trainers/rollout_runtime/`
- server-mode orchestration:
  - `src/launchers/stage2_vllm_server.py`

## Current Supported Contract

- `custom.trainer_variant: stage2_two_channel`
- shadow architecture `surface.id: stage2_two_channel`
- shadow config domains follow the unified contract:
  `run`, `surface`, `data`, `template`, `supervision`, `objectives`,
  `observability`, `artifacts`, and `runtime`
- `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]` are required for active Stage-2 configs
- Channel-A runs a single GT-anchored teacher-forced forward.
- Channel-B clean-prefix behavior is scoped to `token_ce` / `hard_sft`
  baselines. Residual-state trie configs bypass the clean-prefix target path.
- Channel-B final object sequencing is controlled by:
  - `stage2_ab.channel_b.insertion_order: tail_append | sorted`
  - default `tail_append` preserves the historical clean-prefix plus FN-tail path
  - `sorted` applies a final top-left sort over retained current-attempt objects plus FN objects before final teacher-forced serialization, and compact-full FN descriptions remain explicitly tagged for `rollout_fn_desc_weight`
- Channel-B duplicate control is configured only through:
  - `stage2_ab.channel_b.duplicate_control.iou_threshold`
  - `stage2_ab.channel_b.duplicate_control.center_radius_scale`
- Channel-B pseudo-positive mode is opt-in through:
  - `stage2_ab.channel_b.pseudo_positive.enabled`
  - `stage2_ab.channel_b.pseudo_positive.coord_weight`
- `rollout_matching.pipeline.*` is retired; active objective ownership is only
  through `stage2_ab.pipeline.*`
- supported routing/objective presets are:
  - `token_ce.application.preset: anchor_text_only`
  - `stage2_trie_ce.application.preset: rollout_trie_hard_ce`
  - `residual_set_correction.application.preset: rollout_self_prefix`
  - `hard_sft.application.preset: hard_sft`
- `stage2_trie_ce` and `residual_set_correction` are aliases for the same
  residual-state trie semantics. Both consume live Channel-B rollout attempts
  and dynamic residual valid sets; neither uses privileged-rollout
  multiple-positive supervision.
- residual-set smoke handles live under
  `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_*.yaml`;
  they are tiny runnable checks for the correction-event path, not full validation or
  production-quality evidence.
- `stage2_trie_ce` / `residual_set_correction` is an online-learning
  Channel-B objective:
  - Channel-B always generates live rollout attempts from the current batch;
    offline prepared-rollout JSONL inputs are no longer a supported code path.
  - default UL promotion is strict 4-of-4 consensus:
    `expected_num_rollouts: 4`, `ul_consensus_ratio: 1.0`,
    `min_ul_valid_rollouts: 4`
  - loss config consumes `lambda_type`, `lambda_inner`, and residual-set
    runtime construction knobs such as `expected_num_rollouts`, `base_seed`,
    UL thresholds, and `strict_builder_invariants`.
- residual-set loss and telemetry are reported under
  `stage2_ab/channel_b/residual_set/`; key compact metrics include
  `sequence_count`, `atom_count`, `atom_weight_sum`, `sequence_loss`,
  `type_loss`, `inner_loss`, `wrong_type_mass`, `valid_set_mass`,
  dirty-prefix/source counters, STOP/continue counts, and decode-mode slices.
- removed geometry/coordinate modules:
  - `bbox_geo`, `bbox_size_aux`, `coord_reg`, and `coord_diag` are rejected by the active Stage-2 pipeline
- duplicate-burst UL migration state:
  - the live objective module `loss_duplicate_burst_unlikelihood` is removed and current Stage-2 configs must not declare it
  - historical specs and archived run artifacts may still mention the retired objective name for compatibility notes
  - duplicate-control diagnostics and counters remain supported after objective cleanup
- retired adjacent-repulsion state:
  - adjacent-repulsion anti-copy loss and config knobs are no longer live training support
  - current configs must omit `adjacent_repulsion_*` keys; strict config parsing rejects them as unknown
  - duplicate-control diagnostics remain supported and are separate from the retired loss
- Legacy pseudo-positive clean-prefix mode keeps the one-forward contract and is
  mutually exclusive with residual-state trie objectives:
  - retained prefix objects share one global prefix structure CE surface through `token_ce.config.rollout_global_prefix_struct_ce_weight`
  - `matched_clean` -> coord + global prefix structure CE
  - `fn_injection` -> coord + FN desc CE
  - selected `pseudo_positive` current-attempt objects -> coord + global prefix structure CE
  - support-positive retained shield-only objects that stay below promotion threshold -> global prefix structure CE only; no desc, bbox, or coord positive supervision
  - cluster-demoted pseudo-positive candidates -> global prefix structure CE only
  - duplicate control runs before GT matching and target realization on the assembled current-attempt plus peer-attempt evidence surface
  - non-exempt duplicate-control non-survivors are removed from the clean
    prefix and tracked only through duplicate-control diagnostics
  - dead current-attempt object -> no positive supervision, with duplicate-control suppression only
  - pseudo-positive selection is K-of-K peer consensus: an unlabeled object can
    be promoted only when all rollout attempts point to the same unlabeled
    region and normalized description
- Default authored pseudo-positive profile:
  - `triage_posterior.num_rollouts: 4`
  - optional `triage_posterior.rollout_temperatures` supplies one broadcast
    temperature or exactly K ordinal temperatures; omitted values broadcast
    `triage_posterior.explorer_temperature` to every ordinal
  - enabled `K<4` is rejected because pseudo-positive promotion requires full
    4-of-4 consensus in the default profile
- Enabled failure semantics:
  - `stage2_ab.channel_b.rollout_template_family: coordjson` is the explicit
    legacy rollout surface. It uses the legacy CoordJSON parser and records
    rollout template/decode/parser provenance in Channel-B batch metrics.
  - `stage2_ab.channel_b.rollout_template_family: compact_full` is the new
    canonical target surface for compact-full checkpoints. It uses the
    compact-full parser and compact-full Channel-B target renderer, with default
    `rollout_decode_policy: unconstrained`. It must not flow through the legacy
    CoordJSON parser or CoordJSON FN appender.
  - compact-full invalid, malformed, or empty-valid-object rollout policy
    resolves to `fallback_gt_fn_append_only` with `fallback_loss_weight: 1.0`.
    The fallback constructs a compact-full GT/FN-only correction target, records
    `rollout_context: fallback_gt_fn_append_only`, and does not count as a valid
    rollout for readiness gates. Parser/template mismatches remain hard
    failures, not fallback cases.
  - compact-full peer attempts that enter fallback remain visible in raw
    rollout/fallback metrics. They do not contribute positive support, while
    pseudo-positive and recovered-GT support denominators remain the configured
    peer-attempt count (`num_rollouts - 1`) so missing/fallback peers cannot
    silently relax consensus.
  - compact-full Channel-B targets do not use CoordJSON tail-closure or
    semantic-stop supervision. Stop/closure metrics should be interpreted as
    CoordJSON-specific unless explicitly documented otherwise.
  - malformed current-attempt preparation drops that attempt/sample from Channel-B training
  - malformed peer attempts that remain invalid after salvage parsing abort the step by default only when pseudo-positive mode is enabled
  - outside pseudo-positive mode, malformed rollouts fall back to the existing empty-prefix / FN-only handling instead of taking the invalid-rollout abort path
  - `stage2_ab.channel_b.invalid_rollout_policy: dump_and_continue` dumps and skips the offending pseudo-positive sample instead
  - zero-object peer attempts remain valid zero-support evidence
- deprecated authored knobs fail fast in active/training configs:
  - `custom.trainer_variant: rollout_matching_sft`
  - `custom.trainer_variant: stage2_rollout_aligned`
  - `custom.trainer_variant: stage2_rollout_runtime`
  - `stage2_ab.n_softctx_iter`
  - `stage2_ab.softctx_grad_mode`
  - `stage2_ab.softctx_temperature`
  - `stage2_ab.coord_ctx_embed_mode`
  - `stage2_ab.coord_decode_mode`
  - `rollout_matching.coord_decode_mode`
  - legacy flat duplicate-control leaves:
    - `stage2_ab.channel_b.duplicate_iou_threshold`
    - `stage2_ab.channel_b.center_radius_scale`

## Assignment, Duplicate Filtering, And Channel-B Targets

Current design direction:

- Duplicate filtering happens before assignment and before target realization.
  The assignment surface sees only the retained accepted-rollout survivors.
- Greedy IoU assignment is the target architecture for Stage-2 shadow planning.
  `src/training/stage2/assignment.py::GreedyIoUAssignment` builds one-to-one
  prediction-to-GT pairs by descending IoU with stable prediction/GT indices as
  tie-breakers.
- Unmatched GT objects after greedy assignment are false negatives. Channel-B
  inserts them into the final clean-prefix target as `source_role:
  false_negative` objects.
- The default Channel-B ordering remains `tail_append`: retained accepted
  rollout objects first, false-negative GT objects at the tail. `sorted` remains
  the explicit final top-left sort option over retained accepted objects plus
  inserted false negatives.
- Duplicate-control non-survivors are diagnostic evidence only. They must not
  become positive clean-prefix targets after filtering.

Assignment note:

- Live Stage2-AB assignment is routed through
  `src/training/stage2/assignment.py::GreedyIoUAssignment` and accepts only
  `stage2_ab.channel_b.assignment.strategy: greedy_iou`.
  - This intentionally faces rollout quality failures directly instead of
    hiding them behind an alternative assignment mechanism.
  - `stage2_ab.channel_b.assignment.iou_threshold` is optional; when omitted,
    the live matcher gate threshold is reused.
- Policy provenance should record `greedy_iou` plus the duplicate-filter,
  object-ordering, and fallback-loss policy used for target realization.

## Recommended Config Entry Points

- A-only baseline: `configs/stage2_two_channel/prod/a_only.yaml`
- Mixed A/B baseline: `configs/stage2_two_channel/prod/ab_mixed.yaml`
- Pseudo-positive `K=4` production profile: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
- A-only smoke: `configs/stage2_two_channel/smoke/a_only.yaml`
- A-only center-size smoke: `configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml`
- Production-like smoke: `configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml`
- Residual-state trie startup smoke:
  `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`
- `stage2_trie_ce` alias train8 overfit probe:
  `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml`
- Decode-batch=4 train128/val64 runtime gate:
  `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_decode4_hf_gate.yaml`
- Residual-state trie train128/val64 mini matrix:
  - H1 tail-append + weak FP 0.02:
    `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_tail_append_weak_fp_w0p02_lr1e5_decode4.yaml`
  - H2 sorted + zero FP:
    `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_sorted_zero_fp_lr1e5_decode4.yaml`
  - H3 tail-append + zero FP:
    `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_tail_append_zero_fp_lr1e5_decode4.yaml`
  - H4 sorted + weak FP 0.02:
    `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_sorted_weak_fp_w0p02_lr1e5_decode4.yaml`
- Pseudo-positive smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml`
- Enabled `K=2` pseudo-positive control smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_k2_4steps.yaml`
- Server-mode eval smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_triage_posterior_vllm_server_6srv2lr_eval_4steps.yaml`

## Launch Patterns

### Train128/Val64 Decode4 Mini Matrix

This matrix is a two-stage diagnostic. First run the live Channel-B decode
gate to prove DDP plus batch decode plus eval artifacts. Only after the gate is
clean should the residual-state trie H-runs be treated as model-training
evidence.

Config-only checks:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_decode4_hf_gate.yaml \
  --cfg-only

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29650 \
  -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_decode4_hf_gate.yaml \
  --cfg-only
```

G0 runtime gate:

```bash
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29650 \
  -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_decode4_hf_gate.yaml
```

The gate config uses `rollout_matching.rollout_backend: hf` for the current
diagnostic round. This still exercises live Channel-B decode, DDP, batch
decode size 4, and eval artifact materialization. Validate vLLM server or
colocate mode in a separate gate before claiming vLLM production readiness.
The train128/val64 decode4 configs set
`stage2_ab.channel_b.ddp_phase_timeout_s: 600` because HF batch decode and
target construction can create substantial rank skew before the learner step.
Post-rollout packing is DDP-scheduled: each rank first plans its local packs,
the trainer gathers local pack counts, and all ranks execute the same
`global_slot_count`. Ranks with fewer local packs use front-padded zero-weight
shadow slots before their real packs, so DDP forward/backward ordering remains
aligned and the final sync slot is reached together. A rank with zero local
packs while peers have trainable packs fails fast after pack-count gather; that
is treated as a target-construction/fallback data-flow bug, not as a valid empty
training step. The configs also set `global_max_length: 20000` to reduce
avoidable pack splits; DDP safety must come from the shared slot schedule rather
than from assuming equal rank-local pack counts.

Pass criteria:

- train and eval complete without DDP timeout or barrier hang
- batch metrics include `stage2/raw_rollouts`, `rollout/backend_hf`, and
  `rollout/template_family_compact_full`
- batch metrics include pack-schedule counters such as
  `packing/post_rollout_global_slot_count` and
  `packing/post_rollout_empty_slot_count`
- `prompt_tok_mismatch_total == 0`
- eval artifacts are materialized, especially `gt_vs_pred_scored.jsonl`

First-wave 4+4 layout once G0 and online rollout validation pass:

```bash
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29651 \
  -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_tail_append_weak_fp_w0p02_lr1e5_decode4.yaml

PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29652 \
  -m src.sft \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_32steps_stage2_trie_sorted_zero_fp_lr1e5_decode4.yaml
```

Monitor recall first, then F1, with precision interpreted cautiously because
the standard COCO labels may be incomplete. If train recall does not rise,
audit live rollout generation, response-token fidelity, residual target
construction, and loss/gradient flow before trying a hotter learning rate.

### Direct Learner Run

Use this when you do not need the dedicated server-mode launcher split.

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml
```

Legacy center-size experiment note:

- center-size Stage-2 geometry experiments are historical and no longer part
  of the active Stage-2 objective pipeline
- use `experiment_manifest.json` for authored run purpose / hypothesis / key
  deviations plus a runtime summary
- `run_metadata.json` remains the low-level provenance sidecar and does not
  redefine loss semantics

## First Pseudo-Positive Checks

For the first enabled runs, verify:

- `stage2/raw_rollouts` reflects `K` rollout attempts per source sample
- `train/triage/pseudo_positive_selected_count` is non-zero on at least some dense scenes
- `train/triage/shield_only_count` remains the total shield-only count
- `train/triage/pseudo_positive_subthreshold_count` currently mirrors that retained shield-only total; use `train/triage/pseudo_positive_cluster_demoted_count` to separate cluster losers from plain below-threshold support-positive current-attempt objects
- `rollout/peer/*` remains interpretable as mean-over-valid-peer-view aggregates; legacy `rollout/explorer/*` mirrors it for compatibility only
- `dup/raw/duplicate_like_max_cluster_size` and `dup/raw/desc_entropy` move on hard duplicate-collapse scenes before the additive suppression counters do
- `stage2_ab/channel_b/dup/N_clusters_suppressed` and `stage2_ab/channel_b/dup/N_objects_suppressed` remain sparse policy counters rather than raw pathology gauges
- duplicate-control diagnostics remain sparse; do not expect every suppressed
  object or dead current-attempt object to produce a boundary-local diagnostic
  record

Regression gate for Stage-2 objective cleanup:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_training_runtime_sft_integration.py -q
```

### Server-Mode Mixed A/B Run

Use this when Channel-B rollout generation should run through the repo-owned
vLLM server launcher.

```bash
server_gpus=0,1,2,3,4,5 \
train_gpus=6,7 \
config=configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml \
conda run -n ms bash scripts/train_stage2.sh
```

`scripts/train_stage2.sh` is intentionally thin. It delegates YAML preflight,
JSONL validation, GPU-split checks, rollout-server boot, and launcher metadata
export to `src.launchers.stage2_vllm_server`.

## Experiment Authoring

Prefer a concise `training.run_name` and put experiment intent in the top-level
`experiment` block instead of encoding every ablation detail into path names.

```yaml
experiment:
  title: Stage-2 A-only center-size smoke
  purpose: >
    Smoke-test the center-size bbox regression path under the A-only trainer.
  hypothesis: >
    Historical center-size bbox supervision should resolve cleanly without
    changing the canonical xyxy artifact contract.
  key_deviations:
    - Historical center-size geometry path only; not part of new active configs.
    - Caps the run at two optimizer steps.
  runtime_settings:
    - Runs the Stage-2 two-channel trainer in A-only mode.
  comments:
    - Use this for contract validation, not model-quality comparison.
```

`resolved_config.json` remains the authoritative exact-config record.
`experiment_manifest.json` is the operator-facing summary that combines this
authored context with executed-runtime and provenance summaries.

## Smoke Expectations

After a healthy launch, check the run directory for:

- `resolved_config.json`
- `runtime_env.json`
- `effective_runtime.json`
- `pipeline_manifest.json`
- `experiment_manifest.json`
- `train_data_provenance.json`
- `eval_data_provenance.json` when eval data is configured
- `run_metadata.json`
- `logging.jsonl`

What to expect:

- A-only runs finish without Channel-B rollout metric families such as `rollout/*`
- mixed A/B runs emit Channel-B rollout metrics and duplicate diagnostics
- Stage-2 two-channel runs write `stage2_policy_provenance` into
  `effective_runtime.json`, `pipeline_manifest.json`, `run_metadata.json`, and
  `experiment_manifest.json`; verify `assignment_strategy`,
  `duplicate_filter_strategy`, and `object_ordering_policy` before comparing
  assignment or duplicate-filter experiments
- duplicate-control runs should emit both raw gauges under `dup/raw/*` and
  additive policy counters under `stage2_ab/channel_b/dup/N_*`
- eval-enabled server-mode runs emit grouped eval families such as `eval/detection/*`
- eval-enabled runs also materialize offline-compatible eval artifacts under
  `eval_detection/step_<global_step>/` when
  `rollout_matching.eval_detection.materialize_artifacts: true`
  this is the default authored behavior in the base Stage-2 config
  - expect `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `infer_summary.json`,
    `metrics.json`, `per_image.json`, and evaluator sidecars for that window
  - Stage-2 additionally writes `raw_rollouts.jsonl` so rollout text, token
    traces, parsing diagnostics, scores, and matching metadata remain available
    after training finishes
- center-size bbox experiments keep the same run-artifact contract; the proof
  that a run used `parameterization: center_size` lives in `resolved_config.json`
  rather than a new artifact family

Checkpoint restart note:

- `training.save_model_only: true` writes restartable checkpoints with
  optimizer, scheduler, RNG, trainer-state, tokenizer, and repo-owned runtime
  sidecars so resume preflight can fail fast on incomplete checkpoints.
- `training.save_model_only: false` writes inference-only artifacts and accepts
  that interruption may require restarting from the base checkpoint.

Rollout runtime note:

- `src/trainers/stage2_rollout_runtime.py` shares the refactored
  bootstrap/runtime seams and vLLM server infrastructure, but it is not a public
  trainer variant. Repo-owned YAML examples use `stage2_two_channel`.

## Historical Context

Use the historical-context note in this runbook and
[progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md](../../progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md)
for the removed self-context iteration rationale. They are historical context,
not active launch guidance.
