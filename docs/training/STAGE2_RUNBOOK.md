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

The older rollout-matching compatibility surface still exists in
`src/trainers/stage2_rollout_aligned.py`, but the operator-facing config tree
and runbook in this repo are centered on `configs/stage2_two_channel/`.

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
  - `src/trainers/stage2_rollout_aligned.py`
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
- Channel-B keeps the rollout-aligned clean-prefix path.
- Channel-B final object sequencing is controlled by:
  - `stage2_ab.channel_b.insertion_order: tail_append | sorted`
  - default `tail_append` preserves the historical clean-prefix plus FN-tail path
  - `sorted` applies a final top-left sort over the retained anchor objects plus FN objects before final teacher-forced serialization
- Channel-B duplicate control is configured only through:
  - `stage2_ab.channel_b.duplicate_control.iou_threshold`
  - `stage2_ab.channel_b.duplicate_control.center_radius_scale`
- Channel-B pseudo-positive mode is opt-in through:
  - `stage2_ab.channel_b.pseudo_positive.enabled`
  - `stage2_ab.channel_b.pseudo_positive.coord_weight`
- the supported secondary rollout-aligned variant uses:
  - `custom.trainer_variant: stage2_rollout_aligned`
  - `rollout_matching.pipeline.objective[]`
  - `rollout_matching.pipeline.diagnostics[]`
  - exact behavior is specified in `openspec/specs/rollout-matching-sft/spec.md`
- supported routing/objective presets are:
  - `token_ce.application.preset: anchor_text_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
- duplicate-burst UL migration state:
  - the live objective module `loss_duplicate_burst_unlikelihood` is removed and current Stage-2 configs must not declare it
  - historical specs and archived run artifacts may still mention the retired objective name for compatibility notes
  - duplicate-control diagnostics and counters remain supported after objective cleanup
- retired adjacent-repulsion state:
  - adjacent-repulsion anti-copy loss and config knobs are no longer live training support
  - current configs must omit `adjacent_repulsion_*` keys; strict config parsing rejects them as unknown
  - duplicate-control diagnostics remain supported and are separate from the retired loss
- optional `bbox_geo.config` center-size knobs:
  - `parameterization: xyxy | center_size`
  - `center_weight`
  - `size_weight`
  - `parameterization: center_size` keeps outward `bbox_2d` / `xyxy`
    contracts canonical and only changes the internal regression loss-space
- Pseudo-positive mode keeps the one-forward contract:
  - retained prefix objects share one global prefix structure CE surface through `token_ce.config.rollout_global_prefix_struct_ce_weight`
  - `matched_clean` -> coord + global prefix structure CE
  - `fn_injection` -> coord + FN desc CE
  - selected `pseudo_positive` anchors -> coord + global prefix structure CE
  - support-positive retained `shielded_anchor` objects that stay below promotion threshold -> support-weighted coord + global prefix structure CE
  - cluster-demoted pseudo-positive candidates -> global prefix structure CE only
  - duplicate control runs before GT matching and target realization on the assembled anchor + explorer evidence surface
  - non-exempt duplicate-control non-survivors are removed from the clean
    prefix and tracked only through duplicate-control diagnostics
  - `dead_anchor` -> no positive supervision, with duplicate-control suppression only
  - pseudo-positive selection remains anchor-centric: candidates start from unmatched anchor clean objects with explorer support; explorer-only non-GT-backed objects are not promoted into prefix positives
- Default authored pseudo-positive profile:
  - `triage_posterior.num_rollouts: 4`
  - `1` anchor + `3` explorers
  - enabled `K=2` remains the explicit no-promotion control
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
  - compact-full explorer rollouts that enter fallback remain visible in raw
    rollout/fallback metrics, but they are excluded from posterior-support
    denominators used for support rates, recovered-GT rates, and pseudo-positive
    selection.
  - compact-full Channel-B targets do not use CoordJSON tail-closure or
    semantic-stop supervision. Stop/closure metrics should be interpreted as
    CoordJSON-specific unless explicitly documented otherwise.
  - malformed anchor preparation drops that sample from Channel-B training
  - malformed explorer rollouts that remain invalid after salvage parsing abort the step by default only when pseudo-positive mode is enabled
  - outside pseudo-positive mode, malformed rollouts fall back to the existing empty-prefix / FN-only handling instead of taking the invalid-rollout abort path
  - `stage2_ab.channel_b.invalid_rollout_policy: dump_and_continue` dumps and skips the offending pseudo-positive sample instead
  - zero-object explorers remain valid zero-support evidence
- deprecated authored knobs fail fast in active/training configs:
  - `custom.trainer_variant: rollout_matching_sft`
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

Migration note:

- `src/trainers/rollout_matching/matching.py::hungarian_match_maskiou` is a
  legacy compatibility/migration reader until the remaining live adapters and
  historical comparisons are removed.
- Hungarian assignment is not the target architecture for new Stage-2 planning
  or docs. New policy provenance should record `greedy_iou` plus the
  duplicate-filter and object-ordering policy used for target realization.

## Recommended Config Entry Points

- A-only baseline: `configs/stage2_two_channel/prod/a_only.yaml`
- Mixed A/B baseline: `configs/stage2_two_channel/prod/ab_mixed.yaml`
- Pseudo-positive `K=4` production profile: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
- A-only smoke: `configs/stage2_two_channel/smoke/a_only.yaml`
- A-only center-size smoke: `configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml`
- Production-like smoke: `configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml`
- Pseudo-positive smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml`
- Enabled `K=2` pseudo-positive control smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_k2_4steps.yaml`
- Server-mode eval smoke: `configs/stage2_two_channel/smoke/b_majority_coco1024_triage_posterior_vllm_server_6srv2lr_eval_4steps.yaml`

## Launch Patterns

### Direct Learner Run

Use this when you do not need the dedicated server-mode launcher split.

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/a_only_center_size_2steps.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml
```

Center-size experiment note:

- `bbox_geo.config.parameterization: center_size` keeps Channel-A / Channel-B
  decoded boxes and downstream artifacts canonical `xyxy`
- the experimental mode only changes the internal bbox regression term:
  stronger center supervision, softer `log_w` / `log_h`, CIoU still on
  canonical `xyxy`
- verify the intended mode from `resolved_config.json`
- use `experiment_manifest.json` for authored run purpose / hypothesis / key
  deviations plus a runtime summary
- `run_metadata.json` remains the low-level provenance sidecar and does not
  redefine loss semantics

## First Pseudo-Positive Checks

For the first enabled runs, verify:

- `stage2/raw_rollouts` reflects `1 + (K-1)` rollout execution
- `train/triage/pseudo_positive_selected_count` is non-zero on at least some dense scenes
- `train/triage/unlabeled_consistent_count` remains the total shielded-anchor count
- `train/triage/pseudo_positive_subthreshold_count` currently mirrors that retained shielded-anchor total; use `train/triage/pseudo_positive_cluster_demoted_count` to separate cluster losers from plain below-threshold support-positive anchors
- `rollout/explorer/*` remains interpretable as mean-over-valid-explorer-view aggregates
- `dup/raw/duplicate_like_max_cluster_size` and `dup/raw/desc_entropy` move on hard duplicate-collapse scenes before the additive suppression counters do
- `stage2_ab/channel_b/dup/N_clusters_suppressed` and `stage2_ab/channel_b/dup/N_objects_suppressed` remain sparse policy counters rather than raw pathology gauges
- duplicate-control diagnostics remain sparse; do not expect every suppressed
  object or dead anchor to produce a boundary-local diagnostic record

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
    Center-size bbox supervision should resolve cleanly without changing the
    canonical xyxy artifact contract.
  key_deviations:
    - Enables `bbox_geo.config.parameterization: center_size`.
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

Rollout-aligned note:

- `stage2_rollout_aligned` shares the same refactored bootstrap/runtime seams and
  vLLM server infrastructure, but repo-owned YAML examples in this page focus on
  `stage2_two_channel`

## Historical Context

Use the historical-context note in this runbook and
[progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md](../../progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md)
for the removed self-context iteration rationale. They are historical context,
not active launch guidance.
