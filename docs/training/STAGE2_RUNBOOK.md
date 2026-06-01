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

- `custom.trainer_variant: stage2_rollout_correction`
- `stage2_rollout_correction.pipeline.objective[]` contains exactly one enabled `residual_set_correction`
- the training sequence is rollout prefix plus GT/residual correction
- Stage-2 remains YAML-first; no new CLI flags are required

The older split public trainer variants have been removed.
Shared prompt preparation, decode requests, backend dispatch, generated-token
trace normalization, and parser policy route through `src/infer/*`. Stage-2
trainer code owns rollout-correction orchestration, residual target
construction, post-rollout packing, and training metrics.

## Normative References

- [`openspec/specs/stage2-rollout-correction/spec.md`](../../openspec/specs/stage2-rollout-correction/spec.md)
- [`openspec/specs/rollout-matching-sft/spec.md`](../../openspec/specs/rollout-matching-sft/spec.md) only for retired-contract rejection checks
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
  - `src/trainers/stage2_rollout_correction.py`
  - `src/trainers/stage2_rollout_correction/` when package-local helpers are present
  - `src/trainers/rollout_aligned_targets.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/infer/runtime.py`
  - `src/infer/backend.py`
  - `src/infer/backend_vllm_server.py`
  - `src/infer/backend_sync.py`
  - `src/infer/rollout_dispatch.py`
- server-mode orchestration:
  - `src/launchers/stage2_vllm_server.py`

## Current Supported Contract

- `custom.trainer_variant: stage2_rollout_correction`
- shadow architecture `surface.id: stage2_rollout_correction`
- `stage2_rollout_correction.pipeline.objective[]` is required for active Stage-2 configs
- `stage2_rollout_correction.pipeline.diagnostics[]` must be empty
- the only active objective is `residual_set_correction`
- the only active application preset is `rollout_self_prefix`
- removed clean-prefix objectives (`token_ce`, `hard_sft`, `stage2_trie_ce`) fail fast in unified Stage-2 configs
- authored `channels`, scheduler keys, and per-channel namespaces fail fast
- `rollout_matching.pipeline.*` is retired; active objective ownership is only
  through `stage2_rollout_correction.pipeline.*`
- rollout runtime/backend/decode/eval knobs remain under `rollout_matching.*`
  as a classified temporary migration handle until Stage-2 rollout-correction
  schema owns those knobs end to end
- final object sequencing is controlled by:
  - `stage2_rollout_correction.correction.insertion_order: tail_append | sorted | fn_slot_shuffle`
- duplicate control is configured only through:
  - `stage2_rollout_correction.correction.duplicate_control.iou_threshold`
  - `stage2_rollout_correction.correction.duplicate_control.center_radius_scale`
- pseudo-positive clean-prefix knobs are removed from the active Stage-2 contract
- residual-set loss and telemetry are reported under
  `stage2_rollout_correction/residual_set/`; key compact metrics include
  `sequence_count`, `atom_count`, `atom_weight_sum`, `sequence_loss`,
  `type_loss`, `inner_loss`, `wrong_type_mass`, and `valid_set_mass`.
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
  - `stage2_rollout_correction.correction.rollout_template_family: coordjson` is the explicit
    legacy rollout surface. It uses the legacy CoordJSON parser and records
    rollout template/decode/parser provenance in rollout-correction batch metrics.
  - `stage2_rollout_correction.correction.rollout_template_family: compact_full` is the new
    canonical target surface for compact-full checkpoints. It uses the
    compact-full parser and compact-full rollout-correction target renderer, with default
    `rollout_decode_policy: unconstrained`. It must not flow through the legacy
    CoordJSON parser or CoordJSON FN appender.
  - compact-full invalid, malformed, or empty-valid-object rollout policy
    resolves to `fallback_gt_fn_append_only` with `fallback_loss_weight: 1.0`.
    The fallback constructs a compact-full GT/FN-only correction target, records
    `rollout_context: fallback_gt_fn_append_only`, and does not count as a valid
    rollout for readiness gates. Parser/template mismatches remain hard
    failures, not fallback cases.
  - `stage2_rollout_correction.correction.strict_rollout_preflight: true` is the fail-fast
    diagnostic mode for online compact-full training. It rejects truncation,
    parser invalid/fallback outputs, salvage/drop paths, and compact-full
    object-span extraction failures before target construction, so parser or
    token-alignment drift cannot be hidden as fallback supervision.
  - compact-full peer attempts that enter fallback remain visible in raw
    rollout/fallback metrics. They do not contribute positive support, while
    pseudo-positive and recovered-GT support denominators remain the configured
    peer-attempt count (`num_rollouts - 1`) so missing/fallback peers cannot
    silently relax consensus.
  - compact-full rollout-correction targets do not use CoordJSON tail-closure or
    semantic-stop supervision. Stop/closure metrics should be interpreted as
    CoordJSON-specific unless explicitly documented otherwise.
  - Train-time rollout prompt variants are authored with
    `rollout_matching.prompt_variant`; eval-step rollout prompt variants are
    authored separately with `rollout_matching.eval_prompt_variant`. For a
    compact-full checkpoint whose baseline was established under the COCO-80
    closed-class prompt, pin both keys to `coco_80` before comparing no-update
    base-control, short pilots, or production candidates. Leaving the train-time
    key unset falls back to the default dense prompt for backward compatibility
    and is not the same experiment surface as `coco_80`.
  - These `rollout_matching.*` prompt/decode/eval keys are retained migration
    handles, not the target public namespace for new clean-break Stage-2
    schema work.
  - malformed current-attempt preparation drops that attempt/sample from rollout-correction training
  - malformed peer attempts that remain invalid after salvage parsing abort the step by default only when pseudo-positive mode is enabled
  - outside pseudo-positive mode, malformed rollouts fall back to the existing empty-prefix / FN-only handling instead of taking the invalid-rollout abort path
  - `stage2_rollout_correction.correction.invalid_rollout_policy: dump_and_continue` dumps and skips the offending pseudo-positive sample instead
  - zero-object peer attempts remain valid zero-support evidence
- deprecated authored knobs fail fast in active/training configs:
  - `custom.trainer_variant: rollout_matching_sft`
  - `custom.trainer_variant: stage2_rollout_aligned`
  - `custom.trainer_variant: stage2_rollout_runtime`
  - removed split-stage trainer variants and namespaces
  - removed self-context iteration controls
  - `rollout_matching.coord_decode_mode`
  - legacy flat duplicate-control leaves:
    - `stage2_rollout_correction.correction.duplicate_iou_threshold`
    - `stage2_rollout_correction.correction.center_radius_scale`

## Assignment, Duplicate Filtering, And Rollout-Correction Targets

Current design direction:

- Duplicate filtering happens before assignment and before target realization.
  The assignment surface sees only the retained accepted-rollout survivors.
- Greedy IoU assignment is the target architecture for Stage-2 shadow planning.
  `src/training/stage2/assignment.py::GreedyIoUAssignment` builds one-to-one
  prediction-to-GT pairs by descending IoU with stable prediction/GT indices as
  tie-breakers.
- Unmatched GT objects after greedy assignment are false negatives. Rollout-correction
  inserts them into the final clean-prefix target as `source_role:
  false_negative` objects.
- The default rollout-correction ordering remains `tail_append`: retained accepted
  rollout objects first, false-negative GT objects at the tail. `sorted` remains
  the explicit final top-left sort option over retained accepted objects plus
  inserted false negatives.
- Duplicate-control non-survivors are diagnostic evidence only. They must not
  become positive clean-prefix targets after filtering.

Assignment note:

- Live Stage-2 rollout-correction assignment is routed through
  `src/training/stage2/assignment.py::GreedyIoUAssignment` and accepts only
  `stage2_rollout_correction.correction.assignment.strategy: greedy_iou`.
  - This intentionally faces rollout quality failures directly instead of
    hiding them behind an alternative assignment mechanism.
  - `stage2_rollout_correction.correction.assignment.iou_threshold` is optional; when omitted,
    the live matcher gate threshold is reused.
- Policy provenance should record `greedy_iou` plus the duplicate-filter,
  object-ordering, and fallback-loss policy used for target realization.

## Recommended Config Entry Points

- Canonical base: `configs/stage2/rollout_correction/base.yaml`
- New production and smoke leaves should live under `configs/stage2/rollout_correction/`
- Compact-full ckpt3664 + COCO-80 readiness leaves:
  - `configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml`
  - `configs/stage2/rollout_correction/smoke/compact_full_vllm_train64_val32_base_control_lr0_1step_coco80_prompt.yaml`
  - `configs/stage2/rollout_correction/smoke/compact_full_vllm_train64_val32_12steps_coco80_pilot.yaml`
- Every migrated leaf must restate the full `stage2_rollout_correction.pipeline.objective[]` list because config list merging replaces lists wholesale
- Old split-stage and pseudo-positive clean-prefix handles are removed active contracts; use historical records only when interpreting older runs

## Launch Patterns

### Minimal Config And Runtime Gate

Use this first when changing Stage-2 runtime plumbing. It proves config
resolution, DDP launch wiring, live rollout decode, target construction, and eval
artifact materialization on the single active rollout-correction surface.

Config-only checks:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft \
  --config configs/stage2/rollout_correction/base.yaml \
  --cfg-only

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  conda run --no-capture-output -n ms torchrun \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29650 \
  -m src.sft \
  --config configs/stage2/rollout_correction/base.yaml \
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
  --config configs/stage2/rollout_correction/base.yaml
```

The base config uses `rollout_matching.*` only for rollout runtime/backend,
decode, and eval settings. It still exercises live rollout-correction decode,
DDP, batch decode, and eval artifact materialization. Validate vLLM server or
colocate mode in a separate gate before claiming vLLM production readiness.
Rollout backend/decode/trace adapters are implementation-owned by `src/infer/*`;
`rollout_matching.*` is the authored Stage-2 runtime namespace, not a separate
objective or parser implementation root. Server-mode vLLM with adapter sync
records backend-sync identity for metric-bearing eval provenance, while HF eval
ignores stale vLLM sync state.
Set `stage2_rollout_correction.correction.ddp_phase_timeout_s` conservatively
for large HF batch decode runs because rollout and target construction can
create substantial rank skew before the learner step.
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

Monitor recall first, then F1, with precision interpreted cautiously because
the standard COCO labels may be incomplete. If train recall does not rise,
audit live rollout generation, response-token fidelity, residual target
construction, and loss/gradient flow before trying a hotter learning rate.

### Direct Learner Run

Use this when you do not need the dedicated server-mode launcher split.

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage2/rollout_correction/base.yaml
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
- `stage2_rollout_correction/correction/dup/N_clusters_suppressed` and `stage2_rollout_correction/correction/dup/N_objects_suppressed` remain sparse policy counters rather than raw pathology gauges
- duplicate-control diagnostics remain sparse; do not expect every suppressed
  object or dead current-attempt object to produce a boundary-local diagnostic
  record

Regression gate for Stage-2 objective cleanup:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_rollout_correction_contract.py \
  tests/test_training_runtime_sft_integration.py -q
```

### Server-Mode Rollout-Correction Run

Use this when rollout generation should run through the repo-owned
vLLM server launcher.

```bash
server_gpus=0,1,2,3,4,5 \
train_gpus=6,7 \
config=configs/stage2/rollout_correction/base.yaml \
conda run -n ms bash scripts/train_stage2.sh
```

`scripts/train_stage2.sh` is intentionally thin. It delegates YAML preflight,
JSONL validation, GPU-split checks, rollout-server boot, and launcher metadata
export to `src.launchers.stage2_vllm_server`.

For adapter-backed compact-full checkpoints, native Stage-2 vLLM rollout now
requires official ms-swift adapter sync:
`rollout_matching.vllm.enable_lora=true` and
`rollout_matching.vllm.sync.mode=adapter`. Qwen3-VL ViT/connector LoRA also
sets `rollout_matching.vllm.enable_tower_connector_lora=true`, which the
launcher forwards through `--vllm_engine_kwargs`. The old transient full-weight
materialization path was removed because it did not match HF backend behavior.

## Experiment Authoring

Prefer a concise `training.run_name` and put experiment intent in the top-level
`experiment` block instead of encoding every ablation detail into path names.

```yaml
experiment:
  title: Stage-2 rollout-correction smoke
  purpose: >
    Smoke-test the residual rollout prefix plus GT correction trainer path.
  hypothesis: >
    Residual correction should construct trainable target IR from live rollout
    prefixes without changing the canonical xyxy artifact contract.
  key_deviations:
    - Uses the canonical residual_set_correction objective only.
    - Caps the run at two optimizer steps.
  runtime_settings:
    - Runs `custom.trainer_variant: stage2_rollout_correction`.
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

- rollout-correction runs emit rollout metrics and duplicate diagnostics
- Stage-2 rollout-correction runs write `stage2_policy_provenance` into
  `effective_runtime.json`, `pipeline_manifest.json`, `run_metadata.json`, and
  `experiment_manifest.json`; verify `assignment_strategy`,
  `duplicate_filter_strategy`, and `object_ordering_policy` before comparing
  assignment or duplicate-filter experiments
- duplicate-control runs should emit both raw gauges under `dup/raw/*` and
  additive policy counters under `stage2_rollout_correction/correction/dup/N_*`
- eval-enabled server-mode runs emit grouped eval families such as `eval/detection/*`
- eval-enabled runs also materialize offline-compatible eval artifacts under
  `eval_detection/step_<global_step>/` when
  `rollout_matching.eval_detection.materialize_artifacts: true`
  this is the default authored behavior in the base Stage-2 config
  - official Stage-2 eval metrics require materialized scored artifacts and
    fail fast if `materialize_artifacts: false` or `training.output_dir` would
    bypass score-provenance checks
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

- Repo-owned YAML examples use `stage2_rollout_correction`; removed public
  variants fail fast. Shared inference behavior belongs under `src/infer/*`.
  Remaining Stage-2 trainer modules are private orchestration surfaces, not
  public trainer variants or shared inference APIs.

## Historical Context

Use the historical-context note in this runbook and
[progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md](../../progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md)
for the removed self-context iteration rationale. They are historical context,
not active launch guidance.
