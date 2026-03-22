---
doc_id: docs.training.stage2-runbook
layer: docs
doc_type: runbook
status: canonical
domain: training
summary: YAML-first runbook for active Stage-2 training, including direct learner runs and vLLM server-mode launches.
updated: 2026-03-22
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

- `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]` are required for active Stage-2 configs
- the supported secondary rollout-aligned variant uses:
  - `custom.trainer_variant: stage2_rollout_aligned`
  - `rollout_matching.pipeline.objective[]`
  - `rollout_matching.pipeline.diagnostics[]`
  - exact behavior is specified in `openspec/specs/rollout-matching-sft/spec.md`
- supported routing presets are:
  - `token_ce.application.preset: anchor_text_only`
  - `loss_dead_anchor_suppression.application.preset: rollout_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
- deprecated authored knobs fail fast in active configs:
  - `custom.trainer_variant: rollout_matching_sft`
  - `stage2_ab.n_softctx_iter`
  - `stage2_ab.softctx_grad_mode`
  - `stage2_ab.softctx_temperature`
  - `stage2_ab.coord_ctx_embed_mode`
  - `stage2_ab.coord_decode_mode`
  - `rollout_matching.coord_decode_mode`

## Recommended Config Entry Points

- A-only baseline:
  - `configs/stage2_two_channel/prod/a_only.yaml`
- Mixed A/B baseline:
  - `configs/stage2_two_channel/prod/ab_mixed.yaml`
- COCO1024 B-majority continuation:
  - `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority.yaml`
- COCO1024 B-majority + triage-posterior continuation:
  - `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior.yaml`
- A-only smoke:
  - `configs/stage2_two_channel/smoke/a_only.yaml`
- Mixed A/B smoke:
  - `configs/stage2_two_channel/smoke/ab_mixed_20steps.yaml`
- Server-mode eval smoke:
  - `configs/stage2_two_channel/smoke/b_majority_coco1024_triage_posterior_vllm_server_6srv2lr_eval_4steps.yaml`

## Launch Patterns

### Direct Learner Run

Use this when you do not need the dedicated server-mode launcher split.

```bash
config=configs/stage2_two_channel/smoke/a_only.yaml \
gpus=0,1 \
conda run -n ms bash scripts/train.sh
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

## Smoke Expectations

After a healthy launch, check the run directory for:

- `resolved_config.json`
- `runtime_env.json`
- `run_metadata.json`
- `logging.jsonl`

What to expect:

- A-only runs finish without Channel-B rollout metric families such as `rollout/*`
- mixed A/B runs emit Channel-B rollout metrics and duplicate diagnostics
- eval-enabled server-mode runs emit grouped eval families such as `eval/detection/*`

Rollout-aligned note:

- `stage2_rollout_aligned` shares the same refactored bootstrap/runtime seams and
  vLLM server infrastructure, but repo-owned YAML examples in this page focus on
  `stage2_two_channel`

## Historical Context

Use [docs/training/STAGE2_DESIGN.md](STAGE2_DESIGN.md) and
[progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md](../../progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md)
for the removed self-context iteration rationale. They are historical context,
not active launch guidance.
