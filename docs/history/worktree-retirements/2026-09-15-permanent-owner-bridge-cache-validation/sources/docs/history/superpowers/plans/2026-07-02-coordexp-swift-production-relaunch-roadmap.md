# CoordExp-Swift Production Relaunch Roadmap

> **For agentic workers:** execute this roadmap with
> `superpowers:executing-plans` or `superpowers:subagent-driven-development`.
> Fix runtime/config/artifact issues encountered during gates; stop only for
> research semantic changes, destructive cleanup, new production dependencies,
> DeepSpeed production support, active GPU-job interruption, or changed
> hyperparameters.

## Goal

Prepare and relaunch the CoordExp-Swift r16/a32 EBS64 warmup0p1 production
training run after runtime, config, smoke, and artifact gates pass.

## Source Of Truth

- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`
- Follow-on OpenSpec: `openspec/changes/prepare-coordexp-swift-production-relaunch/`
- Completed rebuild baseline: `openspec/changes/rebuild-coordexp-swift-training-infra/`
- Base production config:
  `configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate8_ebs128_4epoch.yaml`

## Implementation Tasks

- Add runtime seed control before Qwen load and DoRA setup, plus a
  manifest-linked `receipts/runtime/seed_control.json`.
- Add a manifest-linked `receipts/optimizer/scheduler_plan.json` so warmup0p1
  resolves visibly to one two-step-smoke warmup step and 92 production warmup
  steps.
- Add the production config
  `configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1.yaml`.
- Add the final smoke config
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof.yaml`.
- Validate OpenSpec, config trace, and targeted unit tests before smoke.
- Run a cheap distributed eval-forward smoke, then the final 8-GPU r16/a32
  EBS64 smoke, before production.

## Acceptance Gates

- Seed order is proven before Qwen load, fresh DoRA setup, selected embedding
  setup, optimizer setup, and `TrainRuntime`.
- Config trace shows rank 16, alpha 32, warmup ratio 0.1, EBS64, seed 17, and
  backend accumulation left null.
- Final smoke artifacts show world size 8, derived grad accumulation 8,
  resolved max steps 2, scheduler receipt with warmup step 1, seed-control
  receipt, r16/a32 adapter receipt, FA2 proof pass, finite metrics,
  eval-forward step 1, rank-local finalization, and checkpoint-final.
- Production early artifacts show expected unchanged-cardinality values:
  `packs_per_epoch=14660`, `resolved_max_steps=917`,
  `tail_fill_pack_count=48`, scheduler receipt with warmup steps 92, and
  checkpoint/eval planned steps 367, 734, 917.

## Stop Conditions

- Production schedule resolves to 459 or 928 with unchanged pack cardinality.
- Pack-cache fingerprint changes unexpectedly from
  `f56f066e6423ca4beb80c2f849a86da032da0972dacff184d4724d9324a03eea`.
- Any rank manifest is incomplete, FA2 proof fails, metrics are non-finite,
  or checkpoint-final is missing.
- A fix would require min-LR scheduler parity, separate dataset seed, token
  embedding LR changes, DeepSpeed production support, or another research
  semantic change.
