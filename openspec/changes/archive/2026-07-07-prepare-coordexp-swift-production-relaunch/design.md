# CoordExp-Swift Production Relaunch Readiness Design

## Context

The V1 training rebuild is complete enough to prepare a production relaunch.
This change governs the final readiness layer before launching the next
8-GPU Accelerate run. It does not reopen the completed rebuild task ledger and
does not broaden the research objective.

The relaunch profile is intentionally narrow: Qwen3-VL 2B, desc-first,
`geo_sorted`, pure CE, language-only DoRA, packed length 12000, COCO len12000
train/val JSONL, seed 17, rank 16, alpha 32, effective batch size 64,
warmup ratio 0.1, max grad norm 1.0, four epochs, Accelerate 8-GPU.

## Decisions

- Seed control is production infrastructure. The configured runtime seed must
  be applied before any stochastic model or adapter initialization, including
  PEFT DoRA initialization.
- Seed setup is evidenced by a runtime receipt rather than a new public config
  knob. Deterministic algorithm forcing remains out of scope unless explicitly
  approved later.
- Public configs continue to expose `training.effective_batch_size`; backend
  accumulation stays derived from world size and must remain unauthored in the
  public config.
- Scheduler warmup derivation is relaunch-readiness infrastructure. Runs must
  emit a scheduler receipt that makes `warmup_ratio: 0.1` resolving to one
  two-step-smoke warmup step and 92 production warmup steps explicit.
- Deterministic supervised packing-cache reuse is approved training
  materialization infrastructure. Hidden-state, KV, and runtime feature caches
  remain outside this relaunch.
- The production relaunch is Accelerate-only. DeepSpeed production support
  remains gated by a separate systems smoke.
- Post-training detection inference/eval remains a follow-on infrastructure
  lane. The current `src.infer` path is not enough to promise val200/val512
  benchmark metrics immediately after training.

## Acceptance Strategy

The readiness path has four gates:

1. OpenSpec and Superpowers docs validate and do not corrupt the completed
   rebuild baseline.
2. Unit/config tests prove seed ordering, seed receipts, strict config loading,
   scheduler receipts, and runtime-batch derivation.
3. Distributed smoke artifacts prove eval-forward health, FA2 proof, finite
   metrics, scheduler warmup derivation, rank-local completion, and final
   checkpoint writing.
4. The 8-GPU production run may launch only after the gates pass and GPU
   occupancy is safe.

The expected production schedule for unchanged pack cardinality is
`resolved_max_steps=917`, `resolved_grad_accum_steps=8`,
`tail_fill_pack_count=48`, and `resolved_warmup_steps=92`. Schedule values of
459 or 928 are stop signals for this relaunch.
