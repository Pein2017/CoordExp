# Wave-7 Cache-Action Packet (task 8.2) — FROZEN

- change: `decompose-coordexp-swift-training-orchestration`
- packet author: Claude Fable lead (session c9895ff9)
- frozen: 2026-08-20, inside one no-commit window at HEAD
  `88391d6bb208931ba3fa537fb9522ea6b106fe60` (tracked tree clean; sole
  untracked path at freeze = `receipts/wave-6-pre-cost-audit.md`, exactly as
  that audit's post-write expectation states)
- gate satisfied: the task-7.7 pre-cost audit
  (`receipts/wave-6-pre-cost-audit.md`, same commit) reports **0 P0 / 0 P1**
  and clears Wave-7 cache materialization explicitly.

## Binding

- bound commit (exact Wave-6 commit): `88391d6bb208931ba3fa537fb9522ea6b106fe60`
- tracked tree must remain clean (untracked receipt/evidence files excepted)
  from this freeze through both invocations; any commit, tracked-file, or
  argv drift = stop without retry
- working directory for both invocations: `/data/CoordExp/.worktrees/CoordExp-swift`
  (cache root resolves from cwd; `COORDEXP_SWIFT_PACK_CACHE_ROOT` must be UNSET)
- config: `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
  - sha256 `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
    (re-hashed identical at freeze)
  - train dataset sha256 `4bc00be57d78d4d76e0874cd83f89329d1dc2cd471c208f3f2e30e0365dd96ee`
  - eval dataset sha256 `81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894`

## Authorization

User blanket pre-authorization granted 2026-08-19 in-session:
"我先提前授权所有的内容,你可以不停下来直接执行" — explicitly covering the full
remaining decompose program including this Wave-7 production cache
materialization and the Wave-8 two-rank GPU smoke (recorded the same day in
memory file `coordexp-swift-reconcile-attempt7-resume.md`). This packet cites
that grant as the fresh authorization required by task 8.2 for exactly one
build-capable invocation. The second argv receives no cache-materialization
authority. Planning/implementation approval is not the basis; the quoted
grant is. The pre-cost audit is gate evidence, not launch authority.

## Frozen environment (both invocations)

```
PYTHONDONTWRITEBYTECODE=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
FLASH_ATTENTION_DETERMINISTIC=1
COORDEXP_SWIFT_PACK_CACHE_ROOT   (verified unset)
CUDA not required; no GPU env changes
```

## Frozen argv 1 — the single build-capable invocation (task 8.3)

```
timeout --signal=TERM --kill-after=30s 600s \
conda run -n ms python -m src.prepare_train_cache \
  --config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml \
  --receipt /data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-7-cache-preparation.json
```

Runs exactly once. May publish the two absent split targets below. Stop
without retry on: commit/argv drift, occupied targets, insufficient headroom,
timeout, or any declared-bound exceedance; retain the terminal receipt and
stop outcome either way.

## Frozen argv 2 — verification, NO materialization authority (task 8.4)

```
timeout --signal=TERM --kill-after=30s 600s \
conda run -n ms python -m src.prepare_train_cache \
  --config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml \
  --receipt /data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-7-cache-verification.json \
  --require-all-hit
```

Must fail before any render/tokenize/pack/build/temporary-publication/
immutable-publication path if either target is missing or invalid; on two
valid hits it only fingerprints/admits/validates and writes its named receipt
(schema `coordexp-swift-pack-cache-verification-receipt-v1`).

## Absent receipt paths (verified absent at freeze; distinct per argv)

1. `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-7-cache-preparation.json`
2. `openspec/changes/decompose-coordexp-swift-training-orchestration/receipts/wave-7-cache-verification.json`

## Absent split targets (from the task-8.1 projections,
`receipts/wave-7-determinant-projections.json`)

Cache root: `.cache/coordexp_swift/packing` — verified **0 entries** at
freeze (no intermediate fingerprint was ever built during Waves 1–6; the
pre-cost audit independently observed the same).

- train: `.cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f` (verified absent at freeze)
- eval:  `.cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662` (verified absent at freeze)

Old immutable evidence that must remain byte-untouched through both
invocations:
- `.cache/coordexp_swift/geometry_flip_aug_5step/{2232c868dae6…, 9c82bc7578…}`
  (2 files each; mtimes 1783394670.996 / 1783394682.933 — recorded by the
  pre-cost audit);
- Wave-4 probe root
  `/data/CoordExp/.claude/jobs/c9895ff9/tmp/wave4-cache-probe/cache-root/coordexp-swift-pack-cache-v3/{ec5baadb…, 76f369a7…}`
  (old-fingerprint publications used as the 8.1 cross-check).

## Task-8.1 projection summary (frozen evidence)

Old commit `2ee6c4959` vs new `88391d6bb`, both splits, 31 determinants
(`receipts/wave-7-determinant-projections.json`):
- `content_identity_changed`: **empty** on both splits (semantic payload
  identical);
- owner changes are exactly the two declared moves:
  `micro_step_runtime_config` `src/training/pipeline.py` → `src/training/cache_contract.py`;
  `micro_step_schema` `src/training/supervised_trainer.py` → `src/training/micro_steps.py`;
- `owner_source_changed` = {cache_serializer, micro_step_runtime_config,
  micro_step_schema, supervision_tokens} (code identity only);
- old fingerprints (train `ec5baadb…`, eval `76f369a7…`) independently
  cross-check the Wave-4 probe `receipt-a.json` byte-for-byte, and both old
  immutable probe targets remain on disk untouched.

## Numeric bounds

Baseline: Wave-4 probe receipt-a (same config bytes, private root, Wave-4
tree): 5.97 s wall, 1.05 GiB peak RSS, ~4.3 MB written.

| bound | limit | baseline |
| --- | --- | --- |
| wall time per invocation | ≤ 600 s (enforced by `timeout 600s` in the frozen argv) | 6 s |
| CPU peak RSS (receipt `resource_high_water.cpu.max_rss_bytes`) | ≤ 4 GiB | 1.05 GiB |
| new bytes under cache root | ≤ 100 MiB | ~4.3 MB |
| free disk on /data at start | ≥ 100 GiB | 1598 GiB at freeze |
| split count | exactly 2 (train, eval.forward) | 2 |
| materialization workers | exactly 16 (`DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS`; CLI passes None) | 16 |

Any bound exceedance (enforced by timeout or observed from the terminal
receipt/measurement) = stop without retry; the terminal receipt and stop
outcome are retained as evidence.

## Acceptance for task 8.3/8.4 (checked by the lead after each invocation)

1. argv-1 receipt `terminal_status = "completed"`, schema
   `coordexp-swift-pack-cache-preparation-receipt-v1`; train/eval
   `fingerprint` equal to the projected `8f11237f…` / `3b30c157…`; both
   targets published complete; bounds within limits; old evidence unchanged.
2. argv-2 receipt schema
   `coordexp-swift-pack-cache-verification-receipt-v1`, `terminal_status =
   "completed"`, both splits admitted as hits with **zero** new bytes under
   the cache root (byte inventory diff before/after) and no third
   fingerprint anywhere under the root.
