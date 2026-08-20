# Wave-8 GPU-Launch Packet (task 9.2) — FROZEN

- change: `decompose-coordexp-swift-training-orchestration`
- packet author: Claude Fable lead (session c9895ff9)
- frozen: 2026-08-20 at HEAD `d598f8894a3e2d0f3a5c407b685b344813a58dd3`
  (the exact Wave-7 commit; tracked tree clean at freeze)
- gate satisfied: the task-7.7 pre-cost audit
  (`receipts/wave-6-pre-cost-audit.md`) reports 0 P0 / 0 P1 and clears the
  Wave-8 two-rank GPU smoke; its finding **L-1** is consumed below.

## Authorization

User blanket pre-authorization granted 2026-08-19 in-session:
"我先提前授权所有的内容,你可以不停下来直接执行" — explicitly covering the
Wave-8 two-rank GPU vertical smoke (recorded in memory file
`coordexp-swift-reconcile-attempt7-resume.md`). This packet cites that grant
as the fresh authorization required by task 9.2. Planning/implementation
approval is not the basis; the quoted grant is.

## Binding

- bound commit (exact Wave-7 commit): `d598f8894a3e2d0f3a5c407b685b344813a58dd3`
  (source bytes identical to the audited Wave-6 commit `88391d6bb…` — Wave 7
  committed receipts + tasks.md only)
- cwd for all commands: `/data/CoordExp/.worktrees/CoordExp-swift`
- named 1-step base config (verbatim, hash-bound):
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
  sha256 `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- launch overlay (consumes audit finding **L-1**: the base config's
  `outputs/smoke/production_mimic` root already exists with
  `collision_policy: timestamp`, which cannot enforce the absent-target stop
  rule; the overlay extends the named config verbatim and overrides only
  `run.*` — the Wave-4 probe proved `run.*` fields are not cache
  determinants):
  `/data/CoordExp/.claude/jobs/c9895ff9/tmp/wave8/wave8_vertical_smoke.yaml`
  sha256 `e2b9321b8fb692d3b3ea918a0cd5ced46c1899b40ec6b2717cc9ba1d1014662b`

  ```yaml
  schema_version: 1
  extends: /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml
  run: {name: wave8_vertical_smoke, artifact_root: /data/CoordExp/.worktrees/CoordExp-swift/outputs/smoke/wave8_vertical_smoke_r1, collision_policy: fail}
  ```

- strict-load verified at freeze: `collision_policy=fail`, `max_steps=1`,
  `ebs=2`, `precision=bf16`;
  `resolve_effective_batch_runtime(world_size=2)` →
  `RuntimeBatchResolution(world_size=2, effective_batch_size=2,
  resolved_grad_accum_steps=1)` — exactly **one planned and applied
  optimizer step**, one train micro-step per rank.
- absent artifact root (verified absent at freeze):
  `outputs/smoke/wave8_vertical_smoke_r1` — `collision_policy: fail` makes an
  occupied output fail closed; the executor also re-verifies absence
  immediately before launch.
- new cache (published by Wave 7, admission-only; any cache build during the
  smoke is a blocking stop):
  train `8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f`,
  eval `3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662`
  under `.cache/coordexp_swift/packing` (root resolves from cwd;
  `COORDEXP_SWIFT_PACK_CACHE_ROOT` unset). Cache-root sha256 inventory is
  taken before and after the smoke; any byte change is blocking.

## Topology and devices

- `world_size=2`, fixed solely by the launch argv (`--nproc_per_node 2`),
  never by a config field.
- exactly two GPUs: `CUDA_VISIBLE_DEVICES=0,1`
  (GPU 0 `GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e`,
  GPU 1 `GPU-f520bc13-34fb-8809-4af8-a6fd8a28aec8`; live map read at freeze).
- occupancy bound revalidated immediately before launch: ≤ 21,474,836,480
  bytes already in use per selected device (at freeze: 597 MiB / 3 MiB).

## Frozen launch command (production-shaped BF16 smoke; runs once)

```
CUDA_VISIBLE_DEVICES=0,1 PYTHONDONTWRITEBYTECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 \
timeout --signal=TERM --kill-after=30s 900s \
conda run -n ms python -m torch.distributed.run --standalone \
  --nproc_per_node 2 -m src.train \
  --config /data/CoordExp/.claude/jobs/c9895ff9/tmp/wave8/wave8_vertical_smoke.yaml
```

(the same `torch.distributed.run --standalone --nproc_per_node 2 -m
src.train` shape the archived Attempt-8 packet qualified on this host; stdout
captured to the session log and retained)

## Numeric config-derived ceilings

| ceiling | limit | derivation |
| --- | --- | --- |
| per-rank train model forwards | exactly 1 | max_steps=1 × grad_accum=1 (train cache micro_step_count=2, one per rank) |
| eval model forwards (total) | ≤ 1 | eval.forward at step 1; eval cache micro_step_count=1 |
| per-rank collective operations | ≤ 100 | rank-convergence phases + 1 gradient sync + finalization barriers (characterized order; generous headroom) |
| wall time | ≤ 900 s (enforced by `timeout`) | Attempt-8 2-step control bound was 600 s; 1 step + model load |
| CPU peak RSS per rank | ≤ 16 GiB | Attempt-8 per-rank bound |
| GPU high-water per rank | ≤ 32 GiB | Attempt-8 per-rank bound; 2B BF16 + DoRA ≪ |
| artifact bytes under the run root | ≤ 2.5 GiB | Attempt-8 control-arm bound (adapter checkpoint + final/best + logs) |
| free disk on /data at launch | ≥ 100 GiB | 1,598 GiB observed at freeze |

## Stop conditions (stop without retry; retain all receipts and stdout)

commit/argv/config-hash drift; occupied artifact root or cache mutation
(byte-inventory diff); unexpected cache build; insufficient device occupancy
headroom; timeout; OOM; non-finite loss gate; collective hang (covered by
timeout); any ceiling exceedance; run terminal status other than success.

## Acceptance (checked by the lead after the run)

1. exit 0 within wall bound; exactly one new directory
   `outputs/smoke/wave8_vertical_smoke_r1/wave8_vertical_smoke*` with
   `run.json`, `resolved_config.json`, `logging.jsonl`, checkpoint step-1,
   `final.json`/`best.json` per the artifacts spec;
2. run consumed the two Wave-7 cache fingerprints (admission receipts in
   run.json), zero cache-root byte change;
3. exactly 1 applied optimizer step; completed-step and eval rows present;
4. resource receipts within ceilings;
5. task 9.3 exact-compare against the Wave-0 compatibility ledger with the
   cache identity turnover and legacy-selector removal as the only intended
   differences; no speedup claim.
