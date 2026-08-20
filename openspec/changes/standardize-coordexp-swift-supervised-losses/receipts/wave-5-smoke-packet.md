# Wave-5 Production-Shaped Vertical Smoke Packet (task 5.3) — FROZEN

- frozen 2026-08-20 by the Claude Fable lead at HEAD `fa8233edf`
  (Wave-5 part-1 commit; tracked tree clean at freeze)
- precondition satisfied BEFORE freezing (task 5.3 ordering): the
  determinant-equality probe ran lead-executed at the same tree —
  `receipts/wave-5-determinant-equality-receipt.json`, both splits EQUAL,
  fingerprints `8f11237f…`/`3b30c157…`, 0 materialization passes. No second
  cache is authorized or possible under this packet.

## Authorization

User standing GPU grant 2026-08-20 (manifest amend-4): host stress tests
skip occupied GPUs; idle GPUs usable without per-action requests. This
packet cites that grant as the distinct authorization task 5.3 requires for
this GPU action. Device selection at launch: two idle GPUs by observed
`nvidia-smi` occupancy (≤ 20 GiB used each; all 8 idle at freeze).

## Binding

- bound commit: `fa8233edf`
- cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- named 1-step base config (migrated, hash-bound):
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
  (gate `mode: zero_weight_ablation`, weight 0.0 — the smoke therefore
  exercises the named ablation row shape)
- launch overlay (run.* only; proven non-determinant):
  `/data/CoordExp/.claude/jobs/c9895ff9/tmp/wave5/wave5_losses_smoke.yaml`
  sha256 `f6b49bc2e487222116b95b77952a3eab742dfb601068743d4ebdd99b3d578d54`
  (extends the base verbatim; `run.name wave5_losses_smoke`, absent root
  `outputs/smoke/wave5_losses_smoke_r1`, `collision_policy: fail`)
- predecessor cache (admission-only): train `8f11237f…`, eval `3b30c157…`
  under `.cache/coordexp_swift/packing`; sha256 inventory before/after must
  be byte-identical; any build/repair is blocking.

## Frozen command (runs once)

```
CUDA_VISIBLE_DEVICES=<two idle GPUs> PYTHONDONTWRITEBYTECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 \
timeout --signal=TERM --kill-after=30s 900s \
conda run -n ms python -m torch.distributed.run --standalone \
  --nproc_per_node 2 -m src.train \
  --config /data/CoordExp/.claude/jobs/c9895ff9/tmp/wave5/wave5_losses_smoke.yaml
```

(bash -c export wrapper on this host; the `torch.distributed.run
--standalone --nproc_per_node 2 -m src.train` shape qualified by the
decompose Wave-8 smoke at the same worktree)

## Bounds

| bound | limit |
| --- | --- |
| devices / world size | exactly 2 GPUs / world_size=2 (argv-fixed) |
| planned steps | exactly 1 planned and applied |
| model forwards | 1 train forward per rank; ≤1 eval forward total |
| cache / materialization passes | REQUIRED 0 |
| wall time | ≤ 900 s (timeout-enforced) |
| peak GPU memory per rank | ≤ 32 GiB |
| CPU RSS per rank | ≤ 16 GiB |
| artifact bytes under run root | ≤ 2.5 GiB |
| free disk at launch | ≥ 100 GiB |

## Acceptance (tasks 5.3 binding + 5.4 inspection)

exit 0 in bound; run status `completed`, `final_optimizer_update_status
= applied`, `final_finite_status = finite`, terminal_error null; both
predecessor fingerprints admitted, cache root inventory unchanged,
`cache_preparation`/`cache_publication` phases not_run; exactly one new
directory `outputs/smoke/wave5_losses_smoke_r1/wave5_losses_smoke`;
logging rows carry the Wave-4 canonical families exactly — per-term
`loss/<T>/{raw,weighted,selected_count,segment_count,token_weighted_diag}`
+ `finite/<T>`, `loss/total` = sum of weighted objective terms, gate
ablation family retained with weighted == 0.0 exactly, ZERO
`coord_gaussian_rps` keys, zero bare `loss/<term>` keys, no
backward/backend keys persisted; raw-to-weighted arithmetic checked
numerically; single-writer rank-zero artifacts. Stop without retry on any
bound exceedance, occupied output, cache byte change, or unexpected
fingerprint.
