# Wave-5.5 Production-Mimic Smoke Packet — FROZEN

- frozen 2026-08-20 by the Claude Fable lead at HEAD `30b563e6c`
  (Wave-5 part-1 commit; tracked tree clean)
- authorization: user standing GPU grant 2026-08-20 (manifest
  `authorization.gpu`); two idle GPUs by observed `nvidia-smi` occupancy at
  launch (≤ 20 GiB used each).

## Binding

- bound commit: `30b563e6c`
- cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- named base config (smallest current two-rank production-mimic, migrated
  with `observability.steps: 1`):
  `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- launch overlay (run.* only; run.* proven non-determinant):
  `/data/CoordExp/.claude/jobs/c9895ff9/tmp/obsw5/obs_wave5_smoke.yaml`
  sha256 `e2c12c960710dc55594b9648aa46b4718b68107e0f46471557daae8b28250d14`
  (absent root `outputs/smoke/obs_wave5_smoke_r1`, `collision_policy: fail`)
- predecessor cache admission-only (train `8f11237f…`, eval `3b30c157…`);
  sha256 inventory before/after must be byte-identical.

## Frozen command (runs once)

```
CUDA_VISIBLE_DEVICES=<two idle GPUs> PYTHONDONTWRITEBYTECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 \
timeout --signal=TERM --kill-after=30s 900s \
conda run -n ms python -m torch.distributed.run --standalone \
  --nproc_per_node 2 -m src.train \
  --config /data/CoordExp/.claude/jobs/c9895ff9/tmp/obsw5/obs_wave5_smoke.yaml
```

## Bounds

| bound | limit |
| --- | --- |
| devices / world size | exactly 2 GPUs / ws=2 (argv-fixed) |
| planned steps | exactly 1 planned + applied (one real finite optimizer update) |
| model forwards | 1 train/rank; ≤1 eval total (scheduled eval at step 1) |
| cache / materialization passes | REQUIRED 0 |
| wall time | ≤ 900 s (timeout-enforced) |
| peak GPU memory per rank | ≤ 32 GiB |
| CPU RSS per rank | ≤ 16 GiB |
| artifact bytes under run root | ≤ 2.5 GiB |
| free disk | ≥ 100 GiB at launch |

## Acceptance (task 5.5)

exit 0 in bounds; ONE shared run tree
(`outputs/smoke/obs_wave5_smoke_r1/obs_wave5_smoke`, no rank-local
run/event trees); run status completed / applied / finite; exactly one
train and one eval canonical row with correct LR (receipt-owned pre-call
value)/pre-clip norm/loss (raw+weighted families)/timing/resource fields
and availability honesty; readable rank-zero TensorBoard events under
`<run>/tensorboard/` (verified with the real event reader: expected
train/eval tags, finite values, canonical global_step); cache root
byte-identical; both fingerprints admitted; record exact config, commit,
command, devices, artifact root, counters, and evidence scope; NO
throughput or model-quality claim. Stop without retry on any bound
exceedance, occupied output, cache byte change, or missing/extra rank
trees.
