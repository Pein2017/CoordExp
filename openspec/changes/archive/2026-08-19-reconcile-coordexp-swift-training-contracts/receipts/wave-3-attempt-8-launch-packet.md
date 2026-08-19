# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 8

Status: `frozen_pending_external_review`. No command in this packet has run.
Attempts 1–7, including Attempt 5's signed `HOLD`, Attempt 6's signed `READY`
invalidated pre-marker by tracked-HEAD movement (`27abc8087e...`), and
Attempt 7's signed `HOLD` after tracked-HEAD movement during its independent
review (`0b0f554e4...`), are immutable evidence and are neither reused nor
modified. No identity in this packet is copied from an earlier attempt: the
three role configs were deterministically re-rendered against the new `-r8`
root and re-hashed, and the commit binding is the current clean HEAD, which
already contains the Attempt-7 retirement planning/evidence commit. The
executor is outside the six commands.

## Frozen identity and new targets

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `f492f36874f036ad4145a45c7b03cd2e0b2fd049`
- Implementation tree (`HEAD^{tree}`): `83d8901bf096eb76bf0b5e2adf131480131a5839`
- Packet author identity: `coordexp-attempt8-packet-author`
- Manifest: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-command-manifest.json`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA-256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8`
- Absent private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r8.pack-cache`
- Absent private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r8.pack-cache-receipt.json`
- Independent review path (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-pre-cost-review.json`
- Exclusive marker (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-attempt-marker.json`
- Outer receipt (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-outer-terminal-receipt.json`
- Inner receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/terminal-receipt.json`

The `-r8` target name keeps the change-local `2026-08-13` attempt-series date
deliberately: it is the successor slot Attempt 7's retirement pinned, not a
calendar claim about the freeze day.

Physical GPU `6` is bound to `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff` and
physical GPU `7` to `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; the live map
was re-read at freeze time and is unchanged from the Attempt-6/7 host state.
At freeze time GPU `6` was drained (3 MiB) while GPU `7` carried transient
external shared-host load (23,491 MiB from a host-namespace PID), so this
packet is not executable until both devices are within the occupancy bound:
the executor revalidates map, occupancy (max 21,474,836,480 bytes per device),
and disk before marker and again before the first GPU command, and the lead
will invoke `execute` only after independently observing both devices within
that bound. Live free space on `/data` at freeze was 1,717,265,702,912 bytes.

## Deterministic config and accumulation binding

The role configs were rendered in a temporary directory by the production
`_write_role_config` path against the r8 root, loaded by production
`load_train_config`, and resolved by
`resolve_effective_batch_runtime(config, world_size=2)`. Their exact values are:

| Role | Generated config SHA-256 | Resolved fingerprint | EBS | Grad accum |
|---|---|---|---:|---:|
| `uninterrupted_control` | `8dd5fbe13bc90dc6cd1615fad9e5377d14c57b944db68b6a8fe340e8f6c07f72` | `7aa2f96c25aad0eb990bb43a02358a242be2418a7902b5d78c36df38cbed7a86` | 2 | 1 |
| `resumed_parent` | `e5803d8da54256bcb9b4152a805d51cdd64cf5f8e1b2588e88b825de43b5d570` | `3e0f1c089c602b231cd690a2904f8386a31ead639e9e95cc4fa566e39773b5de` | 2 | 1 |
| `resumed_child` | `99b0671bcfea3d034a77a81bd8d4e7e4c8779678c15973aa2900cbb273eb6e4b` | `39c1d751aa7e9a081e51751cbe3cf6777dd9ce864f1212665053cf39fb91f479` | 2 | 1 |

Thus the fixed topology (`world_size=2`, bound only from the setup/two-rank
launch argv and this manifest, never from a config field) makes one pack per
optimizer step; the only accepted consumed-pack mapping is `{1: 1, 2: 2}`. The
production shape is intentionally not generalized to arbitrary accumulation.
The exact rendered bytes are bound by the three manifest SHA-256 values and
reproduce:

```yaml
# uninterrupted_control.yaml
schema_version: 1
extends: /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml
run: {name: uninterrupted_control, artifact_root: /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/runs, collision_policy: fail}
runtime: {determinism: {mode: strict_cuda_replay_v1}}
eval: {forward: {steps: []}}
training: {max_steps: 2}
checkpoint: {steps: [1, 2], save_final: true}
resume: {mode: exact_same_world_size, checkpoint_dir: null}
```

`resumed_parent.yaml` is byte-identical except `run.name: resumed_parent`.
`resumed_child.yaml` has `run.name: resumed_child` and
`resume.checkpoint_dir` set to
`.../reconcile_exact_resume_2026-08-13-r8/runs/resumed_parent/checkpoints/step-1`.
The full canonical `yaml.safe_dump(sort_keys=False, allow_unicode=False)` byte
streams were hash-bound above; the compact display is explanatory only.

## Exact six-command order

```text
1 setup
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env -u CUDA_VISIBLE_DEVICES /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py prepare --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8 --base-config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml --world-size 2
2 success.uninterrupted_control
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-control --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8 --commit f492f36874f036ad4145a45c7b03cd2e0b2fd049
3 success.resumed_child
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-resumed --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8 --commit f492f36874f036ad4145a45c7b03cd2e0b2fd049
4 rank_failure
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py rank-failure --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/arms/rank_failure --inject {"rank":1,"kind":"missing"} --world-size 2
5 interruption
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8/arms/interruption --stop-after 1 --world-size 2
6 verification
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8
```

Control and child use only GPUs 6,7. Setup, rank failure, interruption, and
verification are model-free single-process commands. Held parent first calls
the real checkpoint handler, then blocks after committed step 1; its durable
run state is `initialized` with null `completed_at`.

## Cost, evidence, and stop contract

| Command | CPU mode/ranks | GPU ranks | Wall | CPU | GPU/rank | Artifact |
|---|---|---|---:|---:|---:|---:|
| setup | aggregate/`[]` | `[]` | 600s | 16GiB | 0 | 2GiB |
| control | per-rank/`[0,1]` | `[0,1]` | 600s | 16GiB/rank | 32GiB | 2.5GiB |
| resumed | per-rank/`[0,1]` | `[0,1]` | 1200s | 16GiB/rank | 32GiB | 3.5GiB |
| rank failure | aggregate/`[]` | `[]` | 120s | 4GiB | 0 | 64MiB |
| interruption | aggregate/`[]` | `[]` | 120s | 4GiB | 0 | 64MiB |
| verification | aggregate/`[]` | `[]` | 600s | 16GiB | 0 | 64MiB |

Aggregate CPU evidence is the maximum concurrent owned PID/starttime tree RSS
snapshot and needs an owned sample. GPU evidence is exclusively target-bound
`torch_allocator_high_water` from signed success receipt + `run.json` +
canonical `logging.jsonl`, ranks `[0,1]`; NVML process rows are observational.
All commands stop on the first mismatch, with `O_EXCL` marker creation and zero
retries. Total bounds are 3240 seconds and 10GiB; summary limits are 100000
entries, depth 16, path 4096 bytes, and 10GiB under only target/cache/cache
receipt. Invalid review, config, accumulation, map, occupancy, artifact,
receipt, state, topology, row, metric, cleanup, or bound fails closed.

The manifest binds this packet hash. A later non-symlink immutable signed
Attempt-8 review binds manifest and packet hashes, commit, `READY`, and an
independent reviewer; the manifest intentionally does not bind a review hash.
This narrow packet does not claim production interruption writing,
cross-world-size or mid-accumulation resume, performance, model quality, or a
production launch.
