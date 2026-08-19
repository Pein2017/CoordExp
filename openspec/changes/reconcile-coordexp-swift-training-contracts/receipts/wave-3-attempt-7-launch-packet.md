# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 7

Status: `frozen_pending_external_review`. No command in this packet has run.
Attempts 1–6, including Attempt 5's signed `HOLD` and Attempt 6's signed
`READY` that was invalidated pre-marker by tracked-HEAD movement
(`27abc8087e...`), are immutable evidence and are neither reused nor modified.
No identity in this packet is copied from an earlier attempt: the three role
configs were deterministically re-rendered against the new `-r7` root and
re-hashed, and the commit binding is the current clean HEAD. The executor is
outside the six commands.

## Frozen identity and new targets

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `4525a0f73dedc1f31bb88c59ed18f95d22930531`
- Implementation tree (`HEAD^{tree}`): `cbe7925a64804f259c52108cccc879683f312fee`
- Packet author identity: `coordexp-attempt7-packet-author`
- Manifest: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-7-command-manifest.json`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA-256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7`
- Absent private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r7.pack-cache`
- Absent private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r7.pack-cache-receipt.json`
- Independent review path (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-7-pre-cost-review.json`
- Exclusive marker (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-7-attempt-marker.json`
- Outer receipt (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-7-outer-terminal-receipt.json`
- Inner receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7/terminal-receipt.json`

The `-r7` target name keeps the change-local `2026-08-13` attempt-series date
deliberately: it is the successor slot Attempt 6's retirement pinned, not a
calendar claim about the freeze day.

Physical GPU `6` is bound to `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff` and
physical GPU `7` to `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; the live map
was re-read at freeze time and is unchanged from the Attempt-6 host state.
At freeze time both devices carried transient external shared-host load
(50,637 MiB and 15,827 MiB from host-namespace PIDs), so this packet is
explicitly not executable until they drain: the executor revalidates map,
occupancy (max 21,474,836,480 bytes per device), and disk before marker and
again before the first GPU command, and the lead will invoke `execute` only
after independently observing both devices within the occupancy bound. Live
free space on `/data` at freeze was 1,717,320,454,144 bytes.

## Deterministic config and accumulation binding

The role configs were rendered in a temporary directory by the production
`_write_role_config` path against the r7 root, loaded by production
`load_train_config`, and resolved by
`resolve_effective_batch_runtime(config, world_size=2)`. Their exact values are:

| Role | Generated config SHA-256 | Resolved fingerprint | EBS | Grad accum |
|---|---|---|---:|---:|
| `uninterrupted_control` | `87d9ef6fe6dc13992985a9fedcf95ba06ca8b06be3c046e41be9a897fc838dad` | `6ab0dc8f290c67fb7425622434b20f1d5463bd153d7bdcff7d27f62250a6b89c` | 2 | 1 |
| `resumed_parent` | `c4814ceb2a63b5c0b24e756570d2bdd080f429d310b14f1bf0a875d2193ab5e1` | `89fb89e3e49a84322b3ad0351af9e81507323b5f91fc956bd9d0691d4b907451` | 2 | 1 |
| `resumed_child` | `bac865616132b54e4498605df6ba22eb3df8af946945d07ee478fafac641b349` | `cd42089fa92a2154db8d8953de3079df28598206789c3a753b72bb6bc055e4e2` | 2 | 1 |

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
run: {name: uninterrupted_control, artifact_root: /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7/runs, collision_policy: fail}
runtime: {determinism: {mode: strict_cuda_replay_v1}}
eval: {forward: {steps: []}}
training: {max_steps: 2}
checkpoint: {steps: [1, 2], save_final: true}
resume: {mode: exact_same_world_size, checkpoint_dir: null}
```

`resumed_parent.yaml` is byte-identical except `run.name: resumed_parent`.
`resumed_child.yaml` has `run.name: resumed_child` and
`resume.checkpoint_dir` set to
`.../reconcile_exact_resume_2026-08-13-r7/runs/resumed_parent/checkpoints/step-1`.
The full canonical `yaml.safe_dump(sort_keys=False, allow_unicode=False)` byte
streams were hash-bound above; the compact display is explanatory only.

## Exact six-command order

```text
1 setup
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env -u CUDA_VISIBLE_DEVICES /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py prepare --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7 --base-config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml --world-size 2
2 success.uninterrupted_control
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-control --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7 --commit 4525a0f73dedc1f31bb88c59ed18f95d22930531
3 success.resumed_child
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-resumed --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7 --commit 4525a0f73dedc1f31bb88c59ed18f95d22930531
4 rank_failure
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py rank-failure --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7/arms/rank_failure --inject {"rank":1,"kind":"missing"} --world-size 2
5 interruption
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7/arms/interruption --stop-after 1 --world-size 2
6 verification
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r7
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
Attempt-7 review binds manifest and packet hashes, commit, `READY`, and an
independent reviewer; the manifest intentionally does not bind a review hash.
This narrow packet does not claim production interruption writing,
cross-world-size or mid-accumulation resume, performance, model quality, or a
production launch.
