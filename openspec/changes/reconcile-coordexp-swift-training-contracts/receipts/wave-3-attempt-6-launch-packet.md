# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 6

Status: `frozen_pending_external_review`. No command in this packet has run.
Attempts 1–5, including Attempt 5's signed `HOLD`, are immutable evidence and
are neither reused nor modified. The executor is outside the six commands.

## Frozen identity and new targets

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `1fbd9d68be4cf5dee1be1b09886f7c8a7b84205e`
- Packet author identity: `coordexp-attempt6-packet-author`
- Manifest: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-6-command-manifest.json`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA-256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6`
- Absent private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r6.pack-cache`
- Absent private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r6.pack-cache-receipt.json`
- Independent review path (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-6-pre-cost-review.json`
- Exclusive marker (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-6-attempt-marker.json`
- Outer receipt (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-6-outer-terminal-receipt.json`
- Inner receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6/terminal-receipt.json`

Physical GPU `6` is bound to `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff` and
physical GPU `7` to `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`. Live
read-only preflight observed 0 MiB on each and 1,723,438,596,096 free bytes on
`/data`; the executor revalidates map, occupancy, and disk before marker and
again before the first GPU command.

## Deterministic config and accumulation binding

The role configs were rendered in a temporary directory by the production
`_write_role_config` path against the r6 root, loaded by production
`load_train_config`, and resolved by
`resolve_effective_batch_runtime(config, world_size=2)`. Their exact values are:

| Role | Generated config SHA-256 | Resolved fingerprint | EBS | Grad accum |
|---|---|---|---:|---:|
| `uninterrupted_control` | `0ae0b4c75bfb8bb56f105d228d4e350ce10abaaa277f400acbc0dadef5c18cdf` | `17eb444c434f9d22b508d2e5c29e87ccf2ba5d375cb50a7bf6f2e99e2e630494` | 2 | 1 |
| `resumed_parent` | `2df798c29880b7d9ffa1054578cf20635812d8b2d18ea88eb3e57de05715c7eb` | `7c4fc68b06fc0a4be95b8fb520993d23b4598e73554f0b06345c61859a7da5e1` | 2 | 1 |
| `resumed_child` | `dcf9b57bd28ae0484e252efb09d791ba922cf1953c3e58bbcd8b837368fc858e` | `be3bb2fc3c127b32a0c32e5599f60f8474f3d0cb8e3c1ea89845949507443ea4` | 2 | 1 |

Thus the fixed topology (`world_size=2`) makes one pack per optimizer step;
the only accepted consumed-pack mapping is `{1: 1, 2: 2}`. The production
shape is intentionally not generalized to arbitrary accumulation. The exact
rendered bytes are bound by the three manifest SHA-256 values and reproduce:

```yaml
# uninterrupted_control.yaml
schema_version: 1
extends: /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml
run: {name: uninterrupted_control, artifact_root: /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6/runs, collision_policy: fail}
runtime: {determinism: {mode: strict_cuda_replay_v1}}
eval: {forward: {steps: []}}
training: {max_steps: 2}
checkpoint: {steps: [1, 2], save_final: true}
resume: {mode: exact_same_world_size, checkpoint_dir: null}
```

`resumed_parent.yaml` is byte-identical except `run.name: resumed_parent`.
`resumed_child.yaml` has `run.name: resumed_child` and
`resume.checkpoint_dir` set to
`.../reconcile_exact_resume_2026-08-13-r6/runs/resumed_parent/checkpoints/step-1`.
The full canonical `yaml.safe_dump(sort_keys=False, allow_unicode=False)` byte
streams were hash-bound above; the compact display is explanatory only.

## Exact six-command order

```text
1 setup
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env -u CUDA_VISIBLE_DEVICES /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py prepare --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6 --base-config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml --world-size 2
2 success.uninterrupted_control
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-control --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6 --commit 1fbd9d68be4cf5dee1be1b09886f7c8a7b84205e
3 success.resumed_child
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-resumed --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6 --commit 1fbd9d68be4cf5dee1be1b09886f7c8a7b84205e
4 rank_failure
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py rank-failure --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6/arms/rank_failure --inject {"rank":1,"kind":"missing"} --world-size 2
5 interruption
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6/arms/interruption --stop-after 1 --world-size 2
6 verification
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r6
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
Attempt-6 review binds manifest and packet hashes, commit, `READY`, and an
independent reviewer; the manifest intentionally does not bind a review hash.
This narrow packet does not claim production interruption writing,
cross-world-size or mid-accumulation resume, performance, model quality, or a
production launch.
