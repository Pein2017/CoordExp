# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 5

Status: `frozen_pending_external_review`; no command in this packet has
executed. Attempts 1–4 and every associated receipt and produced artifact are
immutable historical evidence. The executor is not one of the six commands.

## Frozen identity and authority

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `9ef726085d2118dca90b373ccde4a88d068bf516`
- Packet author identity: `coordexp-attempt5-packet-author`
- Command manifest: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-5-command-manifest.json`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA-256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5`
- Private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r5.pack-cache`
- Private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r5.pack-cache-receipt.json`
- Independent pre-cost review (absent until independently published): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-5-pre-cost-review.json`
- Exclusive attempt marker (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-5-attempt-marker.json`
- Outer terminal receipt (absent): `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-5-outer-terminal-receipt.json`
- Inner verifier receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5/terminal-receipt.json`
- World size: `2`; the only visible GPUs for the two success commands are
  physical `6,7`, mapped before marker creation and before the first GPU
  command to `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff` and
  `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`, respectively.

The manifest intentionally binds this packet's final SHA-256, while this
packet does not bind a manifest SHA-256. The independent review binds both
finalized hashes; the manifest never binds a review hash, avoiding a circular
freeze. Manifest self-`READY` has no authority.

## Deterministic roles and exact six-command order

Expected rendered configs use `yaml.safe_dump(sort_keys=False,
allow_unicode=False)` and have SHA-256 values `08eae66e9e7476addf747dcb40cba759a02e3c05051cf3aa156eac2dc30b6fa8`
(`uninterrupted_control`), `7e24e6bf427575fd18c22a6a4d506d02cbe6bcbbc9f9f0da5104778582f5f838`
(`resumed_parent`), and `2daf76ee7b9aab49cd561c026ccbb1ce4dbd58036857455bdaefaad0c30b1393`
(`resumed_child`). All roles retain `max_steps=2` and checkpoints `[1,2]`.
The held parent calls the real checkpoint handler before blocking after its
committed step 1; control and child use `-m src.train` through the probe.

```text
1 setup
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env -u CUDA_VISIBLE_DEVICES /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py prepare --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5 --base-config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml --world-size 2

2 success.uninterrupted_control
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-control --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5 --commit 9ef726085d2118dca90b373ccde4a88d068bf516

3 success.resumed_child
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-resumed --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5 --commit 9ef726085d2118dca90b373ccde4a88d068bf516

4 rank_failure
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py rank-failure --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5/arms/rank_failure --inject {"rank":1,"kind":"missing"} --world-size 2

5 interruption
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5/arms/interruption --stop-after 1 --world-size 2

6 verification
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r5
```

`rank_failure` must produce its v2 receipt with expected and serialized ranks
`[0,1]`, published `[0]`, and error `training_state.incomplete_rank_set`.
`interruption` must produce its v2 receipt with expected `[0,1]`, serialized
and published `[]`, and `rank_state_boundary_reached: false`.

## Resource and evidence contract

The only GPU evidence source is `torch_allocator_high_water`: target-bound
signed success receipt + exact `run.json` + canonical `logging.jsonl` join.
It requires ranks `[0,1]` and finite, nonnegative, integer-valued (not boolean)
allocated and reserved values; the bounded value per rank is their maximum.
NVML process rows are optional observations and never GPU-rank evidence.

| Command | CPU mode / ranks | GPU ranks | Wall | CPU bound | GPU bound | Artifact bound |
|---|---|---:|---:|---:|---:|---:|
| `setup` | command tree aggregate / `[]` | `[]` | 600 s | 16 GiB | 0 | 2 GiB |
| `success.uninterrupted_control` | per rank / `[0,1]` | `[0,1]` | 600 s | 16 GiB/rank | 32 GiB/rank | 2.5 GiB |
| `success.resumed_child` | per rank / `[0,1]` | `[0,1]` | 1200 s | 16 GiB/rank | 32 GiB/rank | 3.5 GiB |
| `rank_failure` | command tree aggregate / `[]` | `[]` | 120 s | 4 GiB | 0 | 64 MiB |
| `interruption` | command tree aggregate / `[]` | `[]` | 120 s | 4 GiB | 0 | 64 MiB |
| `verification` | command tree aggregate / `[]` | `[]` | 600 s | 16 GiB | 0 | 64 MiB |

Aggregate CPU evidence is the maximum concurrent sum in an owned
PID/starttime command-tree sampler snapshot, and requires at least one owned
sample. It is not a sum of individual process high-water marks. Total ceilings
are 3,240 seconds and 10 GiB. The preflight requires 20 GiB free disk, 20 GiB
maximum selected-GPU occupancy, and artifact-tree limits of 100,000 entries,
depth 16, path length 4,096 bytes, and 10 GiB over only the artifact root,
private cache root, and cache receipt.

Control requires a signed control receipt, completed world size 2 run at step
2, and exactly one train row at each of steps 1 and 2. Resumed success requires
a signed parent-and-child receipt, parent controlled exit at exactly step 1
with one train row, and child completed world size 2 at step 2 with one train
row; its maxima merge both lifetimes. Any binding, topology, state, row,
measurement, bound, cleanup, summary, map, review, or path failure stops with
no retry.

## Review gate and claim boundary

Before marker creation the executor must validate an immutable non-symlink
regular review file at the bound path with schema
`coordexp-swift-reconcile-resume-probe-pre-cost-review-v1`, `READY`, this
implementation commit, finalized manifest and packet hashes, an independent
reviewer identity distinct from `coordexp-attempt5-packet-author`, and a valid
`receipt_payload_sha256`; it retains the file identity through `O_EXCL` marker
claim. The executor owns each process group, always TERM/KILL/reaps it, and
proves PID/starttime-safe absence before terminal success.

This packet can establish the narrow exact-resume comparison only: it does not
qualify production interruption writing, cross-world-size or
mid-accumulation resume, efficiency, model quality, or a production launch.
