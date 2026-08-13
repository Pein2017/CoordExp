# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 4

Status: `frozen_pending_external_review`; no command in this packet has
executed. Attempts 1–3 and all of their receipts remain immutable historical
evidence. Attempt 4 is bound to the post-repair implementation and the
schema-v2 packet executor. The executor itself is invoked separately by the
lead and is not one of the six inner commands.

## Frozen identity

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e`
- Packet author identity: `coordexp-attempt4-packet-author`
- Command manifest: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-4-command-manifest.json`
- This packet intentionally does not bind the manifest hash. The schema-v2
  manifest binds this packet's final path and SHA-256, and the independent
  review later binds both finalized hashes without a circular dependency.
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA-256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4`
- Private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r4.pack-cache`
- Private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r4.pack-cache-receipt.json`
- Independent pre-cost review: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-4-pre-cost-review.json`
- Exclusive attempt marker: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-4-attempt-marker.json`
- Outer terminal receipt: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-4-outer-terminal-receipt.json`
- Inner verifier receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4/terminal-receipt.json`
- World size: `2`; visible physical GPUs for the two success commands:
  `6,7` only.
- Physical GPU 6: `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff`; frozen
  observation `0 MiB` used.
- Physical GPU 7: `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; frozen
  observation `0 MiB` used.
- Frozen filesystem observation supplied at authoring: `/data` had
  `1,720,460,230,656` bytes available.

The artifact root, private cache root, private cache receipt, exclusive marker,
and outer terminal receipt must all remain absent until the one-shot executor
owns their creation. The review path must remain absent until the independent
reviewer publishes the immutable signed receipt. Repeat GPU mapping,
occupancy, disk, Git, path, manifest, packet, and review checks before marker
creation; repeat the GPU mapping/occupancy and disk checks before the first
GPU command. Shared use is acceptable, but either selected device above
`20 GiB` used memory or free disk below `20 GiB` stops without retry.

## Deterministic setup and expected configs

Setup is CPU-only and starts with `/usr/bin/env -u CUDA_VISIBLE_DEVICES`.
It authors the immutable config bundle and private model-free pack cache only
after the executor validates the frozen identities and independent review.
The expected config bytes below were rendered in memory with the current probe
role overrides and `yaml.safe_dump(sort_keys=False, allow_unicode=False)`;
no config, cache, receipt, marker, or artifact root was created while freezing
this packet.

| Role | Expected SHA-256 | Configured steps/checkpoints | Train forwards per rank | Applied updates per rank |
|---|---|---|---:|---:|
| `uninterrupted_control` | `08eae66e9e7476addf747dcb40cba759a02e3c05051cf3aa156eac2dc30b6fa8` | `max_steps=2`; checkpoints `[1,2]` | 2 total: step 1 boundary plus compared step 2 | 2 total; exactly 1 after boundary |
| `resumed_parent` | `7e24e6bf427575fd18c22a6a4d506d02cbe6bcbbc9f9f0da5104778582f5f838` | `max_steps=2`; checkpoints `[1,2]`; synchronous held-parent route blocks after the real handler commits step 1 | exactly 1 at step 1 | exactly 1 at step 1 |
| `resumed_child` | `2daf76ee7b9aab49cd561c026ccbb1ce4dbd58036857455bdaefaad0c30b1393` | `max_steps=2`; checkpoints `[1,2]`; restores parent step 1 | exactly 1 at step 2 | exactly 1 at step 2 |

Immediately after setup and before the first model command, the executor must
validate the signed `prepare-receipt.json`, implementation commit, all three
actual config digests and resolved fingerprints, completed private-cache
receipt, exact cache paths, GPU mapping, occupancy, and disk headroom. Any
mismatch stops without retry.

## Exact six-command order

Each argv may run at most once and only in this order. The two success commands
are the only GPU/model commands. Setup, rank failure, interruption, and
verification remain ordinary single-process, model-free commands; they must
not be converted to `torchrun` or assigned synthetic rank identities.

### 1. `setup`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env -u CUDA_VISIBLE_DEVICES /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py prepare --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4 --base-config /data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml --world-size 2
```

### 2. `success.uninterrupted_control`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-control --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4 --commit 5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e
```

### 3. `success.resumed_child`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s /usr/bin/env CUDA_VISIBLE_DEVICES=6,7 /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py success-resumed --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4 --commit 5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e
```

### 4. `rank_failure`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py rank-failure --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4/arms/rank_failure --inject {"rank":1,"kind":"missing"} --world-size 2
```

The signed schema-v2 arm receipt must record exact
`expected_ranks: [0,1]`, `serialized_ranks: [0,1]`,
`published_ranks: [0]`, and
`error_code: training_state.incomplete_rank_set`.

### 5. `interruption`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 120s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py interruption --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4/arms/interruption --stop-after 1 --world-size 2
```

The signed schema-v2 arm receipt must record exact
`expected_ranks: [0,1]`, `serialized_ranks: []`,
`published_ranks: []`, and `rank_state_boundary_reached: false`.

### 6. `verification`

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 600s /usr/bin/env CUDA_VISIBLE_DEVICES= /root/miniconda3/bin/conda run -n ms /root/miniconda3/envs/ms/bin/python scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py verify --artifact-root /data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r4
```

## Quantitative execution contract

These are conservative safety ceilings, not efficiency or performance claims.
`per_rank` CPU evidence is required only for the two real two-rank success
commands. For the other four commands, `command_tree_aggregate` requires at
least one exact owned PID/starttime sample and bounds
`cpu_rss_command_tree_max_bytes`, the maximum concurrent RSS sum across the
owned command tree in any one sampler snapshot. It is never a sum of
independent per-PID high-water marks.

| Command | CPU mode / required CPU ranks | Required GPU ranks | Wall | CPU bound | GPU bound per rank | New artifact bytes |
|---|---|---|---:|---:|---:|---:|
| `setup` | `command_tree_aggregate` / `[]` | `[]` | 600 s | 16 GiB command tree | 0 | 2 GiB |
| `success.uninterrupted_control` | `per_rank` / `[0,1]` | `[0,1]` | 600 s | 16 GiB/rank | 32 GiB/rank | 2.5 GiB |
| `success.resumed_child` | `per_rank` / `[0,1]` | `[0,1]` | 1200 s | 16 GiB/rank | 32 GiB/rank | 3.5 GiB |
| `rank_failure` | `command_tree_aggregate` / `[]` | `[]` | 120 s | 4 GiB command tree | 0 | 64 MiB |
| `interruption` | `command_tree_aggregate` / `[]` | `[]` | 120 s | 4 GiB command tree | 0 | 64 MiB |
| `verification` | `command_tree_aggregate` / `[]` | `[]` | 600 s | 16 GiB command tree | 0 | 64 MiB |

- Total wall-time ceiling: `3,240 s`.
- Total new-artifact ceiling: `10 GiB`.
- Required free disk before setup and the first GPU command: `20 GiB`.
- Artifact-tree summaries cover only the artifact root, private cache root,
  and private cache receipt, with ceilings of `100,000` entries, depth `16`,
  path length `4,096` bytes, and `10 GiB` total bytes.
- Model-forward, update, and collective ceilings remain: setup/failure/
  interruption/verification `0`; control `2` forwards and `2` updates per
  rank with at most `64` collective rounds; resumed parent+child `2` forwards
  and `2` updates per rank total with at most `128` collective rounds.

## Held-parent ordering and process ownership

Only `success-resumed` routes the parent through the probe-local synchronous
held-parent entry. That entry wraps and first calls the real pipeline
checkpoint handler; only after successful authoritative step-1 commit does it
block so the trainer cannot enter step 2. Control and resumed child continue
to launch `src.train`. Handler failure propagates without holding. The bounded
controller terminates and reaps the dedicated parent process group, proves
PID/starttime-safe group absence, re-admits the unchanged step-1 authority,
and only then launches the child.

The outer executor owns exactly one returned Popen-like process and one
process group at a time. Every normal, sampler, artifact-summary, timeout,
bound, nonzero-exit, or exception path must TERM, KILL if necessary, reap, and
prove group absence before a successful terminal result can be written. GPU
evidence counts only for the selected UUID and an exact PID/starttime-owned
descendant row. Foreign, stale, index-swapped, or unowned rows fail closed.

## Comparison and claim boundary

The verifier admits and compares control step 1 against parent step 1 before
the corresponding next forward, then control step 2 against resumed-child
step 2 after each branch's compared forward/update. It requires exact equality
for all production-strict identities except role-specific `resolved_config`,
plus cursor/next-pack, trainable, optimizer, scheduler, scaler, every rank RNG,
topology, structure, and semantic step-2 objective/loss/update fields.
Timing and resource measurements are excluded; semantic training fields remain
exact.

The failure arm is representative `rank=1, kind=missing`; interruption remains
representative `stop_after=1` with `stub_not_production`. The model-free matrix
and schema-v2 receipts own their semantic rank evidence. This packet does not
qualify the production inference-writer interruption boundary, cross-world-
size or mid-accumulation resume, efficiency, model quality, or any production
training launch.

## Review gate and stop rule

Before marker creation, the executor must open the exact review path as one
non-symlink, read-only regular file and validate schema
`coordexp-swift-reconcile-resume-probe-pre-cost-review-v1`, status `READY`,
exact implementation commit, finalized manifest SHA-256, this packet SHA-256,
an independent reviewer identity distinct from
`coordexp-attempt4-packet-author`, and `receipt_payload_sha256`. It retains the
review device/inode identity through exclusive marker acquisition. The
manifest does not bind a review hash. A manifest self-declaration, missing or
writable review, `HOLD`, stale identity, invalid digest, replaced file, or
non-independent reviewer executes nothing and creates no marker.

There is no retry. Stop on Git/manifest/packet/config/path/review/GPU drift;
an occupied target; insufficient disk or GPU headroom; setup/cache/receipt/
fingerprint failure; command duplication/order violation; held-parent or
process-group cleanup failure; unexpected forward/update/collective/event
count; missing required rank evidence or aggregate owned sample; timeout,
hang, OOM, rank loss, nonzero exit, artifact-summary overflow, resource-bound
exceedance, verifier mismatch, or missing signed outer receipt. Exit zero or
an inner verifier receipt without the signed outer terminal receipt is not
qualification evidence.
