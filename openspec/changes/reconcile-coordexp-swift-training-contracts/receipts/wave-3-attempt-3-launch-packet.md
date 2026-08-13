# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 3

Status: `not_requested`; no command in this successor manifest has executed.
Attempts 1 and 2 are immutable failures. Attempt 1 stopped during model-free
setup because an empty `CUDA_VISIBLE_DEVICES` is not the same as an absent one.
Attempt 2 stopped after the parent completed step 2, when production read-only
admission correctly rejected parent step 1 as no longer the latest completed
publication event; the child performed no state apply or forward. This packet
retains both failure records and uses a new target, commit, and controller that
interrupts the parent after authoritative step-1 production admission.

## Frozen identity

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `037ab6683f9eeeb99157960f9fcf5bb3176a7044`
- Command manifest: `receipts/wave-3-attempt-3-command-manifest.json`
- Command-manifest SHA256: `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r3`
- Existing direct parent: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift`
- Private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r3.pack-cache`
- Private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r3.pack-cache-receipt.json`
- World size: `2`; visible physical GPUs for model commands: `6,7` only.
- GPU 6 UUID: `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff`; preflight `0 MiB` used.
- GPU 7 UUID: `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; preflight `0 MiB` used.
- Filesystem preflight: `/data` has `1,721,310,781,440` bytes available.

Repeat occupancy and disk checks immediately before setup and before the first
model command. Shared use is acceptable, but either selected device above
`20 GiB` used memory or free disk below `20 GiB` stops without retry.

## Setup and frozen configs

The model-free setup argv retains `/usr/bin/env -u CUDA_VISIBLE_DEVICES`.
The two model commands use exactly `CUDA_VISIBLE_DEVICES=6,7`; all other
model-free commands retain an empty mapping. Immediately after setup and before
model launch, verify all three actual config digests, the signed prepare
receipt, implementation commit, resolved fingerprints, and completed private-
cache receipt. Any mismatch stops without retry.

| Role | Expected SHA256 | Configured steps/checkpoints | Train forwards per rank | Applied updates per rank |
|---|---|---|---:|---:|
| `uninterrupted_control` | `f554f887ca2642db68506b11546ac8810b4ab3c0b714a88569b2c01321d5ed50` | `max_steps=2`; checkpoints `[1,2]` | 2 total: step 1 boundary plus compared step 2 | 2 total; exactly 1 after boundary |
| `resumed_parent` | `3c3a0203f94344db91a576a80b6cf2f920c1c91c1cec4cf24ac7a3f477267da8` | `max_steps=2`; checkpoints `[1,2]`; controller stops after production admission of authoritative step 1 | exactly 1 at step 1 | exactly 1 at step 1 |
| `resumed_child` | `07add7502de73f67b90e4a2c2d585ed9b5cdb5ed27d218514c4164428c8bca5e` | `max_steps=2`; checkpoints `[1,2]`; restores parent step 1 | exactly 1 at step 2 | exactly 1 at step 2 |

The resumed command therefore performs exactly two train forwards and two
applied updates per rank total: one parent forward/update at step 1 plus one
child forward/update at step 2. It does not authorize a third parent forward or
update.

## Quantitative bounds

These are safety ceilings, not efficiency or performance claims.

| Command / arm | Model-forward ceiling | Applied-update ceiling | Collective-round ceiling per rank | Wall-time limit | CPU RSS per rank | GPU reserved per rank | New bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| model-free `prepare` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 2 GiB cache + receipt/config bundle |
| `uninterrupted_control` | 2/rank | 2/rank, one after boundary | 64 | 600 s | 16 GiB | 32 GiB | 2.5 GiB |
| `resumed_child` command (parent + child) | 2/rank total: 1 + 1 | 2/rank total: 1 + 1 | 128 | 1200 s | 16 GiB | 32 GiB | 3.5 GiB |
| representative `rank_failure` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| representative `interruption` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| durable `verify` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 64 MiB receipt overhead |

- Total wall-time ceiling: `3,240 s`.
- Total new-byte ceiling: `10 GiB`.
- Required free disk: `20 GiB` before setup and model launch.
- Collective ceilings are conservative execution-stop bounds from the frozen
  step/checkpoint counts, not efficiency claims. Timeout, divergent rank, or
  any observed bound exceedance terminates the packet without retry.

## Parent interruption ordering

The resumed command must first obtain real production read-only admission of
the parent's authoritative step-1 exact-resume publication. It then terminates
the dedicated parent process group with controlled `SIGTERM`, escalating to
`SIGKILL` only if the bounded grace period expires, and proves the entire group
has exited. Only after termination does it re-admit parent step 1 and require
the authority record to be unchanged. The child may launch only after that
post-termination admission and parent-progress validation succeed. An early
parent exit, admission timeout or mismatch, surviving process-group member,
non-signal-driven parent exit, cleanup failure, or child launch before those
checks complete stops the packet without retry.

## Comparison and claim boundary

The verifier admits and compares control step 1 against parent step 1 before
the corresponding next forward, then control step 2 against resumed-child step
2 after each branch's compared forward/update. It requires exact equality for
all production-strict identities except role-specific `resolved_config`, plus
cursor/next-pack, trainable, optimizer, scheduler, scaler, all rank RNG,
topology, structure, and semantic step-2 objective/loss/update fields.
`input_build_seconds`, `input_wait_seconds`, rank measurements, and resource/
timing fields are excluded; semantic training fields remain exact.

The failure arm remains representative `rank=1, kind=missing`; interruption
remains representative `stop_after=1` with `stub_not_production`. The model-free
failure/boundary gate covers its existing matrix. This packet does not qualify
the production inference-writer interruption boundary, cross-world-size or
mid-accumulation resume, efficiency, or model quality.

## Exact execution and stop rule

Execute each frozen argv at most once and only in this order: `setup_command`
(`prepare`), `success.uninterrupted_control` (`control`),
`success.resumed_child` (`resumed`), `rank_failure`, `interruption`, then
`verification_command` (`verify`). The two success commands are the only
model/GPU commands. Record start/end observations, exits, command hashes,
canonical per-rank maxima, total artifact bytes, controller admission and
termination evidence, and the signed terminal result.

There is no retry. Stop on Git/manifest/config/path/GPU drift; an occupied
target, cache, or cache-receipt path; insufficient disk or GPU headroom; setup,
cache, receipt, fingerprint, or digest failure; command duplication or ordering
violation; parent boundary/admission/termination/cleanup failure; unexpected
forward, update, collective, step, pack, or checkpoint-event count; timeout,
hang, OOM, rank loss, nonzero exit, or any resource-bound exceedance; missing
rank measurements; comparison or verifier mismatch; or an absent/invalid signed
terminal receipt. Verification success requires exit 0 and signed
`terminal-receipt.json` status `verified`. No deletion, repair, alternate
command, partial-success promotion, or interpretation beyond the stated claim
boundary is authorized.

## Pre-cost review and execution authority

Execution remains pending one Sol/xhigh pre-cost review of this exact packet;
`authorization_status: not_requested` records that pending review and does not
claim pre-cost approval. If that review returns READY, the lead-only executor
may proceed under the current goal-level authority without another user prompt.
The executor must bind this exact packet path and command-manifest digest in the
terminal receipt. Any packet edit invalidates the review and requires the exact
edited packet to be reviewed again.
