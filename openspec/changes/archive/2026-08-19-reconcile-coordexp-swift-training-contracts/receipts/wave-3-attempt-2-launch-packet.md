# Wave 3 Two-Rank Exact-Resume Launch Packet — Attempt 2

Status: `not_requested`; no command in this successor manifest has executed.
Attempt 1 is immutable under `receipts/wave-3-attempt-1-*` and stopped before
GPU because an empty `CUDA_VISIBLE_DEVICES` is not the same as an absent one.
This packet changes only setup to `/usr/bin/env -u CUDA_VISIBLE_DEVICES`, uses a
new target and commit, and retains the reviewed success/failure/verification
semantics.

## Frozen identity

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `47dd04a29828c90fe0f7507acbf41284733bd5d1`
- Command manifest: `receipts/wave-3-command-manifest.json`
- Command-manifest SHA256: `6b63caa3eefba86715f72330f7dc3d50f46299cfccbb2786fe9074f209be6836`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r2`
- Existing direct parent: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift`
- Private cache root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r2.pack-cache`
- Private cache receipt: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-13-r2.pack-cache-receipt.json`
- World size: `2`; visible physical GPUs for model commands: `6,7` only.
- GPU 6 UUID: `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff`; preflight `3 MiB / 81920 MiB`, `0%` utilization.
- GPU 7 UUID: `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; preflight `3 MiB / 81920 MiB`, `0%` utilization.
- Filesystem preflight: `/data` has `1,723,254,804,480` bytes available.

Repeat occupancy and disk checks immediately before setup and before the first
model command. Shared use is acceptable, but either selected device above
`20 GiB` used memory or free disk below `20 GiB` stops without retry.

## Setup correction and pre-authorized configs

The production test
`test_production_cache_environment_crosses_real_strict_cpu_seam` passes only
the model-free single-rank seam with `CUDA_VISIBLE_DEVICES` absent. A direct
read-only probe confirmed that `/usr/bin/env -u CUDA_VISIBLE_DEVICES` yields
`_launcher_device_mapping(rank=0, world_size=1).cuda_visible_devices is None`.
An empty string is forbidden and is not used anywhere in this setup argv.

| Role | Expected SHA256 | Train forwards per rank | Applied updates per rank |
|---|---|---:|---:|
| `uninterrupted_control` | `a02f2352248e0240763074f420f8aaac7a3020eb0ef5005396ca3aa0e09c0a01` | 2 total: step 1 boundary plus compared step 2 | 2 total; exactly 1 after boundary |
| `resumed_parent` | `4ef86048c29bf17839ef9f0480fdf4c2c547a05536d86cb8f51230c29fb55ef1` | 2 total; step 1 supplies the compared boundary | 2 total; only step 1 is the resume source |
| `resumed_child` | `fe2001312ed876a1d49fdad00b5a48830eba26b184707db4fd147fb5fe9d5bdc` | 1 compared step after restoring parent step 1 | exactly 1 after boundary |

Immediately after setup and before model launch, verify all three actual file
digests, signed prepare receipt, implementation commit, resolved fingerprints,
and completed private-cache receipt. Any mismatch stops without retry.

## Quantitative bounds

These are safety ceilings, not performance claims.

| Command / arm | Model-forward ceiling | Applied-update ceiling | Collective-round ceiling per rank | Wall-time limit | CPU RSS per rank | GPU reserved per rank | New bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| model-free `prepare` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 2 GiB cache + receipt/config bundle |
| `uninterrupted_control` | 2/rank | 2/rank, one after boundary | 64 | 600 s | 16 GiB | 32 GiB | 2.5 GiB |
| `resumed_child` command (parent + child) | 3/rank total: 2 + 1 | 3/rank total: 2 + 1 | 128 | 1200 s | 16 GiB | 32 GiB | 3.5 GiB |
| representative `rank_failure` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| representative `interruption` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| durable `verify` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 64 MiB receipt overhead |

- Total wall-time ceiling: `3,240 s`.
- Total new-byte ceiling: `10 GiB`.
- Required free disk: `20 GiB` before setup and model launch.
- Collective ceilings are conservative execution-stop bounds from the frozen
  step/checkpoint counts, not an efficiency claim. Timeout, divergent rank, or
  any observed bound exceedance terminates the packet without retry.

## Comparison and claim boundary

The verifier admits and compares control step 1 against parent step 1 before
the corresponding next forward, then control step 2 against resumed-child step
2 after each branch's compared forward/update. It requires exact equality for
all production-strict identities except role-specific `resolved_config`, plus
cursor/next-pack, trainable, optimizer, scheduler, scaler, all rank RNG,
topology, structure, and semantic step-2 objective/loss/update fields.
`input_build_seconds`, `input_wait_seconds`, rank measurements, and resource/
timing fields are excluded; semantic training fields remain exact.

The failure arm is representative `rank=1, kind=missing`; interruption is
representative `stop_after=1` with `stub_not_production`. The 119-test gate
covers the full model-free failure/boundary matrix. This packet does not qualify
the production inference-writer interruption boundary, cross-world-size or
mid-accumulation resume, efficiency, or model quality.

## Exact execution and stop rule

Execute each frozen argv once in order: `setup_command`,
`success.uninterrupted_control`, `success.resumed_child`, `rank_failure`,
`interruption`, `verification_command`. The two success commands are the only
model/GPU commands. Record start/end observations, exits, command hashes,
canonical per-rank maxima, total artifact bytes, and the signed terminal result.

Stop without retry on Git/manifest/config/path/GPU drift; occupied target or
cache paths; insufficient headroom; receipt failure; unexpected forward/update
count; timeout, hang, OOM, rank loss, nonzero exit; bound exceedance; missing
rank measurements; or verifier mismatch. Verification success requires exit 0
and signed `terminal-receipt.json` status `verified`. No deletion, repair,
alternate command, or partial-success promotion is authorized.

## Fresh authorization required

Attempt-1 approval does not authorize this successor. After a narrow
independent pre-cost review is READY, present this exact commit, manifest digest,
target, GPUs, setup correction, and unchanged limits for fresh authorization.
Any edit after approval invalidates it.
