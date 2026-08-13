# Wave 3 Two-Rank Exact-Resume Launch Packet

Status: `not_requested`; no command in the bound manifest has executed.

## Frozen identity

- Cwd: `/data/CoordExp/.worktrees/CoordExp-swift`
- Implementation commit: `fcd05b38c12de5b522b5bd29b61bfd15cbacc28d`
- Command manifest: `receipts/wave-3-command-manifest.json`
- Command-manifest SHA256: `39ff5851853ded501e8cba87c1ccc9de80cbbf26b86598814528c5b8aa1c47da`
- Base config: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
- Base-config SHA256: `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
- Absent artifact root: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-12-r1`
- Existing direct parent: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift`
- Private cache root created by the authorized setup command only: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-12-r1.pack-cache`
- Private cache receipt created by the authorized setup command only: `/data/CoordExp/.worktrees/CoordExp-swift/outputs/probes/coordexp_swift/.reconcile_exact_resume_2026-08-12-r1.pack-cache-receipt.json`
- World size: `2`; visible physical GPUs: `6,7` only.
- GPU 6 UUID: `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff`; preflight `0 MiB / 81920 MiB`, `0%` utilization.
- GPU 7 UUID: `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`; preflight `0 MiB / 81920 MiB`, `0%` utilization.
- Filesystem preflight: `/data` has `1,735,257,751,552` bytes available.

The GPU occupancy and free-disk observations above are point-in-time evidence.
They MUST be repeated immediately before setup and again before the first GPU
command. Shared use is acceptable, but either selected device exceeding
`20 GiB` used memory or free disk falling below `20 GiB` stops this packet
without retry.

## Pre-authorized generated-config identities

`prepare` has not run. The following values were computed without creating the
artifact root by applying the qualification tool's deterministic renderer to
the frozen base config and absent target path:

| Role | Expected SHA256 | Train forwards per rank | Applied updates per rank |
|---|---|---:|---:|
| `uninterrupted_control` | `c71c937b924fd795803a920f3c02bd34d41918b7457d0fde8e3797d51f82dc40` | 2 total: step 1 boundary plus compared step 2 | 2 total; exactly 1 after boundary |
| `resumed_parent` | `50cec54512ba0db1f49fcdfe53941ffa03de039d8585a4654651399b9c320a98` | 2 total; step 1 supplies the compared boundary and step 2 proves the parent config is continuation-compatible | 2 total; only step 1 is the resume source |
| `resumed_child` | `2f74f823e79659b6ef61bf6ac6e864374866a088291672d47f9ae4c9c5a80b48` | 1 compared step after restoring parent step 1 | exactly 1 after boundary |

All roles resolve `eval.forward.steps: []`; therefore no eval forward is
included. Immediately after setup and before any GPU command, the executor
MUST verify all three actual file digests, the signed prepare receipt, the
implementation commit, the resolved fingerprints, and the completed private
cache receipt. Any mismatch stops without regeneration or retry.

## Quantitative bounds

These are safety ceilings, not performance claims. They use the frozen step
counts plus conservative margins over the historical eight-rank Wave-7 exact
resume run, whose observed per-rank maxima were about `9.55 GB` CPU RSS and
`15.5 GB` GPU reserved memory and whose one/two-step branches completed well
inside these limits.

| Command / arm | Model-forward ceiling | Applied-update ceiling | Collective-round ceiling per rank | Wall-time limit | CPU RSS per rank | GPU reserved per rank | New bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| model-free `prepare` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 2 GiB cache + receipt/config bundle |
| `uninterrupted_control` | 2/rank | 2/rank, one after boundary | 64 | 600 s | 16 GiB | 32 GiB | 2.5 GiB |
| `resumed_child` command (parent + child) | 3/rank total: 2 + 1 | 3/rank total: 2 + 1 | 128 | 1200 s | 16 GiB | 32 GiB | 3.5 GiB |
| representative `rank_failure` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| representative `interruption` | 0 | 0 | 0 distributed rounds | 120 s | 4 GiB process RSS | 0 | 64 MiB |
| durable `verify` | 0 | 0 | 0 distributed rounds | 600 s | 16 GiB process RSS | 0 | 64 MiB receipt overhead |

- Total packet wall-time ceiling: `3,240 s`.
- Total new-byte ceiling, including private cache and every arm: `10 GiB`.
- Required free disk before setup and GPU launch: `20 GiB`.
- The collective ceilings are conservative execution-stop bounds derived from
  at most two train steps and at most two exact-state checkpoint publications,
  not an efficiency metric. The production path has explicit finite/metric
  report gathers plus checkpoint gather/broadcast/barrier and exact-state
  gather/barrier owners. A timeout, divergent rank, or any evidence of a
  ceiling exceedance is a terminal failure; it is not retried or reinterpreted.

## Comparison and claim boundary

The exact rules are frozen in the manifest. The verifier must admit and compare
control step 1 against parent step 1 before the corresponding next forward, then
control step 2 against resumed-child step 2 after each branch's one compared
forward/update. It requires exact identity, cursor/next-pack, trainable,
optimizer, scheduler, scaler, all per-rank RNG, topology, structure, and durable
step-2 objective/loss/update fields under the manifest's exclusions. The
role-specific `identities.resolved_config`, `input_build_seconds`, and
`input_wait_seconds` are explicitly excluded; all production-strict identities
and semantic objective fields remain exact.

The failure arm is only the representative `rank=1, kind=missing` signed
model-free receipt; the interruption arm is only representative
`stop_after=1`, with a non-production inference stub explicitly marked
`stub_not_production`. The already-executed CPU gate covers all missing,
duplicate, malformed, corrupt, and stop-after 0/1/2 cases. This packet cannot
qualify the production inference-writer interruption boundary and makes no
cross-world-size, mid-accumulation, throughput, quality, or exact-resume claim
beyond same-world-size optimizer-step-boundary continuation.

## Exact execution and stop rule

Execute the six argv arrays from the frozen manifest exactly once and only in
this order: `setup_command`, `success.uninterrupted_control`,
`success.resumed_child`, `rank_failure`, `interruption`,
`verification_command`. The two success commands are the only GPU/model
commands. The executor records start/end timestamps, exit codes, command hashes,
per-rank maxima from canonical `logging.jsonl`, total artifact bytes, and the
terminal verifier receipt in the change-local Wave-3 terminal receipt.

Stop without retry on any Git/manifest/config/path/GPU-identity drift; occupied
target/cache paths; insufficient disk/GPU headroom; receipt or digest failure;
unexpected forward/update count; timeout, hang, OOM, rank loss, or nonzero exit;
resource/artifact ceiling exceedance; missing per-rank measurements; or any
verifier mismatch. Verification success requires both process exit `0` and the
signed `terminal-receipt.json` field `status: verified`; exit status alone is
never sufficient. No deletion, target repair, alternate command, retry, or
partial-success promotion is authorized by this packet.

## Fresh authorization required

Planning approval and the user's earlier general eight-GPU permission do not
authorize these commands. After the independent pre-cost review is READY, the
executor must present this unchanged packet identity and obtain explicit fresh
approval immediately before execution. Any planning/code/manifest/packet edit
after that approval invalidates it.
