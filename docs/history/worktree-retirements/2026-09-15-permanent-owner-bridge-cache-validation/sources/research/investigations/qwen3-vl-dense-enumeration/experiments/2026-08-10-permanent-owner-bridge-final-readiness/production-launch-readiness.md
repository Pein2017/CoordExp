# Permanent owner bridge production launch readiness

## Decision boundary

- Question: can the exact four-presentation Stage 1 production run be admitted
  to one eight-rank activation without changing the accepted mechanism, data,
  objective, schedule, or inference contract?
- Contrast: the exact production leaf and source-S lineage on the frozen tree,
  versus any alternate config, stale cache, dirty tree, occupied GPU set, or
  repeated activation.
- Decision-owning outcome: task 4.13 passes if a dry invocation of the same
  launch guard that owns activation re-inspects the full source, cache,
  schedule, repository, and GPU boundary and leaves the host-global claim free.
- Strongest alternative: retain the completed W1/W8 smokes as mechanical
  evidence but do not launch production.
- Stop rule: any tree/config/source/cache/GPU drift, existing global claim,
  guard rejection, ambiguous activation, non-finite update, rank failure, or
  missing artifact heartbeat stops the route. There is no naked Accelerate
  fallback and no blind retry after a claim is published.

This record closes production *readiness* only. It does not claim that the
production process has started or that a production optimizer step has
completed; those remain tasks 4.14 and 4.15.

## Frozen evidence revalidation

- Worktree: `/data/CoordExp/.worktrees/permanent-owner-bridge`
- Dry-preflight tree: `de1d26d943b54bd4bd2fbfc1340d372dd97ae8c8`
- Dry-preflight Git state: clean, with an empty tracked and untracked status.
- `git diff 80ded9ad2..de1d26d -- src/ scripts/ configs/coordexp_swift/prod
  configs/coordexp_swift/smoke` is empty. The accepted W1/W8 executable tree
  and launch-bearing configs therefore have not drifted.
- Frozen readiness receipt SHA-256:
  `4d6299009e87d25aad702da6fd1f5b54385fd6cca2db3e28021af2924135f2ec`.
- Canonical JUnit SHA-256:
  `5c24457b7313c00c23454b633e06aaf78000db68bf6e07753a06d9987a3dd120`
  (2001 collected, 2000 passed, one explicit CUDA-hidden skip, no failure or
  error).
- Final Wave 5 algorithm receipt SHA-256:
  `349671c4788137ab684807d3ef29d767d74ff77034acad40f91b70be9cff15e7`
  (23 of 23 exact nodes passed on its recorded clean execution tree).
- Final smoke acceptance record SHA-256:
  `2c7ffdefc432aa352b4ff68088a9ad1534f40779e46ccda44d242daeb119b3b5`.
- The accepted `80ded9a` W1 and W8 materialized inference configs and receipts
  remain tracked; the dry guard does not substitute for their fresh-load HF
  lifecycle evidence.

## Exact production preflight

Command, intentionally without `--execute`:

```bash
conda run -n ms python -m scripts.coordexp_swift.launch_owner_bridge_stage1 \
  --config configs/coordexp_swift/prod/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1.yaml \
  --cache-root .cache/coordexp_swift/packing
```

The command exited zero with `status=preflight_passed`,
`execute=false`, and `global_claim_exists=false`.

- Raw immutable receipt:
  `/data/CoordExp/outputs/prod/coordexp_swift/.owner_bridge_stage1_launch_ledger/preflights/48b399690788875f353f08f0ad935660e74e2fef02330065c32cdbe2d9ce1772-20260810T174230.045461Z.json`
- Raw receipt SHA-256:
  `279af32baa50546574f3a5c24b8c2eb2a2e3bdcbf1d4a2333db735527a3092b0`.
- Canonical receipt fingerprint:
  `1c5af7eed4eae805ae1d911b2e40a4c0f8ee0496220d9b5e403ce11647e7561a`.
- Independent canonical-JSON recomputation after removing only the fingerprint
  field produced the same value.
- Dry intent key:
  `48b399690788875f353f08f0ad935660e74e2fef02330065c32cdbe2d9ce1772`.
- Inspection time: `2026-08-10T17:42:30.045461+00:00`.

The execute path re-runs the full preflight twice on its then-current clean
tree before publishing the singleton claim. This evidence-only commit will
therefore change the deterministic intent key. The dry key above is evidence
of guard behavior at `de1d26d`, not the receipt that will gate activation; the
execute invocation must derive a new key and new preflight receipts. Cache
compatibility remains bound to its own code-identity digest and will be
re-inspected.

## Identities accepted by the guard

- Production config fingerprint:
  `770f705f5357fdfc08d1d6fb332d828874265d296c5f567ae4c64266cd3c056c`.
- Source checkpoint: exact step 2444 under the accepted four-coordinate-xy
  training root.
- Source resolved-config fingerprint:
  `89e5af1269c42bdcc28b7175c8c701a4b57673e4b2fa8255c8912ee8800283ae`.
- Source adapter fingerprint:
  `5a59270e1ed1bf2062590fd0eb820e534e39e8914c7a20d91d642a8f2a99452d`.
- Source selected-embedding fingerprint:
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`.
- Source training-state manifest SHA-256:
  `133a87b5d119d8e373b07b0c2cbeb5d2fc311d46409811f101f6791fd9b98527`.

All four caches were inspected in no-build mode and returned `hit`, with the
same code-identity digest
`c81524a2426cfd7ee5c3b632a6e080eaccac1a51cd6991f9aa194e01c0beeb44`:

| role / presentation | fingerprint | packs | uses |
| --- | --- | ---: | --- |
| train / geo_sorted | `e80ccdaa05003564c0663aaf7e7bcc538a8b959e439449a642781bd215433c8e` | 14660 | p0, p2 |
| train / random-1 | `08eec860dda9a38b186b6a8860732817d791e8cc0cfa0a63f96efc537024c9b0` | 14660 | p1 |
| train / random-2 | `21cf8cec5b51f1eb94d22e84631a0cc04b7e0a0ccdec5d13e996f732fa66e439` | 14660 | p3 |
| eval / geo_sorted | `7d24c40231a634e5011640df2251d3643af2a61cecd5d2bec64843b830eb854d` | 619 | eight TF landmarks |

The schedule is W8/A3/EBS24, with four presentations in exact order
`geo_sorted, random-1, geo_sorted, random-2`; each has 14,660 real packs,
611 planned steps, and four final shadow slots. The total is 58,640 real packs,
2,444 planned steps, 16 shadow slots, and eight teacher-forced eval landmarks.

## Live host boundary

The guard's own snapshot at `2026-08-10T17:42:30.045461Z` found all eight A100
80GB devices idle and above its headroom threshold. A separate operator-side
snapshot at `2026-08-10T17:47:50Z` again found no compute process,
zero-percent utilization, and at least 81,034 MiB free. The ordered UUIDs were:

1. `GPU-8d43cb78-19ca-2f59-3179-7ea166cb1a4e`
2. `GPU-f520bc13-34fb-8809-4af8-a6fd8a28aec8`
3. `GPU-bf5f21dd-c4cb-6731-e611-6c6d55001f55`
4. `GPU-7dfec513-1c8a-e86d-ea3f-1ec0eef82208`
5. `GPU-2f35ef2a-9a05-1b8e-391e-06000c1380ab`
6. `GPU-40445989-5638-1e7c-165b-41926b1fc947`
7. `GPU-3d113ebf-fcac-c7d9-d5a6-a6abefeb32ff`
8. `GPU-4e5382fb-e2a6-1057-32bc-12697f2b09fd`

The guard threshold is 71,680 MiB free per device. No `src.train`, Accelerate,
or torchrun process and no port-29500 listener was present. The selected run
artifact root resolves to
`/data/CoordExp/.worktrees/permanent-owner-bridge/outputs/prod/coordexp_swift`;
it did not yet exist, and the exact config uses timestamp collision policy.
This is a different physical tree from the host-global ledger at
`/data/CoordExp/outputs/prod/coordexp_swift/.owner_bridge_stage1_launch_ledger`.
The ledger contains only the dry preflight receipt and no
`stage1-production-launch-claim.json`.

## Retained launch risks

- The singleton claim, process activation, process-bound eight-rank admission,
  rank quorum, and nonce-bound run discovery are fail-closed and covered by
  focused multiprocess tests, but have not yet executed under the exact
  production fingerprint. The accepted W8 smoke used its distinct smoke
  fingerprint. A spurious production-only failure can therefore consume the
  singleton claim without producing a usable run; recovery is user-owned and
  must not delete the claim or retry blindly.
- The rank admission implementation currently bounds its activation-receipt
  and eight-rank quorum waits at ten seconds. Repository tests cover the focused
  protocol surfaces. A separate current-host, CPU-only disposable diagnostic
  launched eight real subprocesses that each imported `src.train`, loaded the
  exact production config, and performed process-bound admission with a
  synthetic authorized claim and the default bound: all eight exited zero,
  total wall time was 9.079 seconds, import/config work took 6.743--7.539
  seconds, rank-slot arrival skew was 0.795 seconds, and the earliest rank
  waited 0.822 seconds for quorum. Its temporary directory was removed on exit,
  so these timings are transcript evidence rather than a durable receipt. This
  does not execute GPU training or the full production guard chain; measured
  concurrent production worker arrival remains a first-run observation. Treat
  any admission timeout as a retained, non-retryable failure receipt rather
  than silently increasing the bound after activation.
- The production artifact root is inside this worktree and is gitignored.
  Retiring or pruning the worktree would remove the run's checkpoints and
  artifacts; the worktree and its output tree must therefore be retained for
  the full training and evaluation lifecycle.
- The launch inherits the current process environment, records its identity,
  and overrides only the cache root, UUID-ordered visible devices, activation
  nonce, and unbuffered output. Before execute, reject any unexpected
  `ACCELERATE_*`, `CUDA_VISIBLE_DEVICES`, `MASTER_PORT`, or launch-nonce value.
  The first heartbeat also remains the watch point for the previously recorded
  post-unsafe-veto DDP reducer-lifecycle P2.

## Independent decision review

- Sol frozen-tree launch-gate re-audit: `PASS task 4.13`, P0=0, P1=0. It
  independently matched the raw receipt, current live host, guard ordering,
  OpenSpec wording, and cache/config/source identities.
- Opus-xhigh independent preflight audit: `PASS`, no P0. It independently
  recomputed both receipt hashes, traced the claim/nonce/admission/binding
  chain, and required the retained first-production-execution and worktree-root
  risks above to remain visible.
- Sol-xhigh admission-window audit: `PASS`, P0=0, P1=0. It traced both ten-second
  windows, executed the disposable current-host diagnostic recorded above, and
  found no evidence justifying a launch-bearing timeout change on the frozen
  tree.

All three reviews authorize checking task 4.13 only. None substitutes for the
task 4.14 execute receipt or the task 4.15 first-heartbeat receipt.

## Launch authority

Task 4.13 may close on this evidence. Task 4.14 still requires exactly one
`--execute` invocation on a newly committed clean tree. That invocation owns a
fresh GPU snapshot, two complete preflights, a host-global no-replace claim,
fresh activation nonce, UUID-pinned visibility, process-bound W8 admission,
durable logs, and nonce-bound run-root discovery. Task 4.15 remains open until
all ranks initialize and the run publishes its first finite, applied optimizer
update plus artifact heartbeat.
