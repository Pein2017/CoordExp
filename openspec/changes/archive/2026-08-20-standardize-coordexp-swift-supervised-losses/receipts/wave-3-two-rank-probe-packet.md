# Wave-3 Two-Rank Reduction-and-Update Probe Packet (task 3.6) — FROZEN

- frozen 2026-08-20 by the Claude Fable lead at HEAD
  `a20673078` (Wave-3 code commit; tracked tree clean)
- manifest review: this packet freezes the `wave3-two-rank-probe`
  placeholder (`TO-FREEZE-IN-WAVE-3-PACKET`) per the manifest amendment rule;
  the frozen argv is below.
- authorization basis: **no GPU is used** (CPU gloo two-rank), so no GPU
  authorization is required; the user's 2026-08-20 standing GPU grant
  (manifest amend-4) is on record but not consumed. Distributed action
  reviewed against the frozen manifest as task 3.6 requires.

## Frozen argv (runs once; wrapper form per host `env`-prefix hook)

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; \
  unset COORDEXP_SWIFT_PACK_CACHE_ROOT COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE \
        COORDEXP_SWIFT_EVAL_REDUCTION_MODE COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS; \
  cd /data/CoordExp/.worktrees/CoordExp-swift; \
  timeout --signal=TERM --kill-after=30s 600s \
  conda run -n ms python scripts/probes/coordexp_swift/losses_wave3_two_rank_probe.py \
    --output openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/wave-3-two-rank-probe-receipt.json'
```

## Bounds (task 3.6)

| bound | value |
| --- | --- |
| world size | exactly 2 (CPU gloo) + in-process ws-1 reference |
| planned steps | 1 per arm × 3 arms (base_ce, enabled gate, positive auxiliary) |
| model forwards | 0 full-model forwards (synthetic loss-boundary tensors; smallest real component set) |
| cache / materialization passes | REQUIRED 0 — no path under `.cache/` read or written |
| wall time | ≤ 600 s (enforced by `timeout`) |
| peak GPU memory | 0 (no CUDA context) |
| artifact bytes | receipt JSON only, ≤ 1 MiB, at the absent path above |
| parity tolerance | rtol=1e-5 / atol=1e-6 (declared in-probe); exit nonzero on any finding |

Stop without retry on: nonzero exit, timeout, occupied output path, any
cache-root byte change (sha256 inventory before/after), or a bound
exceedance. Builder dry-run receipt (informational, not gate evidence):
exit 0, `status: OK`, `findings: []`, 12 collective ops per rank.
