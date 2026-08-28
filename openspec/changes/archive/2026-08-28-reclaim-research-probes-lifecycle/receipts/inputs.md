# Research-probes Input Baseline

**Frozen**: 2026-08-28

## (a) research-probes HEAD and working tree status

- HEAD SHA: `521183ea00f59ee6c9300633bb74c51b7d501e72`
- Dirty files: `0`

## (b) Retirement buckets: tip SHAs verified

### Bucket A: probe-final (15 lanes)

All tips verified with `git rev-parse --short <branch>`:

| Branch | Design SHA | Verified SHA | Match |
|--------|------------|--------------|-------|
| codex/human13-runner | 832dd63f8 | 832dd63f8 | ✓ |
| codex/human13-analyzer | 68fafe2fb | 68fafe2fb | ✓ |
| codex/human13-live-model | a6bfb1c94 | a6bfb1c94 | ✓ |
| codex/human13-loss-census | a4cb7e89e | a4cb7e89e | ✓ |
| codex/human13-manifest-collector | 10c37852e | 10c37852e | ✓ |
| codex/human13-materializer-launcher | fd2a7542b | fd2a7542b | ✓ |
| codex/human13-discovery-adapter | 4d3d01800 | 4d3d01800 | ✓ |
| codex/rp-crossover-analyzer | b796b5ebb | b796b5ebb | ✓ |
| codex/rp-crossover-integration | efc57dc11 | efc57dc11 | ✓ |
| codex/rp-crossover-launcher | 78d0069d0 | 78d0069d0 | ✓ |
| codex/rp-crossover-live-integration | 7cb17a832 | 7cb17a832 | ✓ |
| codex/rp-crossover-materializer | dbc36730c | dbc36730c | ✓ |
| codex/rp-crossover-production | e7e373037 | e7e373037 | ✓ |
| codex/rp-crossover-runtime | 2c5632e10 | 2c5632e10 | ✓ |
| codex/rp-crossover-wave5-correction | 51d518f75 | 51d518f75 | ✓ |

### Bucket B: mixed retirement targets

| Branch/Checkout | Design SHA | Verified SHA | Match |
|-----------------|------------|--------------|-------|
| codex/human13-nk-factorial-probe | a904e3ae3 | a904e3ae3 | ✓ |
| probe/human13-standalone-recovery | accd80ded | accd80ded | ✓ |
| codex/human13-scientific-fast-path | bed82651d | bed82651d | ✓ |
| research-probe-infras | 62274a97d | 62274a97d | ✓ |

### Bucket C: archive (12 lanes)

| Branch | Design SHA | Verified SHA | Match |
|--------|------------|--------------|-------|
| codex/coverage-ledger-mechanistic-probing | 77acee47c | 77acee47c | ✓ |
| codex/vllm-mechanistic-round-gaussian-rps | cd9c92211 | cd9c92211 | ✓ |
| codex/permutation-bundle-coordinate-noise-pilot | 9ddbe4e91 | 9ddbe4e91 | ✓ |
| codex/regionlock-simplified-pointer | 444cd7ba3 | 444cd7ba3 | ✓ |
| codex/ledger-auxiliary-loss | 91cddea1a | 91cddea1a | ✓ |
| codex/prefix-denoising-sft | b3919b49f | b3919b49f | ✓ |
| codex/owner-commit-binding | 210ad8c0c | 210ad8c0c | ✓ |
| codex/permanent-owner-bridge | 84fc6b318 | 84fc6b318 | ✓ |
| codex/qwen3-vl-painted-gt-transcription-probe | f90381b15 | f90381b15 | ✓ |
| codex/historical-markdown-recovery | 93deed826 | 93deed826 | ✓ |
| codex/8-coords-bbox | e7d3724b3 | e7d3724b3 | ✓ |
| codex/continue-handoff-session | 2fde14b37 | 2fde14b37 | ✓ |

## Discrepancies

None. All branch tips match design.md exactly.

## (c) Worktree status per retirement bucket

### Bucket A worktrees

| Worktree | Dirty | HEAD SHA |
|----------|-------|----------|
| /data/CoordExp/.worktrees/human13-runner | 0 | 832dd63f8 |
| /data/CoordExp/.worktrees/human13-analyzer | 0 | 68fafe2fb |
| /data/CoordExp/.worktrees/human13-live-model | 0 | a6bfb1c94 |
| /data/CoordExp/.worktrees/human13-loss-census | 0 | a4cb7e89e |
| /data/CoordExp/.worktrees/human13-manifest-collector | 0 | 10c37852e |
| /data/CoordExp/.worktrees/human13-materializer-launcher | 0 | fd2a7542b |
| /data/CoordExp/.worktrees/human13-discovery-adapter | 0 | 4d3d01800 |
| /data/CoordExp/.worktrees/rp-crossover-analyzer | 0 | b796b5ebb |
| /data/CoordExp/.worktrees/rp-crossover-integration | 0 | efc57dc11 |
| /data/CoordExp/.worktrees/rp-crossover-launcher | 0 | 78d0069d0 |
| /data/CoordExp/.worktrees/rp-crossover-live-integration | 0 | 7cb17a832 |
| /data/CoordExp/.worktrees/rp-crossover-materializer | 0 | dbc36730c |
| /data/CoordExp/.worktrees/rp-crossover-production | 0 | e7e373037 |
| /data/CoordExp/.worktrees/rp-crossover-runtime | 0 | 2c5632e10 |
| /data/CoordExp/.worktrees/rp-crossover-wave5-correction | 0 | 51d518f75 |

### Bucket B worktrees and detached checkouts

| Worktree/Checkout | Dirty | HEAD SHA |
|-------------------|-------|----------|
| /data/CoordExp/.worktrees/human13-nk-factorial-probe | 0 | a904e3ae3 |
| /data/CoordExp/.worktrees/research-probe-human13-standalone-recovery | 0 | accd80ded |
| /data/CoordExp/.codex/worktrees/3f15/research-probes | 0 | b36216f10 |
| /tmp/coordexp-base-check2 | 0 | 2a297a93a |

### Bucket C worktrees

| Worktree | Dirty | HEAD SHA |
|----------|-------|----------|
| /data/CoordExp/.worktrees/coverage-ledger-mechanistic-probing | 0 | 77acee47c |
| /data/CoordExp/.worktrees/geometry-aware-denoising-sft | 0 | b3919b49f |
| /data/CoordExp/.worktrees/owner-commit-binding | 0 | 210ad8c0c |
| /data/CoordExp/.worktrees/permanent-owner-bridge | 0 | 84fc6b318 |
| /data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot | 0 | 9ddbe4e91 |
| /data/CoordExp/.worktrees/regionlock-simplified-pointer | 0 | 444cd7ba3 |
| /data/CoordExp/.worktrees/ledger-auxiliary-loss | 0 | 91cddea1a |
| /data/CoordExp/.worktrees/vllm-mechanistic-round-gaussian-rps | 0 | cd9c92211 |

## (d) tmux wait-for human13 processes

```
1641167  4-17:33:31 tmux wait-for human13-gpu0-stage1-ok
1641172  4-17:33:31 tmux wait-for human13-gpu0-stage2-ok
1641176  4-17:33:31 tmux wait-for human13-gpu0-stage3-ok
1641180  4-17:33:31 tmux wait-for human13-gpu1-stage1-ok
1641183  4-17:33:31 tmux wait-for human13-gpu1-stage2-ok
1641188  4-17:33:31 tmux wait-for human13-gpu1-stage3-ok
```

## (e) git worktree list

```
/data/CoordExp 29e368144 [main]
/data/CoordExp/.codex/worktrees/3f15/research-probes b36216f10 (detached HEAD)
/data/CoordExp/.worktrees/CoordExp-swift 8d12eab28 [coordexp-swift]
/data/CoordExp/.worktrees/codex-rtk-correctness-first 38b30ebc1 [codex/rtk-correctness-first]
/data/CoordExp/.worktrees/codex-wake-me-up-event-monitor 8dfb8102a [codex/wake-me-up-event-monitor]
/data/CoordExp/.worktrees/coverage-ledger-mechanistic-probing 77acee47c [codex/coverage-ledger-mechanistic-probing]
/data/CoordExp/.worktrees/geometry-aware-denoising-sft b3919b49f [codex/prefix-denoising-sft]
/data/CoordExp/.worktrees/human13-analyzer 68fafe2fb [codex/human13-analyzer]
/data/CoordExp/.worktrees/human13-discovery-adapter 4d3d01800 [codex/human13-discovery-adapter]
/data/CoordExp/.worktrees/human13-live-model a6bfb1c94 [codex/human13-live-model]
/data/CoordExp/.worktrees/human13-loss-census a4cb7e89e [codex/human13-loss-census]
/data/CoordExp/.worktrees/human13-manifest-collector 10c37852e [codex/human13-manifest-collector]
/data/CoordExp/.worktrees/human13-materializer-launcher fd2a7542b [codex/human13-materializer-launcher]
/data/CoordExp/.worktrees/human13-nk-factorial-probe a904e3ae3 [codex/human13-nk-factorial-probe]
/data/CoordExp/.worktrees/human13-runner 832dd63f8 [codex/human13-runner]
/data/CoordExp/.worktrees/image2299-mechanism-microscope 60a0b25a1 [codex/image2299-mechanism-microscope]
/data/CoordExp/.worktrees/ledger-auxiliary-loss 91cddea1a [codex/ledger-auxiliary-loss]
/data/CoordExp/.worktrees/owner-commit-binding 210ad8c0c [codex/owner-commit-binding]
/data/CoordExp/.worktrees/permanent-owner-bridge 84fc6b318 [codex/permanent-owner-bridge]
/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation 477b376a3 (detached HEAD)
/data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot 9ddbe4e91 [codex/permutation-bundle-coordinate-noise-pilot]
/data/CoordExp/.worktrees/regionlock-simplified-pointer 444cd7ba3 [codex/regionlock-simplified-pointer]
/data/CoordExp/.worktrees/research-probe-human13-standalone-recovery accd80ded [probe/human13-standalone-recovery]
/data/CoordExp/.worktrees/research-probe-infras 74609d2b1 [codex/research-probe-infra-foundation] locked
/data/CoordExp/.worktrees/research-probes 521183ea0 [research-probes] locked
/data/CoordExp/.worktrees/rp-crossover-analyzer b796b5ebb [codex/rp-crossover-analyzer]
/data/CoordExp/.worktrees/rp-crossover-integration efc57dc11 [codex/rp-crossover-integration]
/data/CoordExp/.worktrees/rp-crossover-launcher 78d0069d0 [codex/rp-crossover-launcher]
/data/CoordExp/.worktrees/rp-crossover-live-integration 7cb17a832 [codex/rp-crossover-live-integration]
/data/CoordExp/.worktrees/rp-crossover-materializer dbc36730c [codex/rp-crossover-materializer]
/data/CoordExp/.worktrees/rp-crossover-production e7e373037 [codex/rp-crossover-production]
/data/CoordExp/.worktrees/rp-crossover-runtime 2c5632e10 [codex/rp-crossover-runtime]
/data/CoordExp/.worktrees/rp-crossover-wave5-correction 51d518f75 [codex/rp-crossover-wave5-correction]
/data/CoordExp/.worktrees/vllm-mechanistic-round-gaussian-rps cd9c92211 [codex/vllm-mechanistic-round-gaussian-rps]
/tmp/coordexp-base-check2 2a297a93a (detached HEAD)
```
