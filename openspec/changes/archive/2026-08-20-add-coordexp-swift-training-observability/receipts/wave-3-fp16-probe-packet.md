# Wave-3.8 fp16/Injected-Outcome Probe Packet — FROZEN

- frozen 2026-08-20 by the Claude Fable lead at HEAD `305b17eb7`
  (probe scripts committed; Wave-3A `bdb29b3fa` + Wave-3B `4db58e972` code
  underneath; tracked tree clean)
- authorization: user standing GPU grant 2026-08-20 (manifest
  `authorization.gpu`); devices selected by observed idleness
  (`nvidia-smi`: all 8 GPUs idle at freeze; GPUs 0,1 selected)

## Frozen commands (each runs once per attempt; --output must be absent)

1. Single-rank CUDA fp16 (finite + overflow arms):
   `conda run -n ms python scripts/probes/coordexp_swift/obs_wave3_fp16_cuda_probe.py --output <receipts>/wave-3-fp16-ws1-receipt.json`
   with `CUDA_VISIBLE_DEVICES=0`.
2. Two-rank CUDA fp16:
   `conda run -n ms python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/probes/coordexp_swift/obs_wave3_fp16_cuda_probe.py --two-rank --output <receipts>/wave-3-fp16-ws2-receipt.json`
   with `CUDA_VISIBLE_DEVICES=0,1`.
3. Injected-outcome (CPU gloo, six arms; lead re-execution of the builder's
   CPU probe):
   `conda run -n ms python scripts/probes/coordexp_swift/obs_wave3_injected_outcome_probe.py --output <receipts>/wave-3-injected-outcome-receipt.json`

All through the host `bash -c 'export …; unset …; cd worktree; timeout
--signal=TERM --kill-after=30s <T>s <cmd>'` wrapper.

## Bounds

| bound | limit |
| --- | --- |
| devices / world size | ≤2 GPUs; ws=1 then ws=2 (cmd-fixed); injected probe CPU-only |
| planned steps | 2 per fp16 run (1 per arm); 6 across injected arms |
| model forwards | 2 per fp16 run (tiny 16→32→16 Linear stack, batch 8) |
| cache / materialization passes | REQUIRED 0 (no path under .cache/ read or written) |
| wall time | ≤300 s per command (timeout-enforced) |
| peak GPU memory | ≤2 GiB per device (trivial model) |
| artifact bytes | receipts only, ≤1 MiB each, absent paths |

Stop without retry on: nonzero exit, timeout, occupied output, cache-root
byte change, bound exceedance. Known failure mode (from the script author):
a rank-divergent raise in ws=2 hangs the peer until the gloo control
timeout and torchrun kills the job with no receipt — treat a hang as a
rank-divergence failure, not infra.

## Known-defect protocol (scaler_found_inf silent truth)

The probe author identified that `_scaler_found_inf`/`_accelerator_overflow`
key GradScaler state on the wrong optimizer object under real Accelerate
(structurally False; boundary decision remains correct via the independent
unscaled-gradient scan). The probes emit a non-fatal FINDING when
ground-truth found_inf is invisible to the runtime. Protocol: attempt-1 runs
are expected to fire the FINDING (real-CUDA RED evidence); a bundled 3A
correction round then fixes the keying; attempt-2 runs (fresh absent
receipt paths, `-r2` suffix) must be FINDING-free and are the final 3.8
evidence. Both attempts are retained.
