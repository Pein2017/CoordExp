# Bounded acceptance — 2026-09-18

**Lead-accepted:** independent special-token deltas, eight-GPU training,
two-tensor checkpoint publication, exact resume, and native HF reload.
**Scientific status:** not assessed. **Full production training:** not launched.

Machine-readable evidence:
[acceptance.json](/data/CoordExp/outputs/infra_base/untie-20260918/acceptance.json).
Launch instructions and the exact semantic boundary are in [README.md](README.md).

## Evidence

- [Config comparison](/data/CoordExp/outputs/infra_base/untie-20260918/config-parity.json):
  archived `four-coordinate-xy/step-2444` training settings, data, optimizer,
  checkpoint/eval cadence and full prompt match. Tied/untied resolved controls
  differ only in run name and the selected-token tying flag.
- [Data identity](/data/CoordExp/outputs/infra_base/untie-20260918/data-identity.json):
  117,266 train and 4,952 val rows; both file hashes equal the archived provenance.
- [Regression tests](/data/CoordExp/outputs/infra_base/untie-20260918/focused-tests.log):
  279 passed. Subsequent final delta assertions passed all 31 module tests;
  another 7 config tests passed. These sets overlap and should not be added.
  New tests first failed against missing untie/XY support, then passed. They
  check equal initialization, independent nonzero gradients, tied gradient equal
  to the sum of the untied gradients, optimizer membership/LR, exact round trip,
  and malformed output rejection before input mutation.
- [Training summary](/data/CoordExp/outputs/infra_base/untie-20260918/training-summary.json):
  eight A100 80GB GPUs; two optimizer updates; 256 available training rows and
  16 eval rows; original BF16/FA2, 12000 packing and EBS24 retained. Each update
  consumed 24 global packs / 192 example presentations; each rank accumulated
  three packed micro-steps. Both train/eval boundaries were finite and both
  checkpoint publications completed. Eight-rank eval used only two packs,
  exercising ranks with unequal local eval work.
- [Exact resume comparison](/data/CoordExp/outputs/infra_base/untie-20260918/resume-comparison.json):
  resume step 1 -> step 2 reproduced all 588 adapter tensors and both deltas
  bitwise. All eight ranks' decoded trainable-model and optimizer states also
  match exactly. Both independent delta parameters occur in every rank state.
- [Fresh HF reload](/data/CoordExp/outputs/infra_base/untie-20260918/hf-reload-exact.json):
  both FP32 `[1004, 2048]` tensors equal the published file exactly, remain
  independent and nonzero, and both influence a real Qwen forward. After a
  deliberate perturbation of each side, restoring it restores logits bitwise.
  The runnable check is [verify_reload.py](verify_reload.py).

## Observed costs

The successful parent took 86.23 seconds from training entry to terminal
publication. Rank-zero peak allocated GPU memory was 10,015,994,368 bytes
(9.33 GiB), reserved 14,566,817,792 bytes (13.57 GiB). The observed steady-state
all-rank maximum CPU RSS was 9,538,834,432 bytes. These describe the bounded
smoke, not full-data timing or worst-case production memory.

Each saved checkpoint with eight-rank exact state was approximately 2.22 GB:
adapter 72.1 MB, both embedding deltas plus metadata 16.5 MB, training state
2.13 GB. Checkpoint publication took about 10.9 seconds. Default production
inference-payload saving omits exact optimizer/RNG state; enable the documented
resume overlay when restartability is needed.

## Limits and retained failures

1. An initial cache-preparation attempt omitted strict CUDA environment variables
   and failed closed. After source changes, the old smoke cache was also rejected;
   current cache preparation and `--require-all-hit` subsequently passed.
2. The first eight-rank launch timed out before model loading while creating its
   Gloo control group. No optimizer update occurred. Explicit loopback rendezvous
   and Gloo/NCCL interfaces passed parent and resume launches. This establishes a
   working command on this host; it does not establish the transient failure's
   underlying cause. See `eight-gpu-train.log` and `eight-gpu-loopback.log`.
3. Two-row HF generation loaded the checkpoint, then rejected a pad token before
   EOS for image 285. A one-row run on image 139 completed and produced all raw,
   scored, trace, image-plan, diagnostic and manifest artifacts. However, it hit
   512 generated tokens, had one parser failure, 112 dropped predictions, and
   zero scoreable predictions. The evaluator completed with
   `benchmark_eligible=false`; its zero AP is **not a model-quality result**.
   No parsing, stop, generation or scoring rule was weakened. See
   [single-row summary](/data/CoordExp/outputs/infra_base/untie-20260918/untied-smoke-hf-one-row/summary.json)
   and [evaluation receipt](/data/CoordExp/outputs/infra_base/untie-20260918/untied-smoke-hf-one-row/evaluation/evaluation_receipt.json).
4. Full production cache admission reports `training.pack_cache_not_prepared`.
   Full cache construction and long training are still unexecuted; use the
   documented prepare -> require-all-hit -> torchrun sequence.
5. Untied dense folding and vLLM are explicitly unsupported. Native HF is the
   validated reload route. Old tied payload loading remains covered by existing
   compatibility tests. No full-vocabulary training or historical bitwise replay
   is claimed.

The implementation remains uncommitted in the requested worktree. Source hashes
and all failed/successful artifacts are retained under the evidence root.
