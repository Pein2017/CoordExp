# COCO GT-correction production evaluation execution

Date: 2026-09-07

Status: **launched; scientific evaluation outstanding**. This receipt records
execution liveness only and makes no model-quality claim.

## Frozen identity

- Root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-gt-correction-portfolio/pilot-v1`.
- Bank ID: `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
- Source topology receipt SHA256:
  `9e805279a7fde9574260ab4888120c452088158e1747ccf02953961246f43a21`.
- Final checkpoint IDs: R
  `4efbec695067ec4f5a7c1de30add419046e0ac210067a27bd013aa9df71c033d`,
  B `67684f1a842c3c454b45c7efd0c2706c9a7248cb571496353025471117da5d9d`,
  M `4dcccaa341763841d008441bd22bd66ac6f90ba810c2ec38cfeab64b6e79b03b`,
  W `1a068bf17c237bda42c4f73c592c81331dfa001e3fc4228383f42b089b54a0da`.
- The production cold loader accepted all four payloads at completed update 64
  before launch, including their saved payload-file hashes and declared
  surfaces.
- Eval entry SHA256:
  `8f72bd23a4e363022da010c4e3c2b1e83dee68fd7557c9f31fddb08d4c3bcde8`.
- Reducer SHA256:
  `3fc93466b0b2c0bc6c6dfeacc58714e9912464472cb5f6824e42dfe20bf3633e`.
- Driver SHA256:
  `f665be5c459f70a017f21683d22c0be4b141ff393ab3455e0f717f5eb5ebd907`.

## Launch receipt

- Driver:
  `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/run-evaluation.sh`.
- Launch command:
  `tmux new-session -d -s coco-gt-correction-eval-v1 'exec bash /data/CoordExp/.worktrees/coco-gt-correction-portfolio/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-gt-correction-portfolio/run-evaluation.sh'`.
- tmux session: `coco-gt-correction-eval-v1` (`$911`); driver PID `1588621`;
  start `2026-09-07T09:35:44Z`.
- Durable logs: `$ROOT/launch-evaluation/`; outputs: `$ROOT/evaluation/`.
  `driver.log`, per-panel command/start/finish/exit/log files, reducer files and
  `terminal-marker.txt` define the execution receipt.
- The driver runs exactly R/B/M/W x train256/dev128 in that serial order. Each
  panel exposes GPUs 0-7, requires eight active ranks, and inherits the sealed
  FP32 SDPA, repetition-penalty 1.0, batch-2 and 3,084-token-cap configuration.
  It then invokes the fixed reducer once against the sealed Source pair.
- Collision checks reject either pre-existing output root. There are no retries,
  borrowed GPUs, alternate arms, changed panels or scheduler layer; the first
  nonzero panel exit stops later work and is retained.

## Initial production liveness

The first real invocation is R/train256. It wrote the resolved configuration
and eight-rank shard plan, and spawned one live worker for every rank 0-7 under
the single controller. This proves entry/topology startup, not completion or
metric validity. The historical matching Source topology took roughly 17
minutes for train256 and 3 minutes for dev128. Allowing cold-load, arm variation,
shared-GPU contention and reduction, the bounded launch estimate is about
80-120 minutes from start; the terminal marker, not this estimate, owns status.
