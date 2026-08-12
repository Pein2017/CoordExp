# Human-13 K-Union-to-Greedy Planning Checkpoint

Source task: the 2026-08-12 design discussion and independent Sol/Fable review.

Source handles:

- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/review.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/handoff.md`
- `openspec/changes/add-human13-k-union-greedy-overfit-probe/`
- `docs/superpowers/plans/2026-08-12-human13-k-union-greedy-overfit-probe.md`

Captured and last verified: 2026-08-12.

The user chose an overfit-only Human-13 screen as the next practical route.
Its first question is not support expansion: freeze K=16 natural samples,
target owners retrieved at least once but absent from Source greedy, and ask
whether native parameter updates can consolidate them into one original-prompt
clean-greedy completion. K-miss owners remain unknown/gradient-neutral until a
later support-expansion decision.

The approved Stage-1 family is Frozen Source, separate full-GT capacity, A0
shared no-H background, A1 coherent full-H CE, A3 independent-H1 owner-mean
CE, A4 once-per-image any-valid native-row mass, A7 no-replay, A8-prime
coherent token bottleneck, and A6 only when an eligible native donor exists.
A2/A5, candidate-tree training, online refresh, GT-IoU search, K-miss
supervision, random-permutation sweeps, and external owner bridges remain out.

Two implementation decisions are easy to lose:

1. Discovery uses four physical batches of four explicit `n=1` requests per
   image, seeds `21001..21016`, temperature `0.4`, top-p `0.95`, repetition
   penalty `1.10`, max-new-tokens `512`. Clean greedy remains Source-matched HF
   batch-size one with repetition penalty `1.0`.
2. The user requires every later complete class-agnostic pred-pred
   `IoU>0.95` row to count as duplication in chronological order. Therefore
   duplicate classification precedes matching: that row is excluded from
   `G/U/H`, replay, targets, A4, and final owner credit, while every Source/K
   event receives stable raw-state final-coordinate unlikelihood. Never allow
   one row to be both a dense-owner positive and a duplicate negative.

The initial update contract is language-tower DoRA only, AdamW `1e-5`, betas
`(0.9,0.999)`, epsilon `1e-8`, no weight decay, clip `1.0`, cosine zero-warmup
sixteen-update horizon, with normalized `(H,replay,dup)` coefficients A0
`(0,1,1)`, A1/A3/A4/A6/A8-prime `(1,1,1)`, A7 `(1,0,1)`, full-GT `(1,0,0)`.
Packing is isolated no-padding varlen under 12,000 tokens; all packs share one
parameter state before one optimizer step. Across GPUs, parallelize arms as
independent world-size-one processes, at most eight. This does not claim
prefix/image/FLOP reuse.

The first frozen packet drew no P0 from Sol but three P1, and one P0 plus one
P1 from Fable. Reconciliation froze dose/stops, added the full-panel
Source-plus-208-K discovery and census gate, renamed A0, made duplication
matching-consistent, and used stable logit-space unlikelihood. Both original
reviewers then returned `PASS_FOR_USER_APPROVAL` on corrected packet SHA-256
`d0eaf04d58910b152ef2c9b55bf7049f05775c509f7a1ba5f95d81c14ece6830`.

Nothing has been implemented or run. The next action is a user decision on
whether to begin the OpenSpec implementation. Model/GPU discovery, the
one-image update slice, the full matrix, and any 100-update long run remain
separate later gates.
