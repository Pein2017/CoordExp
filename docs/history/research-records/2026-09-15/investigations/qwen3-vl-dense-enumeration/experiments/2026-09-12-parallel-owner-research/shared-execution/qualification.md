# Shared two-rank mechanical qualification

Status: root granted the frozen packet and tolerances; the one two-rank launch,
cold load, and comparison all passed. See `results.md`. This is not model-quality evidence.

Question: does the new literal-record consumer preserve the old C2 objective
and produce a numerically compatible adapter/update when normal56 is sharded
over two rather than eight ranks?

Contrast: old three complete rows, old three conditional successors, unchanged
normal56/6047 KL positions/6030 margin positions, same Stable50, fresh AdamW,
two fixed updates. Coefficients 1/10/100/10 and denominators 3/3/56/56.
No sampled negatives, no new labels or history interventions.

Oracle: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/smoke-weight10/receipt.json`
SHA256 `2a9dcdf21b4f47943ba691e32dcd261e840732f00ca35973f8c6dc2d910098be`.
Current official old verifier passes. It contains no archived raw-gradient
tensors: gradient-norm agreement is not gradient-vector parity. The acceptance
claim is faithful objective plus numerically compatible update and consumer.

## Frozen numeric acceptance (before new execution)

- Each global objective component at each step: absolute error <=
  `1e-5 + 1e-4*abs(old component)`.
- Each raw gradient norm: absolute error <= `1e-5 + 1e-4*abs(old norm)`.
- Each parameter update L2 norm: absolute error <= `1e-7 + 1e-3*abs(old norm)`.
- Final saved adapter: same 588 tensor keys, 18,006,016 scalars, fp32 shapes;
  maximum absolute element difference <= `2e-6`; full adapter-change vector
  relative L2 error <= `0.005`, cosine >= `0.99999`, using unchanged Stable50
  as the origin for both change vectors.
- Final positive replay: exact token counts and target-argmax counts;
  each summed/mean target log probability and mean/min target margin within
  `5e-4` absolute of old C2.
- Saved/cold same route: exact discrete counts and <= `1e-5` absolute error
  for those four scores; identical saved adapter and training input identities.
- All ranks: exact same reduced gradient, optimizer and adapter hashes within
  the run; exactly one synchronized backward per step; explicit schedule counts;
  frozen nonadapter parameters unchanged; successful whole-lifecycle terminal.

Any failed invariant yields a technical qualification failure, retained raw,
not a scientific null. Diagnose/correct faithful implementation once evidence
identifies a mechanism. Do not relax the tolerances after looking at the result.

## Model work and resource bound

Reserved on request: physical GPUs 2,3. Two ranks, two updates. Per rank:
31 reference replays, 68 training replays/backwards, 28 final normal readbacks;
rank0 additionally six positive score replays. Total133/127 model forwards,
260 training-route forwards globally; no generation. The separate cold load
adds one model load and three positive replay forwards on physical GPU2.
Three model loads globally including cold. Known old8 C2 used roughly113s and
12.5GB peak RSS/rank; two-rank normal-cache load is four times the old per-rank
normal cache. Estimate 3–6min training plus cold, not measured yet.

Per invocation bounds: 1200s/rank, 24GiB CUDA allocated/reserved, 32GiB RSS,
160 model/image forwards/rank. These bounds freeze one verification command,
not the portfolio cost. One new C2 route plus cold/CPU comparison is the initial
mechanical stop. No extra eight-rank rerun without root decision.

## Exact commands after packet preparation and root grant

Working directory `/data/CoordExp/.worktrees/research-probes`.
`RAW=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/shared-execution`.

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m probes.parallel_owner_research.training launch \
  --input "$RAW/preparation-v1/inputs.json" --arm C2 --world-size 2 \
  --output-root "$RAW/qualification-C2-two-rank-v1"
CUDA_VISIBLE_DEVICES=2 python -m probes.parallel_owner_research.training cold-check \
  --input "$RAW/preparation-v1/inputs.json" \
  --output-root "$RAW/qualification-C2-two-rank-v1"
```

CPU comparison calls `training.compare_qualification(output_root=Path(...),
oracle_root=Path(the old smoke-weight10 root))`; it writes a full comparison
before asserting pass. Launch log/exit, rank logs, update rows, terminal resources,
saved adapter, cold score deltas and comparison remain under the run root.
