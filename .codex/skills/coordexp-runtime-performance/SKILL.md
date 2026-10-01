---
name: coordexp-runtime-performance
description: Find runtime bottlenecks and verify infrastructure speedups in CoordExp training, inference, or frozen post-training probes. Use for slowness and throughput; routine launches and model-quality questions belong to their existing workflows.
---

# CoordExp Runtime Performance

Own performance attribution, optimization priority, and measured benefit. Use the
selected checkout and the existing task/config contract. Optimize the requested
completion latency or sustained throughput, with resource cost reported alongside
it; reuse existing run evidence before adding instrumentation.

## Interpret the measurements

- Read timing scope and work counters together. Separate startup/cache preparation
  from steady execution when relevant. Parallel job or rank durations cannot be
  summed into elapsed wait time; inspect the slowest dependent path.
- Current training receipts may expose `step_duration_seconds` and
  `throughput/physical_tokens_per_second`, `throughput/supervised_atoms_per_second`,
  or `throughput/packs_per_second`. Their work units differ. Verify aggregation
  in the selected checkout before comparing runs; the maintained training rate
  uses global work divided by the all-rank maximum step duration.
- Inference performance marked `parallel_rank_backend_decode_capacity_estimate`
  with `controller_wall_time_observed: false` assumes perfect rank overlap. It is
  not observed end-to-end wall time. Use command elapsed receipts or the applicable
  run's entry-to-terminal measurement for a completion-time claim.
- Attribute only the unresolved cost: input preparation, model load, vision or
  prefill, decode/replay, backward/synchronization, or output/save/evaluation.
  Select existing counters first; do not instrument every stage by default.

## Select the smallest useful change

Use measured stage shares, work counts and rank skew to rank candidates. A small
input-build fraction is weak justification for a data-pipeline rewrite. Rank
skew can reflect assigned work or different execution conditions; compare work
and timing before choosing a load-balancing change.

Discover the current checkout's cache-preparation, batching and parallel paths
before prescribing a command. Available model-free preparation and bounded
preprocessing differ across worktrees. Prefer an existing supported mechanism;
this skill supplies no batch size, worker count, backend or precision defaults.

Count repeated work as a possible cost, but let the existing run/attempt owner
decide whether results can be reused. Required full-prefix forwards, generated
history, supervision and decode budgets are part of the active contract. A
smaller workload does not establish a faster implementation of the same task.

## Verify the claimed benefit

Compare a representative baseline and candidate through the relevant real entry.
Separate cold and warm measurements when cache/startup reuse affects the task;
account for changed hardware, concurrency and timing variability. Report the
realized work, elapsed time and relevant resource tradeoff, including peak
RSS/VRAM or GPU time when they affect the decision. Local timings explain a
result but do not substitute for the requested completion-time or throughput
measurement. Stop when the scoped comparison settles the choice; no demonstrated
benefit is a valid outcome.

Keep correctness and numeric acceptance at their existing owners. Use only the
reference implicated by the candidate:

- Forward, packing, replay or gradient equivalence: the relevant section of
  [Qwen execution checks](../qwen3-vl-execution/references/execution-checks.md).
  Use the active contract's equivalence criterion and tolerance.
- Inference launch, qualification, artifacts or stage repair:
  [infer/eval workflow](../coordexp-infer-eval-workflow/SKILL.md).
- A named real-entry, distributed, scale or persistence risk:
  [full-pipeline-smoke](../full-pipeline-smoke/SKILL.md).
- Frozen cell identity, attempts or recovery evidence:
  [probe execution packet](../research-flow/references/probe-execution-packet.md).

These are conditional references, not a required sequence of skills or approvals.
