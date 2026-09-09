---
name: qwen3-vl-execution
description: Explain, implement, or debug local CoordExp Qwen3-VL internals, multimodal encoding, forward/replay, generation, packing, and differentiable loss or gradient behavior. Use for model-execution work; routine inference/evaluation launches stay with coordexp-infer-eval-workflow.
---

# Qwen3-VL Execution

Use this as a technical entry, not an additional review or launch gate. Work in
the user-selected checkout and respect the requested read/write boundary.
Resolve its actual model/runtime and scientific owner before choosing an API:
the research base supports direct native execution; packed training and strict
scored inference have different contracts. Do not move a research caller into
either framework just to obtain a model or token scores.

## Load only the relevant knowledge

- For architecture, processor/image grids, MRoPE, compact logits, cache or
  DeepStack hooks, read the matching section of the selected checkout's
  `docs/standards/upstream/QWEN_VL.md`. The maintained research copy is
  [here](/data/CoordExp/.worktrees/research-probes/docs/standards/upstream/QWEN_VL.md).
  Its installed-version scope is explicit; inspect the actual loaded dependency
  when a version-sensitive behavior matters.
- For precision, normalization, causal alignment, batches or template assembly,
  use only the applicable section of [execution checks](references/execution-checks.md).
  It points to code/spec/test owners rather than another execution framework.
- For a training launch, discover the selected checkout's config/launcher;
  this skill owns no GPU count, batch size, precision default or launch budget.

Start from the named source/artifact and retrieve one relevant contract or
counterexample. Familiar tensor operations do not require a tutorial or a
full checklist. Correct shapes and a finite scalar do not by themselves prove
correct target alignment, reduction, autograd or conditioning.

## Close the specific question

State the relevant input-to-output relationship and its owner, then inspect or
exercise the smallest real caller that can distinguish the suspected mistake.
Reuse existing tests and receipts; choose a real model check only when mocks or
saved inputs cannot settle the claim. Keep generation, gradients and numerical
parity claims separate. An existing diagnostic does not become a new acceptance
gate unless the active contract makes it one.

Keep research populations, rewards, owner acceptance and stop rules in their
direction. Use `model-diagnosis` for an observed behavioral symptom;
`model-innovation-risk-audit` only for a decision-bearing mechanism-fidelity
question; `full-pipeline-smoke` when the real entry/distributed/persistence path
is the unresolved risk. Do not run them as a sequence of approvals.
