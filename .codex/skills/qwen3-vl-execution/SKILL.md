---
name: qwen3-vl-execution
description: Explain, implement, or debug local CoordExp Qwen3-VL encoding, forward/replay, generation, packing, and loss or gradient behavior. Routine inference and evaluation launches belong to coordexp-infer-eval-workflow.
---

# Qwen3-VL Execution

Use this as a technical entry, not an additional review or launch gate. Work in
the user-selected checkout and respect the requested read/write boundary.
Resolve its actual model/runtime and scientific owner before choosing an API:
the research base supports direct native execution; packed training and strict
scored inference have different contracts. Do not move a research caller into
either framework just to obtain a model or token scores.

For runtime migration, follow [the project runtime contract](/data/CoordExp/AGENTS.md)
and the selected checkout's dependency/admission owners. Migrate actual upstream
callers and preserve token, image, position and likelihood semantics; historical
interface receipts do not qualify the changed source.

## Load only the relevant knowledge

- For architecture, processor/image grids, MRoPE, compact logits, cache or
  DeepStack hooks, use exact-checkout CodeGraph to find the actual caller and
  inspect its source plus the installed dependency implementation. Verify loaded
  paths and versions for version-sensitive behavior; do not maintain a parallel
  upstream implementation manual or use another branch's notes as runtime truth.
- For precision, normalization, causal alignment, batches or template assembly,
  use only the applicable section of [execution checks](references/execution-checks.md).
  It points to code/spec/test owners rather than another execution framework.
- For candidate-set probability, row scoring, finite-set mass or reducers, read
  its `Objective, denominator and distributed reduction` section. Bind the real
  consumer schema and test empty support, `null`/finite representation and EOS
  behavior rather than inferring them from nonempty examples.
- For a training launch, discover the selected checkout's config/launcher;
  this skill owns no GPU count, batch size, precision default or launch budget.

Start from the named source/artifact and retrieve one relevant contract or
counterexample. Familiar tensor operations do not require a tutorial or a
full checklist. Correct shapes and a finite scalar do not by themselves prove
correct target alignment, reduction, autograd or conditioning.

When implementing or running an already frozen research probe, also use the
[Frozen Probe Execution Packet](../research-flow/references/probe-execution-packet.md)
for producer identity, attempt handling and dependent-artifact readiness.

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
