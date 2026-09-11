## Why

Recent Image2299 work and the retired Human13 N/K lineage repeatedly rebuild
source/runtime binding, immutable result publication, failure receipts, and
continuation plumbing even though most of those mechanics already have stable
owners.  The missing piece is a small, discoverable composition contract: new
probe directions need to know which existing capability to reuse without being
forced through a generic runner whose topology or scientific meaning does not
fit.

## What Changes

- Establish a research-probe infra-base as a set of independently selectable
  capability profiles, not a coordinator: immutable one-shot publication,
  journaled multi-item execution, mechanics-only launch admission, and the
  existing inference frontend/backend session are composed only when a probe
  actually needs them.
- Make the stable artifact operations needed by those profiles available from
  one documented public `src.artifacts` surface, while keeping serialization,
  journal, and admission behavior in their current deep owner modules.
- Add one canonical operator guide and route it from the existing research
  worktree policy/index.  The guide will name the selection rule, exact public
  imports, caller obligations, failure meaning, and the boundary between the
  infra integration lane, the canonical `research-probes` baseline, and a
  direction worktree.
- Add small CPU contract examples/tests for a one-shot producer and two
  independently schedulable journaled producers.  They will prove that shared
  mechanics do not impose a global phase order, result schema, retry decision,
  or scientific status.
- Require a concrete second live cross-direction consumer before introducing
  any further shared execution wrapper.  Image2299 remains the live reference
  specimen; Human13 tags and records remain historical characterization rather
  than a revived consumer.
- Explicitly defer a generic runner, phase DSL, optimizer/RNG transaction,
  trainable-HF session, checkpoint abstraction, monitor, and lifecycle
  automation.  Image2299 and Human13 differ materially on these surfaces, so
  promoting them now would encode experiment semantics as infrastructure.

## Capabilities

### New Capabilities

- `coordexp-infras-research-probe-infra-base`: defines the minimal composable
  profiles, their public owners, per-producer topology rule, mechanics-only
  evidence boundary, and the evidence threshold for adding another shared
  layer.

### Modified Capabilities

None.  The journal, admission, inference, and worktree-lifecycle contracts keep
their existing semantics; this change standardizes their composition and
public discoverability rather than widening them.

## Impact

- Expected implementation surfaces are `src/artifacts/__init__.py`, focused
  artifact/profile tests, one operator-facing guide under `docs/`, and links
  from `docs/AGENT_INDEX.md` and `docs/BRANCH_AND_WORKTREE_POLICY.md`.
- Existing deep owners remain `src/artifacts/json_values.py`,
  `src/artifacts/evidence_journal.py`,
  `src/artifacts/research_probe_admission.py`,
  `src/inference/runtime.py`, and `src/inference/backend.py`; no new runtime
  package or dependency is introduced.
- No active Image2299 file, historical Human13 tag, scientific unit, external
  output, GPU process, worktree, branch, tag, or stable lifecycle policy is
  mutated by this change.
- A later promotion may add a new runtime or rollback capability only after a
  live second direction demonstrates the same contract and a bounded
  counterexample shows that direct composition is insufficient.
