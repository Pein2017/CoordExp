# Tasks

These tasks are proposal and implementation gates. Production code must not be
edited until the user explicitly approves implementation after subagent
convergence.

## 1. Proposal And Review

- [x] 1.1 Draft the OpenSpec change for unified training runtime boundary
  hardening.
- [x] 1.2 Draft the associated Superpowers design and plan documents.
- [x] 1.3 Review the OpenSpec proposal with independent subagents for
  governance scope, Stage-2 trainer boundary risk, shared inference/artifact
  risk, and Superpowers planning clarity.
- [x] 1.4 Revise proposal, design, spec deltas, and Superpowers documents from
  reviewer findings.
- [x] 1.5 Record convergence status in the final user-facing summary.
- [x] 1.6 Receive explicit user approval for the P0 eval-validity
  implementation slice.
- [x] 1.7 Write a separate P0 implementation plan before production-code edits.

## 2. P0 Eval-Validity Slice After Approval

- [x] 2.0 Add P0 eval-validity tests before broader runtime-boundary work:
  Stage-2 metric-bearing eval must fail before official artifact
  materialization when source image identity/dimensions are missing, when
  source record identity is missing, when parser output is diagnostic/salvage
  or raises before metric-bearing output, or when prompt provenance is missing,
  synthetic, or not bound to exact prompt-token metadata.
- [x] 2.0a Implement the P0 eval-validity slice before broader runtime
  projection, target-construction, DDP/packing, or A/B deletion work.

## 3. Characterization Gates After P0 Slice

- [x] 3.1 Add tests proving `custom.trainer_variant` rejects unknown non-empty
  variants unless an explicit extension contract exists.
- [x] 3.2 Add tests proving Stage-2 runtime projection records authored and
  resolved policy sources without relying on hidden `custom.extra` fallbacks.
- [x] 3.3 Add tests proving rollout-correction target construction can be
  exercised without vLLM lifecycle, DDP coordination, or full trainer setup.
- [x] 3.4 Add tests proving post-rollout packing/DDP coordination has a bounded
  responsibility surface around pack production/consumption, rank behavior,
  zero-pack behavior, shadow slots, final-sync barriers, and pack metrics.
- [x] 3.5 Add tests proving shared inference modules do not depend on broad
  trainer/offline-owner private methods outside designated edge adapters.
- [x] 3.6 Add tests proving raw/scored artifact provenance has one writer and
  metric-bearing parser status is auditable.
- [x] 3.7 Add search gates for retired A/B or Channel-B public surfaces, dead
  `rollout_matching` manifest family paths, and retired current-authority docs.

## 4. Runtime Projection And Bootstrap After P0 Slice

- [x] 4.1 Move Stage-2 runtime policy projection into a dedicated
  config/runtime module while preserving public YAML namespaces.
- [x] 4.2 Keep `src/sft.py` as the launcher that obtains resolved runtime state
  and passes it into trainer/bootstrap setup.
- [x] 4.3 Preserve run manifest, policy provenance, and pipeline manifest output
  contracts.
- [x] 4.4 Update docs/spec references only after tests prove behavior parity.

## 5. Stage-2 Trainer Boundary After Runtime Projection

- [x] 5.1 Extract or deepen the rollout-correction target-construction
  responsibility around parsed rollout/GT/correction-policy inputs.
- [x] 5.2 Keep rollout backend lifecycle, DDP coordination, model forward, loss
  execution, and metric projection outside that target-construction boundary.
- [x] 5.3 Narrow packing/DDP coordination from `owner: Any` toward explicit
  responsibilities after characterization tests exist.
- [x] 5.4 Convert obsolete A/B and Channel-B behavior tests into absence or
  rejection tests before renaming or deleting compatibility aliases.
- [x] 5.5 Classify remaining A/B or Channel-B old-name matches as active public
  surface, private implementation identifier, rejection/absence test, migration
  adapter, historical fixture, or archived history before deleting/renaming.

## 6. Shared Inference And Artifact Boundary After P0 Slice

- [x] 6.1 Confine trainer/offline owner introspection to designated edge
  adapters.
- [x] 6.2 Route prompt/decode/backend/artifact helpers through resolved facts
  rather than arbitrary private owner attributes.
  - [x] 6.2a Route rollout prompt normalization through a resolved prompt-facts
    helper while keeping `*_from_owner` as the edge adapter.
  - [x] 6.2b Complete the deeper backend/decode/artifact lifecycle facts
    migration so shared inference no longer needs broad owner-shaped helper
    files as migration adapters.
- [x] 6.3 Single-own score sidecar construction, score policy fingerprinting,
  parser policy, and metric-bearing status.
- [x] 6.4 Preserve raw/scored artifact separation and Stage-2
  `eval_detection/step_<global_step>/` artifact names.
- [x] 6.5 Ensure provenance-free raw or F1-ish eval paths report
  `inspection` / `non_comparable` status and cannot be reported as official
  comparable evaluation.

## 7. Final Cleanup And Validation After Approval

- [x] 7.1 Delete or quarantine dead `rollout_matching` manifest family branches.
- [x] 7.2 Demote retired specs and progress notes from current routing
  authority.
- [x] 7.3 Tighten architecture gates to match the moved boundaries.
- [x] 7.4 Run narrow contract suites first, then broaden only across affected
  config/runtime/infer/eval surfaces.
