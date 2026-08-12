## Why

The live CoordExp-Swift training route now contains a candidate opt-in, same-world-size training-state path and stricter cache, runtime, and provenance controls, while stable specs and canonical docs still describe every checkpoint as inference-only. The candidate path is not a supported exact-resume capability yet: its distributed publication, interruption, admission, and continuation claims remain conditional on this change's executed gates. This change reconciles those authority surfaces only to the behavior that survives qualification, without inheriting unfinished claims from the superseded broad infrastructure change.

## What Changes

- Qualify, and only then specify, an explicitly typed opt-in `training_state/` sibling of the independently loadable minimal inference payload; a failed qualification narrows or removes the proposed exact-resume contract instead of promoting the live candidate implementation.
- Preserve the compatibility default: when exact state is disabled, checkpoint publication writes no training-state payload and inference behavior remains unchanged.
- If qualification passes, bound the supported continuation contract to optimizer-step boundaries, the same world size, strict fail-closed admission, and additive parent/child run lineage; inference loaders ignore the training-state sibling.
- Reconcile strict config, checkpoint/run artifacts, executed-runtime provenance, and pack-cache identity/admission requirements only where current source, tests, and accepted receipts establish the behavior.
- Classify provider authority before implementation: the stable `coordexp-swift-packing-forward` contract admits only explicit `synchronous` and `overlapped` modes; `legacy_fused` and `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` are unsupported live residue, not capabilities to synchronize into stable specs.
- Update canonical operator documentation only after qualification so it no longer denies the resulting bounded capability or conflates model payloads with reproducible training state.
- Re-run the bounded interruption/incomplete-rank and final compatibility gates rather than inheriting their old unchecked status. Leave changed-order packing promotion, speculative efficiency work, logging enhancements, loss-objective changes, and architecture decomposition outside this change.

## Capabilities

### New Capabilities

- `coordexp-swift-training-resume`: Conditionally defines the bounded opt-in exact-training-state sibling, same-world-size optimizer-step-boundary admission and restore contract, and append-only continuation lineage. It becomes stable authority only after every qualification gate passes and the delta is synchronized.

### Modified Capabilities

- `coordexp-swift-config-runtime`: Reconciles the strict authored resume and deterministic-runtime controls already accepted by the live training route.
- `coordexp-swift-training-artifacts`: Distinguishes the minimal inference payload from optional exact training state and records only the live, evidenced checkpoint, provenance, and run-publication behavior.
- `coordexp-swift-pack-cache-semantic-identity`: Reconciles current immutable cache identity and fail-closed admission behavior that exact continuation binds to, without promoting an unverified packing policy.

## Impact

- Stable contracts under `openspec/specs/coordexp-swift-{config-runtime,training-artifacts,pack-cache-semantic-identity}/` and one new `coordexp-swift-training-resume` contract.
- Canonical training and artifact documentation, especially `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and `docs/ARTIFACTS.md`.
- Existing config, checkpoint, training-state, cache-admission, run-lineage, and provenance tests become the acceptance evidence; source changes are limited to resolving a demonstrated contract mismatch and do not add a new runtime mode, objective, telemetry sink, packing policy, or inference requirement.
