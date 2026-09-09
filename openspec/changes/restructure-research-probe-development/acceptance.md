# Slice acceptance

Implementation is in progress. A passed slice does not certify the unfinished model path or scientific results.

## Aligned scoring and unused scaling diagnostic

Lead inspected the seven-file change and the actual runtime neutral-accumulation guard. Aligned token scores preserve device/gradients and expose no reduction. Packed CE uses the same operation. Removed the definitions/exports/two dedicated tests of the unconsumed scaling diagnostic; no runtime guard was removed.

- Before change: packed full/compact gradient characterization and normalization/runtime checks,44passed.
- Worker final: `python -m pytest -q tests/losses tests/runtime/test_train_runtime.py`,111passed.
- Lead replay: `python -m pytest -q tests/losses/test_token_scores.py tests/losses/test_context_and_terms.py tests/runtime/test_train_runtime.py::test_accelerator_accumulation_must_be_neutral`,26passed.
- Standards: scoped diff, meaningful caller/gradient checks, and source/API residue checked.
- Intent: fewer caller assumptions and removal of an unused receipt with the real scaling guard retained. Lead-accepted for tasks5.1/5.2.

## Assignment and independent row-cross reducer

Extracted the existing cardinality-first, quantized-IoU matcher and pixel IoU without changing the greedy visualization contract. The reducer now uses same-checkout parsing/matching and an explicit historical-code root; seven old manifest bindings were verified from preserved bytes and never imported.

- Worker targeted suite:27passed, including original matcher characterization.
- Original and migrated saved-input reduction agree on every historical result field:4 qualification cases,16 complete outputs. The added `execution` field distinguishes current reducer identity.
- Lead executed the documented CLI into `preservation-20260909/row-cross-lead-replay`; exact equality after removing only `execution` was asserted.
- Worker independent-copy run forbade all original-worktree file opens and had no scripts directory; same result. This is offline qualification-subset acceptance, not final32-case or model evidence.
- Standards: public owner, original counterexample, no import fallback and exact saved-input consumption verified. Intent: shared assignment plus usable package without sibling code. Lead-accepted for tasks3.1-3.4.

## Research profile entry

A real Human13/logit package entry reproduced the production-directory rejection before model work. The shared research loader uses the same V1 resolution/value/fingerprint code, without production-only authoring rules. Two new tests failed before the API existed; six targeted production/research/invalid-value tests then passed. The packages compare against original effective-config snapshots, including debug.smoke=False. No model or data semantics changed.

## Explicit parameter and attachment inputs

Existing config entry points now use the same attachment/group core as direct research callers. Named parameter selection preserves identity/order without changing gradient flags; state-key normalization is the unchanged transformation exposed publicly.

Worker pre-change122passed/2failed and post-change126passed/the same2failed. The two source-gate cases reference absent historical owner-commit-binding paths; their checks were not relaxed. Lead inspected all three core diffs and replayed adapter setup/reload, parameter groups and special-embedding checks, excluding those exact pre-existing failures:85passed. Direction consumers use direct public attachments/selection; simple native AdamW remains valid. Standards and intent both accepted for the changed inputs; no claim that the external source-gate fixture failures are repaired.
