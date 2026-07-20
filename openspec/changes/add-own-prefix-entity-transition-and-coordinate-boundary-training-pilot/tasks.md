## 0. Lead-Owned Pre-Fork Fixture

- [ ] 0.1 Before implementation worktrees are forked, publish one immutable smoke fixture manifest that binds the shared source checkpoint, state-bank checksum, exact transition and geometry event identifiers, optimizer and loss values, and random seeds.

## 1. Frozen State-Bank Contract

- [ ] 1.1 After the shared smoke fixture exists, add focused failing tests for checkpoint identity, exact token and hash preservation, image-grouped splits, physical-entity aliases, separate entity and geometry eligibility, axis-specific unknown-as-zero-gradient behavior, and coordinate acceptable-set axis and range validation.
- [x] 1.2 Implement the minimum offline assembler that combines exact rollout artifacts with the reviewed physical-entity ledger and emits immutable state-bank records without making new review judgments.
- [x] 1.3 Implement the minimum state-bank manifest and record loader that satisfies the contract tests without inferring review labels inside training code.
- [x] 1.4 Add one compact validation receipt that binds bank identity, source checkpoint, split counts, event-family counts, and rejection reasons.

## 2. Exact Replay and Atomic Event Materialization

- [x] 2.1 Add a reference parity test proving that one stored image and exact token prefix reaches the same pre-candidate logits as its frozen source event without decode-and-retokenize reconstruction.
- [x] 2.2 Implement the narrow exact-token replay seam by reusing current image preparation, Qwen position construction, forward behavior, and segment isolation.
- [x] 2.3 Implement companion calibration metadata and atomic event planning so every positive and harmful candidate group remains complete within one planned optimizer step.

## 3. Research Losses

- [x] 3.1 Add 32-bit floating-point unit tests and implement grouped entity-transition preference with maximum-over-alias owner reduction, multiple valid owner scores, and the actual admitted harmful branch, including a null-owner single-token premature-terminal case.
- [x] 3.2 Add 32-bit floating-point unit tests and implement first-wrong-coordinate preference over a reviewed discrete acceptable-token set.
- [x] 3.3 Add tests and implement intended token-type gating for every selected positive and harmful research site, including object-row schema rather than terminal type at a premature-terminal boundary, identical-declaration deduplication, conflicting-type rejection, and event-balanced normalization.
- [x] 3.4 Add separate complete-planned-step eligibility denominators, joint-arm weighting, margin diagnostics, and finite checks for all enabled research terms.

## 4. Config, Pipeline, and Artifacts

- [x] 4.1 Extend strict config validation with transition-only, coordinate-boundary-only, coordinate-boundary-gate-only, and joint rollout-calibration profiles while preserving all ordinary supervised-training defaults.
- [x] 4.2 Wire the profiles through the existing training assembly, Weight-Decomposed Low-Rank Adaptation source-checkpoint loading, optimizer, gradient clipping, Accelerate runtime, and checkpoint writer without adding a second trainer or runtime.
- [x] 4.3 Extend existing rank-zero run and logging artifacts with bank identity, enabled research profile, raw and weighted losses, eligibility counts, target margins, token-type legal mass, ignored and unknown counts, and finite status.
- [x] 4.4 Add negative config tests proving that canonical supervised-fine-tuning mixtures, full-row base cross-entropy, zero rollout-site token-type-gate weight, Kullback-Leibler divergence anchoring, online bank refresh, and cross-checkpoint bank reuse are rejected in this profile.

## 5. Smoke Evidence

- [x] 5.1 Build and run the deterministic one-transition-plus-one-geometry smoke; verify exact replay, masks, selected positions, intended token types, finite gradients, and intended margin movement after one tiny update.
- [x] 5.2 Build and run the 13-reviewed-coordinate-state smoke for coordinate-boundary and learning-rate-matched coordinate-boundary-gate-only profiles; verify parseable ordinary free-row inference from each produced adapter and prove that no canonical supervised-fine-tuning examples were consumed. Keep transition and joint profiles deferred until a visually trusted same-prefix transition positive exists.
- [x] 5.3 Verify that each smoke checkpoint loads through the existing inference composition and requires no state bank or custom controller at inference time.

## 6. Implementation-Race Handoff Gate

- [x] 6.1 Run targeted config, state-bank, replay, packing, loss, training-pipeline, checkpoint, and inference tests plus strict OpenSpec validation and residue checks for forbidden alternate trainers or inference paths.
- [x] 6.2 Produce one compact implementation receipt listing changed owner surfaces, exact commands, test and smoke outcomes, runtime and peak-memory observations, known limitations, and any deviation from the frozen research unit.
- [x] 6.3 Stop before the formal 256-image and 12-job training screen; leave that launch for lead-agent comparison of the independently implemented worktrees and a separate user authorization.
