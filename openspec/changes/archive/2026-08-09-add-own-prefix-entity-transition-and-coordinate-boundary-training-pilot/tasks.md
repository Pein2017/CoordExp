## 0. Lead-Owned Pre-Fork Fixture

- [ ] 0.1 Before implementation worktrees are forked, publish one immutable smoke fixture manifest that binds the shared source checkpoint, state-bank checksum, exact transition and geometry event identifiers, optimizer and loss values, and random seeds.

  This procedural pre-fork requirement was not satisfied retroactively. Later
  lead-run research units used their own hash-bound StateBanks and real-model
  smoke receipts; that evidence does not rewrite the original implementation-
  race condition.

## 1. Frozen State-Bank Contract

- [ ] 1.1 After the shared smoke fixture exists, add focused failing tests for checkpoint identity, exact token and hash preservation, image-grouped splits, physical-entity aliases, separate entity and geometry eligibility, axis-specific unknown-as-zero-gradient behavior, and coordinate acceptable-set axis and range validation.

  The implementation now has focused coverage for these contracts, but this
  task remains open as written because it was conditional on the missing
  shared pre-fork fixture.
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

## 7. Compatible Off-Policy StateBank Replay

- [x] 7.1 Add focused config and pipeline tests proving strict replay remains the default, compatible off-policy replay permits only adapter and selected-token embedding-delta differences, and all base/token/processor identity mismatches fail before forward.
- [x] 7.2 Implement one default-false off-policy replay switch without adding online collection, a second trainer, or a multi-bank scheduler.
- [x] 7.3 Record both the immutable trajectory-source checkpoint identity and the training-warm-start checkpoint identity in rank-zero run evidence.
- [x] 7.4 Extend the inference-to-StateBank builder so a refreshed bank is bound to the checkpoint that actually generated its trajectories rather than copying an older reference-bank checkpoint identity.

## 8. Mixed Old-Prefix and Refreshed-Prefix Coordinate Correction

- [x] 8.1 Run a real smoke from Treated-v1 that consumes the old Source-prefix StateBank through explicit off-policy replay and verify truthful dual-checkpoint evidence.
- [x] 8.2 Train one common old-prefix epoch from Treated-v1, run clean inference from the shared intermediate checkpoint, and build a refreshed StateBank from its actual prefixes.
- [x] 8.3 Freeze one paired image cohort admitted by both the old and refreshed builders, then create equal-sized old-prefix and refreshed-prefix branch banks over exactly those images.
- [x] 8.4 From the identical shared intermediate checkpoint, train the Old-Prefix Repeat Control and the Refreshed-Prefix Treatment with equal event and optimizer budgets.
- [x] 8.5 Compare exact-prefix 32-bit floating-point margins and clean self-rollout geometry, detection, duplicate, invalid, and truncation behavior; update the research unit with the decision and artifact handles.

## 9. Exact Greedy Prefix Uncovered-Object Row Treatment

- [x] 9.1 Close the proposed first-divergence loss and target-scoped coverage
  extensions without implementation after the event census showed that the
  exact-terminal family was too sparse for the intended 256-image screen.
- [x] 9.2 Add the smallest exact-prefix terminal-rescue collector and assembler
  that reuses ordinary inference, preserves integer token identifiers, samples
  sixteen independent complete rows, and emits at most one trusted event per
  training image.
- [x] 9.3 Run a one-real-event smoke proving exact replay parity, intended
  branch-margin and continuation-likelihood movement, finite gradients, and
  ordinary greedy inference from the produced checkpoint.
- [x] 9.4 Cancel multi-image training after only three unique events survived
  the 256-image census and the one-event checkpoint produced a broad person
  repetition burst; publish the bounded negative result.

## 10. Best Sampled Trajectory Positive Row Imitation

- [x] 10.1 Add a backward-compatible `positive_path_imitation_only` profile,
  event metadata, 32-bit floating-point complete-row loss, image-balanced event
  weighting, and focused config, StateBank, loss, and pipeline tests.
- [x] 10.2 Add the experiment-local deterministic route selector and StateBank
  assembler; freeze the 256-image, sixteen-sample receipt; enforce physical-
  owner trust and a batch-fit event count; and audit representative enlarged
  crops before training.
- [x] 10.3 Run one real one-event, one-step smoke proving exact replay, selected
  sites, finite gradients, checkpoint loading, and ordinary greedy inference.
- [x] 10.4 Build the expected 512-event bank from 118 images, train one full
  eight-Graphics-Processing-Unit epoch at learning rate `1e-5`, and save steps
  5, 10, 15, and final step 16.
- [x] 10.5 Compare Source and treatment milestones under identical clean greedy
  inference on train-256 and the twelve human-refined development images, with
  entity discovery and geometry reported separately.
- [x] 10.6 Decide whether the combined evidence is promising enough for a
  1,024-image replication; do not require a predeclared numerical margin. The
  identical replication is not promoted because all evaluated milestones
reduce aggregate unique annotated-owner coverage. A targeted follow-up shows a
route-level shift toward the fixed route-added owner set but nearly equal loss
of ordinary owners. Only 118 of 238 route-added owners are direct event
targets, so direct owner-wise imitation is not established. The next candidate
is a matched-arm, preservation-aware 256-image screen rather than the unchanged
1,024-image run.

## 11. Truthful Source-Route Preservation for the Matched Treatment Screen

- [x] 11.1 Add focused StateBank and config tests for a new
  `source_route_imitation_eligible` event family and
  `sampled_path_and_source_route_imitation_only` profile, including truthful
  greedy-versus-sampled provenance, mutual exclusivity, and historical-profile
  isolation.
- [x] 11.2 Reuse the existing complete-row imitation loss, token-type gate,
  exact replay, packing, normalization, and checkpoint path for both event
  families; expose separate admitted-event counts without adding another loss
  implementation.
- [x] 11.3 Build two matched 992-event StateBanks for the 256-image screen:
  one sampled route plus Source preservation and multiple complementary sampled
  routes plus the identical Source preservation set. Verify 496 events per
  family, the same 118 images, equal total image credit, and exclusion of the
  twelve human-refined development images.
- [x] 11.4 Run one real one-event smoke for a greedy Source event and one mixed
  optimizer-step smoke, verifying exact provenance, selected sites, finite
  gradients, checkpoint reload, and ordinary inference before the two formal
  training arms launch.
- [x] 11.5 Train the matched single-route and multiple-route 992-event arms for
  31 optimizer updates and save the declared milestones.
- [x] 11.6 Compare ordinary clean greedy rollout on the full 256-image cohort,
  the admitted and non-admitted subgroups, and the twelve human-refined images.
  The selected-owner signal is real, but unique-owner transfer is negative on
  non-admitted images at every milestone. Do not promote this objective to an
  unchanged 1,024-image epoch. The only authorized follow-on is a separate,
  constant-dose image-breadth discriminator.
