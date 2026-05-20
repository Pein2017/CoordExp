## 1. Config And Contracts

- [ ] 1.1 Add typed Stage-2 config entries for the residual-set correction objective, UL mining, roll-in policy, coordinate span policy, STOP margin diagnostics, and artifact toggles.
- [ ] 1.2 Add config validation that rejects removed duplicate/coord/bbox objective names in the residual-set path.
- [ ] 1.3 Add baseline-preserving configs for hard SFT/current Stage-2 objective versus residual-set correction ablations.
- [ ] 1.4 Update run metadata/provenance so residual-set configs record objective id, base seed `17`, roll-in policy, UL thresholds, and STOP margin weight.
- [ ] 1.5 Add the `residual_set_correction` objective module name with `application.preset: rollout_self_prefix` and strict config keys.
- [ ] 1.6 Align baseline module/preset names across docs, specs, config validation, and runtime registry before adding residual-set smoke configs.

## 2. Residual State Machine

- [ ] 2.1 Implement a Stage-2 residual-state model for grammar-valid compact-full prefixes.
- [ ] 2.2 Implement desc-gated one-to-one matching for completed emitted objects before they enter `E_t`.
- [ ] 2.3 Implement token-level `ValidAction` enumeration with transition-validated `next_state`.
- [ ] 2.4 Implement active-candidate transitions for description, schema, bbox coordinate slots, and STOP.
- [ ] 2.5 Add invariant tests that `valid_actions` and transitions cannot produce empty corrected roll-in states.
- [ ] 2.6 Add tests that shared next tokens produce one coalesced `ValidAction` whose candidate subset is the union of compatible objects.

## 3. Correction Events And Roll-In

- [ ] 3.1 Implement unified `CorrectionEvent` records with anchor position, residual state, active candidates, valid actions, observed bad token, and provenance tags.
- [ ] 3.2 Implement earliest-actionable event selection across transition failures, premature STOP, FP boundary, repeated-object boundary, and matched-object repair.
- [ ] 3.3 Implement corrected teacher-forced sequence construction where raw bad tokens are provenance only.
- [ ] 3.4 Implement deterministic `random_valid_branch` roll-in with `fixed_event` resampling and base seed `17`.
- [ ] 3.5 Add validator coverage for `logit_position + 1 == target_position`, selected-token equality, and prompt/assistant boundary safety.

## 4. Coordinate And STOP Semantics

- [ ] 4.1 Implement `bbox_tail_from_anchor` coordinate spans.
- [ ] 4.2 Implement dynamic commitment inside coordinate spans: valid-set marginal while ambiguous, hard CE after singleton.
- [ ] 4.3 Implement STOP as a token-level `ValidAction` using `<|im_end|>` and `TokenRole.STOP`.
- [ ] 4.4 Enforce core STOP/continuation exclusivity and keep continuation margin disabled by default.
- [ ] 4.5 Add tests for repeated same-description object branch points at `x1`, shared-`x1` branch points at `y1`, and true/premature STOP.

## 5. UL Consensus Mining

- [ ] 5.1 Implement K-valid rollout eligibility accounting with invalid/ineligible skip reasons.
- [ ] 5.2 Implement per-rollout same-description pre-dedup so one rollout contributes at most one vote to a UL cluster.
- [ ] 5.3 Implement strict complete-link UL clustering with same canonical desc id, all-pairs geometry gates, `K_valid` denominator, `ul_consensus_ratio=1.0` validation, and default `min_ul_valid_rollouts=3`.
- [ ] 5.4 Implement rollout-local `G*_k = labeled GT union ul_promoted_local(k)` and UL emission accounting into `E_t`.
- [ ] 5.5 Apply default `lambda_ul_promoted=0.5` and keep labeled-GT and UL-promoted metrics separate.
- [ ] 5.6 Implement mixed labeled/UL marginal atom weighting: mixed support weight `1.0`, UL-only support weight `lambda_ul_promoted`, with separate mass diagnostics.
- [ ] 5.7 Implement consumed-target overlap quarantine/rejection for UL clusters so cross-rollout duplicate bursts cannot promote into positives.

## 6. Event-To-IR And Loss Integration

- [ ] 6.1 Compile correction events into `TeacherForcingTargetIR` / `SupervisionAtom` without exposing Stage-2 provenance logic to loss modules.
- [ ] 6.2 Add residual-set valid-action marginal support for Stage-2 correction atoms.
- [ ] 6.3 Preserve selected-path hard CE behavior for committed labeled GT and committed UL branches.
- [ ] 6.4 Ensure existing hard-SFT/current Stage-2 baselines remain selectable and unchanged unless their config explicitly opts into the new path.

## 7. Diagnostics And Artifacts

- [ ] 7.1 Emit residual-set correction counters by provenance tag and anchor role.
- [ ] 7.2 Emit valid-action mass, selected-token probability, illegal/STOP mass, and STOP/continuation margin diagnostics.
- [ ] 7.3 Emit labeled remaining, UL remaining, and mixed remaining premature-stop diagnostics.
- [ ] 7.4 Emit UL consensus counters and `ul_clusters.jsonl` rows under monitor/debug/smoke artifact flags.
- [ ] 7.5 Keep repeated-object boundary diagnostics while ensuring no duplicate-specific live loss is emitted.
- [ ] 7.6 Emit residual-set metrics under `stage2_ab/channel_b/residual_set/` and UL metrics under `stage2_ab/channel_b/residual_set/ul/`, with artifact-root provenance for `ul_clusters.jsonl`.

## 8. Verification

- [ ] 8.1 Add unit tests for grammar-valid eligibility, malformed-prefix drops, desc-gated matching, residual-state valid actions, and corrected roll-in invariants.
- [ ] 8.2 Add unit tests for event anchoring, next-token logits alignment, corrected-sequence target positions, and coordinate bbox-tail spans.
- [ ] 8.3 Add unit tests for UL consensus promotion/rejection/quarantine, K-valid denominator, per-rollout pre-dedup, UL artifact rows, and UL loss weights.
- [ ] 8.4 Add config validation tests for residual-set configs and removed objective names.
- [ ] 8.5 Add compatibility tests that legacy hard-SFT/stage2-trie configs still use legacy clean-prefix target construction and residual-set configs do not.
- [ ] 8.6 Add config tests that residual-set `num_rollouts >= 3` works with legacy pseudo-positive disabled, while non-residual pseudo-positive-disabled K>2 behavior remains unchanged.
- [ ] 8.7 Run targeted tests with `conda run -n ms python -m pytest <targets>`.
- [ ] 8.8 Run Stage-2 smoke configs for hard SFT/current baseline and residual-set correction paths, recording config path, checkpoint, seed, output dir, metric files, parse/drop counters, and artifact roots.

## 9. Docs And Handoff

- [ ] 9.1 Update Stage-2 runbook and implementation map after implementation validates the final config/artifact names.
- [ ] 9.2 Record experiment scope and results in `progress/` before interpreting model-quality changes.
- [ ] 9.3 Prepare the super-power implementation plan with file ownership, task decomposition, verification commands, and review checkpoints.
