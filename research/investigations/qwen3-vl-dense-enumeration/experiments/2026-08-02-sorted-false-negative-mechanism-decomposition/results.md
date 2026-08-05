# Sorted False-Negative Mechanism Decomposition and Backfill - Results

## Outcome

The unit closed at the predeclared Stage-A control gate. The executed evidence
is a verified fixed-budget Level 1 (`L1`) subset: 2,885 raw 32-bit floating-point
likelihood rows over five exact contexts for three image-`7511` owners. The
declared wider protocol was not executed, so [the unit](unit.md) is complete
with `evidence_status: partial`.

The decision-bearing execution root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-fn-mechanism-decomposition-smoke-v1`

The owning evidence chain is:

- merged `L1` scores and receipt under `l1-stage-a-controls-merged/`;
- accepted scale-run attestation under `l1-stage-a-controls-attested/`;
- likelihood-only control analysis under `l1-stage-a-controls-analysis/`;
- corrected behavior artifact
  `behavior-due-turn-gt7511-17-greedy-rp1p00-rp1p10-corrected-v2.json`
  with SHA-256
  `9c2e3c659f85a6add6c251578a494b1086a308a1c7ef65df5ac2a2eee7b3b26d`;
  and
- behavior-bound analyzer v3 under
  `l1-stage-a-controls-behavior-analysis-corrected-v3/`, whose analysis and
  receipt SHA-256 values are respectively
  `89c3f9a111c7baa584cbfc77a4b894c4a23bbc03785d7ab0f31b36703219ef17`
  and `f7896b51ea00afeeb50cd1c70d6e5a60bb2e38314107fec925a5092839c2caed`.

The attestation accepted the uncached full-prefix raw model likelihood channel,
bound the model and tokenizer identities, and recorded no scientific conclusion
of its own. Interpretation below is owned by this result record, not by the
artifact directory.

## Observed

- The merged receipt contains 2,885 rows: 577 rows for each of
  `due_turn:gt:7511:17`, `due_turn:gt:7511:22`, `root:gt:7511:17`,
  `root:gt:7511:22`, and `root:gt:7511:26`. Target and decoy populations were
  equal-count, and all rows used the raw unmodified full-prefix likelihood
  channel. Repetition-penalty views `1.00` and `1.10` were behavioral policy
  views, not relabelled model likelihoods.
- At the exact due-turn prefix for strict-rescue control `gt:7511:17`, the same
  target peak survived intersection-over-union thresholds `0.4`, `0.5`, and
  `0.6`: target peak `-46.4691`, background prominence `+22.2252`, localized
  rank `0`, and registered other-owner margin `+3.0237`. All matched rank,
  prominence, margin, and threshold-band gates passed.
- The corrected greedy behavior artifact transferred that likelihood result to
  strict owner recovery at the same exact prefix under both repetition-penalty
  views. The forced-description row strictly recovered `gt:7511:17`; the
  downstream accounting recorded `final_set_gain: false`, no gained or lost
  owner IDs, and `target_recovery_exchange: false`.
- No low-temperature sample was drawn. The behavior artifact contains zero
  sampling arms, and analyzer v3 has no sampling admission or parameter source.
- The due-turn `gt:7511:22` control did not expose usable target support. Its
  target peak was `-58.7200` at thresholds `0.4` and `0.5`, but the registered
  other owner peaked at `-55.8078`; the other-owner margin was `-7.0680`, and
  peak identity changed at threshold `0.6`.
- The root `gt:7511:17` read had target peak `-52.7156` versus background peak
  `-51.9843`, for prominence `-0.7313`. Review of the winning background
  comparator found wrist/hand/extent spill, so it is not a clean background
  control for contrasting root with due-turn accessibility.
- Root `gt:7511:22` was nonpositive against its controls: background prominence
  `-4.1311`, localized rank `0.2031`, and registered other-owner margin
  `-0.5189`. Root `gt:7511:26` was also nonpositive against background, with
  prominence `-2.6316` at thresholds `0.4` and `0.5` and `-3.6535` at `0.6`;
  its peak identity was not stable across the threshold band.

## Supported

- `gt:7511:17` is a positive route-conditioned accessibility control in the
  sampled-context stratum. At its exact due-turn prefix, fixed-budget
  likelihood support transfers to greedy strict recovery.
- That recovery is an accessibility observation, not a final-set repair:
  analyzer v3 records `final_set_gain: false`. It also is not owner exchange,
  because the conditioned final set lost no owner.
- The conditional sampling branch is closed for this unit. Greedy already
  released the target under the admitted exact-prefix condition, so the
  predeclared sampling escalation had no remaining discriminator to answer.

## Ruled out

- The due-turn `gt:7511:22` arm cannot serve as a usable positive localization
  control in this fixed candidate bank: another registered owner outranked the
  target and the target peak was not stable across the intersection-over-union
  band.
- Root `gt:7511:22` and root `gt:7511:26` do not provide positive fixed-budget
  support under the frozen functional.
- Root `gt:7511:17` does not supply a clean negative comparator to the positive
  due-turn result because its nominal background winner is contaminated by
  wrist/hand/extent spill.
- The executed evidence does not satisfy the control gate for Level 2 (`L2`) or
  broad twenty-context expansion. Neither was run.

## Unresolved

- Whether `gt:7511:17`'s due-turn accessibility reflects traversal state,
  description conditioning, insertion-conditioned release, or another
  exact-prefix effect. The required post-pass and natural-route contrasts were
  not executed.
- Whether the root and due-turn difference would survive a repaired background
  comparator whose candidates exclude wrist, hand, and extent spill.
- The `gt:7511:22` other-owner competition is real in the frozen bank but does
  not by itself identify same-description physical-owner collision or explain
  natural behavior.
- Dense-predecessor reanalysis still has a detached candidate-normalization
  block. It was not imported into this closeout, and its resolution cannot
  retroactively change this fixed-budget result.
- Geometry or extent, route or traversal, same-description collision, semantic
  drift, and absence of usable localization support remain unresolved as owner
  mechanisms outside the single positive control statement above.

## Not claimed

- No natural-route, absence, prevalence, cohort-scale, architecture, training,
  checkpoint, or model-wide mechanism claim is made.
- No `L2` result, broad twenty-context result, sampled-accessibility rate, or
  finite-sampling negative is reported.
- Strict recovery at the conditioned due-turn prefix is not claimed as an
  owner-set gain, route repair, or evidence that the original natural greedy
  miss was caused by traversal.
- The nonpositive root arms are not evidence that the corresponding owners are
  visually absent or lack localization support under another valid context or
  candidate bank.

## Next discriminator

Keep successor work detached from this closed unit. The approved next
discriminator is the prospectively registered
[all-person owner-relative route landscape](../2026-08-03-sorted-all-person-owner-relative-route-landscape/unit.md):
it removes the contaminated background comparator, gives all forty-one
confirmed `person` owners an equal score-independent candidate budget, and
keeps realized generated boxes in a separate bank-coverage diagnostic. That
successor may describe context sensitivity, scan proximity, and extent effects;
it does not inherit authority to call the due-to-post contrast a causal
coverage or route-collision effect. The detached dense-predecessor
normalization block remains outside both units. Do not reopen `L2`, broad
twenty-context expansion, or a natural-route claim from this result without a
new predeclared discriminator.
