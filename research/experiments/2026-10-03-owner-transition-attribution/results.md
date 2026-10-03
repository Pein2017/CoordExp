# Round05 accepted saved-output attribution

Current evidence: [attribution artifacts](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/) and [retained round04 raw evidence](../../../outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/retained-execution-01/); see the [shared retirement record](../2026-10-03-greedy-prefix-branching/retirement-01.md). Old absolute locations remain historical provenance and logical hash keys. Execution/readback commands tied to the old checkout are archived; new execution requires separate qualification.

All54 saved measurements reproduce the accepted round04 records exactly, including raw/category owner IDs, matches, IoUs, denominator IDs, invalid-row details and burdens. All114 category transition instances reconcile to the accepted ID sets: single31 gains/29 losses; multiple24 gains/30 losses. Coverage remains anchor274, single276, multiple268 on18 images/570 annotations. The lead has accepted and closed this unit. Technical status is `completed_as_frozen`; scientific status is `descriptive_evaluator_conditions_only`. User scientific acceptance remains false; no next unit is scheduled.

## Current acceptance

The [lead ruling](lead-ruling-01.md) accepts the immutable [lead-acceptance-01.json](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/lead-acceptance-01.json), SHA256 `c2263d7d57a69d64cb0ba5dcd45b2bff52367a6d9525bd2e99ce8c30341f9ddf`. The lead verified all60 input bindings, source identity and five check logs, reused the worker's54-measurement reconstruction and focused tests, and independently recomputed all114 witness maxima, eligibility, assignment links and classes.

The historical worker candidate at commit `a075e5270dedd46a7c8c1a41f2f71afb3e266e50` was unreviewed and pending lead acceptance. Its [terminal candidate](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/terminal-candidate-01.json), artifact, check logs and candidate evidence remain unchanged. This closure updates only state/results and runs the research-record consumer; it does not rerun the analysis or tests.

## Descriptive result

For gains the table describes the uncovered anchor; for losses it describes the uncovered endpoint. Assignment remains separate from these four exclusive descriptive classes.

| Arm/direction | Eligible same category not credited | Only other categories at threshold | Same category overlap below threshold | No positive same category overlap | Total |
|---|---:|---:|---:|---:|---:|
| R-single gain | 1 | 1 | 25 | 4 | 31 |
| R-single loss | 0 | 0 | 23 | 6 | 29 |
| R-multiple gain | 0 | 0 | 19 | 5 | 24 |
| R-multiple loss | 0 | 0 | 23 | 7 | 30 |
| Total | 1 | 1 | 90 | 22 | 114 |

The59 losses comprise46 positive same-category overlaps below0.5 and13 no-positive-overlap instances (five absent-description, eight zero-overlap). None has an eligible same-category candidate that lost credit through assignment competition. The55 gains comprise44 below-threshold cases, nine no-positive-overlap cases, one eligible same-category competition case and one other-category-at-threshold case.

All114 uncovered sides are unassigned. Of the22 no-positive-overlap instances,11 lack any valid same-description candidate and11 have valid same-description candidates with zero overlap. The absent/zero splits are single gain3/1, single loss2/4, multiple gain3/2 and multiple loss3/4. Instances can concern the same annotation in both arms;114 is not a count of distinct physical owners.

The one eligible same-category case is `R-single/16228/-57/gain`: anchor row `16228:greedy:0:p32` overlaps owner-57 at0.514021 but the existing assignment gives that row to owner-45 at0.573082. The endpoint credits owner-57 with its rowp32 at0.845846. The only-other-category case is `R-single/477415/-3032364130617015/gain`: the chair reference has best anchor chair IoU0.258179 and person IoU0.677309; the endpoint credits a chair row at0.565574. These are separate trajectories; equal row numbers do not establish object identity.

## Witness and artifact locators

The single [attribution.json](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/attribution.json) retains all114 entries, both sides' best same/any-description rows, eligible counts, every eligible same-description row and its actual assignment recipient, reference descriptions/boxes, raw locators, raw identities, compressed candidate order, raw parser order, character/token intervals and coordinate positions. Token indices are zero-based and intervals half-open.

| Selected instance | JSON pointer | Uncovered witness | Covered-side witness |
|---|---|---|---|
| R-single/1584/541559/gain | `/entries/0` | Anchorp5 IoU0.440741 | Singlep5 IoU0.554970 |
| R-single/5001/2146793/gain | `/entries/3` | No valid scissors description; best-any person IoU0.019789 | Singlep16 scissors IoU0.914634 |
| R-single/16228/-57/gain | `/entries/14` | Anchorp32 assigned to owner-45 | Singlep32 assigned to owner-57 |
| R-single/477415/-3032364130617015/gain | `/entries/28` | Anchor personp3 IoU0.677309 | Single chairp3 IoU0.565574 |

## Identity and verification

The artifact SHA256 is `dbf575b14f9b18cfa6c7582b74ba488c5c7d2e0941f316934b7746da9ede0b59`. Its provenance contains60 consumed-file hashes, verified once per invocation against predecessor bindings:54 natural records, the released contract, native terminal candidate, lead acceptance, accepted package terminal, label snapshot and tokenizer JSON. The tokenizer JSON supplies structural IDs only; no tokenizer, processor or model was loaded and no text was redecode-revalidated. Accepted token/text validation is reused.

Label identity: `research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json`, SHA256 `1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792`. Saved execution source is `fef0320926a5cde311caf203a5fbfa6a3e042ad8`; new analysis source is base revision `9cf94d70269c76576f99ee1da7b7ec5a8376ecec` plus the complete owned probe/check diff stored in the artifact. The scoped final commit and final consumer identities are in [terminal-candidate-01.json](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/terminal-candidate-01.json).

The caller remains `natural_measure`'s offline seam: `rollout_row_credit.assess_outputs` -> `hidden_human_recovery.candidates` via `iterative_positive` -> `saved_rows.one_to_one_matches` -> `assignment.global_matches`. All valid duplicates participate. Assignment is class agnostic, maximum cardinality then quantized total IoU, inclusive0.5; exact category equality follows assignment without rematching. Geometry uses full-image normalized bins without pixel rounding. `online_row_credit.observations` is not used as the matching pool.

Actual command exits:

- Development `build-01`: exit1; my initial label-array/contract-order equality check was too strong. Image7116 has a different snapshot position. Explicit image-ID joining now retains accepted natural order and original reference positions/locators. No evaluator or frozen input changed.
- Development `build-02`: exit0; first production-shaped54-record seam plus serialized consumer passed in3.952s, peak RSS135864320 bytes.
- Final `python -m probes.owner_transition_attribution build`: exit0, [build-03.log](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/build-03.log); wall4.055s, peak RSS135716864 bytes, attribution646466 bytes. Neither300s nor8GiB bound was exceeded.
- `python -m pytest -q tests/probes/test_owner_transition_attribution.py`: exit0, five checks; [checks-03.log](../../../outputs/research/physical-fn-recovery/2026-10-03/owner-transition-attribution-05/checks-03.log). Fixtures distinguish class-agnostic postassignment from category-aware matching, cardinality from greedy matching, all-valid duplicate participation, exact0.5, parser-drop order gaps and null versus zero. The actual54-record serialized consumer rejects fabricated/omitted IDs, assignment recipients, IoUs, classifications, coordinate positions, parser orders and source diffs after hash refresh. A changed inherited measurement fails closed before attribution.
- Post-commit readback and research-record consumer exits/logs are recorded in the terminal candidate. The same research-record consumer also ran after state/results creation and before the scoped commit.

No inherited evaluator mismatch remains. Token positions map existing parser spans to the bound raw structural-ID sequence; lexical tokens are never retokenized. Original label-array positions and accepted natural-image order are deliberately distinct.

## Evidence boundary

These results locate annotation credit changes within eligibility and one-to-one assignment for the frozen saved outputs. They do not establish physical absence, semantic forgetting, causal interference, physical track identity, or a useful training/policy intervention. Incomplete reference truth and alternative assignments remain interpretation limits; no label, threshold, matching rule, generation or acquisition changed. The lead owns the next research decision. These annotations provide no training or acquisition feedback; below-threshold overlaps do not establish small movement of the same physical entity.
