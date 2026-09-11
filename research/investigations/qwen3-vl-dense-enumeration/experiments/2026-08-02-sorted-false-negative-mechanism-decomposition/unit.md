---
title: Sorted False-Negative Mechanism Decomposition and Backfill
description: Uses canonical-description-conditioned geometry landscapes and exact self-prefix contrasts to separate extent errors, traversal misses, physical-owner collision, semantic drift, and absence of usable localization support on the sorted step-4887 checkpoint.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-02-sorted-false-negative-mechanism-decomposition
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: partial
updated: 2026-08-03
---

# Sorted False-Negative Mechanism Decomposition and Backfill

## Decision and outcome

This exploratory unit decides which mechanism should own the next false-negative
research step for the geometry-sorted step-`4887` checkpoint on the frozen
twelve-image human-refined panel. The decision unit is one registered physical
owner, not one token, candidate box, prediction row, or image-level metric.

The candidate mechanisms are:

1. **geometry or extent error**: the model selects the intended entity but the
   emitted box is a part, whole, or nearby extent that fails strict matching;
2. **route or traversal miss**: a usable owner-localized geometry basin exists,
   but the natural route does not select it, or passes it and does not return;
3. **same-description physical-owner collision**: adding one covered physical
   owner selectively suppresses a distinct owner with the same normalized
   description;
4. **semantic drift**: spatial support exists near the owner, but the emitted
   row uses a different description and cannot enter strict matching;
5. **no usable localization support under the tested interface**: none of the
   declared full-image, canonical-description, exact-self-prefix tests exposes
   a target-localized output basin;
6. **annotation-neutral**: the apparent error cannot be assigned to model
   behavior without changing or extending the frozen annotation.

The fifth outcome is deliberately narrower than a claim that the vision tower
cannot see the owner. This unit does not isolate vision-tower representation
from multimodal binding, language routing, coordinate realization, or image
resolution.

## Falsifiable question

For a stratified set of greedy false negatives and positive controls, does a
canonical-description-conditioned target-localized geometry basin exist before
the sorted route passes the owner, and if it exists, is its later loss or
behavioral non-release explained selectively by traversal state or by a
same-description covered physical owner?

The strongest alternative is that an apparent target peak is only a generic
spatial or extent prior. That alternative is separated by registered
same-description physical-owner basins, non-overlapping same-description foils,
different-description spatial foils where available, and equal-size background
controls.

## Originating intent and semantic deltas

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Geometry-sorted step-`4887`, twelve frozen images, permanent owner and row IDs | User direction and predecessor unit | scientific invariant | Scope and joins | inherited and approved |
| Forced-continuation rows do not admit owners or prefixes | User direction and predecessor unit | scientific invariant | Cohort | inherited and approved |
| Crop, rescale, annotation mutation, training, and architecture work are excluded | User direction | scientific invariant | Claim and cost | approved |
| Region membership is defined from geometry and registered identity rather than a label-first peak-shape classifier | Predecessor Stop Rule 3 and user discussion | conservative design choice | Primary estimand | approved |
| Same-description collision requires a prospectively bound non-overlapping physical foil | Predecessor Stop Rule 9 and user discussion | scientific invariant | Collision claim | approved |
| Greedy is primary; low-temperature sampling is conditional positive-support evidence only | User-delegated dynamic choice | conservative design choice | Execution branch | approved |
| Recursive successor units may be opened only when they separate live mechanisms inside the same checkpoint and panel | User authorization | conservative design choice | Stop rule and cost | approved |

## Frozen source boundary

### Model

- Base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Geometry-sorted adapter:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/adapter`
- Special-token embedding delta:
  `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/special_token_embeddings`

Decision-bearing likelihoods use raw pre-penalty logits. Repetition penalty
`1.0` is the primary mechanism stratum. Repetition penalty `1.10` may be used
only as a separately labelled production-policy view.

### Panel and immutable ledgers

- Panel:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl`
- Owner and prediction ledgers:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task0-v2-global-ambiguity-neutral-final/`
- Predecessor cohort assignments:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-cohorts-v2-foil-sealed-final-v2/cohort-assignments.jsonl`
- Matched-policy greedy and sixteen-sample natural rollouts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/sorted/`

The predecessor census contains 343 decision-eligible owners: 141 natural
greedy strict positives, 64 strict sampling rescues, 54 loose-only owners, 26
positive-overlap-neutral owners, and 58 owners with no semantic-compatible
free spatial support. These are source facts, not mechanism labels.

## Permanent identity and prefix rules

Existing `gt_owner_id`, `diagnostic_owner_id`, and `pred_row_id` values are
reused without renumbering. New diagnostic rows receive immutable identifiers
that include checkpoint, policy, image, source context, decode mode, seed, and
original row position. Context-dependent sampled rows also include the exact
prefix-token digest; the natural-rollout prediction identifier is not reused
across counterfactual prefixes.

For a missed owner `C`, define from the natural greedy trajectory:

- **pre-pass self-prefix**: the longest exact model-generated prefix before the
  first later sorted owner that establishes that the route has passed `C`;
- **post-pass self-prefix**: that same prefix after appending the exact natural
  successor row `D`;
- **covered-owner contrast prefix**: the pre-pass prefix plus a prospectively
  registered same-description covering-owner row;
- **physical-foil prefix**: the pre-pass prefix plus a prospectively registered
  same-description, spatially non-overlapping owner row at the closest feasible
  sorted-scan offset.

Greedy contexts remain primary for traversal, geometry, root/due-turn, and
no-usable-localization decisions. The collision causal contrast is a separately
labelled sampled-context stratum when a frozen sampled trajectory supplies a
token-identical on-manifold `P`, natural next row `G`, and same-trajectory foil
`F`. Such a deterministic teacher-forced contrast may be decision-bearing for
collision because it is matched to the null envelope; it cannot by itself show
that collision caused the natural greedy miss and is never pooled with greedy
traversal evidence.

Covering and physical-foil rows must be provenance- and format-matched. Both
are registered self-generated rows under the same policy, or both are explicitly
labelled oracle rows. A natural/self-generated covering row may not be compared
with a Ground-Truth-clean foil row. If a matched-provenance foil cannot be
constructed, collision remains unresolved.

Novelty is evaluated against the literal rows inside `P`, strictly before the
`cut_before_pred_row_id`; the natural row at the cut is not already covered in
`P`. Every collision role records separately whether `G` names the natural
successor owner `D` and whether the inserted row is the exact natural successor
row. When `G` is `D`, `P + G` is also the post-pass route surface: decline
against `P` is traversal evidence, while the difference against `P + F` is the
collision readout. A counterfactual-`G` candidate is structurally unmatched to
the current witnessed-successor null envelope unless it binds a class-matched
null; otherwise any collision interpretation carries an explicit
structural-mismatch caveat and remains unresolved-leaning.

The canonical reference order is `(y1, x1)`, matching the sorted checkpoint's
training order. `D` is the first unambiguous new physical owner after `C` in
that reference order. This ordering is recorded separately from the emitted
coordinate serialization `x1,y1,x2,y2`; the unit may diagnose but does not
change that mismatch.

Prefixes contain exact self-generated rows. Ground-Truth rows may appear only
as explicitly labelled oracle controls. A case without an unambiguous pre-pass
and post-pass transition stays neutral for the traversal contrast.

Owners without a clean pre-pass/post-pass pair use a frozen fallback menu:

1. root context before the first row;
2. natural-stop context immediately before the model's terminal token;
3. a reference scan-position self-prefix immediately before the first clear
   native owner following the target under `(y1, x1)`, when constructible.

When the reference scan-position context is unavailable, report localization
or order-conditioned accessibility as unresolved. Such a case cannot use a
convenient single context to support the no-usable-localization disposition.

## Stratified exploratory cohort

The decision-bearing **mechanism cohort registry** contains exactly twenty-four
owners. It becomes frozen only after CPU validation of every cohort status,
control role, physical foil, and row provenance:

| Stratum | Owners and roles |
| --- | --- |
| Image `7511`, tiny crowded persons | No-free targets `gt:7511:6`, `gt:7511:11`, `gt:7511:13`, `gt:7511:41`; shared controls strict positive `gt:7511:22`, strict rescue `gt:7511:17`, and loose-only `gt:7511:26` |
| Image `1584`, mid-size persons | No-free targets `gt:1584:11`, `gt:1584:9`; strict positive `gt:1584:3`; strict rescue `gt:1584:8` |
| Image `16228`, person extent and collision | Loose or collision candidates `gt:16228:33`, `gt:16228:30`; strict positive `gt:16228:32`; strict rescue and proposed physical foil `gt:16228:41` |
| Image `2685`, semantic drift and collision | No-free candidates `gt:2685:28`, `gt:2685:9`; loose-only collision candidate `gt:2685:15`; strict positive `gt:2685:17` |
| Image `14038`, annotation-neutral books | `gt:14038:42` and `gt:14038:41`; these test ambiguity handling and cannot supply a mechanism verdict |
| Image `13923`, wall bowls | `gt:13923:0`, `gt:13923:3`, and `gt:13923:4` |

Additional rows such as `gt:7511:24`, `gt:7511:29`, `gt:16228:35`,
`gt:16228:17`, `gt:2685:22`, `gt:2685:18`, and `gt:2685:26` may be bound as
physical controls without becoming mechanism targets. Their exact roles and
row provenance are frozen before target scoring.

The separate context/control registry may also reference auxiliary owners such
as first-skip target `gt:7511:1`, successor `gt:7511:2`, and null-pair owners.
These references do not change the exact twenty-four-owner mechanism cohort and
cannot enter a cohort prevalence denominator.

A proposed physical control whose only registered rows are sampled
(`strict_rescued`) or absent (`no_free_spatial_support`) may serve only in a
provenance-matched sampled or oracle contrast, never against a natural greedy
covering row.

The three wall bowls have no same-image same-description strict-positive,
strict-rescue, or loose-only control. They remain high-priority case studies,
but cannot receive the no-usable-localization disposition or estimate its
prevalence. They may support positive geometry, route, or semantic evidence;
otherwise they close unresolved.

Each target also binds the controls required by its mechanism claim. A target
without the required physical control may still contribute to geometry or
route evidence but cannot contribute to a collision conclusion.

Every target stratum also binds at least one positive control matched as closely
as the panel permits on normalized description, box area, and same-description
crowding. If no credible matched positive exists, a negative localization
finding in that stratum remains unresolved.

## Primary observation: geometry-defined owner landscape

The predecessor's label-first `localized_peak` classifier is retired. Candidate
boxes are assigned to declared regions directly from registered geometry:

- **target strict region**: intersection over union at least `0.5` with the
  target owner;
- **target extent halo**: positive overlap with the target but intersection over
  union below `0.5`;
- **other-owner region**: intersection over union at least `0.5` with another
  registered physical owner of the same normalized description;
- **background region**: zero overlap with every registered owner of that
  description;
- ambiguous multi-owner boxes remain neutral rather than being assigned by a
  tie-breaking label.

The predecessor's unbounded dense full-interior bank is not reused at cohort
scale: its target population contained thousands of boxes while foil banks
often contained one, making maximum-score prominence an order-statistic
comparison and making large owners prohibitively expensive. The successor first
uses the immutable dense control rows to validate a deterministic budget ladder.

- **L0 mandatory core** contains approximately 24 to 40 candidates: exact
  Ground-Truth box, corners, center, every reviewed natural box, and the frozen
  minimal shift and scale families. It is present at every rung but cannot
  decide a negative result alone.
- **L1 primary** contains exactly 256 target candidates and 256 family-mirrored
  decoys. At least 64 target candidates are in the strict region by
  construction. The remaining frozen quotas cover natural boxes, center shifts,
  scale and aspect changes, and an auditable interior lattice.
- **L2 escalation** contains exactly 1,024 target candidates and 1,024
  family-mirrored decoys. It halves the primary lattice spacing and may use up
  to eight extents per retained anchor.

The predecessor's 2,000-target-candidate cap is not a default rung in this unit.
Using it requires a named successor decision after L2 fails to resolve an
otherwise viable stratum. Every rung includes the exact Ground-Truth box,
corners, center, reviewed natural boxes, and auditable pruned anchors.

For every target population, construct an equal-count decoy population using
equal-size background and matched sorted-scan positions. A statistic comparing
unequal target and decoy counts is invalid. The candidate bank combines:

1. the exact target box;
2. auditable fixed-budget target-interior anchors plus size-aware center shifts and
   scale changes around the target;
3. registered same-description physical-owner boxes and matched local
   perturbations;
4. equal-count, equal-size background and sorted-scan controls;
5. reviewed natural prediction boxes relevant to extent or semantic drift.

For every declared prefix, record raw complete-box log likelihood, coordinate-
slot conditional log probabilities, rank, target-versus-background prominence,
target-versus-other-owner margin, and geometry-cluster membership. Any regional
"mass" is a proposal-weighted log-sum-exp over the frozen finite candidate bank,
not the model's full probability of all boxes; it is used only when proposal
weights are comparable across the paired regions. Non-maximum
suppression or connected components may summarize the number and location of
peaks for visualization, but inferred peak-shape names do not gate execution or
mechanism assignment.

L1 is admitted only if CPU reanalysis of the existing dense strict-positive,
loose-only, and B2 before/after control rows preserves their geometry-first
usable status, strict-region peak within a frozen score tolerance, localized
rank side, and paired-change sign across intersection-over-union thresholds
`0.4`, `0.5`, and `0.6`. Reanalysis selects exact existing candidate rows; a
nearby candidate is not assigned an old score. The strict-rescue class has no
completed predecessor dense surface, so it is never validated by CPU reuse:
`gt:7511:17` instead emits both an unconstrained, positive-direction-only
scalar smoke rung and an unconditionally frozen `L1` rung, freshly scored by
the successor scorer. Scalar smoke can only ever support a positive claim (a
usable target strict-region peak exists); it never by itself establishes an
absence or negative claim. `L1` is the rung that carries marginal or negative
evidence, and it is scored before any stop-rule-4 judgment is made.

Every scorer execution must explicitly select its rung or rungs. A context
allowlist never implies all rungs, and omission of the rung allowlist fails
closed. The scorer receipt, merged receipt, and run attestation bind the exact
selected `(owner_context_id, rung)` domain so the scalar smoke cannot silently
absorb the already-frozen `L1` population. `scalar_smoke` is an exclusive run
selection and cannot be mixed with another rung or attested as scale mode.

Control failure at L1 permits target-blind expansion of that stratum to L2.
Control failure at L2 holds the stratum. A target null at L1 never triggers
expansion merely to chase a peak. The only target-side escalation is a
pre-registered near-miss band: target prominence above the matched decoy
90th percentile but below the usable threshold permits one L2 rescore, whose
only scientific effect can be to rescue positive support or retain the narrower
L2-conditioned null.

The primary context is the exact due-turn reference scan-position self-prefix.
Root is secondary. Natural-stop is used only when the due-turn context cannot
be constructed. Root and due-turn controls are never pooled. The smallest
decision-changing observation is the paired change in the target strict-region
peak and registered candidate-bank target score from pre-pass to post-pass,
compared with equal-count physical-foil and background contrasts.

### Frozen quantitative decision functional

Before any target row is read, a successor-owned
`mechanism-decision-rules.json` freezes the candidate generator, proposal
weights, geometry regions, numerical tolerance, matched-control strata,
sampling escalation, and the following statistics. For raw complete-box score
`s_P(b)` at prefix `P`:

- `target_peak(P)` is the maximum score in the target strict region;
- `background_prominence(P)` is `target_peak(P)` minus the maximum equal-count, equal-size
  background-region score;
- `other_owner_margin(P)` is `target_peak(P)` minus the maximum registered
  same-description other-owner-region score;
- `localized_rank(P)` is the normalized rank of `target_peak(P)` in the union
  of the equal-count target and decoy populations;
- `target_bank_score(P)` is the proposal-weighted regional log-sum-exp, used
  only when its target and control proposal measures are comparable.

A target peak is **usable** only when its prominence is at least the frozen
lower control quantile and its localized rank is at most the frozen upper
control quantile for the same description/size/crowding stratum and context.
The initial calibration proposal is lower quantile `0.10` for prominence and
upper quantile `0.90` for rank; the quantile algorithm, perturbation band, score
tolerance, family quotas, and near-miss band are frozen before target scoring.
Rank and prominence thresholds are recalibrated separately at every admitted
budget rung and never transferred from the dense bank or another rung. Every
margin must also exceed measured scalar full-reforward numerical tolerance. No
pooled threshold may substitute for a missing matched control.

For covering row `G` and provenance-matched physical foil row `F`, the primary
collision statistic is:

```text
selective_decline =
  [target_peak(P + G) - target_peak(P)]
  - [target_peak(P + F) - target_peak(P)]
```

The rules freeze a negative-control envelope from at least three prospectively
registered **behaviorally witnessed non-collision pairs** spanning at least two
images or matched strata. For each null target `T`, one already sealed natural
or sampled trajectory must contain a novel same-description row `G` followed
later by a strict match to `T`. `P` is that exact trajectory immediately before
`G`; therefore `P + G` is a previously observed history that still released
`T`. Neither `G` nor the matched physical foil `F` may already be covered in
`P`. The actual inserted `G` row and its physical owner must overlap `T`; the
actual inserted `F` row and its physical owner must have zero overlap with `T`.
At least one null uses sampled provenance matched to the sampled collision
arms. Null membership is frozen from these predecessor trajectory facts before
any landscape score is read and is never reclassified from its observed
`selective_decline`.

A loose-only or missed owner selected because it already has an overlapping
same-description neighbor is a collision candidate, not a non-collision null.
A single pair cannot by itself calibrate the envelope. One of the three
registered nulls is additionally designated as the mechanical smoke vehicle,
but after it is rescored under the same frozen rules/backend it remains in the
numerical max envelope; the designation never changes threshold membership.
The minimum global envelope is prospectively frozen from these three
independent sampled witnesses:

- `wit-16228-46-a`: trajectory `sampled:21005:16228`, target
  `gt:16228:46`, covering row `pred:sorted:sampled:21005:16228:9`, foil row
  `pred:sorted:sampled:21005:16228:12`, and later target release row
  `pred:sorted:sampled:21005:16228:20`;
- `wit-5001-6-a`: trajectory `sampled:21015:5001`, target `gt:5001:6`,
  covering row `pred:sorted:sampled:21015:5001:4`, foil row
  `pred:sorted:sampled:21015:5001:5`, and later target release row
  `pred:sorted:sampled:21015:5001:6`;
- `wit-4134-35-a`: trajectory `sampled:21008:4134`, target `gt:4134:35`,
  covering row `pred:sorted:sampled:21008:4134:28`, foil row
  `pred:sorted:sampled:21008:4134:29`, and later target release row
  `pred:sorted:sampled:21008:4134:30`.

These witnesses span three images without within-trajectory pseudoreplication.
Their trajectory IDs, row indices, exact prefix-token digests, and sampled-row
teacher-forced parity are frozen before scoring. None may be reclassified after
its score is observed. Witness `wit-5001-6-a` is the mechanical sign/join check;
`wit-16228-46-a` and `wit-4134-35-a` are the two calibrating nulls. This keeps
the image-`16228` matched witness conclusion-bearing for candidate
`gt:16228:30`, while satisfying the separate mechanical-check requirement.
With three nulls, including the mechanically exercised witness after its clean
rescore, the global exceedance threshold is the most negative observed null
`selective_decline`, extended by scalar numerical tolerance; a leave-one-out
sign readout is reported as robustness rather than a replacement threshold.

The global envelope has two declared stratum gaps: there is no witnessed null
for the tiny-person image-`7511` stratum and no bottle null for image `2685`.
It may support candidate `gt:16228:30` because witness `wit-16228-46-a` uses
the identical image, description, covering row, and foil row. It cannot support
a collision verdict for `gt:2685:15`; an exceedance there is only
collision-consistent diagnostic evidence pending a bottle-stratum witnessed
null. The same restriction applies to image-`7511` tiny-person candidates.

Collision support requires `selective_decline` to be more negative than the
admitted matched envelope and numerical tolerance, with the same sign over the
frozen candidate-neighborhood perturbation band and a co-moving behavioral
readout. If the minimum witnessed-null registry cannot be built, collision
remains unresolved. The rules file and its digest are bound into every
acceptance receipt.

## Behavioral transfer and sampling escalation

### Greedy first

Two behavioral readouts stay separate:

1. **free next-row and first-divergence readout**: release the model from the
   exact self-prefix to observe STOP, natural description choice, and route
   selection; this is the only readout that may support natural semantic drift;
2. **canonical-description-conditioned geometry readout**: force only the
   canonical description and box opener, then decode the box greedily; this
   tests localization conditional on the supplied description and cannot by
   itself establish semantic drift.

For the second readout, record whether the generated box identifies `C`,
repeats an already covered owner, or selects another spatial mode. If the
generated row strictly or meaningfully loosely identifies `C`, append that
exact generated row and continue natural greedy decoding to the fixed horizon.

For an unambiguous skip `A, B, D`, compare:

1. recover `C` from the self-prefix `A, B`, then continue;
2. recover `C` from the self-prefix `A, B, D`, then continue;
3. the untouched natural `A, B, D` route.

Report final unique-owner gained, retained, and lost sets, successor retention,
same-description duplicates, invalid rows, and natural stop behavior. Recovering
`C` while losing `D` or another retained owner is owner exchange, not repair.
Any suffix gain remains insertion-conditional route value; generic on-manifold
insertion perturbation is unresolved unless a separately registered insertion
control rules it out.

### Conditional low-temperature sampling

Low-temperature sampling is not run by default. It is admitted only when the
likelihood landscape contains a usable target strict-region peak or multiple
separated owner-localized peaks but greedy canonical-description-conditioned
decoding does not release the target. The initial registered policy is
temperature `0.4`, top-p
`0.95`, repetition penalty `1.0`, with a bounded `K` selected after the one-case
smoke. A sampled hit is positive accessibility evidence. Failure of finite
sampling is neutral and never establishes absent support.

The sampling admission must bind the exact prior greedy behavior artifact and
its per-role failure disposition by path and digest. Landscape support without
that role-matched greedy-failure evidence cannot authorize sampling.

Sampling K, seeds, donor selection, and horizon are frozen before execution.
Likelihood is sealed before sampling. When likelihood selects a case or admits
sampling, all sampling conclusions are explicitly conditional on that
selection and are not prevalence estimates.

## Mechanism dispositions

An owner may receive more than one supported mechanism; the unit does not force
an exhaustive single label.

- **geometry or extent supported** requires target-overlapping natural output
  plus canonical-description-conditioned target-region support, with strict failure explained
  by box extent rather than selection of another physical owner;
- **route or traversal supported** requires usable pre-pass target support and
  either a post-pass decline or a failure of natural selection that is released
  by owner conditioning, without selective same-description collision evidence;
- **same-description collision supported** requires a selective decline after
  the covering-owner row relative to the prospectively bound non-overlapping
  same-description physical foil, visible in both likelihood and at least one
  behavioral readout;
- **semantic drift supported** requires spatial support near the target with a
  non-equivalent description and a canonical-description contrast that changes
  owner localization or release;
- **no usable localization support under the tested interface** requires that
  every declared context and target-region probe lacks a target-localized basin,
  positive controls pass, and no registered greedy or sampled row supplies
  contrary support;
- **unresolved** is mandatory when controls, prefixes, identities, annotation,
  or evidence families do not support a narrower statement.

## Alignment and preservation

The proxy unit is canonical-description-conditioned complete-box likelihood and short
canonical-description-conditioned generation. The final behavior surface is natural one-prompt
greedy completion and its unique physical-owner set. Transfer is supported only
when a likelihood distinction predicts canonical-description-conditioned release or a
prefix-insertion effect. Likelihood improvement alone is not a final-set gain.

Preserve exact image, prompt, tokenizer, checkpoint, special-token delta,
coordinate conversion, row grammar, matcher, self-prefix tokens, policy
stratum, and output horizon. Suspected missing annotations, ambiguous
same-description assignment, invalid rows, and finite-sampling misses remain
neutral.

## Execution outline

1. Freeze a CPU-only cross-description overlap census and the twenty-to-
   twenty-four-owner registry with all physical controls.
2. Reuse the predecessor's native multimodal materialization and uncached
   full-reforward primitives only after an end-to-end successor case proves
   stable IDs, exact prefix reconstruction, geometry-region assignment,
   raw-logit extraction, and downstream receipt reconstruction. A predecessor
   score may be reused only when the successor candidate maps one-to-one to the
   same box tokens, candidate ID, raw-row identity, and source digest. Every
   unmatched successor candidate is rescored by the successor restricted-box
   runner into the successor artifact root; it is never assigned a nearby
   predecessor score, and no fixed-budget receipt claims a free-tree surface.
3. Require native replay for every prefix receiving traversal or behavioral
   interpretation: the current runtime must reproduce the stored natural next
   row, terminal action, and suffix under the same policy. Stored sampled rows
   require teacher-forced token parity rather than greedy reproduction.
4. Run one positive geometry control, one strict-rescue control, and one
   no-free target through every conclusion-bearing link before cohort scale-up.
5. Shard independent owners across the available eight Graphics Processing
   Units. Batch equal-length full-prefix reforward requests where exactness is
   retained.
6. Exploration may use a declared relaxed cache or reduced precision to rank
   candidates. Every decision-bearing target, control, and sentinel is rescored
   by full reforward under the accepted precision contract before closure.
7. Run greedy behavioral transfer only for owners admitted by the landscape.
   Add low-temperature sampling only under its declared escalation rule.
8. Close with per-owner evidence tables and representative landscape and
   rollout visualizations. Prevalence over all greedy false negatives is a
   successor decision, not an output of this exploratory cohort.

The successor owns a versioned manifest planner, restricted fixed-budget
complete-box scorer, arbitrary-role merger, attestor, and behavioral wrapper.
The predecessor's closed planner, production candidate contract, scorer, and
merger are not relaxed or silently repurposed. In particular, the predecessor
production scorer requires the complete declared `x1` domain and one dynamic
free-tree root per owner-context; feeding the successor's L0/L1 lattice to it
would either fail validation or silently restore the dense/free-tree cost.
The successor scorer therefore imports the already-attested multimodal
materialization and uncached full-reforward primitives, but emits a separately
versioned restricted-complete-box receipt and makes no free-surface claim.
Identical prefix tensors may be execution-deduplicated by exact token and image
digest, but every logical role remains a separate manifest entry and receipt
row.

## Representative smoke

The first real smoke centers its geometry and traversal controls on image
`7511`, while the prospectively frozen collision and null-envelope roles span
their separately registered sampled images. Context count is derived from the
manifest and is not the old numeric shorthand "six contexts." The following
logical roles are all present:

1. root and due-turn contexts for strict positive `gt:7511:22`, whose target
   region must outrank equal-size background and a same-description other-owner
   region;
2. root context for loose-only geometry control `gt:7511:26`, whose strict
   region and extent halo must remain separately visible;
3. root and due-turn contexts for strict-rescue control `gt:7511:17`;
4. the collision baseline prefix `P`, plus distinct `P + G` covering-row and
   `P + F` provenance-matched physical-foil roles for each registered sampled
   seed pair below;
5. the clean first-skip `P_pre` self-prefix for `gt:7511:1` and the exact
   `P_post` self-prefix after native successor `gt:7511:2`;
6. mechanical non-collision null roles `P_null`, `P_null + G_null`, and
   `P_null + F_null`, whose exact owner, row, and prefix identities are frozen
   in the registry before scoring; and
7. any additional matched-null roles needed to reach the prospectively frozen
   minimum of three non-collision pairs across at least two images or strata.

Two roles may share an execution tensor only when image and token digests are
identical. Such execution deduplication never merges their semantic roles,
controls, or receipt rows.

The first reviewed non-collision pair is a mechanical null check that exercises
the exact collision machinery. It does not by itself calibrate the collision
false-positive envelope. That envelope becomes decision-bearing only after the
minimum multi-pair null registry above passes.

For the image-`7511` collision contrast, the natural greedy
`pred:sorted:greedy:0:7511:3` row remains an observed route transition but is
not an admitted controlled insertion. CPU physical-owner review rejected one
of the originally proposed sampled pairs and did not select the other for the
decision-bearing registry:

- in seed `21003`, covering owner `gt:7511:22` appears at the cut boundary and
  is therefore novel relative to the literal prefix `P`; this pair is
  mechanically admissible after correcting the prefix-membership rule, but it
  is not selected because the sampled-prefix image-`16228` target/null design
  below provides the stronger same-image controlled contrast;
- in seed `21011`, the proposed covering prediction has zero overlap with
  target `gt:7511:26`, while the proposed foil prediction and physical owner
  overlap the target.

The default registry therefore contains no image-`7511` causal collision pair.
Any replacement must satisfy the witnessed-null or collision-target contract,
teacher-forced chosen-token parity, novelty, actual-row geometry, and
on-manifold grammar before it is added. A clean oracle row is never silently
substituted. Geometry and traversal smoke may proceed while collision remains
explicitly unresolved.

Two separately registered sampled-provenance triples are admissible as
**collision targets**, never as members of the negative-control envelope:

- target `gt:16228:30`, sampled seed `21005` immediately before row `9`,
  covering owner `gt:16228:37` from
  `pred:sorted:sampled:21005:16228:9`, and zero-overlap physical foil
  `gt:16228:22` from `pred:sorted:sampled:21005:16228:12`;
- target `gt:2685:15`, sampled seed `21007` immediately before row `16`,
  covering owner `gt:2685:17` from
  `pred:sorted:sampled:21007:2685:16`, and zero-overlap physical foil
  `gt:2685:22` from `pred:sorted:sampled:21007:2685:19`.

In both triples the inserted rows strict-match their declared owners, share
one sampled trajectory and policy, are novel relative to the literal sampled
prefix `P`, the actual covering row is the witnessed natural next row and
overlaps the target, and the actual foil row does not overlap the target.
For `gt:16228:30`, candidate and null `wit-16228-46-a` use token-identical
`P`, `G`, and `F`, so an envelope exceedance plus co-moving behavior may support
collision in this tested sampled context. It does not establish that collision
caused the greedy miss. For `gt:2685:15`, the same structure is diagnostic only:
the missing bottle-stratum witnessed null keeps the collision verdict
unresolved even on global-envelope exceedance.

The smoke first performs CPU reanalysis of the predecessor's immutable raw
strict-positive, loose-control, and B2 before/after rows under the new
geometry-defined criterion. Reuse is admitted only for exact one-to-one
candidate mappings with matching box tokens, candidate ID, raw-row identity,
and source digest. Missing physical-foil, pre-pass/post-pass, null, or unmatched
fixed-budget candidates, plus the new strict-rescue surface, receive new
likelihood scoring in the successor root.

The acceptance reference is scalar batch size `1`, uncached FP32 full-prefix
reforward. Cache reuse, reduced precision, or batched reforward cannot decide
the smoke. They may be admitted later only by comparison with this reference.

Acceptance requires one immutable receipt that binds source digests, owner and
row IDs, prefix-token digests, candidate-region assignments, scoring backend,
raw score rows, control outcomes, parser status, and the exact disposition. The
receipt also binds dirty-diff identity, exact context membership, native replay,
requested and effective batch size, prefix-dependent sample IDs, the frozen
decision-rules digest, strict-positive, strict-rescue, and loose-control
outcomes, every mechanical-null outcome, and the multi-pair non-collision
envelope when available.

## Stop rules

Stop or narrow before scale-up if any of the following occurs:

1. checkpoint, panel, prompt, tokenizer, special-token delta, policy, matcher,
   owner ID, row ID, or prefix lineage is not exact;
2. the strict positive does not expose target-region prominence over its
   registered controls;
3. the loose-only control cannot keep strict-region and extent-halo evidence
   separate;
4. the strict-rescue control's scalar smoke exposes no positive-direction
   usable target support, and its unconditionally frozen L1 rung -- scored
   before this stop judgment, on marginal or negative evidence -- also fails
   to expose usable target support; scalar smoke alone, being
   positive-direction-only, never establishes this stop by itself;
5. L1 and its permitted target-blind L2 control expansion both fail to preserve
   the geometry-first control result across the frozen intersection-over-union
   perturbation band, or any target and decoy population has unequal count;
6. a target stratum lacks the prospectively frozen matched positive controls
   required for a negative localization disposition;
7. a collision target lacks the prospectively bound same-description,
   non-overlapping, provenance-matched physical foil;
8. a relaxed-cache or reduced-precision ranking changes the candidate or sign
   needed for the decision and full reforward cannot resolve it;
9. a first-skip case has no unambiguous pre-pass/post-pass self-prefix;
10. native replay or stored sampled-token parity fails for a context receiving
   traversal or behavioral interpretation;
11. a semantic-drift claim is attempted without the free next-row and
   first-divergence readout;
12. canonical-description-conditioned recovery adds `C` only by losing the successor or another
   retained owner;
13. annotation uncertainty determines the apparent mechanism;
14. two consecutive successor units add no separation between surviving
   mechanisms, or the next discriminator requires training, annotation
   mutation, crop/rescale, architecture work, or expansion beyond this panel.

## Rough cost and recursive authority

The CPU registry and three-owner smoke precede any broad Graphics Processing
Unit run. One unit may use up to eight GPUs for at most six wall-clock hours;
recursive units inside the same checkpoint and panel may use at most eighteen
wall-clock hours in aggregate before pausing for a new user decision. The lead
may change batching, greedy versus conditional sampling, case allocation, and
worker routing when doing so preserves the question, controls, and claim
boundary.

Before execution, every recursive successor names the exact surviving
mechanism pair it separates and the observation that changes the disposition.
Each emits an immutable lead-review receipt. A post-hoc statement that a unit
"added separation" cannot override its predeclared discriminator.

Claude workers may implement bounded, mechanically verifiable surfaces.
Codex Sol independently audits code, runtime semantics, receipts, and silent
correctness. Claude Fable independently reviews protocol freezes and
conclusion-changing claims. The research lead owns integration and acceptance;
reviewer agreement is not a vote and does not replace executable evidence.

## Artifact handle

Planned durable root:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-false-negative-mechanism-decomposition/<immutable-run-id>/`

No run identifier is allocated until the representative smoke has an executable
command and a frozen case registry.
