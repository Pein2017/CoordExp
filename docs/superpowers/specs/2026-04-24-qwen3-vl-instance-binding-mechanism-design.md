# Qwen3-VL Instance Binding Mechanism Study

Date: 2026-04-24
Status: first-pass evidence loop converged; follow-up controls recommended
Owner: Codex

## Goal

Design a mechanism-first study for one fixed CoordExp / Qwen3-VL detection
checkpoint:

`/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

Implementation worktree:

`/data/CoordExp/.worktrees/qwen3-vl-instance-binding`

Path contract:

- source/config/test files live in the worktree
- heavyweight checkpoint/data/output paths resolve through shared root
  `/data/CoordExp`
- prepared COCO data is under `/data/CoordExp/public_data/...`; the worktree's
  `public_data/` directory is only the tracked preparation-code skeleton

The study asks whether autoregressive same-desc instance identity is already
meaningfully bound before the first coordinate token, or whether the decisive
instance split still happens at early coordinate generation, especially `x1` and
`y1`.

This is not a generic benchmark design. It is an onset-local mechanism study
for desc-first CoordJSON with bare `<|coord_*|>` coordinate tokens in norm1000
space. The model and autoregressive architecture remain fixed. No new detection
head, objective, RL loop, pseudo-label pipeline, or giant rollout sweep is in
scope.

## Current Result Snapshot

Artifact root:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/`

Current report:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report/report.md`

First-pass conclusion:

`converged_first_pass_mixed_soft_pre_x1_coordinate_hardening`

Interpretation:

- Weak/partial pre-`x1` binding exists, but the first coordinate tokens still
  form the hard instance-disambiguation boundary in difficult same-desc scenes.
- Pre-`x1` probe accuracy is above weak baselines but not strong:
  `0.422` best pre-`x1` vs `0.594` after `x1/y1`.
- Pre-`x1` `x1` mass remains multi-modal: target strict-best rate `0.547`,
  target/other mean mass `0.168 / 0.145`.
- Schema-context attenuation is causally high impact:
  mean absolute margin delta `0.068` and top-candidate flip rate `0.375`.
- Same-case donor patching strengthens the schema-context result:
  schema-context copies increase donor `x1` mass by `0.029` and reduce target
  mass by `0.064` on average, while current-desc and previous-geometry donor
  mass deltas stay near zero.
- Rollout contrast is present on the curated subset:
  `38` healthy vs `26` failure-like labels across `64` cases.

Open uncertainty:

- Donor patching is still an activation-level intervention and can introduce
  positional or serialization-distribution artifacts.
- The next highest-value controls are randomized-donor patching, wrong-image
  schema-context controls, and a larger same-desc rollout split.

## Starting Thesis

Prior CoordExp diagnostic notes already make the x1/y1 coordinate-basin story
plausible: duplication failures often look like weak early coordinate escape
from a previous or nearby same-desc local basin. This study should not merely
repeat that conclusion. It should test whether hidden-state evidence and causal
interventions show that the instance was already selected before `x1`, or
whether `x1/y1` is still the first hard split.

The key distinction:

- semantic category/objectness can be strong without instance binding being
  fixed
- post-x1 sharpness can be autoregressive self-conditioning, not proof of
  pre-x1 binding
- punctuation/schema states may carry context-conditioned binding information
  even though their token identities are constant

## Iterative Convergence Loop

This study should loop through evidence-producing stages until the mechanism
conclusion stops changing under stronger controls. A loop is considered
converged only when all of the following are true:

1. Position-wise probe results agree across at least two target-selection
   surfaces:
   - hard same-desc cases with varied target ordinal positions
   - sparse/easy controls
2. Pre-`x1` coordinate-family mass is measured over the full coord-token
   softmax, not only top-k bins.
3. The conclusion is stable after separating:
   - target identity uncertainty
   - boundary-style uncertainty around the same target
   - serialization-order or left-to-right heuristics
4. At least one causal intervention has been run for the positions implicated
   by the probe. Probe-only evidence may suggest a boundary, but it cannot
   close the causal claim.
5. Good-basin and bad-basin cases are compared side by side, or the report
   explicitly marks that the failure-mode conclusion is not yet converged.

Interim conclusions must therefore be labeled as one of:

- `not_converged_extraction_only`
- `not_converged_probe_only`
- `not_converged_missing_causal`
- `not_converged_missing_good_bad_failure_split`
- `not_converged_missing_rollout_failure_split`
- `converged_first_pass_mixed_soft_pre_x1_coordinate_hardening`
- `converged_mixed_partial_pre_x1_binding`
- `converged_no_meaningful_pre_x1_binding`
- `converged_strong_pre_x1_schema_routing`

## Hypotheses

### H0: late coordinate split

After the object description, the model still has a multi-modal same-desc
posterior over multiple candidate instances. The transition from description to
`x1`, and sometimes to `x2`, remains unresolved or multi-peak. `x1/y1` is the
first decisive instance-disambiguation event.

### H1: pre-x1 implicit binding

The model already performs meaningful implicit instance binding before `x1`.
Binding may emerge at desc end, the closing quote, structural delimiters, the
`bbox_2d` key, or the opening bracket before coordinates. `x1/y1` then mostly
reads out an already selected instance.

### Mixed view

Pre-x1 binding exists but is soft, partial, or unstable. It can be visible in
late hidden states or easy same-desc scenes, but difficult repeated-object scenes
still require `x1/y1` to harden the choice.

## Primary Questions

1. At what token position does intended same-desc instance identity become
   decodable from hidden states?
2. Is pre-x1 binding absent, weak, partial, or strong?
3. Are punctuation/schema states causally important, or are they mostly
   scaffolding while geometry-related state is the real binder?
4. In hard same-desc scenes, does `x1/y1` remain the main
   instance-disambiguation event?
5. Which mechanism-level conclusion is best supported after the first-pass
   experiments?

## Sample Strategy

The case bank is small by design. It should maximize ambiguity and causal
separation, not validation-set coverage.

### Case buckets

1. Sparse single-instance controls
   - 8 to 12 object sites
   - one clear candidate for a desc
   - used to confirm the machinery can recover easy binding and does not invent
     same-desc ambiguity
2. Healthy same-desc transitions
   - 24 object sites
   - repeated exact desc in the same image
   - target and distractors have clearly separated boxes
   - the model continuation or teacher-forced target remains healthy
3. Hard same-desc / first-burst failures
   - 16 to 24 object sites
   - repeated-object scenes, preferably near a duplicate-collapse onset
   - each site has an intended target, a previous same-desc or duplicate anchor,
     and at least one local same-desc distractor
4. Duplicate-collapse or near-duplicate cases
   - 8 to 12 object sites
   - drawn from existing duplication diagnostics when available
   - used for the good-basin vs bad-basin side-by-side read

The first GPU pass should target roughly 48 to 72 object sites, plus the sparse
controls. If the repeated-desc pool is larger, the probe lane can use a separate
larger teacher-forced extraction slice of roughly 300 to 500 images, but the
causal and distributional headline should stay on the curated high-information
bank.

The target instance must not always be the first same-desc object in the record.
For repeated-desc groups, include multiple target ordinal positions when the
serialized object list contains several instances with the same description.
Otherwise, an order-only or "first same-desc" heuristic can look like successful
instance binding.

### Priority descs

Prioritize exact repeated descriptions including:

- `book`
- `person`
- `chair`
- `baseball bat`
- `bowl`
- `sheep`

Similar crowded same-class scenes are allowed if they satisfy the same exact-desc
and clear-candidate constraints.

### Candidate definition

For each object site, define:

- `target_instance`: the same-desc GT instance whose bbox span is being emitted
- `remaining_same_desc_candidates`: same image, exact normalized desc, not
  already emitted under the teacher-forced target prefix
- `full_same_desc_candidates`: same image and exact normalized desc, including
  already-mentioned instances
- `previous_same_desc_anchor`: the most recent same-desc object when present
- `local_same_desc_distractor`: the nearest competing same-desc object by IoU,
  center distance, or previous duplicate match

Exclude sites where candidate boxes quantize to the same norm1000 coord-token
surface or where annotation ambiguity makes the intended identity visually
unclear.

## Experiment Matrix

### A. Position-wise binding probe

Question:

Can the intended same-desc instance be decoded from hidden states before `x1`?

Positions:

1. end of desc content
2. closing quote after desc
3. colon or structural delimiter around the field transition
4. `bbox_2d` key token region
5. opening bracket before coordinates
6. pre-x1 residual state, immediately before predicting `x1`
7. post-x1 state
8. post-y1 state

Layers:

- early layer
- middle layer
- late layer
- final layer
- the last several layers if the first pass shows a late-layer transition

Readout:

- candidate-conditioned low-capacity retrieval probe
- score each candidate box with `s(h, c) = W(h) dot phi(c)` over that example's
  candidate set
- main `phi(c)`: standardized center/log-size box features
- auxiliary `phi(c)`: xyxy norm1000 features for direct-coordinate upper-bound
  comparison

Metrics:

- top-1 accuracy
- chance-normalized lift over `1/K`
- mean reciprocal rank
- rank percentile
- target-vs-top-distractor margin
- entropy of the predicted candidate distribution
- expected IoU or center/log-size error under the candidate distribution

Controls:

- full same-desc candidate set, not just remaining candidates
- order-only baseline: next unseen same-desc by serialization order
- left-to-right and top-to-bottom heuristic baselines
- token-only baseline for post-x1/post-y1 so coordinate-token leakage is visible
- image split for train/test and bootstrap confidence intervals by image
- explicit target-ordinal distribution in the case summary

Falsification value:

- strong pre-x1 lift over controls supports H1
- chance-level pre-x1 with a sharp post-x1/post-y1 jump supports H0
- weak pre-x1 lift plus a larger x1/y1 jump supports the mixed view

### B. Pre-x1 multi-modality analysis

Question:

Is `desc -> x1` intrinsically multi-modal in same-desc scenes?

Readout:

- inspect next-token distribution over coord-token ids at the pre-x1 state
- aggregate coordinate-token mass into candidate-aligned windows around each
  same-desc candidate's `x1`
- optionally repeat for `x2` as a boundary-style uncertainty check
- use full coord-family softmax for candidate-window mass; top-k bins are only
  for human-readable mode display

Metrics:

- coord-token entropy
- top-k coord modes
- mass in target window
- mass in previous-anchor window
- mass in local-distractor windows
- target-vs-duplicate x1 margin
- boundary-style mass: nearby tight/loose variants around the target rather
  than other same-desc instances

Interpretation:

- mass spread across several same-desc candidate windows supports unresolved
  instance identity
- mass concentrated around target but broad within the target window supports
  boundary uncertainty rather than identity uncertainty
- sharpness that appears mainly after forcing `x1` or `y1` supports the
  coordinate-split story

### C. Causal patching and intervention

Question:

Which states actually change the next-object choice?

Use residual-stream patching at block boundaries. Start with span-level patches,
not head-level or single-token patches. Keep the recipient image and prefix
fixed unless the intervention is explicitly an image-swap control.

Intervention spans:

1. current-object desc span
2. current-object schema span from the closing desc quote through
   `"bbox_2d":[`
3. previous-object full geometry span
4. previous-object `x1/y1` subspan
5. desc + schema combined, only if individual spans show a possible interaction

Layer bands:

- approximately 25 percent depth
- approximately 50 percent depth
- approximately 75 percent depth
- approximately 90 percent depth
- drill into exact layers only after a band matters

Donor/recipient pairs:

- healthy same-desc pair, both directions A -> B and B -> A
- hard wrong-instance site with `donor_intended` and `donor_competitor`
- duplicate-collapse site with `donor_dup` and `donor_missing`
- schema-only null donor from an unrelated object with the same JSON role but no
  plausible same-desc competitor

Outcomes:

- delta in next-x1 target-vs-competitor margin
- delta in intended-instance probe margin
- delta in coord-token entropy
- parse/routing sanity: probability that the next token is a coord token
- greedy or short controlled continuation flip toward intended vs duplicate

Interpretation:

- schema patch changes parse/routing but not target-vs-competitor margin:
  schema is mostly scaffolding
- schema patch causes donor-directed x1/probe/continuation shifts:
  punctuation/schema-position state participates in binding
- previous geometry patch dominates and `x1/y1` reproduces it:
  prior anchor carry-over is the main causal lever
- previous geometry patch dominates but `x1/y1` does not reproduce it:
  broader box-memory state is involved
- desc + schema is stronger than either alone:
  schema helps route or stabilize semantic binding

### D. Good-basin vs bad-basin comparison

Question:

Is pre-x1 binding already strong in good same-desc cases but weak in
duplication-collapse cases, or is binding generally unresolved until `x1/y1`?

Side-by-side read:

- position-wise probe curves for matched healthy and bad sites
- pre-x1 x1-mode mass over target, previous anchor, and local distractor
- causal patch deltas for previous geometry and schema spans
- short qualitative panel with image, GT candidates, natural continuation, and
  x1/y1 margins

Decision value:

- good cases show pre-x1 target binding and bad cases do not:
  mixed view with failure-specific binding weakness
- both good and bad cases remain unresolved until x1/y1:
  H0-like coordinate-split mechanism
- both good and bad cases show strong pre-x1 binding:
  investigate why rollout still duplicates despite apparent binding

### E. Optional attention / representation read

Attention is supporting evidence only. Use it after causal/probe results suggest
where to look.

Allowed uses:

- confirm whether causally important schema or geometry spans also receive
  concentrated attention
- inspect layer-local timing around the first causal band
- provide qualitative context for the final report

Disallowed use:

- using attention maps as the primary evidence for binding or causality

## Minimal First-Pass GPU Subset

The first worktree execution should run:

1. A position-wise binding chronogram on the curated case bank
   - all eight token positions
   - four layer anchors plus final several layers if cheap
   - remaining-candidate and full-candidate controls
2. Pre-x1 x1 distribution analysis on the same bank
   - target/previous/distractor window mass
   - entropy and top-k coord modes
   - x2 check on a smaller subset only
3. Causal patching on 12 to 16 sites
   - desc span
   - schema span
   - previous full geometry span
   - previous x1/y1 span
   - four layer bands
4. Good-vs-bad side-by-side report
   - 6 healthy and 6 bad cases with images and table rows

Use the 8 available GPUs as the default execution assumption for the first pass.
The study is still small and mechanism-first, but GPU-heavy stages should be
sharded by case id or stage cell rather than run serially when parallel devices
are free. Any generation or short controlled continuation used for flips should
use per-GPU generation batch size `8` unless a smoke run shows memory pressure.

Defer:

- full validation sweeps
- large Monte Carlo rollout
- exhaustive 0..999 heatmaps for every slot
- attention-head sweeps
- EOS/repetition penalty experiments
- new objectives or heads

## Compute Estimate

This estimate is for planning only and should be updated after a smoke run in
the implementation worktree.

- case mining and serialization audit: CPU, minutes
- teacher-forced hidden-state extraction on the curated bank: 8-way sharded
  GPU pass, target under 30 minutes after model-load overhead if batching works
- larger repeated-desc probe slice, if used: 8-way sharded GPU pass, roughly
  30 to 90 minutes depending on image resolution and case count
- pre-x1 coord-token distributions on the curated bank: 8-way sharded GPU pass,
  target under 30 minutes after model-load overhead
- span-level patching on 12 to 16 sites and four layer bands: 8-way sharded GPU
  pass, roughly 30 to 90 minutes depending on hook implementation and
  controlled-continuation use
- generation or short controlled continuation: per-GPU batch size `8`
- probe fitting and bootstrap summaries: CPU, minutes to under 1 hour

The first-pass work should still be runnable on one GPU for debugging, but the
research execution plan should use all 8 GPUs for throughput. Split
hidden-state extraction, multimodality, and patching into independent shards,
then merge manifests before fitting probes and writing the report. Do not scale
beyond the curated subset until the first report shows a real mechanism signal
worth widening.

## Artifact Contract

Use one isolated study root:

`output/analysis/qwen3-vl-instance-binding-mechanism-20260424/`

Expected artifacts:

- `config.resolved.yaml`
- `checkpoint_audit.json`
- `case_bank.jsonl`
- `case_bank_summary.json`
- `token_position_inventory.jsonl`
- `hidden_states/manifest.json`
- `probe_results.json`
- `pre_x1_multimodality.jsonl`
- `patching_results.jsonl`
- `donor_patching/donor_patching_results_merged.jsonl`
- `donor_patching/summary.json`
- `good_bad_case_panels.jsonl`
- `summary.json`
- `report.md`

After the study is reviewed, durable report copies may be placed under:

`progress/diagnostics/artifacts/qwen3_vl_instance_binding_mechanism_20260424/`

and the decision memo may be promoted to:

`progress/diagnostics/2026-04-24_qwen3_vl_instance_binding_mechanism.md`

Only promote after actual results exist.

## Decision Standard

The final report must separate evidence, interpretation, and open uncertainty.
It should choose one of:

1. no meaningful binding before `x1`; `x1/y1` is the first real split
2. partial pre-x1 binding exists, but it is soft and unstable; `x1/y1` remains
   the decisive split
3. strong pre-x1 binding already exists, and punctuation/schema states play a
   real causal routing or binding role

Evidence should be considered strong only if:

- the probe signal appears in same-desc controls and survives order/image/null
  baselines
- pre-x1 multimodality metrics agree with the probe read
- causal patching moves target-vs-competitor margins in donor-directed ways
- good/bad case panels tell the same story as the aggregate

Evidence should be downgraded if:

- signal appears only after x1/y1
- signal disappears under full same-desc candidates
- image-swap controls preserve the same margin
- schema patches only improve parse/routing without moving target identity
- candidate labels are visually ambiguous or quantized to identical coord-token
  surfaces

## Open Risks

- A linear probe failure does not prove binding is absent; it only shows the
  selected low-capacity readout cannot recover it.
- Teacher-forced hidden states can overstate recoverability relative to free
  rollout.
- Counterfactual patching can create out-of-distribution residual states.
- Same-desc object order can masquerade as binding if the case bank and controls
  are weak.
- The checkpoint is a merged coord-token model, so raw-text digit-token tooling
  may be useful only as a structural reference, not as a direct method.
