# Candidate-Field Cardinality Tomography Design

## Status

Date: 2026-06-03

Status: active diagnostic; Phase A2 dual-checkpoint comparison in progress

Scope: diagnostic design, not measured evidence and not a stable OpenSpec
contract.

Primary decision record:

- `progress/diagnostics/2026-06-02_fn_rescue_attention_binding_findings.md`

Review inputs integrated into this draft:

- research-design review: projection-collision gates, A3 subtype split,
  teacher/self prefix transition matrix, and unmatched-peak ambiguity handling;
- artifact-validity review: exhaustive universe versus sampled GPU probes,
  schema/join-key contract, policy consistency, denominator reporting, and
  manifest validation;
- implementation-layout review: project-wise `analysis/` subtree, thin CLI,
  model-touching modules isolated from pure artifact/report logic.

This design is for Phase A only.  It must not launch painting, sorted/random
SFT, multiple-positive supervision, background suppression, or any production
training.  Those directions remain Phase B/C candidates and are promoted only
after Phase A produces an artifact-backed dominant mechanism.

## Pure-CE Contrast Addendum

This addendum extends Phase A with one checkpoint contrast.  The purpose is to
test whether the desc-conditioned pre-x1 candidate field is a generic
autoregressive V-LLM tendency, or whether ET-RMP-CE changes the number and
validity of exposed same-desc x1 modes.

Primary ET-RMP-CE checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Pure-CE contrast checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664
```

The contrast is admissible because both checkpoints use:

- Stage-1 2B Qwen3-VL adapter checkpoints;
- `compact_full` detection template;
- coord-token `xyxy` serialization;
- `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`;
- `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`;
- `object_ordering = random_permutation`;
- token rows enabled and tied.

The intended objective difference is:

| Checkpoint | Objective ID | Variant | Trie Support | Trie Balance | State Weighting | Normalization |
| --- | --- | --- | ---: | ---: | --- | --- |
| ET-RMP-CE | `recursive_detection_ce` | `random_permutation_et_rmp_ce` | 2.0 | 1.0 | `uniform_permutation` | `semantic_image_bucket_balanced` |
| pure CE | `sft` | `random_order_sft` | 0.0 | 0.0 | `none` | `token_mean` |

The first contrast must be apples-to-apples with the completed ET-RMP-CE
`representative8192` run:

```text
sampling.max_cases = 8192
sampling.num_shards = 8
sampling.seed = 3664
peak.absolute_mass_floor = 0.002
peak.relative_floor = 0.10
peak.primary_merge_radius = 24
peak.gt_x1_neighborhood_radius = 24
peak.raw_topk_k = 32
```

Primary comparison metrics:

- `multi_peak_row_rate`
- `mean_peak_count`
- `median_peak_count`
- `valid_peak_share`
- `valid_peak_mass_share`
- `coverage_fraction`
- `A1_cardinality_collapse` rate
- target x1 rank and `p_gt_cond`

Allowed interpretation:

- If pure CE is more single-peaked or lower-coverage than ET-RMP-CE, ET-RMP-CE
  may already improve candidate-field breadth, though not necessarily enough
  for instance-level coverage.
- If both checkpoints are similarly low-coverage, the bottleneck is more likely
  a generic desc-first autoregressive candidate-field compression tendency.
- If pure CE exposes more valid modes than ET-RMP-CE, ET-RMP-CE may improve
  rollout stability while compressing pre-x1 candidate diversity; this requires
  follow-up on objective credit assignment.

Disallowed interpretation:

- Do not treat this as production training.
- Do not treat the contrast as a pure isolated objective ablation; batch shape,
  normalization, and state weighting also differ.
- Do not use the comparison to select an algorithmic intervention until the
  missing Phase A controls and sensitivity checks are inspected.

## Phase A2 Dual-Checkpoint Analysis Addendum

Phase A2 coordinates the ET-RMP-CE and pure-CE `representative8192` artifacts as
a paired diagnostic surface.  The unit of comparison is one shared probed row:
same dataset row, same image, same desc, same target object row, same peak
policy, and different checkpoint.

Phase A2 asks four questions:

1. Does either checkpoint behave like a locally single-instance x1 selector at
   `pre_x1`, or does it expose multiple same-desc x1 candidates in one step?
2. When pure CE exposes more peaks than ET-RMP-CE, are the extra peaks near
   annotated same-desc GT x1 values, or mostly unmatched under the COCO
   annotation universe?
3. Are A1 differences robust to reasonable peak extraction policy changes, or
   are they an artifact of one merge radius or mass floor?
4. Does crowded-cardinality compression remain even for the broader checkpoint?

Required Phase A2 artifacts:

```text
outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/
  et_rmp_ce_vs_purece_representative8192/
    phase_a2_dual_checkpoint_analysis/
      phase_a2_summary.json
      phase_a2_report.md
      plots/
      unmatched_review/
        unmatched_peak_rows.jsonl
        unmatched_peak_summary.json
        unmatched_peak_review.md
        manual_review_template.csv
        gallery/
```

Required Phase A2 metrics:

- paired A1 transition matrix: `both_A1`, `ET_only_A1`, `pure_only_A1`,
  `neither_A1`;
- paired peak-count delta, coverage delta, valid-peak delta, and unmatched-peak
  delta;
- same-desc-count bucket breakdown for `count1`, `count2`, `count3`,
  `count4_5`, and `count6_plus`;
- approximate top-32 sensitivity over merge radius and mass thresholds;
- top descs by pure-minus-ET valid peak gain and unmatched peak gain;
- representative cases for the main paired transitions.
- unmatched-peak manual review sidecar with row-level manual-label fields and
  a rendered x1 gallery.

Unmatched review semantics:

- `unmatched` means no annotated same-desc GT x1 lies within the configured
  valid x1 radius of the peak.
- `unmatched` is not a hallucination label.
- manual labels are filled after visual inspection, with suggested values
  `unlabeled_object`, `hallucination`, `duplication`, `ambiguous`, and `other`.
- gallery overlays must show same-desc annotated GT boxes and checkpoint x1
  peaks while preserving the row-level catalog as the source of truth.

Phase A2 interpretation rules:

- If pure CE has more peaks and higher GT coverage under multiple sensitivity
  settings, the "V-LLM is inherently single-peaked at pre-x1" explanation is
  weakened.
- If pure CE also has more unmatched peaks, treat this as a breadth-versus-
  annotated-precision tradeoff, not as automatic hallucination, because COCO is
  incomplete.
- If both checkpoints still under-cover high-cardinality rows, keep the
  instance-cluster compression hypothesis alive even when pure CE is broader.
- If ET-RMP-CE has fewer peaks but higher rollout stability elsewhere, separate
  candidate-field breadth from final decode quality.  Do not collapse these
  into one "better checkpoint" claim.

## Phase A3 Prefix-State Transition Addendum

Phase A3 moves the diagnostic surface from an empty-prefix candidate field to
teacher-conditioned prefix-state transitions.  Its purpose is to test whether
ET-RMP-CE and pure CE differ in how the next object distribution changes after
some objects have already been emitted.

Primary resolved design decision:

- Phase A3.1 uses teacher GT prefixes as the source-of-truth prefix state.
- Self-rollout prefixes are deferred to Phase A3.2 as a realism and
  compounding-error contrast.
- Phase A3 uses a dual readout at each prefix state:
  1. object-boundary next-desc / stop-token distribution;
  2. forced residual-desc `pre_x1` candidate field.

Rationale:

- Teacher GT prefixes isolate whether the model has a useful residual-object
  state transition under a correct already-emitted ledger.
- Self-rollout prefixes are closer to deployment but mix in reconstruction
  failure, duplication, false negatives, unlabeled positives, and local decode
  errors.  They should not be the first source of truth for mechanism
  separation.

Scope:

- Compare the same ET-RMP-CE and pure-CE checkpoint pair used by Phase A2.
- Increase object-count coverage beyond same-desc crowded rows and include
  mixed-desc images.
- Use both train and val hard pools for Phase A3.1 headline diagnostics.
  Reports must split train and val before presenting any combined summary.
- Use controlled counterfactual teacher prefixes as the primary Phase A3.1
  surface.  Original dataset/training sequence order is retained only as a
  control slice.
- Keep easy scenes as sanity controls.  Headline Phase A3.1 evidence must come
  from hard prefix-state strata with enough residual ambiguity to separate the
  two checkpoints.
- Treat `x1=0` and `x1=999` peaks as boundary-token artifacts or unknown
  coordinate-token behavior until specifically controlled; do not use them as
  evidence for instance discovery by default.
- Do not spend Phase A3 effort on top-1 accuracy ranking.  The primary question
  is how prefix state changes next-object and desc-conditioned coordinate
  distributions.

Prefix-state strata:

| Stratum | Purpose |
| --- | --- |
| `same_desc_prefix_k` | Emit `k` GT objects with desc `d`, then probe residual `d` objects. |
| `different_desc_prefix_k` | Emit one or more GT objects with desc `a`, then probe residual desc `b`. |
| `class_block_prefix` | Emit all or most GT objects of desc `a`, then test whether the boundary prefers continuing the class block, switching class, or stopping. |
| `spatial_prefix` | Emit objects in left-to-right / top-to-bottom order to test spatial-order compatibility. |
| `size_salience_prefix` | Emit larger or visually dominant objects first to test salience-order compatibility. |
| `original_order_prefix` | Preserve the dataset/rendered order as a control, not as the primary mechanism surface. |

Hard-strata requirements:

- Same-desc hard rows should have at least two residual annotated objects with
  the probed desc after the prefix.
- Mixed-desc hard rows should have at least three annotated desc groups and at
  least six annotated objects when possible.
- Prefixes should include both shallow states and deeper states where several
  objects have already been emitted.
- Count-1 and otherwise easy rows are retained as extractor sanity checks, but
  cannot dominate headline Phase A3 rates.
- Train rows test whether residual-state tracking holds on the supervised data
  distribution; val rows test whether the same mechanism generalizes.  A
  train/val gap must be reported as such, not hidden inside a pooled aggregate.

Sampling unit:

- Phase A3.1 samples prefix-state rows, not image rows.
- The source-of-truth sampling table is:

```text
prefix_state_index.jsonl
```

- Each row represents one explicit teacher-controlled prefix state and includes
  at minimum:
  - `prefix_state_id`;
  - `image_id`, `source_line_idx`, and `split`;
  - `transition_type = same_desc_transition | different_desc_transition`;
  - `prefix_depth`;
  - `prefix_order_policy_id`;
  - emitted GT indices and residual GT indices;
  - emitted desc groups and residual desc groups;
  - selected `probe_desc` values.
- Sampling must be stratified over:

```text
split
transition_type
prefix_depth
desc_count_bucket
object_count_bucket
```

- Image-level counts may be reported as a side diagnostic, but headline
  denominators are prefix-state rows.
- `prefix_state_index` rows are CPU planning rows, not checkpoint readout rows.
  They use `checkpoint_role = paired_index` and
  `readout_type = prefix_state_index`.  GPU readout rows use the paired
  checkpoint roles `et_rmp_ce | pure_ce` and readout types
  `boundary_full_desc_span | forced_desc_pre_x1`.

Execution stages:

Phase A3.1 is split into an index-only CPU stage and a GPU probe stage.

1. `prefix_state_index`
   - CPU-only.
   - Builds `prefix_state_index.jsonl` and `prefix_state_index_summary.json`.
   - Reports train/val, transition type, prefix depth, desc-count bucket,
     object-count bucket, and hard/easy strata counts before any model forward.
   - Computes a deterministic `launch_eligible` field from the GPU launch
     gates below.  The summary remains human-readable, but the pipeline does
     not require manual user approval before the GPU probe.
2. `paired_checkpoint_probe`
   - GPU stage.
   - Runs the paired ET-RMP-CE and pure-CE readouts for approved
     prefix-state rows.
   - Uses 8 GPUs as analysis shard capacity, not as production training.
   - First GPU scale target: 4096 paired prefix-state rows after index-gate
     approval.  If the results are high-variance or key strata remain sparse,
     extend to 8192 or 16384 with the same paired-row contract.
   - The first 4096-row probe does not dump full attention.  It records
     boundary full-desc span scores and forced-desc `pre_x1` posterior rows as
     the primary evidence.

Attention policy:

- Full attention is not a Phase A3.1 headline artifact for the 4096 paired
  probe.
- Attention may be run later as a sampled sidecar after the mechanism quadrant
  table is materialized.
- Attention sidecar sampling should prioritize `boundary_bad_x1_good`,
  `boundary_good_x1_bad`, paired ET-vs-pure quadrant disagreements, high
  emitted-attraction same-desc cases, and surprising `class_block_done` cases.
- Attention evidence can explain or stratify probability-transition findings,
  but it must not replace boundary/full-desc span scoring or forced-desc x1
  residual-state metrics.

Project layout:

Phase A3.1 uses a new idea-wise project layout instead of appending artifacts to
the Phase A/A2 empty-prefix candidate-field run roots.

Artifact root:

```text
outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/
  et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/
```

Code, config, and tests:

```text
configs/analysis/prefix_state_transition_tomography/
scripts/analysis/prefix_state_transition_tomography/
src/analysis/prefix_state_transition_tomography/
tests/analysis/prefix_state_transition_tomography/
```

The existing candidate-field cardinality tomography project remains the source
of prior Phase A/A2 evidence.  Phase A3 may reuse utility logic from it, but
new prefix-state transition scripts and artifacts must live under the
`prefix_state_transition_tomography` project namespace.

The GPU probe should not start unless the index-only summary passes the fixed
launch gates below.  This is an automatic contract, not a manual approval
checkpoint.

GPU launch gates:

`paired_checkpoint_probe` is blocked until `prefix_state_index_summary.json`
sets `launch_eligible = true` by passing these fixed gates:

- train and val both contain `same_desc_transition` and
  `different_desc_transition` rows;
- each split covers at least three of:
  `shallow_1`, `mid_half`, `late_one_left`, `class_block_done`;
- same-desc headline rows have at least two residual annotated same-desc
  objects after the teacher prefix;
- mixed-desc headline rows report the share of states with at least three desc
  groups and at least six annotated objects;
- easy sanity rows are present but do not exceed 20% of the headline sample;
- all gate denominators are reported by split and transition type.

If any gate fails, `prefix_state_index_summary.json` must set
`launch_eligible = false`, list `failed_launch_gates`, and the linked launcher
must stop before GPU work.  The next run should adjust the fixed indexing or
sampling policy rather than relying on ad hoc manual approval.

Sampling fallback policy:

- Prefix-state sampling is not itself a research object in Phase A3.1.
- Use a deterministic fixed-seed best-effort stratified sampler.
- If a narrow stratum is sparse, the sampler may top up from nearby available
  rows under the same split and transition type without additional design
  discussion.
- Any such fallback is recorded in `sampling_adjustments`, but reports should
  not over-interpret the exact sampling path.
- The important invariant is paired checkpoint comparability on the final
  sampled prefix-state rows.

Prefix-depth grid:

Phase A3 uses a stratified fixed-depth grid rather than exhaustive prefix
permutation enumeration.

| Prefix depth | Meaning |
| --- | --- |
| `empty` | Empty-prefix baseline compatible with Phase A2 semantics. |
| `shallow_1` | One GT object has been emitted; tests earliest state transition. |
| `mid_half` | Approximately half of the relevant objects have been emitted; tests mid-sequence state maintenance. |
| `late_one_left` | One residual object remains; tests final-object continuation versus EOS/stop. |
| `class_block_done` | A desc group has been fully emitted; tests switch-to-other-class, repetition, or stop behavior. |

The grid is crossed with the controlled prefix-state strata when the case has
enough annotated objects.  It is not a full permutation search.

Primary readout contract:

- At the object boundary, record the probability/logit ranks for EOS/stop
  tokens and all residual annotated desc strings in the image.
- Boundary desc readout uses teacher-forced full-desc span scoring, not only
  the first desc token.  For each candidate desc, score the compact-full entry
  prefix:

```text
<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>
```

  using length-normalized log probability over the desc/entry-prefix path.
  This is trie-like in spirit: each candidate desc is evaluated as a valid
  token path, while EOS/stop is read from the same boundary prompt.
- Phase A3.1 boundary candidate desc universe is image-local: annotated desc
  groups present in the current image.  COCO-80-wide desc scoring is deferred
  to an optional `coco80_open_desc_sidecar` for checking mass assigned to
  unannotated, background-adjacent, or off-image categories.  It cannot drive
  the Phase A3.1 headline residual-state alignment metrics.
- For each selected residual desc, force the compact-full desc prefix through
  `<|box_start|>` and record the `pre_x1` coordinate posterior using the same
  peak policy as Phase A2.
- Each sampled prefix-state probes at least two descs when available:
  - `target_residual_desc`: the desc whose residual-object behavior the state
    is designed to test;
  - `hard_competitor_desc`: a competing desc, selected from already-emitted
    descs, high-count same-image descs, or the highest boundary-scored desc
    that is not the target.

  This keeps the forced-desc cost bounded while preserving evidence about
  desc competition and repeated/emitted attraction.
- Forced-desc `pre_x1` peak attribution is x1-only in Phase A3.1.  Peaks are
  partitioned into:

| Peak partition | Meaning |
| --- | --- |
| `residual_same_desc_x1_peak` | Peak is near an annotated same-desc GT x1 not included in the teacher prefix. |
| `emitted_same_desc_x1_peak` | Peak is near an annotated same-desc GT x1 already included in the teacher prefix. |
| `ambiguous_x1_collision` | Peak is simultaneously near residual and emitted objects, or same-desc x1 projections are too overlapped to assign cleanly. |
| `unmatched_x1_peak` | Peak is not near any annotated same-desc GT x1; this is annotation-relative and not a hallucination label. |
| `boundary_artifact_x1_peak` | Peak is at or near boundary coordinates such as `0` or `999` without annotated support. |

  Full `y1/x2/y2` instance binding is deferred to an optional tail/basin
  sidecar and cannot be inferred from an x1 stripe alone.
- Boundary and forced-desc rows must share one `prefix_state_id`, so reports can
  separate:
  - `boundary_desc_or_stop_failure`;
  - `same_desc_residual_x1_failure`;
  - `different_desc_transition_failure`;
  - `class_block_or_ordering_tendency`.
- EOS/stop is interpreted only as a boundary-level competing score.  It is not
  sufficient by itself to explain the final continuation-versus-stop decision.
  Phase A3 uses EOS/stop to separate boundary suppression from post-desc x1
  state transition:
  - residual desc scores low and EOS/stop high -> boundary-level suppression;
  - residual desc scores high but forced-desc x1 peaks return to emitted
    objects -> post-desc emitted-attraction / x1 transition problem.

Primary metric contract:

Phase A3 reports residual-state alignment as the primary comparison surface,
not raw peak-count or top-1 accuracy.

| Metric | Meaning |
| --- | --- |
| `boundary_alignment` | Whether residual desc strings receive higher boundary probability/rank than already-emitted desc strings and EOS/stop. |
| `forced_x1_residual_coverage` | After forcing a residual desc, how much of the residual same-desc GT set is covered by x1 peaks. |
| `emitted_attraction_rate` | After forcing a desc, how often x1 peaks fall back onto already-emitted same-desc objects in the teacher prefix. |
| `state_transition_delta` | Within the same image/desc, how boundary mass, residual coverage, emitted attraction, and EOS/stop tendency change from empty prefix to prefix depth `k`. |

Interpretation:

- More peaks are not automatically better; extra peaks must be separated into
  residual coverage, already-emitted attraction, unmatched peaks, and boundary
  artifacts.
- If a checkpoint has a broad empty-prefix candidate field but high
  emitted-attraction after prefixing, that is a state-transition failure rather
  than evidence of strong multi-instance enumeration.
- If a checkpoint is single- or low-peaked but consistently moves mass to the
  correct residual instance under teacher prefixes, that is evidence for
  usable prefix-state tracking even if the empty-prefix field looks narrow.

Paired-checkpoint contract:

- Phase A3 headline metrics use paired state rows across the ET-RMP-CE and
  pure-CE checkpoints.
- The pairing key is:

```text
image_id
source_line_idx
prefix_state_id
prefix_condition
prefix_depth
prefix_order_policy_id
probe_desc
readout_type
```

- A paired headline row exists only when both checkpoints materialize the same
  boundary readout and the same forced-desc `pre_x1` readout for that prefix
  state.
- Unpaired rows are retained in sidecar diagnostics but cannot drive the
  primary ET-vs-pure comparison.
- Reports must present paired deltas for boundary alignment, forced-x1 residual
  coverage, emitted-attraction rate, EOS/stop tendency, and unmatched/artifact
  peak rates.

Mechanism quadrant table:

Phase A3 reports a four-quadrant mechanism table crossing boundary alignment
with forced-desc x1 residual coverage:

| Quadrant | Meaning |
| --- | --- |
| `boundary_good_x1_good` | Residual desc is favored at the boundary and forced-desc x1 covers residual objects. |
| `boundary_good_x1_bad` | Residual desc is favored, but forced-desc x1 returns to emitted objects, unmatched peaks, or artifacts. |
| `boundary_bad_x1_good` | Boundary does not favor the residual desc or EOS/stop competes strongly, but forced-desc x1 can cover residual objects. |
| `boundary_bad_x1_bad` | Neither boundary nor forced-desc x1 supports the residual state. |

The quadrant table must be reported separately for same-desc and different-desc
transitions, for train and val, and as paired ET-vs-pure deltas.

Manual review gallery sidecar:

Phase A3.1 includes a sampled manual-review gallery, but it is not rendered for
all rows.  The gallery is for mechanism inspection and manual ambiguity review;
the headline statistics remain JSONL/report driven.

Gallery sampling should prioritize:

- `boundary_bad_x1_good` cases, especially where the model appears able to
  localize residual objects only after the desc is forced;
- `boundary_good_x1_bad` cases, especially emitted-attraction, unmatched, or
  boundary-artifact x1 peaks;
- paired rows where ET-RMP-CE and pure CE fall into different quadrants;
- high `emitted_attraction_rate` same-desc cases;
- `class_block_done` states where the boundary repeats the completed desc or
  strongly favors EOS/stop.

Each rendered case should show:

- teacher-prefix emitted GT boxes;
- residual GT boxes;
- forced-desc x1 peaks with residual/emitted/unmatched/artifact partitions;
- boundary desc score table over image-local annotated desc groups;
- EOS/stop score from the same boundary prompt;
- checkpoint labels for ET-RMP-CE and pure CE when the case is paired.

Required Phase A3 question families:

1. Same-desc transition: after teacher-prefixing one or more objects with desc
   `d`, does the next `d -> pre_x1` field move toward residual `d` objects,
   stay attracted to emitted `d` objects, collapse to EOS, or remain broad?
2. Different-desc transition: after teacher-prefixing objects of desc `a`, how
   does the model allocate next-desc probability and forced-desc `pre_x1`
   fields for residual desc `b`?
3. Natural ordering tendency: under teacher prefixes, does the model prefer to
   continue by class block, spatial order, size/salience, or another stable
   ordering signal?

Resolved experiment split:

- `same_desc_transition` and `different_desc_transition` are two peer primary
  experiments.
- `same_desc_transition` primarily diagnoses whether the model tracks emitted
  instances and reallocates probability mass to residual same-desc objects.
- `different_desc_transition` primarily diagnoses class switching, class-block
  tendency, and whether category-sorted training is a plausible follow-up
  hypothesis.
- Aggregate Phase A3 reports must present these experiments separately before
  any pooled summary.

Interpretation boundaries:

- Phase A3 cannot decide whether pure CE has better final rollout ability just
  because it exposes broader one-step candidate fields.
- Phase A3 should separate "candidate discovery potential" from "state
  transition/use of already-emitted prefix" and from "tail coordinate binding".
- If class-block transitions look cleaner than mixed-desc transitions, this is
  evidence for a possible category-sorted training hypothesis, not yet an
  algorithmic recommendation.
- Phase A3 includes a `category_sorted_hypothesis_evidence` sidecar.  This
  sidecar routes evidence for or against category-sorted training, but it must
  not launch training or present category sorting as a resolved intervention.

Category-sorted hypothesis sidecar:

| Pattern | Interpretation boundary |
| --- | --- |
| `class_block_done` cleanly switches to other residual descs | Category sorting may not be necessary for cross-class transition. |
| In-class residual transitions are stable but cross-class transitions are weak | Category-block ordering may reduce next-object entropy and deserves Phase B consideration. |
| In-class transitions still show high emitted attraction or EOS/stop suppression | Category sorting alone is unlikely to solve recall; residual instance-ledger failure remains plausible. |
| ET-RMP-CE and pure CE differ strongly under class-block prefixes | Objective/state weighting likely changes natural ordering tendency; do not attribute the effect to architecture alone. |

## Research Question

Phase A asks whether, under a desc-conditioned pre-coordinate state, the model
exposes enough distinct same-desc instance modes to explain all annotated
same-desc objects in crowded scenes.

The main example is:

```text
desc = person
same_desc_gt_count = 5
question = does pre_x1 expose about 5 instance modes, or only 2-3 merged modes?
```

The primary measurement surface is the `x1` coordinate posterior at `pre_x1`.
Attention components are retained as secondary visual-field evidence, but they
must not be used as the headline criterion for candidate cardinality or as
attention-head causal proof.

## Non-Goals

- Do not train or fine-tune models.
- Do not run Phase B painting as part of Phase A.
- Do not run sorted-vs-random SFT or object-marginal training.
- Do not call extra peaks hallucinations by default.
- Do not use attention-only evidence to assign A1/A2/A3 mechanism buckets.
- Do not treat `progress/` notes as stable behavior contracts.

## Inputs

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Primary dataset JSONL:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Overlay artifacts:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase5_logit_binding_coordslot
```

The overlay is a failure-surface link, not the primary universe.  Headline
candidate-cardinality claims use the train/val primary pool under
`teacher_set_empty_prefix`.

## Case Universe

The case-index stage is the only source of truth for every Phase A pool role.
It scans train and val exhaustively and materializes:

```text
case_index.jsonl
case_index_summary.json
```

Headline crowded-pool condition:

```text
same_desc_gt_count >= 3
```

This condition applies only to the `headline_crowded` pool.  Count-1 and
count-2 controls are indexed as first-class case-index rows with their own
`pool_role`, so control probes share the same provenance, sampling, and join
contract as crowded probes.

Allowed pool roles:

| Pool Role | Condition | Headline Use |
| --- | --- | --- |
| `headline_crowded` | `same_desc_gt_count_annotated >= 3` | yes |
| `same_desc_count_1_control` | exactly 1 annotated same-desc GT object | control only |
| `same_desc_count_2_control` | exactly 2 annotated same-desc GT objects | control only |
| `fn_rescue_overlay_linked` | case linked to FN-rescue artifacts | overlay only unless also `headline_crowded` |
| `overlay_only_control` | overlay-linked row outside headline crowded pool | control/qualitative only |

Desc equality:

- `desc_normalization_policy_id = lower_strip_collapse_ws_v1`;
- normalize desc by stripping outer whitespace, lowercasing, and collapsing
  internal whitespace runs to one ASCII space;
- compare exact canonical desc text after that normalization;
- preserve raw desc text in `desc_text_raw`;
- preserve stripped-only text in `desc_text_stripped`;
- preserve canonical text in `desc_text_canonical`;
- set join/report desc text to `desc_text_canonical`;
- do not collapse synonyms or map through LVIS categories in Phase A.

Crowding buckets:

| Bucket | Condition |
| --- | --- |
| `same_desc_3` | exactly 3 annotated GT objects with the desc |
| `same_desc_4_5` | 4 or 5 annotated GT objects with the desc |
| `same_desc_6_plus` | 6 or more annotated GT objects with the desc |

The control slices are not part of the crowded headline denominator, but they
are required to verify that peak extraction behaves sanely on easier cases.

## Exhaustive Universe Versus GPU Probes

Phase A separates the case universe from the planned/probed subset.

`case_index.jsonl` is exhaustive over configured train/val inputs.
`probe_plan.jsonl` is the durable source of truth for which indexed rows are
planned for GPU-heavy stages.  GPU-heavy tables may be exhaustive or
stratified, but every probed row must point back to one `case_index.jsonl` row
and one `probe_plan.jsonl` row.

`probe_plan.jsonl` fields:

- `probe_plan_row_id`
- `case_id`
- `case_index_row_id`
- `probe_sampled`
- `sampling_policy_id`
- `sampling_policy_sha256`
- `sampling_seed`
- `sampling_weight`
- `strata_key`
- `planned_shard_id`
- `planned_gpu_id`
- `planned_stage_set`
- `planned_status`
- `planned_skip_reason`

Required probe bookkeeping fields:

- `case_id`
- `case_index_row_id`
- `probe_plan_row_id`
- `probe_sampled`
- `sampling_policy_id`
- `sampling_weight`
- `strata_key`
- `shard_id`
- `gpu_id` when GPU-bound
- `probe_status`

`summary.json` must report separate counters:

- `case_index_total_cases`
- `gpu_probe_planned_cases`
- `gpu_probe_attempted_cases`
- `gpu_probe_valid_cases`
- `taxonomy_assigned_cases`

Each counter is stratified at least by:

- `split`
- `desc_text`
- `same_desc_count_bucket`
- `pool_role`
- `fn_rescue_overlay_membership`

No report may use a GPU-probed denominator as if it were the exhaustive
case-index denominator.

`summary.json` must include both structural validity and conclusion eligibility:

- `validation_status`: artifact/schema/join correctness;
- `headline_eligibility_status`: whether coverage, controls, ambiguity gates,
  and required strata are sufficient for headline Phase A interpretation.

Artifact validity may be `ok` while headline eligibility is
`ineligible_low_coverage`, `ineligible_missing_controls`,
`ineligible_partial_label_ambiguous`, or `ineligible_sensitivity_unstable`.

## Annotation Boundary

COCO annotations are treated as an annotated universe, not complete object
truth:

```text
same_desc_gt_count = annotated same-desc count
gt_universe = coco_annotated
```

Extra peaks without annotated GT neighborhoods are not called hallucinations by
default.  They enter a review bucket until gallery review or cross-dataset
evidence can classify them.

Required fields:

- `gt_universe`
- `same_desc_gt_count_annotated`
- `extra_peak_without_gt_neighborhood_count`
- `peak_over_gt_count`
- `unmatched_peak_review_status`
- `unmatched_peak_interpretation`

If the unmatched peak rate is high, taxonomy rows must be marked
`partial_label_ambiguous` and cannot drive a Phase B/C headline until reviewed.
The default ambiguity threshold is:

```text
unmatched_peak_rate >= 0.25
```

where:

```text
unmatched_peak_rate =
  extra_peak_without_gt_neighborhood_count / max(1, merged_peak_count)
```

## Prefix Conditions

Phase A uses three prefix conditions with explicit hierarchy:

| Condition | Role |
| --- | --- |
| `teacher_set_empty_prefix` | primary headline condition |
| `teacher_prefix_at_boundary` | secondary clean coverage condition |
| `self_rollout_prefix` | overlay condition linked to low recall/FN behavior |

### teacher_set_empty_prefix

Construct a prompt that contains the image, the normal detection instruction,
the assistant response prefix up to the next object row, the queried desc, and
`<|box_start|>`, with no previous object rows.

It measures the pure desc-conditioned candidate field.  This is the only
condition allowed to support the headline A1 cardinality-collapse claim.

### teacher_prefix_at_boundary

Construct a clean GT-rendered prefix at object boundaries.  It measures how the
candidate field changes after a clean coverage state.

Ordering views:

- `sorted`
- `dataset_order` when available
- sampled `random_order` views for compatibility analysis only

### self_rollout_prefix

Use the checkpoint's own rollout prefix, reconstructed at complete object-row
boundaries.  It is split into:

| Subcondition | Meaning |
| --- | --- |
| `clean_self` | parse-valid self prefix with no accepted FP/duplicate flags under existing matching artifacts |
| `dirty_self` | parse-valid self prefix with FP, duplicate, mixed, or otherwise contaminated prefix quality |

Self-prefix rows are not used as pure cardinality evidence.  They are used to
build the transition matrix:

```text
teacher_empty -> teacher_boundary -> clean_self -> dirty_self
```

If teacher-empty has enough modes but self-prefix loses modes, the result is a
prefix/coverage/readout failure, not candidate-field collapse.

### Prompt And Policy Identity

All prompt-dependent tables must share the same prompt identity fields so that
x1 posterior, residual-row scoring, basin decode, attention rows, and taxonomy
rows can be joined without ambiguity:

- `prompt_instance_id`
- `prefix_condition`
- `prefix_row_count`
- `prefix_text_sha256`
- `prompt_text_sha256`
- `tokenizer_id`
- `tokenizer_sha256` when materializable
- `eos_token_id`
- `pad_token_id`
- `prompt_template_id`
- `object_field_order`
- `bbox_format`
- `coord_surface`
- `normalization`

Policy ids are required for any row that scores, decodes, or aggregates model
outputs:

- `posterior_policy_id`
- `row_score_policy_id`
- `score_token_reduction`
- `decode_policy_id`
- `attention_aggregation_policy_id`

Taxonomy assignment is invalid if evidence rows for one assignment disagree on
prompt identity or policy identity.

## Coordinate Posterior And Peak Semantics

The primary field is the coordinate-only conditional posterior over
`<|coord_0|>` through `<|coord_999|>` at `pre_x1`.

Each posterior row must also record:

- `coord_vocab_mass`
- `noncoord_vocab_mass`
- `p_cond_topk`
- `full_vocab_topk`

If:

```text
coord_vocab_mass < 0.05
```

or non-coordinate leakage is high enough that coordinate evidence is unreliable,
the taxonomy row must be marked `coordinate_channel_low_mass_or_leakage`, not
A1.

### Peak Extraction

Default peak policy:

```text
absolute_mass_floor = 0.002
relative_floor = 0.10
primary_merge_radius = 24
sensitivity_merge_radii = [16, 32]
gt_x1_neighborhood_radius = 24
raw_topk_k = 32
topk_sensitivity = [16, 64]
```

Peak definition:

```text
x1_peak = local maximum over coord bins
keep if prob_mass >= max(absolute_mass_floor, top1_mass * relative_floor)
merge peaks if abs(x1_a - x1_b) < merge_radius
```

Required metrics:

- `raw_topk_coverage@32`
- `merged_peak_count_r16`
- `merged_peak_count_r24`
- `merged_peak_count_r32`
- `gt_instance_coverage_by_peak_r24`
- `gt_instance_coverage_by_topk`
- `sensitivity_flip_flags`

If A1/A2/A3 labels flip across radius or mass-floor sensitivity enough to
change the dominant mechanism, the case is marked:

```text
sensitivity_unstable
```

and is excluded from headline promotion.

### X1 Projection Collision

X1-only cardinality can falsely collapse vertically separated objects that
share similar x1 values.  A case must be flagged:

```text
x1_projection_collision = true
```

when same-desc GT objects satisfy:

```text
abs(x1_i - x1_j) < primary_merge_radius
```

and are separated by either:

```text
abs(y1_i - y1_j) >= 48
```

or center distance at least `64` norm1000 units.

Projection-collision cases cannot be used as headline A1 evidence in Phase A.
They are reported as `projection_collision_unresolved` slices.  A future
joint x1/y1 or forced-x1 conditional-y1 sensitivity can be specified later, but
it is not part of this Phase A implementation roadmap.

## Residual-Row Logprob Scoring

Residual-row scoring answers whether multiple remaining GT rows are high
probability alternatives, whether EOS dominates, or whether desc/x1/tail spans
fail separately.

Required fields:

| Field | Meaning |
| --- | --- |
| `logp_row_mean` | mean teacher-forced log probability over the complete residual row |
| `logp_desc_span_mean` | mean log probability over desc-entry tokens |
| `logp_x1_token` | log probability of the residual row's first coordinate token |
| `logp_bbox_tail_mean` | mean log probability over y1/x2/y2 tokens |
| `logp_eos_at_boundary` | EOS/stop log probability at the row boundary |
| `margin_best_residual_vs_teacher_next` | best residual score minus teacher-next score |
| `margin_best_residual_vs_eos` | best residual score minus EOS boundary score |
| `preference_violation` | whether best residual differs from teacher-next |

Scoring rule:

```text
row_score_policy_id = residual_row_mean_v1
score_token_reduction = mean_logprob
best_residual_score = logp_row_mean
```

Length-normalized mean is the primary row score.  Summed logprob may be
reported as a diagnostic but cannot drive `preference_violation`.

EOS policy:

- record tokenizer EOS id;
- record compact/chat stop token ids if distinct;
- `logp_eos_at_boundary` must be computed from the same prompt instance as the
  residual row scores.

## Basin-Attraction Probe

The primary basin probe is greedy continuation after forced GT x1:

```text
prompt = desc + <|box_start|> + GT x1_i
decode = greedy generate y1/x2/y2
measure = generated box IoU to GT_i versus same-desc competitors
```

Auxiliary teacher-forced tail scoring:

```text
score target tail_i = log P(y1_i, x2_i, y2_i | desc, x1_i)
score competitor tails_j = log P(y1_j, x2_j, y2_j | desc, x1_i)
```

A3 is split into:

| Bucket | Meaning |
| --- | --- |
| `A3a_decode_basin_failure` | teacher-forced target tail beats competitors, but greedy tail does not separate |
| `A3b_tail_representation_failure` | teacher-forced target tail does not beat competitor tails |
| `A3_unresolved` | required tail-score evidence is missing or policy-inconsistent |

Required fields:

- `forced_x1_gt_idx`
- `greedy_box`
- `greedy_target_iou`
- `greedy_target_iou_rank`
- `best_same_desc_competitor_iou`
- `teacher_forced_tail_margin_target_vs_best_competitor`
- `target_tail_beats_competitor_tail`
- `decode_policy_id`
- `max_new_tokens`

## Attention Visual Field

Attention rows are secondary evidence.  They may explain x1 posterior behavior
or stratify cases, but they cannot be the sole evidence for A1/A2/A3.

Allowed uses:

- localization-head component coverage;
- background/sink mass diagnostics;
- target-vs-same-desc-competitor mass;
- gallery examples and stratification tags.

Disallowed uses:

- assigning A1/A2/A3 from attention alone;
- claiming attention-head causality;
- calling background mass a causal sink without a matched intervention.

Taxonomy rows whose only evidence is attention must be labeled:

```text
attention_only_unassigned
```

## Negative Controls And Sensitivity Gates

Phase A must include these controls or explicitly mark missing controls in
`summary.json`:

| Control | Purpose |
| --- | --- |
| `same_desc_count_1_control` | verify peak extractor works on non-crowded cases |
| `same_desc_count_2_control` | verify ordinary two-object ambiguity is separable |
| `wrong_desc_same_image` | prevent generic objectness peaks from masquerading as desc-conditioned modes |
| `wrong_image_same_desc` | prevent desc priors from generating apparent GT coverage on wrong images |
| `x1_projection_collision_slice` | separate x1 projection artifact from true cardinality collapse |
| `gt_x1_jitter` | verify forced-x1 basins are locally stable |
| `competitor_x1_control` | verify forced competitor x1 redirects basin |
| `merge_radius_sensitivity` | verify bucket stability under 16/24/32 radii |
| `mass_floor_sensitivity` | verify bucket stability under peak threshold changes |
| `p_cond_vs_coord_vocab_mass` | verify low peak count is not coordinate-channel leakage |

If controls fail, the related mechanism conclusion is marked
`control_failed_or_missing`, not promoted as a Phase B/C trigger.

`summary.json` must include a control matrix:

```text
control_status_by_type = {
  same_desc_count_1_control: pass | fail | missing,
  same_desc_count_2_control: pass | fail | missing,
  wrong_desc_same_image: pass | fail | missing,
  wrong_image_same_desc: pass | fail | missing,
  x1_projection_collision_slice: pass | fail | missing,
  gt_x1_jitter: pass | fail | missing,
  competitor_x1_control: pass | fail | missing,
  merge_radius_sensitivity: pass | fail | missing,
  mass_floor_sensitivity: pass | fail | missing,
  p_cond_vs_coord_vocab_mass: pass | fail | missing
}
```

Any `fail` or `missing` entry blocks the related headline claim unless the
report explicitly scopes the conclusion away from that control.

## Taxonomy

Taxonomy rows are materialized in:

```text
phase_a_case_taxonomy_rows.jsonl
```

Each row assigns a primary bucket and optional secondary flags for a
`case_id + desc_id + prefix_condition + prompt_instance_id`.

Primary buckets:

| Bucket | Criteria |
| --- | --- |
| `A1_cardinality_collapse` | non-collision, sensitivity-stable x1 candidate field has fewer covered GT instance modes than annotated same-desc GT |
| `A2_readout_or_coverage_failure` | candidate modes exist, but viable continuation/readout fails because residual rows are low, EOS is high, or modes disappear under coverage/self-prefix state |
| `A3a_decode_basin_failure` | x1 modes exist, viable continuation/readout passes, teacher tail separates, but greedy tail fails |
| `A3b_tail_representation_failure` | x1 modes exist, viable continuation/readout passes, but teacher-forced target tail does not beat competitor tails |
| `unassigned_or_inconclusive` | required evidence missing, ambiguous, or insufficient |

Invalid or guard buckets:

- `policy_mismatch`
- `sensitivity_unstable`
- `partial_label_ambiguous`
- `coordinate_channel_low_mass_or_leakage`
- `attention_only_unassigned`
- `projection_collision_unresolved`
- `control_failed_or_missing`

Continuation/readout gate:

`A3a` and `A3b` are allowed only when all are true:

- candidate x1 modes cover the target same-desc GT under the configured peak
  policy;
- residual-row continuation is viable under the same prompt and policy;
- EOS/stop is not dominant at the same boundary;
- prompt identity and scoring policy are consistent across x1, residual, and
  basin evidence.

If tail evidence fails while EOS is dominant or all residual rows are
suppressed, the primary bucket is `A2_readout_or_coverage_failure`, not A3.

Conflict fixtures required in tests:

| Conflict | Expected Primary Bucket |
| --- | --- |
| EOS high plus target tail loses | `A2_readout_or_coverage_failure` |
| residual rows low plus greedy binds competitor | `A2_readout_or_coverage_failure` |
| residual viable plus teacher tail loses | `A3b_tail_representation_failure` |
| residual viable plus teacher tail wins but greedy fails | `A3a_decode_basin_failure` |

Precedence:

1. policy/schema/parse mismatch;
2. missing required evidence;
3. coordinate channel low mass/leakage;
4. sensitivity unstable;
5. partial-label ambiguity requiring human review;
6. unresolved x1 projection collision;
7. A1;
8. A2 continuation/readout/coverage failure;
9. A3a/A3b basin/tail failure;
10. unassigned.

Secondary flags may include:

- `order_path_ce_friction`
- `same_desc_competitor_dominant`
- `prefix_transition_failure`
- `high_overlap_possible_merge`
- `small_object_slice`
- `person_heavy_slice`

Every assignment must include:

- `taxonomy_version`
- `primary_bucket`
- `secondary_flags`
- `reason_codes`
- `reason_metric_refs`
- `primary_evidence_chain`
- `invalid_flags`

## Macro And Stratified Reporting

Reports must include:

- micro aggregate;
- macro-by-desc aggregate;
- macro-by-image aggregate;
- person-excluded view;
- top-5-desc-capped view;
- train vs val;
- prefix condition;
- same-desc bucket;
- FN-rescue overlay membership;
- object-size bucket;
- overlap bucket.

If a dominant mechanism appears only in micro/person-heavy view but not in
macro-by-desc, report it as:

```text
dominant_in_frequent_crowded_classes_only
```

not as a general CoordExp mechanism.

## Artifact Contract

Core JSONL:

```text
case_index.jsonl
probe_plan.jsonl
x1_candidate_field_rows.jsonl
residual_row_score_rows.jsonl
basin_attraction_rows.jsonl
attention_component_rows.jsonl
phase_a_case_taxonomy_rows.jsonl
```

Support artifacts:

```text
case_index_summary.json
controls_summary.json
summary.json
report.md
resolved_config.yaml
manifest.json
plots/plot_manifest.json
gallery/gallery_rows.jsonl
```

Every row must include:

- `schema_version`
- `project_id = candidate_field_cardinality_tomography`
- `phase_id = phase_a`
- `run_id`
- `checkpoint_id`
- `case_id`
- `case_index_row_id`
- `probe_plan_row_id` when probe-dependent
- `split`
- `pool_role`
- `source_dataset_jsonl`
- `dataset_manifest_id`
- `dataset_manifest_sha256`
- `fn_rescue_overlay_membership`
- `fn_rescue_case_id` when applicable
- `prefix_condition` when prompt-dependent
- `prompt_instance_id` when prompt-dependent
- `shard_id` when sharded

Minimum join keys:

```text
case_id
case_index_row_id
split
source_line_idx
image_id
image_path
desc_id
desc_text
same_desc_cluster_id
target_gt_idx or gt_idx
prefix_condition
prompt_instance_id
residual_gt_idx
region_instance_id
x1_peak_id
shard_id
run_id
```

Table-specific keys:

| Table | Extra keys |
| --- | --- |
| `probe_plan` | `probe_plan_row_id`, `sampling_policy_id`, `sampling_policy_sha256`, `sampling_seed`, `planned_shard_id`, `planned_status` |
| `x1_candidate_field_rows` | `posterior_snapshot_id`, `x1_peak_id` |
| `residual_row_score_rows` | `row_score_id`, `residual_gt_idx` |
| `basin_attraction_rows` | `forced_x1_gt_idx`, `decode_attempt_id` |
| `attention_component_rows` | `role`, `layer`, `head`, `aggregation_scope`, `region_kind` |
| `phase_a_case_taxonomy_rows` | `taxonomy_assignment_id`, `reason_metric_refs` |

## Provenance

`manifest.json` must include:

- `artifact_contract_version`
- `created_at_utc`
- `code_git_commit`
- `worktree_path`
- `config_path`
- `resolved_config_sha256`
- `command_argv`
- Python, Torch, Transformers versions when available
- checkpoint path and identity
- model source
- processor/tokenizer identity
- prompt template id
- object field order
- bbox format
- coordinate surface
- `do_resize=false` provenance
- data manifest/hash or input JSONL identities
- input artifact identities and hashes where available
- row counts and sha256 for every artifact

## Execution

Required order:

1. `unit_tests`
2. `case_index_only`
3. `probe_plan_only`
4. `post_approval_case_index_validate`
5. `smoke`
6. `smoke_merge_validate_report_gallery`
7. `full_8card_tmux`
8. `full_merge_validate_report_gallery`

`case_index_only` and `probe_plan_only` perform no model forward.  Full GPU probing may be
exhaustive or stratified, but denominator reporting always comes from
`case_index.jsonl`, and planned-probe reporting always comes from
`probe_plan.jsonl`.

Full shard allocation uses strata-balanced round-robin over at least:

```text
split
same_desc_count_bucket
desc_frequency_bucket
fn_rescue_overlay
object_size_bucket
overlap_bucket
```

Each shard writes `shard_manifest.json` with:

- `shard_id`
- `gpu_id`
- `case_count`
- `strata_histogram`
- `stage_status`
- `input_case_ids`
- `output_row_counts`

Shard output layout:

```text
artifact_root/
  shards/
    shard_000/
      shard_manifest.json
      x1_candidate_field_rows.jsonl
      residual_row_score_rows.jsonl
      basin_attraction_rows.jsonl
      attention_component_rows.jsonl
```

The `merge` stage is the only stage allowed to create authoritative merged
GPU-probe tables at `artifact_root/*.jsonl`.  Duplicate active rows after merge
are validation failures unless they are exact byte-identical retry outputs with
one selected canonical row recorded in the merged manifest.

## Validation

Artifact validation fails if:

- any required core table is missing;
- JSON/JSONL parsing fails;
- schema versions mismatch the manifest;
- probe rows cannot join to `case_index`;
- taxonomy metric refs cannot resolve to source rows;
- policy ids mismatch within a taxonomy assignment;
- coverage counters are non-monotonic;
- shard merge has missing, malformed, stale, or duplicate shard outputs;
- A1/A2/A3 is assigned from attention-only evidence;
- report headline lacks scope and denominator;
- unmatched peaks affect interpretation without review status;
- sensitivity instability is present but ignored.

Smoke acceptance:

- `case_index_only` has run and produced `case_index.jsonl` plus
  `case_index_summary.json`;
- smoke sample is drawn from the case index, not handwritten;
- all six core JSONL tables exist and are non-empty;
- taxonomy rows include reason codes and metric refs;
- `summary.json.validation_status == "ok"`;
- `manifest.json` lists every artifact with row counts and hashes;
- `report.md` labels the scope as `smoke`.

Full acceptance:

- shard plan is built from inspected `case_index_summary.json`;
- shard manifests cover the planned probe set with no duplicate active rows;
- coverage counters distinguish indexed, planned, attempted, completed, valid,
  invalid, and assigned cases;
- headline tables are stratified by split, prefix condition, same-desc bucket,
  pool role, and FN overlay membership;
- taxonomy uses only policy-consistent rows;
- attention is supporting evidence only;
- annotation incompleteness boundary is present in `summary.json` and
  `report.md`.

## Implementation Layout

Use project-wise subtrees under the existing analysis namespaces:

```text
src/analysis/candidate_field_cardinality_tomography/
scripts/analysis/candidate_field_cardinality_tomography/
configs/analysis/candidate_field_cardinality_tomography/
tests/analysis/candidate_field_cardinality_tomography/
```

Module split:

```text
config.py
runner.py
case_index.py
prefixes.py
x1_candidate_field.py
residual_row_scoring.py
basin_attraction.py
attention_components.py
taxonomy.py
artifacts.py
report.py
plots.py
```

The CLI stays thin and delegates to `runner.py`.  Heavy model imports are
allowed only inside GPU-facing functions, not at module import time.

## Phase B/C Promotion Rules

After Phase A completes, choose one primary next phase:

| Dominant Result | Primary Next Phase |
| --- | --- |
| `A1_cardinality_collapse` | instance-mode separation / candidate-field sharpening |
| `A2_readout_or_coverage_failure` | Phase B painting and coverage-memory probes |
| `A3a` or `A3b` | coordinate-chain binding and post-x1 basin objectives |
| high residual alternatives and teacher-next not best | Phase C ordering / multiple-positive probe |

No Phase B/C direction is promoted without matched Phase A evidence on the
same case universe.
