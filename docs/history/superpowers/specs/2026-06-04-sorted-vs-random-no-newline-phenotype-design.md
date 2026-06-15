# Sorted Vs Random No-Newline Pure-CE Phenotype Design

## Decision

Run an A3.2 checkpoint-phenotype comparison for the two new full-object
pure-CE checkpoint-3668 runs:

- `fullobj_random_pure_ce_ckpt3668`
- `fullobj_sorted_pure_ce_ckpt3668`

The user hypothesis about external scratchpads / emitted-object ledgers is
temporarily parked.  This experiment should first characterize the two
checkpoints and relate them to the prior A3.1 ET-RMP-CE vs old pure-CE
findings.

## Checkpoints

Random-order pure-CE:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Sorted-order pure-CE:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
```

Training provenance says both use the same data family:

```text
public_data/coco/rescale_32_1024_bbox/{train,val}.coord.jsonl
```

Resolved config difference:

```text
random: resolved.data.object_ordering = random_permutation
sorted: resolved.data.object_ordering = sorted
```

## Scope

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2
```

Do not write A3.2 artifacts under the older A3.1
`prefix_state_transition_tomography` namespace.

Primary evidence scope should be labeled `a3_2_4096` if it reuses the 4096-row
prefix-state budget.

This is not production training.  GPU use is for paired mechanism probes and
rollout phenotype analysis.

## Scale Decision

Resolved scale:

```text
prefix-state tomography: 4096 sampled prefix states
native greedy rollout: 1024 images
FN-specialized probe: max 512 FN objects per checkpoint
GPU sharding: 8 shards when GPU work is parallelizable
sampling: hard-biased, same-desc multi-instance prioritized
easy sanity cap: <= 20%
```

Hard-biased sampling should favor:

- images or prefix states with same-desc object count >= 2;
- object count >= 6;
- desc count >= 2;
- crowded / repeated categories;
- train and val coverage;
- a small easy sanity slice no larger than 20%.

Rationale:

- 4096 prefix states keeps A3.2 comparable to A3.1 mechanism scope.
- 1024 rollout images should produce enough FNs for bucket analysis without
  becoming a full validation run.
- 512 FN objects per checkpoint bounds the hint-ladder cost while preserving
  high-value FN diversity.
- same-desc multi-instance cases are prioritized because they are most
  diagnostic for residual transition and instance-accounting behavior.

## External Val-200 Phenotype Context

The user's independent offline detector eval over `val-200` already reports a
clear sorted-over-random behavior advantage:

| metric | random | sorted | sorted - random |
| --- | ---: | ---: | ---: |
| `AP@[.50:.95]` | 0.3914 | 0.4072 | +0.0158 |
| `AP50` | 0.5351 | 0.5642 | +0.0291 |
| `AP75` | 0.4054 | 0.4226 | +0.0173 |
| `AR100` | 0.4498 | 0.4836 | +0.0338 |
| `F1-ish@0.50` | 0.4659 | 0.5892 | +0.1233 |
| `F1-ish recall@0.50` | 0.4952 | 0.5637 | +0.0686 |
| `F1-ish precision@0.50` | 0.4400 | 0.6171 | +0.1771 |
| `F1-ish FN@0.50` | 729 | 630 | -99 |
| `F1-ish FP@0.50` | 910 | 505 | -405 |

Use this as external phenotype context, not as mechanism evidence by itself.
A3.2 should test which prefix-transition, rollout, and FN-rescue readouts are
consistent with sorted reducing both FN and FP under the no-newline pure-CE
comparison.

## Resolved Experiment Policy

Use canonical sorted teacher prefix states for the main paired prefix-state
probe:

```text
same images
same prefix states
same residual sets
same probe descs
random checkpoint readout vs sorted checkpoint readout
```

Rationale:

- This isolates checkpoint dynamics under identical states.
- It avoids confounding the readout with different teacher prefix trajectories.
- It directly tests whether sorted training changed boundary dynamics,
  forced-desc x1 candidate fields, and same-desc residual transition behavior.

Use native greedy rollout separately:

```text
random checkpoint greedy rollout
sorted checkpoint greedy rollout
```

Rationale:

- Rollout captures each checkpoint's natural autoregressive trajectory.
- Prefix-state probe and rollout answer different questions and should not be
  collapsed into one metric.

Rollout decode policy:

- First-round native rollout uses greedy decoding only (`temperature=0` /
  deterministic decode).
- Main FN cases are derived only from greedy rollout.
- Native rollout is free text and unconstrained.  Diagnostic parser issues,
  malformed local rows, and non-metric-bearing parse policies are recorded as
  rollout phenotype evidence instead of failing the whole mechanism run.
- Official metric-bearing eval remains strict.  The diagnostic gt-vs-pred rows
  are for A3.2 mechanism analysis, not a substitute for mAP-bearing evaluator
  input.
- Low-temperature multi-sample rollout is reserved for later targeted follow-up
  on high-value FN subsets, not for the first-round primary FN buckets.

Rationale:

- Greedy rollout exposes the checkpoint's default conservative / deterministic
  behavior.
- Sampling can reveal latent recall potential, but it changes the FN set and
  introduces path variance.
- If a greedy FN appears in later sampled paths, that should be interpreted as
  decode/path/state preference evidence, not as first-round FN ground truth.

## Required Readout Axes

Boundary dynamics:

- residual-favored rate
- EOS-favored rate
- emitted-favored rate
- hard-competitor-favored rate
- other-desc-favored rate
- true tie / low-margin cases
- residual-vs-EOS margin
- residual-vs-winner margin

Boundary candidate desc set:

- Use image-local candidate descs, not the full COCO class list.
- Candidate descs are the de-duplicated union of:
  - all GT descs in the image;
  - emitted rollout descs;
  - the current FN desc;
  - the selected hard competitor desc.
- Record each candidate's role:
  - `target_fn_desc`
  - `residual_same_desc`
  - `residual_other_desc`
  - `emitted_same_desc`
  - `emitted_other_desc`
  - `hard_competitor_desc`
  - `other_gt_desc`

Rationale:

- The probe is about image-local residual object selection, not open-vocabulary
  class prior over all COCO labels.
- Full-class scoring is more expensive and introduces irrelevant descs that can
  obscure the boundary decision being tested.

Forced-desc pre-x1 candidate field:

- residual x1 coverage
- coverage positive rate
- full coverage rate
- merged x1 peak count
- valid peak count
- unmatched peak rate
- x1 target rank
- adaptive x1 evidence under the focused R95 axis rule

Rollout phenotype:

- class-any recall
- instance recall
- class-found-but-instance-missed rate
- same-desc FN rate
- same-desc duplication rate
- predicted row count
- EOS-after-partial-coverage rate
- output order agreement with sorted GT order

FN-specialized phenotype:

- FN object count and same-desc FN count
- FN class-found rate: whether the FN object's desc appears anywhere in the
  rollout
- FN pre-boundary residual score: whether the next-row boundary favors the FN
  desc or EOS / emitted / other desc
- FN forced-desc pre-x1 evidence: whether forcing the FN desc exposes a valid
  x1 peak near the FN object
- FN forced-desc-plus-x1 evidence: whether forcing desc+x1 makes y1/x2/y2
  converge toward the FN object
- prefix sensitivity: whether empty, canonical-sorted, rollout-prefix, and
  teacher-prefix conditions change the FN object's desc/x1 evidence
- emitted-ledger sensitivity: whether removing, shuffling, or replacing
  previous same-desc rows changes the FN object's desc/x1 evidence
- visual rescue boundary: whether even strong desc+x1 prompting fails to bind
  the FN object

FN hint coordinate surface:

- `fn_bbox` in replayable FN cases preserves the source rollout/GT bbox
  surface.  For current COCO `len12000` rollout rows this is pixel `xyxy`, so
  values can exceed 1000 on wide images.
- Hint prompts and generated-coordinate accounting operate on coord-token
  surface.  The FN hint runtime must derive a `target_box_coord_token` from
  the pixel `fn_bbox` using the case `width` and `height`, then seed
  desc+x1/desc+x1+y1 hints from that coord-token box.
- IoU for hint decode success is measured between generated coord-token boxes
  and `target_box_coord_token`.  Pixel `fn_bbox` remains available for raw
  case review and matching provenance.
- Out-of-range pixel coordinates such as `x2 > 1000` are therefore valid in
  `fn_bbox` but invalid inside prompt coord tokens.

FN matching definition:

- Main matching is greedy one-to-one same-desc matching with IoU >= 0.5.
- A GT object is a main FN if no same-desc prediction matches it at IoU >= 0.5.
- `near_miss` is recorded when a same-desc prediction overlaps the GT at
  0.3 <= IoU < 0.5.
- `wrong_desc_overlap` is recorded when a non-same-desc prediction overlaps the
  GT at IoU >= 0.5.
- `same_desc_duplicate` is recorded using the previously accepted simple rule:
  same-desc IoU > 0.95.
- Near-miss, wrong-desc-overlap, and duplication tags do not alter the main FN
  set; they are explanatory side labels.

Prior-A3.1 comparison:

- clean comparison: new random vs new sorted
- historical comparison only: old pure-CE / ET-RMP-CE vs new pair
- do not treat old-vs-new comparisons as clean ablations because data root,
  recipe, and template details may differ.

## FN-Specialized Probe

This module should focus only on false negatives.  False positives are not a
primary optimization target for this probe because many COCO FPs may be
unlabeled objects.

Start from greedy rollout for each checkpoint, match predictions to GT, and
select FN GT objects.  For each FN, run controlled probes to classify the likely
failure surface.

### FN Buckets

`vision_unreachable_fn`:

- Even after forcing the GT desc and a near-GT x1 hint, later coordinate
  readouts do not converge toward the FN object.
- This bucket is the closest to "recognize the visual encoder / visual feature
  stack failed" and should be rare before making strong claims.

`prefix_suppressed_fn`:

- The FN object has desc/x1 evidence under empty or teacher prefix, but that
  evidence weakens under the model's rollout prefix.
- This bucket means the object is conditionally visible but the autoregressive
  state suppresses saying it.

`residual_accounting_fn`:

- The FN desc was already emitted at least once, and forced-desc x1 evidence
  exists for the FN, but boundary scoring does not favor the residual FN desc
  under the current prefix.
- This bucket means the model may know the class and can bind the object when
  asked, but fails to decide that this object remains un-emitted.

`desc_selection_fn`:

- The FN x1 can be recovered if the desc is forced, but boundary scoring favors
  EOS, emitted desc, hard competitor, or other desc.
- This distinguishes desc/continuation selection failure from coordinate
  binding failure.

`coord_binding_fn`:

- Boundary / desc evidence is healthy, but forced-desc pre-x1 or later
  coordinate slots do not bind to the FN object.
- This should be tracked by slot: x1, y1, x2, y2.

`low_margin_ambiguous_fn`:

- The FN desc/residual candidate is close to the winner but not selected.
- This bucket is important for algorithm design because it may be helped by
  margin shaping, multiple-positive supervision, or coverage-aware decoding.

### Primary Bucket Priority

Each FN may carry multiple side labels, but summary tables should assign one
primary bucket using this priority:

```text
1. vision_unreachable_fn
2. coord_binding_fn
3. desc_selection_fn
4. residual_accounting_fn
5. prefix_suppressed_fn
6. low_margin_ambiguous_fn
```

Rationale:

- `vision_unreachable_fn` is the strictest and most consequential claim, so it
  must be checked first and only assigned when strong desc/x1 hints still fail.
- `coord_binding_fn` separates visual/coordinate binding failure from boundary
  desire failure.
- `desc_selection_fn` captures cases where forcing desc reveals evidence but
  natural boundary scoring did not select that desc.
- `residual_accounting_fn` captures same-desc emitted/residual confusion after
  class evidence exists.
- `prefix_suppressed_fn` is a state-sensitivity side mechanism that can coexist
  with the above, but should be primary only when no stronger bucket applies.
- `low_margin_ambiguous_fn` is the fallback for close calls where the residual
  candidate is near the winner but not selected.

The implementation should still persist all side labels, margins, and slot
errors so bucket priority does not hide mixed mechanisms in later analysis.

### Prefix Conditions

Probe each FN under at least these conditions:

- `empty_prefix`: no previous object rows.
- `rollout_prefix`: the model's own greedy rollout before it stopped or before
  the FN would be inserted.
- `teacher_sorted_prefix`: GT sorted rows before the FN.
- `teacher_oracle_remaining_prefix`: GT rows that exclude the FN but include
  other rows, used to test residual accounting without rollout drift.
- `same_desc_removed_prefix`: remove previous same-desc emitted rows.
- `same_desc_shuffled_prefix`: keep previous same-desc rows but change their
  order.

### Hint Ladder

Use a monotonic hint ladder for each FN:

1. no hint: score continuation / candidate descs at the boundary.
2. desc hint: force the FN desc and inspect pre-x1.
3. desc + x1 hint: force desc and near-GT x1, inspect y1/x2/y2.
4. desc + x1 + y1 hint: inspect x2/y2 if needed.

Adaptive coordinate evidence:

Use the focused Gaussian R95 axis rule for FN rescue evidence instead of a
fixed x1 radius:

```text
R95(axis_len) = floor(min(8, 0.04 * axis_len))
sigma = R95 / 1.96
```

For x1, `axis_len = x2 - x1`; for y evidence, `axis_len = y2 - y1`.

An FN x1 peak is strict evidence only when:

```text
abs(x1_peak - gt_x1) <= R95(width)
```

If `R95(width) == 0`, strict evidence is exact-token evidence.  Broader
diagnostics may still record a side label such as `x1_broad_near_24`, but this
must not be used as the main rescue-success criterion.

The previous fixed-radius A3.1 setting:

```text
gt_x1_neighborhood_radius = 24
```

is retained only as a broad candidate-field diagnostic, not as the FN rescue
success definition.

Interpretation:

- Failure at all levels suggests visual or coordinate-representation
  unreachable behavior.
- Rescue by desc suggests boundary/desc selection failure.
- Rescue by desc+x1 suggests x1 binding or instance-selection failure rather
  than visual invisibility.
- Large prefix-conditioned changes suggest state / residual-accounting failure.

### Output Artifacts

Write FN-focused artifacts separately from general A3.2 outputs:

```text
fn_probe/fn_cases.jsonl
fn_probe/fn_probe_rows.jsonl
fn_probe/fn_bucket_summary.json
fn_probe/fn_prefix_sensitivity.json
fn_probe/fn_slot_rescue_summary.json
fn_probe/gallery/index.md
fn_probe/gallery/images/*.jpg
```

Each row should include:

```text
checkpoint_role
image_id
source_line_idx
fn_gt_idx
fn_desc
fn_bbox
rollout_prefix_id
prefix_condition
hint_level
boundary_winner_role
boundary_margin_residual_vs_eos
forced_x1_residual_coverage
x1_target_rank
y1_rescue_error
x2_rescue_error
y2_rescue_error
assigned_fn_bucket
```

### Comparison Questions

For sorted vs random:

- Does sorted reduce `prefix_suppressed_fn` or `residual_accounting_fn`?
- Does sorted reduce EOS-driven FN but increase coordinate-binding FN?
- Does random have more recoverable FN under desc/x1 hints, implying broader
  latent candidate fields?
- Are same-desc FNs disproportionately `residual_accounting_fn`?
- Are different-desc FNs disproportionately EOS / desc-selection failures?

## Data Root Decision

The current machine has:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox
```

but only `train.jsonl` / `val.jsonl` are present there; the `coord` files are
not currently present at that exact path.  The machine also has:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/{train,val}.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/{train,val}.coord.jsonl
```

These are close data-family candidates but do not exactly match the checkpoint
provenance path.

Resolved policy:

- Use the locally available full-object-like coord-token data root:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/{train,val}.coord.jsonl
```

Rationale:

- The exact checkpoint-provenance coord files are not present at the recorded
  `rescale_32_1024_bbox/{train,val}.coord.jsonl` path in this worktree.
- The local `rescale_32_1024_bbox_len12000` coord files are the closest
  available full-object data-family surface.
- The user accepted treating the local dataset differences as negligible for
  this mechanism comparison.
- A3.1 remains a historical comparison rather than a strict same-data
  ablation.

## Template Contract

The A3.2 probes must match the two checkpoint training template surfaces:

```text
compact_full
coord_token
xyxy
row_separator = none
```

Completed object rows must be concatenated directly:

```text
row_1row_2row_3
```

not:

```text
row_1\nrow_2\nrow_3
```

This matters because newline insertion would alter boundary logits, EOS score,
and forced-desc continuation states.  The current A3.1 renderer already uses
direct concatenation:

```text
src/analysis/prefix_state_transition_tomography/prefix_rendering.py::render_teacher_prefix
```

Implementation requirements:

- Add an explicit A3.2 config/report field such as
  `template_contract.row_separator: none`.
- Reuse `render_teacher_prefix(...)`, `render_boundary_assistant_text(...)`,
  and `render_forced_desc_pre_x1_assistant_text(...)` for prefix-state and FN
  probes.
- Add or preserve a unit test asserting completed compact rows concatenate
  without newline.
- Include a runtime/provenance assertion or summary field showing no newline is
  inserted between compact object rows.
- Do not mix old newline-bearing teacher prefixes with these no-newline
  checkpoint readouts.

## Expected Interpretive Patterns

If sorted improves same-desc residual-favored boundary rates and lowers EOS,
sequence ordering entropy is likely a meaningful contributor to low recall.

If sorted narrows x1 peak count while improving top candidate stability, sorted
training may encourage a more deterministic traversal policy but reduce
candidate-field breadth.

If random keeps broader x1 fields but rollout remains more duplicate-prone or
less ordered, random ordering may provide weak multi-next-object supervision
without a reliable selection policy.

If random and sorted are similar, ordering is probably not the dominant root
for the previously observed A3.1 state-transition bottleneck.
