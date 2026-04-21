# Raw-Text Coordinate Mechanism Redesign

Date: 2026-04-21
Status: proposed
Owner: Codex

## Goal

Design a brand-new, mechanism-first research program for two **raw text-only**
grounding inference objects:

1. base-only:
   `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
2. base + raw-text pure-CE adapter:
   `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`

The program must answer two coupled mechanism questions:

1. whether raw-text digit-token coordinate chunks are internally composed into
   numeric-like states with local structure
2. whether that structure explains both:
   - good GT-centered grounding continuity
   - bad duplicate-burst onset and pre-burst anchor-collapse

It must also separate labeled false negatives into mechanism classes rather
than collapsing them into a single “model missed it” bucket.

The study is explicitly **not** a reuse of the older continuity report. Prior
artifacts may be reused as tooling references, candidate generators, or smoke
surfaces, but not as evidence.

## Hard Constraints

- Only the two fixed inference objects above are allowed.
- Both are **raw text-based** checkpoints with **no coord-token surface**.
- No auxiliary comparison models may be added.
- The study should maximize mechanism insight rather than minimize runtime.
- Eight GPUs are available, with one inference-heavy task per GPU and parallel
  execution encouraged.
- Human review must minimize effort and use a UI-friendly Notion surface rather
  than raw markdown or JSONL inspection.
- Human review budget is fixed to:
  - `15` FP / extra-prediction cases
  - `5` FN-mechanism cases

## Primary Thesis

The redesigned study tests two linked hypotheses.

### H1: Coordinate continuation

Digit-token sequences such as `8,1,9` are internally composed into
numeric-like states rather than treated as unrelated symbolic fragments, and
those states induce local structure in grounding behavior.

### H2: Emission suppression

Some labeled false negatives are not pure perceptual failures. Instead, the
model may internally support the object but fail to emit it because of rollout
state, prefix competition, stop pressure, or decode-selection effects.

## Primary Failure Surfaces

The study uses two linked failure surfaces.

### Observable failure surface

- `duplicate burst onset`

### Mechanism surface

- `pre-burst anchor-collapse`

The observable surface is the first visibly dangerous duplicate-like emission.
The mechanism surface is the earlier local basin or anchor failure that makes
that burst likely.

## High-Level Design

The study is divided into two explicit lanes.

### Lane A: Confirmatory core

Purpose:

- establish the strongest claim that can survive strict controls
- remain blind to duplicate-mined “hard” cohorts at the headline stage
- preregister metrics, layers, pooling rules, and null models

Main properties:

- random or blind-confirmatory object cohorts
- fixed serializer comparison
- fixed representation metrics
- no post-selection by wrong-anchor advantage

### Lane B: Exploratory mechanism lane

Purpose:

- investigate duplicate-burst onset, pre-burst anchor-collapse, FN suppression,
  and onset-local causality after the confirmatory core is frozen

Main properties:

- curated onset bank
- orthogonal prefix interventions
- heatmaps, hidden-state extraction, and FN-specific probes
- human review on a small gold subset

## Required Separation Of Concerns

The study must keep three objects distinct:

1. **immutable case bank**
   - one frozen object-level table with stable ids
2. **derived branch bundles**
   - duplicate-burst, FN, heatmap, perturbation, and representation subsets
3. **Notion review projection**
   - user-facing audit queue derived from local artifacts, never the authority

The case bank must not be a continuously mutating shared object. Earlier
artifacts are prepared, frozen, and then reused downstream.

## Serialization Surface Rule

The study must not treat repo-canonical `pretty_inline` as the only valid text
surface.

All confirmatory and mechanism-critical analyses must be run on two explicit
surfaces:

1. **model-native bbox surface**
2. **repo-canonical `pretty_inline` surface**

If an apparent numeric-composition effect disappears when the serializer
changes, it must not be promoted as a strong mechanism result.

## Case Bank

### Case-bank categories

The immutable object-level case bank must label each row as one of:

- `clean_continuation`
- `preburst_anchor_collapse`
- `first_burst_onset`
- `clean_extra_prediction`
- `labeled_fn`

Mined duplicate-like or hard-scene cases may be used only as candidate
generators. They must not be the main evidence bank.

### Required case-bank fields

Each row should carry at least:

- `case_uid`
- `model_object`
  - `base_only`
  - `base_plus_adapter`
- `image_id`
- `line_idx`
- `record_idx`
- `bucket`
- `source_object_index`
- `onset_object_index`
- `pred_idx` or `gt_idx`
- `preburst_prefix_objects`
- `source_object`
- `burst_object`
- `gt_next`
- `clean_extra_prediction`
- `selection_rank`
- `serializer_surface`
- `evidence_paths`

## Behavioral Measurement Contract

### Core measurement

The core observable remains full changed-chunk scoring, not single-token logits.

For coordinate-slot probes:

`score_chunk(k | prefix, image, slot, serializer_surface, model_object)`

For FN / stop-pressure probes, the same principle extends to:

- object-start continuation chunks
- EOS-vs-next-object continuation choices
- GT-object candidate chunks

### Required behavioral outputs

- `mass@1`, `mass@2`, `mass@4`, `mass@8`, `mass@16`
- expected absolute error
- local monotonicity around GT
- wrong-anchor advantage
- anchor-shift delta under prefix intervention
- vision lift

## Representation-Mechanism Contract

The hidden-state lane must be preregistered and heavily controlled.

### Required extraction surfaces

For each model object and serializer surface, extract:

- input embeddings for digit tokens
- per-layer hidden states at each digit token
- preregistered pooled span states:
  - `last_digit`
  - `mean_digits`
- final residual / readout-space state before LM head

### Required contexts

Each number representation should be tested in:

1. plain numeric string
2. bbox-slot context
3. non-bbox numeric control context

And repeated under:

- correct image
- swapped image
- onset-local prefix intervention where relevant

### Required representation metrics

- RSA or partial RSA against true numeric distance
- CKA across contexts
- local linear decode of scalar value
- ordinal pair tests such as:
  - “closer to `820` than `860`”
- nearest-neighbor numeric adjacency within matched lexical strata
- LM-head or readout margin for next-digit competition

### Mandatory controls

- exact tokenizer-level edit distance
- token-count match
- shared-prefix and suffix match
- non-bbox numeric controls
- swapped-image and distractor-image controls
- within-context permutation nulls of numeric labels
- dev-only layer selection
- image/object/value-disjoint evaluation splits

If a decoder or representation effect works only within one narrow serializer or
prefix family and collapses out of domain, it must not be called strong numeric
composition.

## Duplicate-Burst Causal Protocol

### Orthogonal intervention matrix

For each onset-local duplicate-burst case, run at least:

1. baseline
2. drop previous object
3. geometry-only swap with text fixed
4. text-only swap with geometry fixed
5. `x1/y1` only
6. `x2/y2` only
7. full `gt_next` geometry
8. nonlocal same-desc geometry control

### Required falsification gates

No duplicate-burst mechanism claim is accepted unless the case also has:

- correct image
- swapped image
- one local visual ablation or masking control

## FN Mechanism Protocol

### FN taxonomy

The FN lane uses five mutually exclusive buckets:

- `never_supported_fn`
  - not recovered by sampling, not recovered by clean-prefix continuation, and
    no strong teacher-forced support
- `decode_selection_fn`
  - recovered by same-prompt union-of-K sampling with no prefix intervention
- `continuation_blocked_fn`
  - not recovered by image-only sampling, but recovered only by clean-prefix
    continuation
- `stop_pressure_fn`
  - not recovered by normal decode, but recovered only when stop pressure is
    explicitly relaxed or EOS-vs-next-object preference flips
- `unlabeled_positive_or_eval_ambiguity`
  - annotation or evaluator ambiguity remains plausible; do not promote this to
    mechanism

### Extra-prediction taxonomy

Extra predictions are handled in a separate lane:

- `duplicate_burst_extra`
- `wrong_location_but_visually_real_extra`
- `unlabeled_positive_extra`
- `invalid_geometry_extra`
- `dead_hallucination_extra`

### Minimum causal probes

Every FN mechanism claim should be backed by:

1. baseline greedy decode
2. same-prompt union-of-K sampling
3. clean-prefix continuation probe
4. explicit stop-pressure probe
5. teacher-forced GT candidate scoring under fixed context

Old null findings such as “extended length did not help” are useful background,
but they do not replace an explicit stop-pressure probe.

## Human Review

### Review budget

- `15` FP / extra-prediction cases
- `5` FN cases

### FN review allocation

The five FN reviews should optimize for contrast:

- `2` adapter `decode_selection_fn` candidates
- `1` adapter `continuation_blocked_fn` candidate
- `1` ambiguity candidate
- `1` base-only anchor case for true incapacity or invalid-geometry-like
  failure

### Notion-first review surface

Notion should be the primary review interface with one row per case, not one
row per panel.

Required fields:

- `case_uid`
- `bucket`
- `priority`
- `status`
- `model_focus`
- `bbox_judgment`
- `mechanism_label`
- `best_evidence`
- `confidence`
- `notes`
- `asset_links`

Recommended views:

- `Inbox`
- `FP 15`
- `FN 5`
- `Escalations`

The review flow is bbox-first:

1. judge whether the prediction is correct / wrong-instance / GT-missing /
   wrong-location / unclear
2. only then consult heatmaps, perturbations, and representation summaries

## Execution Program

### Stage 0: Contract rebuild

- verify both fixed model objects exist
- rebuild:
  - coordinate chunk scoring
  - object-start continuation scoring
  - EOS-vs-object continuation scoring
- verify span alignment under:
  - GT prefix
  - self prefix

### Stage 1: Immutable case-bank mining

- infer/eval if required
- construct one immutable object-level bank
- freeze stable ids before any mechanism analysis begins

### Stage 2: Confirmatory core

- blind or random object cohort
- dual-surface serializer comparison
- preregistered representation metrics
- no duplicate-mined headline claims

### Stage 3: Review-gated shortlist freeze

- freeze exactly `20` review rows:
  - `15 FP`
  - `5 FN`

### Stage 4: Exploratory mechanism lane

- duplicate-burst onset
- pre-burst anchor-collapse
- orthogonal prefix interventions
- FN suppression lane
- heatmaps and hidden-state extraction on the frozen shortlist

### Stage 5: Synthesis

- confirmatory claims must survive the confirmatory core and all required
  falsification gates
- exploratory findings must be labeled as exploratory

## 8-GPU Execution Graph

Because the current stock runners bundle both model objects onto one device,
parallelization must be per-model, per-branch, per-shard.

### Stage 1 allocation

- `GPU0` base FP mining
- `GPU1` adapter FP mining
- `GPU2` base FN mining
- `GPU3` adapter FN mining
- remaining GPUs available for infer/eval rebuild or overflow

### Stage 4 allocation

- `GPU0` base FP hidden-state extraction
- `GPU1` adapter FP hidden-state extraction
- `GPU2` base FN hidden-state extraction
- `GPU3` adapter FN hidden-state extraction
- `GPU4` base heatmaps
- `GPU5` adapter heatmaps
- `GPU6` base perturbation
- `GPU7` adapter perturbation

## Artifact Contract

### Canonical local artifacts

Per model object:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`
- `summary.json`
- `resolved_config.json`
- `resolved_config.path`
- `per_image.json`
- `matches.jsonl`
- `vis_resources/gt_vs_pred.jsonl`

### Shared case bank

- immutable object-level case table
- stable join keys for every downstream branch

### Deep-probe branch artifacts

- `selected_cases.jsonl`
- `per_coord_scores.jsonl` or `per_cell_scores.jsonl`
- `summary_rows.jsonl`
- `heatmaps.jsonl`
- figure PNGs
- hidden-state or attention summaries keyed by `case_uid`

### Review snapshot artifacts

- `notion_export.csv`
- `review_labels.jsonl`
- `review_manifest.json`
- `report.md`
- `summary.json`

## Output Root

The study should write to a brand-new analysis root such as:

`output/analysis/raw_text_coordinate_mechanism_redesign_<date>/`

Do not overwrite or append to prior continuity bundles.

## Cleanup Policy

The study may begin with a hard cleanup of stale continuity-study artifacts.
This cleanup does not require backup copies.

Preferred cleanup targets include:

- superseded continuity or basin-analysis output bundles
- stale review packs, shortlist manifests, and intermediate deep-probe assets
- obsolete design notes or diagnostic reports from the rough continuity pass
- regenerable caches created solely for those earlier studies

Do not delete:

- source checkpoints
- active base inference infrastructure needed for the new study
- unrelated repo state outside the old continuity-study surface

The purpose of cleanup is to reset the workspace so the new mechanism-first
study starts from a clean artifact surface rather than inheriting stale
evidence.

## Non-Goals

- proving that old continuity verdicts were correct
- comparing coord-token versus raw-text families
- expanding beyond the two fixed raw-text inference objects
- using human review as a substitute for causal probes

## Success Criteria

The study succeeds if it produces:

1. a confirmatory answer on whether numeric-composition-like structure survives
   strict lexical and serializer controls
2. a causally cleaner answer on whether pre-burst anchor-collapse drives
   duplicate-burst onset
3. an FN mechanism breakdown that distinguishes decode selection, continuation
   blocking, stop pressure, and ambiguity
4. a low-effort Notion review surface that calibrates the interpretation of
   extra predictions and selected FN cases
