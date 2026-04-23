# Raw-Text Decode-Time Bias Study

Date: 2026-04-21
Status: proposed
Owner: Codex

## Goal

Design a clean follow-on mechanism study for the two fixed raw-text detection
objects from the 2026-04-21 mechanism line:

1. `base_only`
   - `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
2. `base_plus_adapter`
   - base:
     `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
   - adapter:
     `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`

The study must explain whether decode-time priors, rather than missing visual
evidence alone, are suppressing or distorting valid object emission under the
raw-text `xyxy` / `norm1000_text` contract.

It focuses on two linked decode-time mechanisms:

1. EOS / continue path-length bias
2. repeat-penalty bias under dense same-class enumeration

## Why This Study Exists

The current mechanism study already established several important facts:

- raw-text coordinate continuity is real
- naive summed logprob is strongly length-biased
- the adapter moves hard duplicate-burst cases closer to the GT-vs-duplicate
  decision boundary
- some false negatives are better explained as EOS beating continuation than as
  pure perceptual failure

What remains open is the decode-policy layer:

- how much of conservative raw-text behavior is explained by EOS favoring short
  paths
- whether `repetition_penalty` suppresses valid repeated objects together with
  invalid duplicate bursts
- whether the adapter improves local boundary alignment while becoming more
  sensitive to decode-time priors

## Hard Scope Constraints

- Study only the two fixed raw-text checkpoints above.
- This is a raw-text-only study.
- Do not introduce coord-token comparisons.
- Do not route any scoring, reporting, or interpretation through coord-token
  families.
- Do not use this study to make claims about coord-token value or coord-token
  continuity.
- Backend for end-to-end sweeps is HuggingFace.
- Keep `max_new_tokens` large, defaulting to `3084`, to avoid truncation
  artifacts.
- Use `val200` as the end-to-end benchmark surface.
- Keep the end-to-end EOS leg narrow:
  - targeted stop-pressure ablation only
  - no broad decode-policy search
- Use the repeat-penalty sweep:
  - `1.00`
  - `1.02`
  - `1.05`
  - `1.10`

## Primary Questions

### Q1: EOS / continue path-length bias

Determine whether valid object continuation loses mainly because `EOS now` is a
much shorter sequence, even when continuation is locally plausible on a
per-token basis.

The study must answer:

- whether this mechanism appears in a measurable subset of baseline `val200`
  false negatives, with prevalence reported separately from enriched mechanism
  cases
- whether the effect differs between `base_only` and `base_plus_adapter`
- whether the adapter improves local GT alignment but remains vulnerable to
  stop-pressure at decode time

### Q2: Repeat-penalty bias

Determine whether `repetition_penalty` incorrectly penalizes valid repeated
objects in crowded same-class scenes.

The study must answer:

- whether the penalty harms recall for categories such as:
  - `person`
  - `book`
  - `chair`
  - `bowl`
  - `baseball bat`
- whether the penalty impact is concentrated in:
  - description tokens
  - digit tokens
  - structural tokens
- what tradeoff exists between duplicate-burst suppression and recall / AP

## Study Populations

The study must keep selection populations explicit instead of using one curated
case source for every conclusion.

### P1. Enriched mechanism cases

Used for:

- deep EOS counterfactual reading
- duplicate-burst and pre-burst repeat-penalty analysis

Source:

- canonical mechanism case bank and shortlist

### P2. EOS prevalence slice

Used for:

- coarse prevalence reporting on baseline `val200` false negatives

Source:

- baseline `val200` false negatives sampled or enumerated without oracle-only
  enrichment

The EOS report must distinguish:

- enriched mechanism evidence
- prevalence-style `val200` FN evidence

### P3. Dense same-class valid-repeat slice

Used for:

- repeat-penalty analysis on legitimate repeated-object scenes

Source:

- an explicit dense same-class slice, independent from duplicate-burst or
  pre-burst pathology cases

This population should favor scenes where repeated instances of the same class
are valid rather than already degenerate.

### P4. Hybrid hard-sample research pack

Used for:

- deeper follow-up after the main `val200` and counterfactual lanes finish
- selecting publication-grade examples instead of relying on one monolithic
  shortlist
- building a compact in-depth pack that covers both EOS-like suppression and
  legitimate crowded same-class repetition

Source:

- joint mining over:
  - enriched mechanism cases
  - baseline `val200` prevalence rows
  - dense same-class valid-repeat rows
  - completed counterfactual outputs
  - completed end-to-end decode outputs

This is not a replacement for the explicit study populations above. It is a
derived research pack meant to support sharper qualitative and targeted
follow-up analysis once the main evidence lanes already exist.

## Design Principle

Keep one canonical truth per concern:

1. the existing raw-text mechanism study remains the authority for case
   discovery and review curation
2. the new decode-bias study becomes the authority for decode-time bias
   measurement and sweep reporting

This avoids duplicating case-selection logic or splitting evidence across two
incompatible research harnesses.

## EOS Intervention Refinement

The EOS-side decode intervention should now be understood as a branchpoint
steering problem, not a special-token suppression problem.

The next decode probe should therefore use a local positive-steering mode with
this exact contract:

- activate only at a fresh completed-object boundary inside the top-level
  `objects` array
- suppress any token whose stripped decoded form begins with `]`, because on
  this surface that represents either immediate list closure or fused
  close-plus-schema-drift continuation
- positively bias comma-led continuation tokens, because after a completed
  object the correct next step is a comma that keeps enumeration inside the
  `objects` array
- keep special EOS tokens out of the mechanism claim; they may still appear as
  trailers, but they are not the decisive stop lever

The first live execution surface for this refined probe is the exact
`base_only stop_signature19` slice, with a one-case smoke on source index
`123` before the full 19-case run.

## Hybrid Hard-Sample Mining Protocol

After the main study lanes complete, the decode-bias study should support one
explicit mining pass for harder and more representative follow-up examples.

The purpose is to avoid two common failure modes:

- over-indexing on pathological duplicate bursts that are visually unusual
- reporting only broad aggregate `val200` numbers without concrete,
  interpretable hard cases

The hybrid mining pass should score each candidate case along at least four
axes:

1. `eos_bias_strength`
   - derived from branchpoint preference, sum-vs-mean divergence, and the
     `stop_pressure_signature` rule
2. `valid_repeat_sensitivity`
   - derived from repeat-penalty change on valid continuation, repeat-penalty
     change on exact duplicate, and the resulting continuation-vs-duplicate
     margin shift
3. `crowdedness_and_repeat_density`
   - derived from GT object count, same-class count, and whether target dense
     categories such as `person`, `book`, `chair`, `bowl`, and `baseball bat`
     are repeated
4. `representativeness`
   - derived from whether the scene looks like a normal crowded detection case
     rather than only an extreme pathology case, with both `base_only` and
     `base_plus_adapter` behavior preserved for comparison

The mining pass should emit three explicit packs, not one collapsed shortlist:

- `eos_hard_shortlist`
  - strongest EOS-like suppression cases, especially where `base_only` shows a
    sum-vs-mean signature
- `dense_valid_repeat_shortlist`
  - crowded same-class scenes where repeated instances are legitimate and
    repeat penalty materially changes the continuation-vs-duplicate margin
- `representative_mixed_shortlist`
  - compact publication-style examples spanning both mechanisms without
    over-representing pathological bursts

These packs should preserve exact `image_id`, source index, artifact handles,
and case provenance so later targeted reruns can be driven from them directly.

## Raw-Text Contract

Everything in this study must stay inside the raw-text bbox family:

- raw-text `xyxy`
- raw-text `norm1000_text`

Allowed numeric evidence is limited to raw-text digit spans and raw-text object
continuations.

Disallowed surfaces:

- coord-token generations
- coord-token scoring
- coord-token continuity heatmaps or attribution language
- any “raw-text vs coord-token” framing in the report

All scorer and prompt-construction paths must set raw-text mode explicitly.
They must not rely on scorer defaults.

## Architectural Decision

Implement the work as a sibling study inside the existing raw-text mechanism
family rather than as multiple standalone one-off scripts.

### Canonical upstream sources

- case discovery and shortlist:
  - `scripts/analysis/run_raw_text_coordinate_mechanism_study.py`
  - `src/analysis/raw_text_coordinate_mechanism_study.py`
- authoritative case artifacts:
  - `output/analysis/raw-text-coordinate-mechanism/case_bank.jsonl`
  - `output/analysis/raw-text-coordinate-mechanism/shortlist.jsonl`

### New decode-bias surface

- CLI:
  - `scripts/analysis/run_raw_text_coordinate_decode_bias_study.py`
- implementation:
  - `src/analysis/raw_text_coordinate_decode_bias_study.py`
- configs:
  - `configs/analysis/raw_text_coordinate_mechanism/decode_bias_default.yaml`
  - `configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml`

### Reused scoring utilities

- `src/analysis/raw_text_coordinate_continuation_scoring.py`
- `src/analysis/raw_text_coord_continuity_scoring.py`
- `src/analysis/unmatched_proposal_verifier.py`

These already provide:

- adapter-aware `TeacherForcedScorer`
- matched-span sum / mean scoring
- continuation-span scoring on raw-text surfaces
- coordinate-span scoring for digit-token changes

### Required New Scoring Seam

The current scorer utilities are sufficient for raw teacher-forced span scoring,
but they do not yet define the processed-logprob path required for a clean
repeat-penalty counterfactual.

The decode-bias study therefore must add one explicit new seam:

- a processed span-scoring path that applies HF decode-time logits processors,
  especially `RepetitionPenaltyLogitsProcessor`, against the true history ids at
  each scored position

Implementation should reuse the existing decode-processor pattern already
present in:

- `src/analysis/duplication_collapse_analysis.py`

The repeat-penalty counterfactual lane must report both:

- raw teacher-forced score
- repetition-processed score

for the same fixed continuation, so the analysis can distinguish:

- the model's intrinsic continuation likelihood
- the additional decode-policy penalty imposed at inference time

Authoritative processed-score contract:

- use the full model history available at each scored position
  - prompt tokens
  - already-emitted assistant tokens before the current scored token
- do not use assistant-only history as the headline processed score
- record this choice explicitly in manifests as:
  - `counterfactual.history_scope: full_model_history`

The processed-score path must emit per-position rows before aggregation so later
reports can deterministically build:

- full-span totals
- group-specific totals
- raw-vs-processed deltas

### Required Case-Hydration Contract

The current mechanism case bank is an authority for identifiers and selection,
but not a sufficient frozen replay surface for decode-bias scoring by itself.

The decode-bias study must therefore materialize a versioned hydrated input
bundle before scoring, for example under:

- `counterfactual_inputs/hydrated_cases.jsonl`

Each hydrated row must include enough information to replay the exact scoring
objects without re-deriving hidden logic from upstream source rows:

- baseline assistant text
- stop-now candidate assistant text
- continue-with-GT candidate assistant text when applicable
- exact-duplicate candidate assistant text when applicable
- predicted-object candidate assistant text when applicable
- source artifact handles and record ids
- hydration algorithm version

## Study Lanes

The study is intentionally split into two linked evidence lanes.

### Lane A: Counterfactual lane

Purpose:

- isolate decode-time bias while holding the continuation fixed
- avoid conflating model belief with search-policy effects

This lane uses teacher-forced or exact-span rescoring on selected mechanism
cases from the canonical case bank.

#### A1. EOS / continue counterfactual

In this study, “EOS / stop now” means the shortest valid raw-text assistant path
that closes the current object list immediately. It does not mean the literal
chat-template EOS special token.

On the same baseline prefix, compare:

- `stop_now_path`
- `continue_with_gt`

for each selected FN case.

Required views:

- shared-prefix branchpoint comparison:
  - `logprob(first_token_of_stop_now_path | prefix)`
  - first valid continuation-token logprob or continuation mass at the same
    branchpoint
- continuation-span `sum_logprob`
- continuation-span `mean_logprob`
- token-count delta
- matched-length or continuation-only normalized comparison

#### A1b. Branchpoint token census

After the EOS counterfactual rows exist, the study must run one explicit
branchpoint-token census on the same raw-text prefixes.

This lane exists because the fixed EOS interventions already showed that the
important stop decision is not the later special EOS token. The decisive
competition happens at the structural JSON close boundary.

The census must therefore not be framed as:

- special EOS token vs continuation

It must instead read the actual token competition at the structural boundary.

Required design:

- do not invent a hand-authored `poly` or other irrelevant schema candidate
  string for scoring
- keep the analysis on real next-token distributions from the teacher-forced
  scorer
- group tokens by boundary role rather than by one forced wrong-schema replay

The census should expose two related branchpoints when tokenization allows:

1. `array_close_branch`
   - context: immediately after a completed object while still inside the
     top-level `objects` list
   - compare:
     - structural array-close tokens
     - valid next-object continuation tokens
   - report:
     - actual stop-path first-token logprob
     - actual continue-path first-token logprob
     - top-k token table
     - exact group mass for `close_now` and `next_object`

2. `final_close_branch`
   - context: after the stop path has already closed the `objects` list and is
     deciding whether to close the enclosing JSON object
   - compare:
     - final close-now token
     - generic wrong-schema continuation mass, defined as top-level comma-based
       continuation rather than a hand-injected field name
   - report:
     - actual final-close token logprob
     - top-k token table
     - exact group mass for `close_now` and `wrong_schema`

If tokenization fuses the array-close and final-close decision into one token,
the artifact must record that explicitly rather than pretending the second
branchpoint was measured.

This lane is the primary mechanism bridge between:

- the teacher-forced EOS-like evidence
- the negative results from blunt closure suppression
- the next positive intervention design

#### A2. Repeat-penalty counterfactual

Rescore the same candidate continuations under:

- `repetition_penalty = 1.00`
- `repetition_penalty = 1.02`
- `repetition_penalty = 1.05`
- `repetition_penalty = 1.10`

Candidate set should include, when available:

- valid repeated-object continuation from the dense same-class valid-repeat
  slice
- valid GT repeated-object continuation
- exact duplicate continuation
- model-predicted continuation

The key requirement is to hold candidate text fixed while changing only the
decode-time repetition processor.

#### A3. Hard-sample case cards

For mined hard samples, the study should support one compact case-card surface
that joins:

- source identity:
  - `image_id`
  - source index
  - model alias
  - population pack membership
- GT scene statistics:
  - total GT object count
  - max same-class count
  - repeated target categories when present
- EOS summary:
  - branchpoint preference
  - sum-vs-mean margin
  - whether the sample matches `stop_pressure_signature`
- repeat summary:
  - continuation and exact-duplicate deltas from `1.00` to the selected higher
    penalties
  - token-group deltas for `desc`, `digit`, and `structure`
- downstream recommendation:
  - whether the sample is best used as:
    - EOS exemplar
    - dense valid-repeat exemplar
    - mixed representative exemplar

The case-card output is not a new scoring family. It is a synthesis layer built
on top of the existing machine-readable row artifacts.

### Lane B: End-to-end HF decode lane

Purpose:

- measure whether the same biases materially move behavior in real decoding
- quantify the recall / duplicate-control tradeoff on a benchmark surface

This lane runs on `val200` and reuses the standard infer -> score -> eval path.

#### B1. Repeat-penalty sweep

For both checkpoints, run full HF decode on `val200` with:

- `repetition_penalty ∈ {1.00, 1.02, 1.05, 1.10}`
- large `max_new_tokens`
- fixed prompt and ordering contract aligned with the baseline mechanism study

#### B2. EOS stop-pressure ablation

Run a targeted stop-pressure intervention only.

The study should not perform a large grid over stop-related knobs. The point is
causal sensitivity, not heuristic optimization.

Preferred intervention shape:

- retain the canonical HF decode policy
- add one explicit stop-pressure mode:
  - `decode.stop_pressure.mode: suppress_terminating_tokens_after_object_boundary`
  - `decode.stop_pressure.trigger_rule: raw_text_object_boundary`
  - semantics: when the raw-text decoder is at a valid top-level `objects`
    list boundary immediately after a completed object, suppress close-now
    terminating tokens such as structural `]` / `]}` continuations and known
    end-of-turn special tokens
- record exact keys in the manifest:
  - `decode.stop_pressure.mode`
  - `decode.stop_pressure.min_new_tokens`
  - `decode.stop_pressure.trigger_rule`
  - `decode.stop_pressure.active`

If this increases duplicate bursts or malformed outputs, that is part of the
result and must be reported rather than hidden.

Interpretation rule:

- end-to-end EOS ablation is a sensitivity test, not standalone proof of native
  EOS preference
- causal EOS claims must still be anchored in the shared-prefix branchpoint
  measurements from the counterfactual lane

## Benchmark Contract

The end-to-end lane must pin one exact benchmark surface in config rather than
relying on informal `val200` shorthand.

Default study input should align with the current raw-text mechanism family:

- `public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl`

Canonical `val200` rule:

- headline `val200` results must come from one fixed materialized subset or one
  exact persisted source-index list
- do not define `val200` by fresh reservoir sampling at run time
- if the subset is derived from full val, the exact source indices must be
  written and reused verbatim in manifests and downstream reruns

If an alternate proxy-expanded eval view or lvis-proxy input is used for a
particular slice, the run name, manifest, and report must state that explicitly
rather than silently treating it as the same benchmark surface.

## Measurement Contract

The study must keep score families separated. No single metric is allowed to
stand in for every mechanism claim.

### EOS / continue scoring family

For each case, report:

- `eos_branch_logprob`
- `first_continue_branch_logprob` or `continue_branch_mass`
- `eos_now_sum_logprob`
- `eos_now_mean_logprob`
- `continue_with_gt_sum_logprob`
- `continue_with_gt_mean_logprob`
- `continue_minus_eos_sum_logprob`
- `continue_minus_eos_mean_logprob`
- token counts for both paths
- matched-length or continuation-only normalized comparison
- branchpoint-token census rows when the census lane is enabled:
  - `array_close_branch`
  - `final_close_branch`
  - fused-token status when applicable

#### EOS classification rule

Only classify a case as `stop_pressure_signature` when:

- the shared-prefix branchpoint favors `EOS` over the first valid continuation
  branch
- continuation loses on joint score
- but continuation is competitive or better on mean or matched-length score

This prevents generic low-confidence misses from being mislabeled as EOS bias.

### Repeat-penalty scoring family

For each candidate continuation and penalty setting, report:

- changed-span `sum_logprob`
- changed-span `mean_logprob`
- token count
- delta from the `1.00` baseline

#### Token-group decomposition

Every repeat-penalty counterfactual row must break the affected span into:

- `desc`
- `digit`
- `structure`

and report group-specific penalty deltas.

Authoritative grouping rule:

- `desc`
  - token positions aligned to the serialized `desc` value span only
- `digit`
  - token positions aligned to ASCII digit characters inside raw-text bbox
    numeric spans
- `structure`
  - all remaining token positions in the scored continuation span, including
    quotes, commas, brackets, colons, whitespace, and field names

The grouping must be computed from serialized assistant-text spans first and
then projected to token positions deterministically.

This is a diagnostic decomposition, not a standalone causal proof that one
token family is solely responsible for the suppression pattern. It is mandatory
because the main research question includes whether the penalty appears to be
hurting:

- semantic repetition
- numeric coordinate spans
- JSON-like structural scaffolding

## End-to-End Reporting Contract

For each `val200` decode run, materialize:

- infer summary and resolved config
- scored artifact
- evaluation outputs
- decode-health counters

At minimum, decode-health counters should include:

- parse-valid rate
- nonempty rate
- total prediction count
- duplicate-like or repeated-object rate
- finish-reason summary when available

Required headline reporting:

- standard `val200` metric table, explicitly labeled as `val200`
- object count totals
- false negative counts
- duplicate / repeated-object diagnostics
- same-class-dense category slices for at least:
  - `person`
  - `book`
  - `chair`
  - `bowl`
  - `baseball bat`

Required hard-sample reporting:

- one mined `eos_hard_shortlist`
- one mined `dense_valid_repeat_shortlist`
- one mined `representative_mixed_shortlist`
- one case-card table that can be used to trace each shortlisted sample back to
  the exact source artifacts and replay settings

The report must always state the scope as `val200`; it must not blur these
results into a full-val claim.

When reporting EOS prevalence, the study must state the population explicitly:

- enriched mechanism FN subset
- or prevalence-style baseline `val200` FN slice

## Adapter Comparison Rule

Every causal claim must be framed as a within-metric comparison between:

- `base_only`
- `base_plus_adapter`

The main interpretation target is:

1. whether the adapter improves GT-vs-duplicate boundary alignment
2. whether that improvement also increases or decreases sensitivity to decode
   priors such as EOS preference or repetition penalty

Do not collapse the comparison into one scalar “better” judgment.

## Artifact Layout

The study should emit one clear run root:

- `output/analysis/raw-text-coordinate-decode-bias-<date>/`

Expected contents:

- `summary.json`
- `stage_manifest.json`
- `counterfactual_eos/`
- `counterfactual_repeat_penalty/`
- `decode_val200_repeat_penalty/`
- `decode_val200_stop_pressure/`
- `hard_sample_mining/`
- `report/`

Each lane should also write machine-readable row tables such as:

- `case_rows.jsonl`
- `summary_rows.jsonl`
- `sweep_rows.jsonl`
- `shortlist.jsonl`
- `case_cards.jsonl`

Artifact naming must preserve:

- model alias
- scoring family
- decode setting
- benchmark scope

## Verification Plan

The study must verify on the narrowest realistic surface first, then scale.

### Smoke pass

Before full `val200`, run a smoke configuration that verifies:

- both checkpoints resolve correctly
- the decode-bias config schema parses cleanly
- teacher-forced score rows include token counts and matched-span stats
- repeat-penalty counterfactual outputs include token-group decomposition
- the EOS intervention path records its exact policy in the manifest
- end-to-end artifacts are complete for at least one small subset

### Full pass

For the full `val200` run, verify:

- resolved configs exist
- `max_new_tokens` is the intended large value
- exact decode settings are written into manifests
- scored artifacts exist where required
- evaluation outputs exist and are labeled as `val200`
- report tables distinguish:
  - counterfactual evidence
  - end-to-end decode evidence
- mined hard-sample packs distinguish:
  - EOS-hard examples
  - dense same-class valid-repeat examples
  - representative mixed examples

No result should be promoted from partial logs. Only completed artifacts under
the run directory count as evidence.

## Risks And Failure Modes

### R1. False EOS attribution

Risk:

- generic low-confidence misses may be mistaken for stop pressure

Mitigation:

- require the explicit sum-vs-mean or matched-length signature

### R2. Repeat-penalty claims without token attribution

Risk:

- we might observe a net score shift without knowing whether it comes from
  description tokens, digits, or structure

Mitigation:

- require token-group decomposition for every counterfactual repeat-penalty row

### R3. Heuristic overreach in EOS sweeps

Risk:

- a large stop-policy search would confound causal interpretation

Mitigation:

- keep the EOS end-to-end lane to a targeted stop-pressure ablation only

### R4. Scope drift away from raw-text contract

Risk:

- accidental reuse of coord-token or non-canonical bbox surfaces

Mitigation:

- keep the raw-text `xyxy` / `norm1000_text` contract explicit in config,
  scoring, and report language

### R5. Hard-sample overfitting

Risk:

- the deeper follow-up may accidentally overfit to spectacular but unrepresentative
  cases

Mitigation:

- keep separate packs for `EOS-hard`, `dense_valid_repeat`, and
  `representative_mixed`
- preserve GT crowdedness statistics and source provenance in every case card
- keep aggregate `val200` reporting as the main headline evidence

## Out Of Scope

- coord-token comparisons
- large-scale stop-policy hyperparameter search
- claiming full-val benchmark conclusions from `val200`
- replacing the existing mechanism study as the authority for case discovery

## Deliverables

The design is complete when the repo can support:

1. a smoke decode-bias study run
2. a full `val200` decode-bias run
3. a mined hard-sample pack with case cards for deeper research
4. a concise diagnostic note in `progress/diagnostics/`
5. machine-readable artifacts that let a later agent reconstruct:
   - which cases were scored
   - which decode settings were used
   - how the conclusions were separated between counterfactual and end-to-end
     evidence
   - how shortlisted hard samples were chosen and replayed
