---
title: Raw-Text Decode-Bias Mechanism Findings
date: 2026-04-22
status: active-diagnostic
owner: codex
depends_on:
  - progress/diagnostics/2026-04-21_raw_text_coordinate_mechanism_findings.md
  - output/analysis/raw-text-decode-bias-laneb-val200-bs4-fanout8/report/summary.json
  - output/analysis/raw-text-decode-bias-counterfactual-val200-bs4/report/summary.json
  - output/analysis/raw-text-decode-bias-dense-repeat12-rp-bs4/report/summary.json
  - output/analysis/raw-text-decode-bias-eos-hard12-first-structural-closure-bs4/decode_val200_stop_pressure/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-stop-signature19-first-structural-closure-bs4/decode_val200_stop_pressure/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-stop-signature1-branchpoint-census/counterfactual_branchpoint_census/case_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-stop-signature19-branchpoint-census/counterfactual_branchpoint_census/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-stop-signature1-bbox-tail-then-object-open-bs4/decode_val200_stop_pressure/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-quote-drift7-bbox-tail-then-object-open-bs4/decode_val200_stop_pressure/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-quote-drift7-bbox-tail-then-object-open-once-bs4/decode_val200_stop_pressure/summary_rows.jsonl
  - output/analysis/raw-text-decode-bias-base-only-stop-signature19-bbox-tail-then-object-open-once-bs4/decode_val200_stop_pressure/summary_rows.jsonl
---

# Raw-Text Decode-Bias Mechanism Findings

## Why This Note Exists

This note preserves the current decode-bias conclusions from the raw-text-only
mechanism line so the results do not get lost while the deeper branchpoint work
continues.

It is the historical diagnosis layer for:

- decode-time EOS / continue bias
- decode-time repeat-penalty behavior
- the current failed and successful intervention reads

All claims here are about raw-text `norm1000_text` behavior only.

## Fixed Scope

The study keeps one fixed checkpoint pair:

1. `base_only`
   - `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
2. `base_plus_adapter`
   - base:
     `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
   - adapter:
     `/data/CoordExp/output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`

This note does not compare against coord-token families and should not be read
as evidence for coord-token value.

## Current Artifact Bundle

- full `val200` counterfactual lane:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-counterfactual-val200-bs4`
- full `val200` decode lane:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-laneb-val200-bs4-fanout8`
- dense valid-repeat 12-image subset:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-dense-repeat12-rp-bs4`
- EOS-hard 12-image structural-close intervention:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-eos-hard12-first-structural-closure-bs4`
- exact `base_only stop_signature19` structural-close intervention:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature19-first-structural-closure-bs4`
- one-case branchpoint-census smoke:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature1-branchpoint-census`
- full `base_only stop_signature19` branchpoint census:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature19-branchpoint-census`
- one-case two-step object-open smoke:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature1-bbox-tail-then-object-open-bs4`
- quote-drift 7-image two-step object-open subset:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-quote-drift7-bbox-tail-then-object-open-bs4`
- quote-drift 7-image one-shot object-open subset:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-quote-drift7-bbox-tail-then-object-open-once-bs4`
- exact `base_only stop_signature19` one-shot object-open subset:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature19-bbox-tail-then-object-open-once-bs4`
- broader `EOS-hard12` one-shot object-open rerun:
  `/data/CoordExp/output/analysis/raw-text-decode-bias-eos-hard12-bbox-tail-then-object-open-once-bs4-rerun`

## Preserved Conclusions

### 1. EOS-like suppression is real in counterfactual scoring

On the repaired full `val200` counterfactual EOS lane:

- `base_only`: `19 / 200` cases (`9.5%`) satisfy the current
  `stop_pressure_signature`
- `base_plus_adapter`: `0 / 200`

The important read is not that the model lacks visual evidence. It is that some
`base_only` cases favor stopping on total path score even when continuation is
competitive on a token-normalized read.

### 2. Repeat penalty is the strongest useful decode-time lever so far

On the repaired full `val200` decode sweep, `repetition_penalty = 1.10` was
best for both checkpoints:

- `base_only`: `bbox_AP 0.0949 -> 0.1198`
- `base_plus_adapter`: `bbox_AP 0.2749 -> 0.3794`

The effect is not “higher repeat penalty improves everything.” The cleaner read
is:

- AP usually improves
- valid enumeration often improves
- duplicate-like and repeated-desc behavior are not monotonically reduced in
  every setting

So repeat penalty is moving the continuation-vs-duplicate decision boundary,
not simply suppressing repeated objects everywhere.

### 3. Dense valid-repeat scenes strengthen the repeat-penalty story

On the dense same-class 12-image subset:

- `base_only`: `bbox_AP 0.0682 -> 0.0787` from `rp=1.00 -> 1.10`
- `base_plus_adapter`: `bbox_AP 0.0563 -> 0.2849`

This is the clearest current evidence that higher repeat penalty can help
legitimate crowded raw-text enumeration rather than only suppressing bursts.

The fixed 12-image dense-repeat pack is also concentrated in the categories we
explicitly care about for valid-repeat behavior:

- `person`: `8 / 12` pack rows
- `chair`: `5 / 12`
- `bowl`: `3 / 12`
- `book`: `2 / 12`

Average repeat-margin gain from `rp=1.00 -> 1.10` on the curated target
categories is:

- `bowl`: `+2.597`
- `person`: `+1.961`
- `book`: `+1.914`
- `chair`: `+0.956`

So the dense-repeat result is not being carried by an unrelated long tail. It
is already centered on the intended crowded repeated-object categories.

### 4. Special EOS tokens are not the decisive stop mechanism

The special-token-only suppression runs were exactly inert on both:

- `EOS-hard12`
- exact `base_only stop_signature19`

So the harmful ending decision is not made at `<|im_end|>` or
`<|endoftext|>`-like tokens.

### 5. The real stop branchpoint is structural JSON closure

The structurally targeted interventions changed behavior, while special-token
suppression did not. This localizes the real stop mechanism to structural
raw-text closure of the top-level JSON object / `objects` list.

### 6. But naive closure suppression is the wrong intervention

The corrected structural-close suppressor is active and harmful.

On exact `base_only stop_signature19`:

- `bbox_AP 0.1560 -> 0.1267`
- predictions `62 -> 36`

On `EOS-hard12`:

- `base_only`: `0.1656 -> 0.1271`, predictions `67 -> 41`
- `base_plus_adapter`: `0.2326 -> 0.2034`, predictions `121 -> 94`

So “block the first closure token” is not a rescue intervention. It removes
objects and lowers AP on the very slice where we hoped it would help.

## Updated Mechanism Read

The current best mechanism statement is:

1. the harmful stopping decision is made at structural JSON closure, not at
   special EOS tokens
2. the decode problem is not binary `stop now` vs `continue correctly`
3. the real local competition is:
   - close the list now
   - continue with the next object correctly
   - continue into the wrong schema path

That third branch is essential. It explains why blunt closure suppression
failed: blocking closure did not reliably redirect probability into the valid
next-object path.

## Branchpoint-Census Read

The exact `base_only stop_signature19` branchpoint census is now complete under:

- `/data/CoordExp/output/analysis/raw-text-decode-bias-base-only-stop-signature19-branchpoint-census`

Headline result:

- `array_branch_close_prefers_stop_count = 19 / 19`
- `avg_array_branch_stop_minus_continue_raw_logprob = +6.9836`
- `final_close_available_count = 0`
- `final_close_fused_count = 19 / 19`

At the decisive array-close boundary, the group masses are:

- average `wrong_schema` mass: `0.7491`
- average `close_now` mass: `0.2481`
- average `next_object` mass: `0.000153`

And on the exact 19-case slice:

- `wrong_schema_mass > next_object_mass` in `19 / 19`
- `close_now_mass > next_object_mass` in `19 / 19`

The top tokens on the strongest cases are dominated by fused
close-array-then-continue tokens such as:

- `"],"`
- `"]}"`
- `"]"`

The token-level census is even cleaner than that summary suggests:

- wrong-schema top token: `"],"` in `19 / 19`
- close-now top token: `"]}"` in `18 / 19`, `"]"` in `1 / 19`
- next-object top token: `" ,"` in `18 / 19`, `",\""` in `1 / 19`
- final-close status: `fused_with_array_close` in `19 / 19`

Representative hardest cases by wrong-schema mass include:

- `base_only:val200:112`
- `base_only:val200:54`
- `base_only:val200:8`
- `base_only:val200:190`
- `base_only:val200:93`

This is the strongest mechanism evidence so far. On the actual stop-signature
population, the valid next-object path is not merely weaker than clean closure.
It is almost absent compared with both:

- close-now mass
- and a fused close-array-plus-wrong-schema path

The one-case smoke on `base_only:val200:123` already previewed this:

- `stop_minus_continue_raw_logprob = +8.5`
- `close_now` mass: `0.1826`
- `next_object` mass: `0.00000146`
- `wrong_schema` mass: `0.8166`

The completed 19-case run confirms that this was not a one-off outlier.

## Artifact Hygiene Note

Top-level `poly` leakage in `raw_output_json` was fixed in
`src/common/prediction_parsing.py`, but that was an artifact-hygiene repair,
not a metric-changing eval correction. The current AP conclusions above still
stand.

## Two-Step Quote-Drift Rescue Read

The first positive follow-up after the branchpoint census was a two-step
intervention:

1. keep the bbox-tail `]}` to `]},` steering at the true structural branch
2. then, after `]},` plus optional whitespace, steer the next non-whitespace
   token toward object-open (`{`) and away from a bare top-level quote (`"`)

The one-case smoke on `base_only:val200:123` was not informative. It matched
the earlier one-step bbox-tail steer exactly, which means this case was never a
quote-drift failure in the first place.

The informative targeted run is the 7-image `quote_drift7` subset:

- source indices: `55, 89, 93, 130, 138, 165, 190`
- image ids: `5600, 8762, 9400, 13546, 14226, 16958, 19109`

These are the exact cases where the earlier bbox-tail-only intervention had
already forced `]},`, but then often continued with a bare top-level quote and
drifted into keys like `"poly"` or `"bbox_2d"`.

On the older one-step bbox-tail steer, these 7 cases had:

- parse-valid rows: `1 / 7`
- nonempty rows: `1 / 7`
- total predictions: `1`
- `raw_output_json = null` rows: `6 / 7`

On the new two-step `steer_bbox_tail_then_object_open` run, the same 7 cases
had:

- parse-valid rows: `7 / 7`
- nonempty rows: `7 / 7`
- total predictions: `7`
- `raw_output_json = null` rows: `0 / 7`

The local branchpoint behavior changed exactly as intended. On all 7 cases, the
first non-whitespace token after the first steered `]},` became:

- `{\n`
- then `"desc":`

rather than the earlier bare quote path that led into top-level key drift.

So the two-step intervention does solve the immediate wrong-schema continuation
problem.

## But The Rescue Still Fails Operationally

Even though the new two-step steer fixes the immediate quote-drift branch, it
still underperforms the within-run baseline on the same 7-image subset:

- `bbox_AP 0.0777 -> 0.0455`
- total retained predictions `77 -> 7`
- repeated-desc rate `0.5714 -> 0.0`

At first glance this looks like another “one object only” collapse, but the raw
token traces show a more interesting failure mode.

For all 7 `quote_drift7` cases, the steered generation:

- stayed inside the `objects` schema
- continued producing many additional objects after the first one
- ran all the way to `max_new_tokens = 3084`
- did not close the full JSON object/list cleanly

Representative trace facts:

- generated token length: `3084` for all `7 / 7`
- first post-`]},` continuation: always object-open plus `"desc"`
- number of `]},` occurrences in the generated text: roughly `83-88` per case
- no final special EOS close; no clean full JSON closure

Representative tail patterns show runaway enumeration rather than clean
termination. The model keeps emitting more objects until truncation, for
example:

- repeated `traffic light` objects marching across the frame
- repeated `person` boxes on a sliding coordinate grid
- a late drift into object lists like `spoon`, `bowl`, `banana`

So the two-step rescue does **not** convert the failure into correct dense
enumeration. It converts:

- wrong-schema immediate drift

into:

- schema-correct but unclosed runaway enumeration to truncation

The saved `raw_output_json` remains small because the parser can only recover a
valid early prefix from the unclosed overlong generation.

## Re-Armed One-Shot Continuation Read

The next probe tightened the intervention further.

Instead of keeping bbox-tail / object-open steering alive at every later object
boundary, the new one-shot mode does exactly one local rescue:

1. at the first bbox-tail closure branch, steer `]}` toward `]},`
2. immediately after that, steer the next non-whitespace token away from a bare
   top-level quote and toward object-open (`{`)
3. once the next object has actually opened, disable the intervention for the
   rest of the decode

This probe was designed to answer a narrower question:

- is the real missing ingredient simply “repair the first continuation and then
  get out of the way”?

### Result on quote-drift7

On the same `quote_drift7` subset:

- the old one-step bbox-tail steer had `1 / 7` parse-valid rows
- the persistent two-step steer had `7 / 7` parse-valid rows but hurt AP and
  collapsed retained objects
- the new one-shot steer is **exactly baseline-equivalent**

For the one-shot run:

- `bbox_AP` stayed `0.0777 -> 0.0777`
- total retained predictions stayed `77 -> 77`
- parse-valid rows stayed `7 / 7 -> 7 / 7`
- repeated-desc rate stayed `0.5714 -> 0.5714`

More strongly, on all 7 images the one-shot `stop_pressure_on` outputs were
identical to baseline `stop_pressure_off`:

- `pred` lists identical
- `raw_output_json` identical
- `errors` identical
- `generated_token_text` identical

So the one-shot probe successfully avoids the old harmful quote-drift branch,
but it does so by collapsing back to the model's natural baseline decode rather
than by creating a new improved continuation trajectory.

### Result on exact stop_signature19

The same one-shot probe was then run on the full exact
`base_only stop_signature19` slice.

This result is even cleaner:

- `bbox_AP` stayed `0.1560 -> 0.1560`
- total retained predictions stayed `62 -> 62`
- parse-valid rows stayed `19 / 19 -> 19 / 19`
- repeated-desc rate stayed `0.4737 -> 0.4737`

Again, the equality is exact:

- `pred` lists identical on all `19 / 19`
- `raw_output_json` identical on all `19 / 19`
- `errors` identical on all `19 / 19`
- `generated_token_text` identical on all `19 / 19`

So on the strongest mechanistic population we have, the re-armed local rescue
does not merely “cause no measurable aggregate change.” It is literally
baseline-equivalent token-for-token.

### Result on broader EOS-hard12

The one-shot probe was then widened to the broader `EOS-hard12` validation pack
for both checkpoints under:

- `/data/CoordExp/output/analysis/raw-text-decode-bias-eos-hard12-bbox-tail-then-object-open-once-bs4-rerun`

This broader rerun stayed exact-match for both models.

For `base_only`:

- `bbox_AP` stayed `0.1656 -> 0.1656`
- total retained predictions stayed `67 -> 67`
- parse-valid rows stayed `12 / 12 -> 12 / 12`
- repeated-desc rate stayed `0.75 -> 0.75`

For `base_plus_adapter`:

- `bbox_AP` stayed `0.2326 -> 0.2326`
- total retained predictions stayed `121 -> 121`
- parse-valid rows stayed `12 / 12 -> 12 / 12`
- repeated-desc rate stayed `0.75 -> 0.75`

And again the equality is exact, not just metrically flat:

- `pred` lists identical on `12 / 12` rows for both checkpoints
- `raw_output_json` identical on `12 / 12`
- `errors` identical on `12 / 12`
- `generated_token_text` identical on `12 / 12`

So the one-shot intervention is no longer just a negative result on a narrow
`base_only` signature slice. It is now a broader exact no-op across:

- `quote_drift7`
- exact `base_only stop_signature19`
- full `EOS-hard12` for both `base_only` and `base_plus_adapter`

### What this means

This is a high-value negative result.

The local post-bbox-tail continuation repair is **not** the missing intervention
for the model's native EOS-like conservatism.

More precisely:

- persistent continuation steering is harmful because it keeps fighting later
  healthy closure decisions
- but once the steering is limited to a single local rescue, it becomes exactly
  the same as baseline on the actual stop-signature population

So the remaining mechanism is not:

- “after the model decides to continue, it still needs help opening the next
  object”

Instead, the harder remaining problem is earlier:

- whether the decode path enters the relevant continuation branch at all
- and how often the model attempts another object before any bbox-tail rescue
  would even become relevant

## Best Current Research Thesis

Raw-text decode conservatism is real, but it is not best explained as “special
EOS wins because the path is short.”

The stronger current thesis is:

- `base_only` has a real stop-pressure signature population on `val200`
- the crucial decision happens at structural JSON closure
- the wrong alternative to clean closure is often schema drift, not valid next
  object continuation
- even when we redirect away from immediate schema drift, the model can still
  fall into schema-correct but runaway unclosed enumeration
- when we reduce that rescue to a one-shot local intervention, the decode path
  collapses all the way back to exact baseline on every completed slice we have
  tested, including the broader `EOS-hard12` pack for both checkpoints
- repeat penalty is a real and useful decode-time lever, especially for the
  adapter on dense same-class scenes

## Next Live Step

The re-armed local continuation probe is no longer a future idea. On the
completed `base_only` slices, it is now established as exact baseline-equivalent.

So the next live step, if this line continues, is no longer another variant of
“repair the first post-bbox-tail continuation and then get out of the way.”

The harder remaining question is earlier and sharper:

- how do we make the model enter the relevant next-object branch at all
- before the decode ever commits to clean early closure
- without pushing it into wrong-schema drift
- and without pushing it into runaway unclosed enumeration

That points to a more upstream intervention class, such as:

- branch-entry steering before the first bbox-tail close decision
- or a stateful bounded-window controller that stays active longer than one
  token family but still deactivates before runaway enumeration starts
