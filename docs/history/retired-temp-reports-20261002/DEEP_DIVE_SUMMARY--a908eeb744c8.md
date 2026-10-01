# Raw-Text Decode-Bias Hard-Sample Deep Dive

Date: 2026-04-22

This note summarizes the deeper follow-up after the main raw-text decode-bias
study completed. It uses only raw-text `norm1000_text` artifacts and focuses on
the mined hard/representative/crowded subsets plus the completed `val200`
counterfactual and decode-side runs.

## Sources

- main counterfactual run:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-counterfactual-val200-bs4`
- main decode run:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-laneb-val200-bs4-fanout8`
- hybrid hard-sample mining:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-hybrid-hard-samples-val200`
- EOS-hard subset run:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-eos-hard12-stop-pressure-bs4`
- dense-repeat subset run:
  - `/data/CoordExp/output/analysis/raw-text-decode-bias-dense-repeat12-rp-bs4`

## Main Findings

### 1. The current decode-side EOS intervention is globally inert

The stop-pressure implementation currently maps the named mode
`min_new_tokens_after_object_open` to a plain HF `min_new_tokens` floor.
This is not a dynamic object-open-aware intervention.

Empirical consequence on the repaired full `val200` stop-pressure lane:

- `base_only`: stop-pressure `on` and `off` are identical for `200 / 200` rows
- `base_plus_adapter`: stop-pressure `on` and `off` are identical for
  `200 / 200` rows

Generation-length evidence from the same full lane:

- `base_only`: min `92`, avg `1365.78`, max `3084`, `56 / 200` hit
  `max_new_tokens`
- `base_plus_adapter`: min `71`, avg `830.42`, max `3084`, `28 / 200` hit
  `max_new_tokens`
- all `400 / 400` decoded samples already exceed the `24`-token floor

EOS-hard subset evidence:

- `base_only`: min `242`, avg `1214.67`, max `3084`, `4 / 12` hit
  `max_new_tokens`
- `base_plus_adapter`: all `12 / 12` runs hit `3084`
- EOS-hard subset `on` and `off` runs are identical record-by-record for both
  models

Interpretation:

- the existing stop-pressure decode intervention does not probe the
  counterfactually observed EOS bias
- the intervention is inactive in practice because the generated sequences are
  already far longer than the imposed `24`-token floor

### 2. Decode termination pathology is dominated by long post-parse tails

On the full `val200` stop-pressure lane:

- `base_only`: `148 / 200` rows include `<|endoftext|>` in `raw_special_tokens`
- `base_plus_adapter`: `149 / 200` rows include `<|endoftext|>` in
  `raw_special_tokens`

On the EOS-hard subset:

- `base_only`: `8 / 12` rows include `<|endoftext|>` tails
- `base_plus_adapter`: `9 / 12` rows include `<|endoftext|>` tails

This means the decode-side EOS issue is not showing up as a simple
“native model stops too early unless we force more tokens” phenomenon.
Instead, many runs produce a valid parsed object list and then continue into a
special-token or unfinished-tail regime.

### 3. Counterfactual EOS bias is still real, but it is a branchpoint story

The main counterfactual EOS lane remains the best evidence for EOS-like bias:

- `base_only`: `19 / 200` `stop_pressure_signature` cases
- `base_plus_adapter`: `0 / 200`

Representative EOS-hard examples mined from the finished case cards:

- source index `123`, image `12670`, repeated target category `person: 13`
- source index `190`, image `19109`, repeated target category `person: 13`
- source index `2`, image `632`, repeated target category `book: 13`

Interpretation:

- the EOS-like conservative mechanism is visible in teacher-forced branchpoint
  scoring
- the current decode-side stop-pressure knob does not reach that mechanism
- future EOS experiments should suppress the actual terminating tokens or the
  relevant branchpoint mass, not only set a global minimum generation length

### 4. Repeat penalty is the stronger decode-time lever on crowded raw-text scenes

Dense-repeat `12`-image subset decode results:

- `base_only`
  - `rp=1.00`: AP `0.0682`, predictions `17`
  - `rp=1.10`: AP `0.0787`, predictions `119`
- `base_plus_adapter`
  - `rp=1.00`: AP `0.0563`, predictions `194`
  - `rp=1.10`: AP `0.2849`, predictions `248`

Interpretation:

- on this crowded same-class subset, repetition penalty is not mainly hurting
  valid repeats
- instead, raising the penalty often improves decode behavior materially,
  especially for the adapter

### 5. The repeat-penalty mechanism has two distinct modes on hard samples

The dense-repeat subset still mixes two mechanisms:

1. valid-repeat recovery
2. duplicate-burst suppression

Examples:

- image `12120` (`person: 13`, `chair: 8`)
  - `base_only`
  - `rp=1.00`: `1` prediction
  - `rp=1.10`: `44` predictions
  - interpretation: repeat penalty unlocks enumeration rather than suppressing it

- image `18380` (`person: 13`, `bowl: 6`)
  - `base_plus_adapter`
  - `rp=1.00`: `26` predictions
  - `rp=1.10`: `48` predictions
  - interpretation: repeat penalty expands a crowded scene into a more complete
    multi-class parse

- image `14038` (`book: 12`)
  - `base_plus_adapter`
  - `rp=1.00`: `50` predictions, dominated by `47` `book`
  - `rp=1.10`: `6` predictions with a much more diverse category mix
  - interpretation: repeat penalty suppresses a duplicate-burst mode

This means the current dense-repeat shortlist is informative, but not pure.
For publication-quality reporting, it should be split into:

- `valid_repeat_recovery`
- `duplicate_burst_suppression`

instead of treated as one homogeneous population.

Full-lane recovery / suppression mining artifacts are now materialized under:

- `/data/CoordExp/output/analysis/raw-text-decode-bias-hybrid-hard-samples-val200/repeat_mode_split`

Key mined exemplars from the repaired full `val200` repeat-penalty lane:

- `base_only` recovery:
  - image `16958`
  - image `12120`
  - image `7574`
- `base_only` suppression:
  - image `17899`
  - image `3255`
  - image `6471`

- `base_plus_adapter` recovery:
  - image `4134`
  - image `19432`
  - image `18380`
- `base_plus_adapter` suppression:
  - image `14038`
  - image `3934`
  - image `19109`

Aggregate counts on the crowded/repeated full-lane slice:

- `base_only`
  - recovery-like cases with positive prediction-count change: `41`
  - suppression-like cases with negative prediction-count change: `18`
- `base_plus_adapter`
  - recovery-like cases with positive prediction-count change: `44`
  - suppression-like cases with negative prediction-count change: `12`

### 6. Token-group evidence supports the decode-side repeat results

On the dense-repeat shortlist, counterfactual `rp=1.10` token-group deltas show:

- `base_only`, `continue_with_gt`
  - `desc`: `+0.075`
  - `digit`: `+0.508`
  - `structure`: `+0.454`
- `base_only`, `exact_duplicate`
  - `desc`: `-1.189`
  - `digit`: `+0.044`
  - `structure`: `+0.572`

- `base_plus_adapter`, `continue_with_gt`
  - `desc`: `+0.391`
  - `digit`: `+0.743`
  - `structure`: `+0.573`
- `base_plus_adapter`, `exact_duplicate`
  - `desc`: `-1.146`
  - `digit`: `-0.038`
  - `structure`: `+0.670`

Interpretation:

- valid continuation is helped mainly through `digit` and `structure`
- exact-duplicate continuations are hurt mainly through `desc`
- the adapter shows the stronger continuation gain, especially on digits

This is consistent with the decode-side dense-repeat AP gains.

## Current Mechanism Picture

The strongest current interpretation is:

1. EOS-like conservative behavior exists in raw-text branchpoint scoring,
   especially for `base_only`
2. the current end-to-end stop-pressure knob is not a valid causal intervention
   for that mechanism
3. repeat penalty is a much stronger and more operationally effective decode
   lever on crowded same-class raw-text scenes
4. the adapter is especially sensitive to repeat-penalty improvements
5. crowded hard samples are not one mode; they separate into at least:
   - EOS-hard branchpoint cases
   - valid-repeat recovery cases
   - duplicate-burst suppression cases

## Recommended Next Experiments

1. Replace the current decode-side EOS intervention with a true terminating-token
   suppression policy rather than a global `min_new_tokens` floor.
2. Split the current dense-repeat shortlist into:
   - valid-repeat recovery
   - duplicate-burst suppression
3. Render visual case cards for at least:
   - image `12670`
   - image `19109`
   - image `12120`
   - image `18380`
   - image `14038`
