## ADDED Requirements

### Requirement: Policy likelihood remains score authoritative
Artifact `logprob`, selected replay `selected_logprobs`, `pred[*].score`,
`pred_score_version: 1`, and source kind
`token_trace_selected_logprob_mean` MUST continue to use policy likelihood.
Raw-model likelihood MUST be auxiliary and MUST NOT alter prediction inclusion,
ranking, parsing, bbox values, or evaluator behavior.

#### Scenario: Dual trace score replay
- **WHEN** policy and raw likelihood differ for selected tokens
- **THEN** the stored prediction score recomputes only from artifact `logprob`
  policy values

### Requirement: Additive raw-model likelihood evidence
Each generated-token trace row SHALL include nullable `raw_model_logprob` and
MUST record whether the raw channel was disabled, available, or failed. When
enabled, every non-pad generated token including `<|im_end|>` MUST have a finite
non-positive raw value. Selected replay rows MAY omit duplicated raw values
because generated-step indices provide exact lookup.

#### Scenario: Raw trace enabled
- **WHEN** raw tracing is enabled and generation succeeds
- **THEN** every generated non-pad token row contains aligned policy and raw
  likelihood values

### Requirement: Likelihood and backend provenance
Run manifest, scored provenance, shard identity, and merge evidence MUST record
the likelihood definitions, score-owned channel, raw-trace enablement, backend
name/mode/response family/version, effective engine settings, execution-model
fingerprint, processor policy, and generation policy. Rank-local disagreement
in semantic fields MUST fail merge.

#### Scenario: Rank uses raw vLLM default logprobs
- **WHEN** one rank reports raw generated logprobs where policy logprobs are
  required
- **THEN** strict merge fails before top-level artifacts are published

### Requirement: Likelihood trace failure accounting
Missing, duplicate, shifted, token-mismatched, positive, NaN, or infinite
backend likelihood evidence MUST fail the affected scored run. This terminal
integrity rule applies when a backend claims a generated token trace but its
evidence is corrupt or incomplete; it does not replace parser-level partial
salvage for malformed object text. No fallback constant, alternate likelihood
channel, or partial benchmark-looking artifact set may be published.

#### Scenario: Raw replay missing stop token
- **WHEN** raw tracing is enabled but replay omits the generated stop token
- **THEN** the run fails with row and generated-step diagnostics

## MODIFIED Requirements

### Requirement: Selected-token score formula
For V1 scored inference, `pred[*].score` SHALL equal
`exp(sum(selected_token_logprobs) / n_selected)`. Selected token logprobs MUST
be finite natural-log policy probabilities. A parser-salvaged prediction with
no complete selected-token span or the wrong selected-token count MUST be
excluded from scored output and recorded diagnostically. By contrast, missing,
positive, non-finite, shifted, duplicate, or token-mismatched values inside a
backend-claimed token trace constitute backend evidence corruption and MUST
fail the scored run before canonical scored artifacts are published.

#### Scenario: Deterministic score
- **WHEN** selected token logprobs are `[log(0.2), log(0.2), log(0.2)]`
- **THEN** the prediction score equals `0.2` within numeric tolerance

#### Scenario: Empty selected set
- **WHEN** malformed generated object text yields no complete eight-token score
  span while another object remains valid
- **THEN** only the unscoreable prediction is excluded and parser diagnostics
  record the drop

#### Scenario: Non-finite logprob
- **WHEN** a generated non-pad token in the backend-claimed trace has NaN or
  infinite policy likelihood
- **THEN** the scored run fails before canonical scored artifacts are published

### Requirement: Scored artifact row contract
`gt_vs_pred_scored.jsonl` SHALL preserve exactly one scored row per raw row for
a successfully completed run. The row keeps identical image identity, image
dimensions, GT payload, row order, and record index. Its `pred` list SHALL
contain only parser-salvaged predictions with complete finite comparable score
evidence. Rows with no parser-salvaged scoreable predictions SHALL remain
present with `pred: []`. Backend likelihood-integrity failure is terminal and
MUST NOT be converted into a successful empty scored row.

#### Scenario: Trace-missing row
- **WHEN** a raw row contains only malformed or incomplete object spans while
  the backend trace itself is valid
- **THEN** the scored row remains present with the same GT and image metadata
  and an empty `pred` list

#### Scenario: Backend trace evidence is corrupt
- **WHEN** a claimed generated-token trace is missing or token-mismatched
- **THEN** the run fails and no canonical completed scored artifact set is
  published

#### Scenario: Row-count parity
- **WHEN** a successful raw and scored artifact pair is compared
- **THEN** the files have the same row count and row identity order
