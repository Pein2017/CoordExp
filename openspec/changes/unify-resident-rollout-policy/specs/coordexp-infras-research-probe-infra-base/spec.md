## ADDED Requirements

### Requirement: Resident research continuations preserve sampled policy evidence
The resident research continuation interface SHALL support existing greedy selection and opt-in seeded full-support temperature-one selection. When paired traces are requested, it SHALL return the actual emitted token IDs with separate pre-normalization and normalized-policy selected-action log probabilities from the same generation and snapshot. It MUST preserve actual EOS and valid PAD actions, retain exact request and action association through scheduler reordering, and reject unsupported policy settings or incomplete, duplicate, mismatched, or nonfinite evidence. Trace storage SHALL scale with emitted action count plus one transient active logits batch, without an additional model forward.

#### Scenario: Median-normalized seeded sample
- **WHEN** a caller requests seeded full-support sampling with median normalization and paired traces
- **THEN** every emitted action including EOS has both finite likelihood channels associated with its literal token and request
- **AND** the raw channel precedes normalization, even for a non-coordinate action whose probability denominator changes

#### Scenario: Mixed budgets and early EOS
- **WHEN** resident requests have different budgets, one emits EOS early, and another emits PAD before EOS
- **THEN** each request retains its own ordered action trace, valid PAD is preserved, and only verified scheduler work beyond the final emitted boundary is excluded

#### Scenario: Tracing disabled
- **WHEN** an existing caller uses the default greedy interface without paired traces
- **THEN** its continuation contract remains unchanged and no paired-trace buffers accumulate

### Requirement: Resident refresh acknowledges the current training snapshot
The resident research engine SHALL acknowledge a new snapshot only after selected trainables, derived DoRA and normalization factors, and relevant cache state have been updated. The next generation MUST use the acknowledged snapshot identity. Refresh MUST reject active paired traces, and failed refresh or cache invalidation MUST NOT publish the new snapshot. HF replay and resident acquisition SHALL share the declared coordinate-normalization arithmetic while retaining derivatives through current HF factors.

#### Scenario: HF update followed by resident acquisition
- **WHEN** HF completes an optimizer update and the caller synchronously refreshes the resident engine
- **THEN** the acknowledgement binds the updated snapshot and the next acquisition uses its updated weights and factors without restarting the engine

#### Scenario: Refresh fails
- **WHEN** a weight, factor, or cache update fails
- **THEN** the requested new identity is not usable for generation and the failure is observable to the caller

#### Scenario: Replay differentiates current normalization
- **WHEN** HF replays an acquired literal action sequence under the declared median policy
- **THEN** its selected-action probabilities retain gradients through both logits and current normalization factors, with numerical backend differences reported separately from identity failures
