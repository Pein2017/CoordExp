# encoded-training-cache Delta

## ADDED Requirements

### Requirement: Encoded cache eligibility respects teacher-forcing target IR and roll-in

Encoded-sample caching SHALL be disabled or keyed safely for teacher-forcing
target IR payloads.

Normative behavior:

- training v1 with epoch-varying random roll-in MUST NOT use a static
  encoded-sample cache that would freeze `input_ids` and
  `teacher_forcing_target_ir`;
- fixed eval/probe teacher-forcing examples MAY use encoded-sample cache only
  when the cache key includes tokenizer fingerprint, chat-template fingerprint,
  compact-full serialization policy, description normalization policy, roll-in
  policy/version/seed/epoch, target IR schema version, and max length;
- cached teacher-forcing payloads MUST include both `input_ids` and
  `teacher_forcing_target_ir`;
- cached payloads MUST be validated on load.

#### Scenario: Epoch-varying roll-in disables static training cache

- **GIVEN** `objective.id: teacher_forcing`
- **AND** training roll-in varies by epoch
- **WHEN** encoded-sample cache eligibility is resolved
- **THEN** v1 training cache reuse is rejected or bypassed
- **AND** the run records an explicit cache-bypass reason.
