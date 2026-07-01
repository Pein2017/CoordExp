## ADDED Requirements

### Requirement: Shared runtime consumes resolved facts instead of broad owner objects

Shared inference runtime modules SHALL keep caller-specific owner introspection
at explicit edge adapters and SHALL pass resolved prompt, decode, backend,
model, parser, and artifact facts into shared behavior.

Normative behavior:

- prompt, backend, parser, artifact, and rollout-dispatch helpers MUST NOT
  depend on arbitrary trainer-private methods such as `_cfg`,
  `_object_ordering`, `_ensure_vllm_engine`, or `_maybe_rollout_offload_context`
  except inside documented migration or edge adapter code;
- offline inference and Stage-2 rollout callers MAY each have adapters that
  translate their existing state into shared-runtime facts;
- shared runtime modules MUST NOT import Stage-2 correction target builders,
  residual boundary adapters, duplicate-control training logic, or DDP
  coordination helpers;
- backend-heavy imports SHOULD be deferred to the backend path that requires
  them so prompt/config/parser/provenance helpers remain import-light.

#### Scenario: Shared prompt helper does not call trainer-private policy methods

- **GIVEN** the shared prompt helper receives resolved prompt policy and visual
  input facts
- **WHEN** it prepares an offline or Stage-2 rollout prompt bundle
- **THEN** it does not call trainer-private methods to rediscover object
  ordering, prompt variant, or detection sequence format
- **AND** caller-specific state translation remains confined to the edge
  adapter.

### Requirement: Stage-2 runtime facade does not own shared decode behavior

Any remaining Stage-2 rollout runtime class SHALL be a trainer-owned facade and
SHALL NOT own prompt rendering, backend lifecycle, decode request conversion,
trace normalization, parser policy, or shared artifact provenance.

Normative behavior:

- if a Stage-2 runtime class remains after migration, it MUST delegate shared
  decode behavior to `src/infer` or to a narrow dedicated backend adapter;
- trainer-owned responsibilities MAY include orchestration, DDP/packing
  coordination, target/loss handoff, and training metric projection;
- docs and tests MUST describe any remaining runtime class as trainer-owned,
  not as the shared inference runtime.

#### Scenario: Runtime class deletion would not remove shared inference policy

- **WHEN** the remaining Stage-2 runtime facade is reviewed after migration
- **THEN** deleting it would not delete shared prompt, decode, backend, parser,
  trace, or artifact policy
- **AND** those policies remain owned by shared-runtime modules or dedicated
  adapters.

### Requirement: Prompt provenance reflects the actual prompt bundle and visual policy

Shared inference provenance SHALL fingerprint the actual prompt bundle, visual
policy, and parity status used for generation rather than a synthetic summary
assembled after the fact.

Normative behavior:

- prompt provenance MUST be derived from the real prompt policy, prompt text or
  tokenized prompt surface, template family, detection sequence format, bbox
  format, object ordering policy, visual metadata, model/tokenizer/processor
  policy inputs, and prompt-token/visual parity status when available;
- Stage-2 eval score sidecars MUST bind to the prompt bundle and decode request
  used to generate the rollout they score;
- synthetic prompt fingerprints that omit real prompt text, template family,
  detection sequence format, visual metadata, or parity status MUST NOT mark
  artifacts as comparable;
- offline-vs-online parity claims MUST require prompt-token and visual parity,
  not only string-hash or summary-hash equality.

#### Scenario: Prompt policy change invalidates Stage-2 eval comparability

- **GIVEN** two Stage-2 eval runs differ in prompt variant, detection sequence
  format, bbox format, template family, or visual metadata policy
- **WHEN** score provenance is written
- **THEN** the prompt provenance fingerprint changes or comparability is marked
  unavailable
- **AND** the artifacts cannot claim prompt parity without verified
  prompt-token and visual metadata parity.
