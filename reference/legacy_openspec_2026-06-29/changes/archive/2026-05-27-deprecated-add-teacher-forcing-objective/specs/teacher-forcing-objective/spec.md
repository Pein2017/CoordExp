# teacher-forcing-objective Specification

## ADDED Requirements

### Requirement: Teacher-forcing objective SHALL be the active unified objective surface

The system SHALL expose the new unified teacher-forcing objective through
`objective.id: teacher_forcing`.

Normative behavior:

- active profiles MUST be `hard_sft`, `pure_valid_set_marginal`, and
  `hybrid_valid_set_marginal`;
- active configs MUST NOT use `recursive_detection_ce`,
  `prefix_rollin_et_rmp_ce`, `random_permutation_et_rmp_ce`, ET-RMP aliases, or
  support/balance names as runnable training objective identities;
- hard SFT MUST be represented as the singleton-valid-set profile inside this
  objective surface;
- the old ET-RMP comparator behavior MAY be represented structurally as
  `hybrid_valid_set_marginal` with
  `objective.modules.within_valid_coverage.coverage_strength: 1.0`;
- profile names MUST NOT encode hyperparameters, roll-in policy, or alpha-like
  shorthand.

#### Scenario: Old objective id is rejected

- **WHEN** a training config sets `objective.id: recursive_detection_ce`
- **THEN** config validation fails fast
- **AND** the error recommends `objective.id: teacher_forcing`
- **AND** no legacy recursive-detection training wrapper is instantiated.

#### Scenario: Hard SFT uses singleton valid sets

- **GIVEN** `objective.id: teacher_forcing`
- **AND** `objective.profile: hard_sft`
- **WHEN** target IR profile transforms run
- **THEN** each trainable likelihood atom uses a singleton
  `valid_token_ids` set containing the selected teacher token
- **AND** latent legal alternatives may remain available only for diagnostics.

### Requirement: Marginal scope SHALL be sampled-path next-token

The objective SHALL optimize next-token valid-set likelihood at the current
teacher-forced roll-in prefix.

Normative behavior:

- the v1 marginal scope MUST be recorded as `sampled_path_next_token`;
- `valid_token_ids` MUST be computed for the current teacher-forced prefix only;
- the objective MUST NOT claim or compute a full autoregressive recursive
  subtree marginal over alternate hidden-state prefixes in v1;
- configs, resolved manifests, and objective reports SHOULD record the marginal
  scope when they record objective identity.

#### Scenario: Full subtree marginal is not claimed

- **GIVEN** `objective.id: teacher_forcing`
- **WHEN** a training run records objective metadata
- **THEN** the metadata identifies `marginal_scope: sampled_path_next_token`
- **AND** it does not describe the loss as full recursive subtree DP.

### Requirement: TeacherForcingTargetIR SHALL be the shared sidecar contract

The system SHALL build one canonical teacher-forcing sidecar under the key
`teacher_forcing_target_ir`.

Normative behavior:

- the sidecar MUST contain a `TeacherForcingTargetIR`;
- the IR MUST contain `SupervisionAtom` records for causal next-token
  supervision events;
- every trainable atom MUST record `batch_index`, `logit_position`,
  `target_position`, `allowed_token_roles`, `selected_token_role`,
  `valid_token_ids`, `selected_token_id`, `latent_valid_token_ids`, `loss_tags`,
  and `loss_weight`;
- v1 MUST enforce `target_position = logit_position + 1`;
- `selected_token_id` MUST match `input_ids[batch_index, target_position]`;
- `valid_token_ids` MUST be deduplicated token ids;
- non-diagnostic atoms MUST have nonempty `valid_token_ids`;
- loss modules MUST NOT derive ambiguity semantics by reconstructing object
  tries at runtime.

#### Scenario: Causal position mismatch fails fast

- **WHEN** an atom has `target_position != logit_position + 1`
- **THEN** the objective runner fails before loss computation
- **AND** the atom is not silently skipped or shifted.

#### Scenario: Selected token mismatch fails fast

- **WHEN** an atom's `selected_token_id` differs from the live
  `input_ids[batch_index, target_position]`
- **THEN** the objective runner fails before loss computation.

### Requirement: Teacher-forcing sidecars SHALL survive collation and be stripped before forward

The pipeline SHALL treat `teacher_forcing_target_ir` as a runner-owned sidecar,
not as a model-forward argument.

Normative behavior:

- implementation MUST define and reuse a single canonical
  `TEACHER_FORCING_TARGET_IR_KEY` for the sidecar key;
- dataset items, latest-detection sidecar filtering, batch extras, collator
  enrichment, model-input bundling, trainer bridges, and objective runners MUST
  preserve the sidecar until loss computation;
- trainer/model-input code MUST strip `teacher_forcing_target_ir` before calling
  the upstream model forward;
- stripping the teacher-forcing sidecar MUST NOT strip upstream attention or
  flash-attention metadata that legitimately belongs in the model call, such as
  `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, or `max_length_k`.

#### Scenario: Target IR sidecar is not forwarded to the model

- **GIVEN** a batch containing `teacher_forcing_target_ir`
- **WHEN** the trainer prepares model-forward kwargs
- **THEN** `teacher_forcing_target_ir` is absent from model-forward kwargs
- **AND** the objective runner still receives the corresponding target IR.

### Requirement: Runner logits and attention alignment SHALL be fail-fast

The objective runner SHALL validate that explicit atom positions refer to live
unsliced model logits before objective math runs.

Normative behavior:

- v1 MUST consume full unsliced rank-3 logits with shape
  `[batch, seq, vocab]`;
- v1 MUST require `logits.ndim == 3`;
- v1 MUST require `logits.shape[:2] == input_ids.shape[:2]`;
- v1 MUST reject or bypass `logits_to_keep` and other sliced-logit paths for
  teacher-forcing objective computation;
- every atom's `batch_index`, `logit_position`, and `target_position` MUST be
  in bounds for the live batch tensors;
- when an `attention_mask` is available, both
  `attention_mask[batch_index, logit_position]` and
  `attention_mask[batch_index, target_position]` MUST indicate live tokens;
- unsupported packed or padding-free training paths MUST fail before loss
  computation unless a later spec defines an exact atom-position mapping for
  them;
- loss modules MUST consume explicit atom positions and MUST NOT shift labels
  internally.

#### Scenario: Sliced logits are rejected

- **WHEN** teacher-forcing loss receives logits whose time dimension does not
  match `input_ids`
- **THEN** the runner fails before objective math
- **AND** no fallback label shift is attempted.

#### Scenario: Rank-2 logits are rejected

- **GIVEN** `input_ids.shape == [1, T]`
- **WHEN** teacher-forcing loss receives `logits.shape == [T, V]`
- **THEN** the runner fails before objective math
- **AND** it does not infer an implicit batch dimension.

#### Scenario: Padding positions are rejected

- **GIVEN** an atom whose causal logit or target token position is masked out
- **WHEN** the runner validates the atom against `attention_mask`
- **THEN** the runner fails before objective math.

### Requirement: Token roles SHALL support singleton and controlled mixed-role atoms

The objective SHALL use token role sets to express which token-type families are
valid before branch commitment.

Normative behavior:

- token roles MUST include `SCHEMA`, `TEXT`, `COORD`, and `STOP`;
- most atoms SHOULD use singleton `allowed_token_roles`;
- Stage-1 marker-delimited `compact_full` v1 MAY emit `{TEXT, SCHEMA}` only for
  description-boundary ambiguity;
- builders MUST reject unsupported mixed role sets such as `{TEXT, COORD}`,
  `{SCHEMA, COORD}`, `{COORD, STOP}`, and `{TEXT, STOP}` unless a later spec
  authorizes a template state that requires them;
- validators MUST require `valid_token_ids` to be a subset of the union of the
  vocabularies for `allowed_token_roles`;
- `STOP` MUST target `<|im_end|>` only.

#### Scenario: Description boundary supports car and carrot

- **GIVEN** compatible remaining descriptions `car` and `carrot`
- **AND** the current prefix is `<|object_ref_start|>car`
- **WHEN** the builder emits the next-token atom
- **THEN** `allowed_token_roles` includes `TEXT` and `SCHEMA`
- **AND** `valid_token_ids` includes both the text continuation token for
  `carrot` and `<|box_start|>` for `car`.

### Requirement: Token-type mass SHALL use full vocabulary probabilities

The token-type mass module SHALL operate on full-vocabulary probabilities.

Normative behavior:

- training MUST compute full-vocabulary softmax before token-type mass;
- wrong-role logits MUST NOT be hard-masked before training loss computation;
- for an atom, the allowed mass MUST be the sum of probability mass over the
  union of vocabularies for `allowed_token_roles`;
- the token-type loss MUST be `-log P(allowed role union)`;
- mixed-role atom diagnostics MUST report allowed-union mass and selected-role
  provenance, but selected-role provenance MUST NOT become a pre-commitment
  loss.

#### Scenario: Mixed-role atom does not penalize a legal alternate role

- **GIVEN** an atom with `allowed_token_roles={TEXT, SCHEMA}`
- **WHEN** the selected teacher token is a schema token
- **THEN** text-role probability mass remains part of the allowed type mass
- **AND** it is not penalized as wrong-type mass.

### Requirement: Conditional valid-set likelihood SHALL normalize inside the allowed role union

The conditional valid-set likelihood module SHALL compute valid-token
likelihood inside the allowed role union.

Normative behavior:

- the module MUST receive conditional probabilities normalized by the allowed
  role union mass;
- the valid-set loss MUST be `-log sum p_bar(v)` for `v in valid_token_ids`;
- singleton valid sets MUST be equivalent to hard CE inside the allowed role
  union;
- ambiguous atoms MUST NOT add selected-child hard CE before branch commitment.

#### Scenario: Ambiguous valid set is support-only when coverage is disabled

- **GIVEN** an ambiguous atom with multiple valid token ids
- **AND** within-valid coverage is disabled
- **WHEN** the valid-set likelihood is computed
- **THEN** the loss is only `-log sum p_bar(valid_token_ids)`
- **AND** no uniform, count-weighted, or selected-token CE term is added.

### Requirement: Within-valid coverage SHALL be optional and explicitly weighted

The within-valid coverage module SHALL regularize the normalized distribution
inside the valid set only when enabled.

Normative behavior:

- `coverage_strength` MUST live under
  `objective.modules.within_valid_coverage.coverage_strength`;
- production coverage-regularized configs MUST set positive
  `coverage_strength` explicitly;
- `coverage_strength` MUST NOT appear as top-level `objective.alpha`, top-level
  `objective.coverage_strength`, or as part of the profile name;
- coverage targets MUST use raw nonnegative `coverage_target_weights` supplied
  by the builder and normalized by the module;
- coverage weights for branch ambiguity MUST represent residual-object branch
  mass aggregated by next token id, independent of token role;
- `pure_valid_set_marginal` MUST disable coverage or use zero coverage strength.

#### Scenario: Coverage strength one structurally matches old comparator behavior

- **GIVEN** `objective.profile: hybrid_valid_set_marginal`
- **AND** `objective.modules.within_valid_coverage.coverage_strength: 1.0`
- **WHEN** coverage is computed for an ambiguous valid set
- **THEN** the objective applies valid-set marginal plus within-valid
  distribution matching
- **AND** the config name still does not use ET-RMP terminology.

### Requirement: Candidate filtering SHALL follow selected trie transitions

The target builder SHALL filter candidate objects according to selected teacher
tokens before constructing subsequent atoms.

Normative behavior:

- if a selected description token continues a description, the next candidate
  set MUST keep only candidates whose description token path continues with that
  token;
- if a selected token is `<|box_start|>`, the next candidate set MUST keep only
  candidates whose description token path is complete at the current prefix;
- if a selected coordinate token is emitted at an ambiguous coordinate slot, the
  next candidate set MUST keep only compatible object branches;
- branch commitment occurs only when the filtered active candidate set is
  singleton;
- object completion removes the selected object from the remaining set only at
  full object-entry completion.

#### Scenario: Repeated car objects remain ambiguous after description end

- **GIVEN** remaining candidates `car A`, `car B`, and `carrot C`
- **AND** the selected token after `<|object_ref_start|>car` is `<|box_start|>`
- **WHEN** the builder filters candidates
- **THEN** active candidates are `car A` and `car B`
- **AND** coordinate-onset ambiguity remains eligible at `x1`.

### Requirement: Compact-full training serialization SHALL be marker-delimited

The `compact_full` training template SHALL use marker-delimited serialization in
new teacher-forcing configs.

Normative behavior:

- the high-level template id remains `compact_full`;
- production configs MUST explicitly set
  `detection_template.serialization_policy: marker_delimited`;
- renderer/parser implementation MUST be policy-aware and MUST NOT globally
  mutate historical `compact_full` newline behavior;
- object entries MUST be delimited by `<|object_ref_start|>`;
- fields MUST be delimited by `<|box_start|>`;
- row-separator newline MUST NOT be a training schema token, diagnostic-only
  formatting token, or allowed target token for new teacher-forcing training;
- examples and docs for exact serialized payloads MUST show adjacent
  marker-delimited entries with no inserted newline.

#### Scenario: New compact-full rendering has no row separator

- **GIVEN** two objects in a teacher-forcing sample
- **WHEN** `compact_full` is rendered with `serialization_policy:
  marker_delimited`
- **THEN** the second object begins immediately with `<|object_ref_start|>`
  after the first object's fourth coordinate token
- **AND** no newline token is inserted between entries.

### Requirement: Description normalization and token-span mapping SHALL be exact

Description rendering and tokenization SHALL be strict enough to preserve exact
target-position supervision.

Normative behavior:

- descriptions MUST be stripped of leading/trailing whitespace;
- empty descriptions MUST be rejected;
- newline, tab, control characters, reserved schema/control/image/stop marker
  text, and coordinate-token text MUST be rejected inside descriptions;
- canonical description token ids MUST be extracted from tokenizing
  `<|object_ref_start|>{desc}<|box_start|>`;
- marker ids MUST appear as exact single-token boundaries;
- description token spans MUST be uniquely extractable and aligned to the full
  training sample;
- mapping failures MUST be hard errors.

#### Scenario: Marker boundary ambiguity fails fast

- **WHEN** the builder cannot uniquely locate `<|object_ref_start|>` and
  `<|box_start|>` token boundaries around a description
- **THEN** target construction fails
- **AND** no approximate span mapping is used.

### Requirement: Roll-in SHALL be deterministic and epoch-varying for training

The default teacher-forcing roll-in policy SHALL be deterministic random
permutation with base seed `17`.

Normative behavior:

- default roll-in policy MUST be `random_permutation`;
- the authored roll-in policy path MUST be
  `objective.target_ir.rollin_policy.name`;
- training roll-in seed MUST derive from base seed `17`, epoch, stable sample
  id, policy name, and policy version;
- dataloader order or worker process state MUST NOT affect roll-in;
- routine validation/probe roll-in MUST use fixed eval seed/epoch for
  checkpoint-comparable curves;
- permutation-sensitivity analysis MAY use multiple fixed eval roll-in
  epochs/seeds and MUST report them.

#### Scenario: Dataloader order does not change roll-in

- **GIVEN** the same sample, epoch, base seed, policy name, and policy version
- **WHEN** the sample is loaded by different dataloader workers or ranks
- **THEN** the selected roll-in path is identical.

### Requirement: Length overflow and empty detection lists SHALL be explicit

Teacher-forcing target construction SHALL reject unsupported samples before
target IR construction.

Normative behavior:

- v1 MUST NOT partially truncate supervised object entries;
- v1 MUST NOT subsample objects to fit max length;
- overlength samples MUST be rejected or dropped before target construction with
  counters and thresholds;
- mapping/alignment failures MUST always be hard errors;
- Stage-1 COCO v1 MUST drop samples with missing detection lists;
- Stage-1 COCO v1 MUST drop explicit empty detection lists with counters;
- STOP-only empty-object samples are reserved for future intentional
  negative-image datasets.

#### Scenario: Explicit empty COCO object list is dropped

- **GIVEN** a Stage-1 COCO teacher-forcing sample with an explicit empty object
  list
- **WHEN** dataset validation runs for v1
- **THEN** the sample is dropped with an empty-list counter
- **AND** no STOP-only training atom is built.

### Requirement: Objective diagnostics SHALL expose teacher-forcing failure modes

The objective SHALL provide diagnostic families that can be emitted through the
trainer metrics namespace and analysis reports.

Normative behavior:

- diagnostics MUST include token-type mass, valid-set mass, within-valid
  coverage, continuation/EOS margin, coordinate-onset ambiguity, mixed-role
  ambiguity, and builder rejection counters;
- target-side diagnostics MUST expose residual-set and branch-coherence
  information needed to detect object-coordinate mixing;
- decode-time or analysis diagnostics MUST include object coherence, duplicate,
  missed-object, and malformed-sequence rates when generation artifacts are
  available;
- permutation/residual-set probes MUST be supported as a separate targeted
  analysis surface and MUST report the fixed eval roll-in seeds or epochs used;
- trainer metric emission and compact-full parser metric namespaces are owned by
  `trainer-metrics-components`.

#### Scenario: Permutation probe records fixed roll-in seeds

- **WHEN** a permutation-sensitivity report evaluates multiple valid roll-ins
- **THEN** the report records the fixed roll-in epochs or seeds
- **AND** reports residual-set or emitted-vs-remaining diagnostics separately
  from headline AP/AR.
