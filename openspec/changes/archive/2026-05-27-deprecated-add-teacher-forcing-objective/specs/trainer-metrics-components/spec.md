# trainer-metrics-components Delta

## ADDED Requirements

### Requirement: Teacher-forcing metrics use new objective and parse namespaces

Training and inference metrics for the new teacher-forcing objective SHALL use
stable namespaces that do not preserve old recursive-detection aliases.

Normative behavior:

- objective metrics MUST be emitted under `teacher_forcing/...`;
- compact-full parse metrics MUST be emitted under
  `infer/parse/compact_full/...`;
- new training runs MUST NOT emit `recursive_detection_ce/*`,
  `loss/recursive_detection_ce`, `trie_support/*`, `trie_balance/*`, or
  `et_rmp/*` aliases;
- minimum stable objective metric keys MUST include:
  `teacher_forcing/loss/total`,
  `teacher_forcing/loss/token_type_mass`,
  `teacher_forcing/loss/conditional_valid_set_likelihood`,
  `teacher_forcing/loss/within_valid_coverage`,
  `teacher_forcing/valid_set/mass`,
  `teacher_forcing/coverage/kl`,
  `teacher_forcing/continuation/eos_margin`,
  `teacher_forcing/branch_coherence/rate`,
  `teacher_forcing/residual_set/remaining_count`,
  and `teacher_forcing/permutation_probe/nll_std` when the corresponding
  diagnostic is computed;
- compact-full parse diagnostics MUST count strict parser errors under
  `infer/parse/compact_full/error/<code>`;
- permutation/residual-set reports MUST include `rollin_seed` or
  `rollin_epoch` fields for each evaluated fixed roll-in;
- required objective diagnostic families include loss, type mass, valid-set
  mass, coverage, continuation, coordinate-onset ambiguity, mixed-role
  ambiguity, target-side branch/residual-set coherence, decode-time object
  coherence when generation artifacts exist, permutation/residual-set probes,
  and builder rejection counters when available.

#### Scenario: New training logs do not carry recursive metric aliases

- **WHEN** a new teacher-forcing training step logs objective diagnostics
- **THEN** metrics use `teacher_forcing/...`
- **AND** old recursive-detection aliases are absent.

#### Scenario: Parser diagnostics use compact-full namespace

- **WHEN** strict compact-full parsing records a malformed generation
- **THEN** parse diagnostics use `infer/parse/compact_full/...`
- **AND** the artifact records the exact parser mode and stable error code.

#### Scenario: Permutation probe records stable leaf keys

- **WHEN** a permutation probe report is emitted
- **THEN** it includes `teacher_forcing/permutation_probe/nll_std`
- **AND** each evaluated roll-in records `rollin_seed` or `rollin_epoch`.
