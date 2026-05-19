# stage1-latest-detection-objectives Delta

## MODIFIED Requirements

### Requirement: Latest-detection objective authoring is config-first and typed

The latest compact detection objective surface SHALL be authored through the
top-level `objective` section parsed by the latest detection training config.

Normative behavior:

- latest-schema configs MUST use top-level `data`, `prompt`,
  `detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, and
  `validation` sections;
- latest-schema configs MUST NOT use legacy `custom.stage1_set_continuation`,
  `custom.trainer_variant: stage1_set_continuation`, or legacy
  `custom.coord_soft_ce_w1.*` to configure teacher-forcing objectives;
- new active compact-full detection training MUST use
  `objective.id: teacher_forcing`;
- old objective ids and variants, including `recursive_detection_ce`,
  `random_permutation_et_rmp_ce`, `prefix_rollin_et_rmp_ce`, and
  `trie_disabled_full_suffix_ce`, MUST fail fast with migration guidance;
- standard SFT MUST be represented through `objective.id: teacher_forcing` and
  `profile: hard_sft`.

#### Scenario: Old recursive CE objective is rejected

- **WHEN** a latest-schema config authors `objective.id:
  recursive_detection_ce`
- **THEN** config parsing fails fast
- **AND** the error points to `objective.id: teacher_forcing`.

### Requirement: Random-permutation ET-RMP-CE remains historical comparator context only

The previous random-permutation ET-RMP-CE latest compact detection surface SHALL
be treated as historical comparator context after this change.

Normative behavior:

- old checkpoints trained with recursive-detection objective configs MAY be
  evaluated through explicitly historical inference/eval configs;
- new active training configs MUST NOT use the old recursive-detection training
  objective ids, sidecars, or metric aliases;
- structural comparator runs inside the new framework MUST be expressed as
  `coverage_regularized_valid_set_marginal` with explicit
  `objective.modules.within_valid_coverage.coverage_strength: 1.0`, not as
  ET-RMP-named configs.

#### Scenario: Historical checkpoint remains scoreable

- **GIVEN** an already-trained recursive-detection checkpoint
- **WHEN** a legacy inference/eval config scores it
- **THEN** historical compact parser compatibility may be enabled
- **AND** no old recursive-detection training objective is available.

### Requirement: Latest recursive detection runtime rejects unsupported packing and cache paths

The latest compact detection runtime SHALL apply fail-fast runtime checks to the
new teacher-forcing objective rather than the removed recursive objective.

Normative behavior:

- `objective.id: teacher_forcing` MUST reject unsupported packed or
  padding-free paths unless an exact atom-position mapping is implemented;
- training with epoch-varying roll-in MUST bypass static encoded-sample cache;
- fixed eval/probe cache MAY be enabled only when target-IR and roll-in fields
  are included in the cache key;
- runtime errors MUST refer to the teacher-forcing objective and target IR, not
  recursive-detection CE.

#### Scenario: Unsupported packing is rejected for teacher forcing

- **GIVEN** `objective.id: teacher_forcing`
- **AND** a packing mode without exact atom-position support
- **WHEN** training config validation runs
- **THEN** validation fails fast with teacher-forcing migration guidance.

### Requirement: Compact-full token-row adaptation covers geometry and structural rows

Compact-full latest detection SHALL treat trainable token rows as a
teacher-forcing template/tokenizer contract rather than as a recursive-detection
or coord-only adapter contract.

Normative behavior:

- when token-row adaptation is enabled for new teacher-forcing compact-full
  training, the trainable row set MUST include coord-token rows plus active
  schema marker rows required by the marker-delimited template;
- the persisted module name `coord_offset_adapter` MAY remain for checkpoint
  compatibility;
- contract wording MUST describe this surface as token-row adaptation rather
  than coord-only adaptation or recursive-detection adaptation;
- the `coord_geometry` token-row group MUST preserve the expected
  `<|coord_0|>` through `<|coord_999|>` token id range.

#### Scenario: Compact-full token rows include marker schema rows

- **WHEN** compact-full teacher-forcing training enables trainable token rows
- **THEN** the trainable row set includes coord geometry rows
- **AND** it includes `<|object_ref_start|>` and `<|box_start|>`.

### Requirement: EOS and generation-token contracts are explicit

Latest compact teacher-forcing detection SHALL preserve the Qwen chat-template
stop contract across training and generation.

Normative behavior:

- semantic STOP supervision MUST target `<|im_end|>` only;
- text-level terminators such as `<|endoftext|>` or `<|end_of_text|>` MUST NOT
  be used as semantic training STOP targets for this surface;
- HF/Qwen generation MUST use `eos_token_id=id("<|im_end|>")`;
- HF/Qwen generation MUST use `pad_token_id=id("<|endoftext|>")`;
- vLLM inference MUST stop on `"<|im_end|>"` only;
- inference and rollout paths MUST preserve training-time geometry with
  `do_resize=false`;
- inference artifacts MUST record the Qwen chat generation contract when this
  surface is evaluated.

#### Scenario: Teacher-forcing STOP target is the assistant stop marker

- **GIVEN** `objective.id: teacher_forcing`
- **WHEN** training target labels are constructed
- **THEN** semantic STOP supervision targets `<|im_end|>`
- **AND** `<|endoftext|>` is not treated as the semantic STOP label.

### Requirement: Objective-status language separates production baselines, ablations, diagnostics, and retired routes

Objective-status language SHALL treat the teacher-forcing objective as the only
active compact-full detection training surface after this change.

Normative behavior:

- production baselines MUST use `objective.id: teacher_forcing`;
- hard SFT baseline runs MUST use `profile: hard_sft`;
- valid-set marginal runs MUST use `profile: pure_valid_set_marginal`;
- coverage-regularized comparator runs MUST use
  `profile: coverage_regularized_valid_set_marginal` plus explicit
  `objective.modules.within_valid_coverage.coverage_strength`;
- old recursive-detection objectives, sidecars, and metrics MUST be documented
  only as retired training routes or historical inference/eval context.

#### Scenario: Active ablation names are semantic profiles

- **WHEN** active compact-full objective configs are listed for new training
- **THEN** they are grouped by teacher-forcing profiles
- **AND** old ET-RMP, support/balance, or alpha-like names are absent.

## REMOVED Requirements

### Requirement: Random-permutation ET-RMP-CE remains the production baseline/comparator

The old random-permutation ET-RMP-CE training surface is removed from active
latest compact detection training. Historical checkpoints MAY remain scoreable
through explicitly legacy inference/eval configs, and the comparator behavior is
represented structurally through the new teacher-forcing coverage profile.

### Requirement: Prefix-rollin ET-RMP-CE is a compact-full ablation surface

The old prefix-rollin ET-RMP-CE ablation surface is removed from active compact
training. New ablations MUST vary teacher-forcing profile, roll-in policy, and
coverage strength through the new schema rather than old objective ids.

### Requirement: Latest recursive detection coord-soft-target overlays are objective-local

The recursive-detection coordinate soft-target overlay is removed from the
active teacher-forcing core. Any future coordinate-neighborhood ablation MUST be
specified as valid-set or valid-neighborhood marginal semantics, not Gaussian,
Gibbs, IoU/CIoU, W1, or regression-style soft CE.

### Requirement: Recursive detection CE diagnostics expose target mix, support, balance, boundary, EOS, and type-gate health

Recursive-detection CE diagnostic aliases are removed from new training runs.
New diagnostics are emitted through `teacher_forcing/...` and parser diagnostics
through `infer/parse/compact_full/...`.
