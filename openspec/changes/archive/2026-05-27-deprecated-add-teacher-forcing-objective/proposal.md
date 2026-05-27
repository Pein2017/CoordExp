## Why

CoordExp needs a single teacher-forcing objective contract that covers Stage-1
compact detection and Stage-2 rollout-aware training without preserving the old
`recursive_detection_ce` / ET-RMP training surface. The current objective names,
target sidecars, token-type gates, parser assumptions, and auxiliary losses are
split across legacy modules and make decoding-aligned set prediction hard to
audit.

## What Changes

- Introduce a new `objective.id: teacher_forcing` training contract centered on
  a shared `TeacherForcingTargetIR` sidecar and reusable objective modules.
- Define typed valid-set marginal likelihood, full-vocabulary token-type mass,
  optional within-valid coverage regularization, and optional continuation
  diagnostics/margin as the canonical objective decomposition.
- Support sampled-path next-token valid-set marginal supervision over
  teacher-forced roll-in prefixes, including coordinate-onset ambiguity and
  mixed text/schema role-set atoms for shared description-prefix cases.
- Require marker-delimited `compact_full` training serialization with no
  row-separator newline.
- Preserve legacy newline compact parsing only for historical inference/eval of
  already-trained checkpoints.
- **BREAKING** Remove active training support for `recursive_detection_ce`,
  `prefix_rollin_et_rmp_ce`, ET-RMP support/balance aliases, old recursive
  sidecars, and old recursive trainer mixins.
- **BREAKING** Remove duplicate unlikelihood, bbox auxiliary losses,
  coordinate-regression/geometry regularizers, W1, and adjacent-repulsion style
  losses from the active teacher-forcing core.
- Keep standard hard SFT as a first-class comparator profile inside the new
  teacher-forcing framework rather than as a separate objective engine.
- Define new metric/artifact namespaces for teacher-forcing objective diagnostics
  and compact-full parse diagnostics.

## Capabilities

### New Capabilities

- `teacher-forcing-objective`: Stable config, target IR, loss semantics,
  compact-full serialization, diagnostics, cache, and migration contract for the
  unified teacher-forcing objective.

### Modified Capabilities

- `stage1-latest-detection-objectives`: Supersede the active
  `recursive_detection_ce` latest compact detection training surface with the
  new teacher-forcing objective and explicit migration failures.
- `teacher-forcing-unified-loss-registry`: Replace old Stage-2-centric CE/gate
  registry assumptions with the new shared module taxonomy and token-role-set
  semantics.
- `teacher-forcing-objective-pipeline`: Remove legacy auxiliary modules from
  live objective pipelines and route active teacher-forcing semantics through
  the new objective modules.
- `encoded-training-cache`: Define cache eligibility for epoch-varying
  teacher-forcing roll-in and fixed eval/probe target IR payloads.
- `stage2-ab-training`: Resolve Stage-2 dynamic packing against explicit
  teacher-forcing atom-position invariants and require fail-fast behavior until
  exact packed-position mapping is implemented.
- `inference-pipeline`: Add strict marker-delimited compact-full parse mode,
  legacy-compatible parse mode, and explicit compact grammar serialization
  policy.
- `trainer-metrics-components`: Reserve the `teacher_forcing/...` and
  `infer/parse/compact_full/...` metric namespaces for new objective and parser
  diagnostics.

## Impact

- Affected config schema: Stage-1 latest compact detection objective config,
  Stage-2 teacher-forcing pipeline/objective config, inference parsing/generation
  config, encoded-sample cache eligibility, and training metric namespaces.
- Affected code areas for implementation planning: `src/training/objectives/`,
  `src/training/teacher_forcing/`, `src/training/supervision/`,
  `src/detection/`, `src/trainers/`, `src/data_collators/`, `src/infer/`,
  `src/config/`, `src/analysis/`, and current Stage-1/Stage-2 tests.
- Affected docs/artifacts: training docs, implementation map, catalog, OpenSpec
  stable specs, run manifests, inference artifacts, and teacher-forcing analysis
  reports.
- No upstream `ms-swift`, Transformers, or Qwen model files are modified by this
  contract.
