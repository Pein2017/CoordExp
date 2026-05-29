## ADDED Requirements

### Requirement: The study shall be fixed-checkpoint and fixed-decode
The system SHALL provide a manifest-driven duplication-collapse analysis study
that operates only on existing checkpoints and inference/evaluation artifacts.

Normative behavior:

- the study MUST consume only existing checkpoints,
- the study MUST discover all locally available Stage-1 and Stage-2
  checkpoints that are compatible with the current pipeline,
- the authoritative checkpoint cohort MUST include only entries whose
  checkpoint alias, directory name, or paired artifact name contains
  `merged`,
- the study MUST NOT launch training, mutate checkpoints, or depend on new
  optimization runs,
- the authoritative rollout backend MUST be HuggingFace,
- the authoritative decode contract MUST use:
  - `temperature = 0.0`,
  - `top_p = 0.9`,
  - `repetition_penalty = 1.05`,
  - `max_new_tokens = 3084`,
  - `seed = 42`,
- the baseline decode contract MUST be reproduced before any secondary decode
  diagnostic is interpreted,
- the authoritative workflow MUST NOT include temperature sweeps or
  repetition-penalty sweeps,
- `top_k` or alternate `top_p` settings MAY be used only as secondary
  diagnostics after baseline reproduction and MUST NOT become the primary study
  axis,
- the resolved manifest MUST record checkpoint aliases, checkpoint paths,
  source artifact roots, prompt controls, image-root provenance, and the fixed
  decode settings before any probe executes,
- the checkpoint inventory MUST preserve, for each discovered `merged` family:
  - stage,
  - family label,
  - `coord_soft_ce_w1` state when it can be inferred,
  - parent-checkpoint provenance when it is evident from config,
  - whether a paired infer artifact is already available,
  - and the best current probe surface handle,
- the inventory output MUST split the discovered cohort into:
  - ready-to-probe families with usable infer or eval artifact surfaces,
  - and fresh-inference-needed families that exist on disk but do not yet have
    a matched probe-ready artifact package.

#### Scenario: Manifest records a fixed HF decode contract
- **WHEN** the user launches the duplication-collapse analysis study
- **THEN** the study writes a resolved manifest that records the selected
  checkpoint aliases and paths, the artifact roots, the prompt variant, the
  object field order, the HuggingFace backend, the fixed low-temperature decode
  settings, `top_p = 0.9`, `repetition_penalty = 1.05`,
  `max_new_tokens = 3084`, and `seed = 42`
- **AND** the checkpoint inventory records per-family provenance fields such as
  `coord_soft_ce_w1` state, parent checkpoint when known, infer-artifact
  readiness, and best current probe surface
- **AND** the study does not materialize a temperature or repetition-penalty
  sweep matrix.

#### Scenario: Existing checkpoints are analyzed without training
- **WHEN** the study executes against Stage-1 or Stage-2 checkpoints
- **THEN** it reuses existing checkpoints and existing or newly generated
  inference-only artifacts
- **AND** it does not start any training job or alter checkpoint weights.

### Requirement: Historical duplication cases shall drive case selection
The study SHALL prioritize samples with clear duplication behavior from
historical rollout logs and offline inference/evaluation artifacts.

Normative behavior:

- canonical case selection MUST read all locally available Stage-1 and
  Stage-2 checkpoint or artifact roots that are compatible with the current
  pipeline,
- canonical case selection MUST restrict that search to artifact or checkpoint
  names containing `merged`,
- the initial case miner MUST bias toward recall over precision,
- the study MUST preserve deterministic case-selection provenance, including
  the source artifact path, checkpoint alias, sample identifier, and selection
  reason,
- the study MUST support selection from existing artifact fields such as:
  - duplicate-burst telemetry,
  - `gt_vs_pred.jsonl`,
  - `pred_token_trace.jsonl`,
  - per-image evaluation outputs,
  - and monitor-dump duplication summaries when available,
- the study SHOULD prioritize cases whose rollout shows clear same-desc
  duplication onset rather than only broad over-generation.

#### Scenario: Selected case preserves duplication provenance
- **WHEN** the study selects a bad sample from a historical rollout artifact
- **THEN** the case manifest records the checkpoint alias, source artifact root,
  sample id or line index, image id when available, and why the sample was
  selected as a duplication case
- **AND** downstream probes can recover the same sample deterministically.

### Requirement: The study shall preserve causal comparisons across training-signal regimes
The study SHALL explicitly compare available checkpoint families so the final
diagnosis can explain why duplication is absent in some regimes and emerges
after soft coordinate supervision.

Normative behavior:

- when the corresponding checkpoints are available, the study MUST preserve
  family labels or equivalent provenance for:
  - clean pure-CE Stage-1 references when they are locally available,
  - otherwise CE-like or `coord_soft_ce_w1`-disabled Stage-1 continuation
    branches,
  - and Stage-1 checkpoints with soft coordinate supervision such as soft-CE
    or W1,
- checkpoints that do not support the expanded coord-token space used by the
  current training and inference pipeline MUST NOT be treated as formal family
  comparisons for this requirement,
- family-comparison reports MUST align those checkpoints on the same decode
  contract, prompt controls, field order, case-selection rules, and onset-level
  probe outputs whenever that alignment is feasible,
- the final diagnosis MUST explicitly distinguish at least two different
  reasons duplication may be absent:
  - the checkpoint retained sufficiently sharp coord discrimination to avoid
    the local copy basin,
  - or the checkpoint avoids duplication only by drifting into some other
    rollout failure mode,
- the final diagnosis MUST treat attention to prior coord tokens as an onset
  symptom and MUST additionally analyze which training-signal differences made
  that shortcut available, cheap, or dominant,
- the final diagnosis MUST treat a coordinate basin or weak local escape
  barrier as a first-class candidate mechanism, especially at `coord_x1` and
  `coord_y1`,
- if the comparison uses a CE-like or `coord_soft_ce_w1`-disabled continuation
  rather than a clean pure-CE baseline, the report MUST label that family as a
  proxy comparison and MUST preserve its parent-checkpoint provenance,
- when an expected family is unavailable in the local checkpoint inventory, the
  report MUST mark that family as missing rather than silently omitting the
  comparison.

#### Scenario: Family comparison explains why soft supervision introduces a new basin
- **WHEN** the study has access to either a clean pure-CE reference or a
  CE-like soft-disabled continuation, and a soft-coordinate-supervised Stage-1
  checkpoint that are compatible with the current expanded coord-token pipeline
- **THEN** the final report compares their onset-level rollout and re-forward
  evidence under a matched decode contract
- **AND** it records whether the CE-side family is a clean baseline or a proxy
  continuation reference
- **AND** it explicitly states whether the soft-supervised family differs
  because it preserves more local coord ambiguity, a stronger same-prefix
  copy-previous shortcut, or some mixed mechanism relative to the other
  families.

### Requirement: The study shall support matched crowding and class-prior cohort analysis
The study SHALL test whether duplicated cases remain separable from healthy
same-desc controls after conditioning on local same-class crowding.

Normative behavior:

- the study MUST support a follow-up cohort-analysis layer that computes, for
  duplicated and healthy same-desc cases when GT is available:
  - same-class GT count,
  - same-class overlap summaries such as maximum IoU,
  - and nearest-neighbor or local-cluster distance summaries,
- the cohort-analysis layer MUST preserve class labels or equivalent
  same-desc family labels when available from existing artifacts,
- the study MUST support duplicate-versus-control comparisons within matched or
  stratified crowding regimes rather than only raw aggregate counts,
- the report MUST distinguish:
  - crowding-correlated duplicate behavior,
  - class- or scene-family enrichment that persists after conditioning on
    crowding,
  - and insufficient evidence,
- the report MUST NOT treat annotation noise or crowded scenes as a sufficient
  explanation unless the matched analysis supports that claim.

#### Scenario: Crowding-matched analysis preserves duplicate-versus-control separation
- **WHEN** the study runs follow-up cohort analysis on duplicated and healthy
  same-desc cases
- **THEN** it emits machine-readable crowding summaries keyed by case id
- **AND** it records whether duplicated cases remain separable from controls
  after matching or stratifying on same-class crowding metrics.

### Requirement: The study shall capture the exact duplication-onset step
For each selected failure case, the study SHALL identify and probe the exact
decoding step where duplication first emerges.

Normative behavior:

- the study MUST define a deterministic duplicate-family detector that favors
  recall over precision,
- the duplicate-family detector MUST treat an emitted object as a duplicate
  candidate when:
  - the description repeats an earlier object in the same burst,
  - and the geometry either overlaps strongly with that earlier object or shows
    similar bbox size with narrow local drift around the same object cluster,
- the object-level onset MUST be the earliest emitted object that satisfies the
  duplicate-family detector,
- the token-level onset MUST be the first description token of the onset object,
- the probe output MUST record an onset field-phase marker chosen from:
  - `continue_or_open`
  - `desc`
  - `coord_x1`
  - `coord_y1`
  - `coord_x2`
  - `coord_y2`
  - `close`,
- for that onset step, the study MUST preserve:
  - the preceding decoded context,
  - a fixed onset window around the emitted token sequence,
  - raw forward logits at the onset step,
  - decode-processed token competition at the onset step,
  - attention probabilities across layers and heads,
  - hidden-state or residual summaries needed to analyze inter-layer signal
    propagation,
  - and available cross-step state needed to analyze signal accumulation,
- the study MUST also capture at least one pre-onset step and one post-onset
  step so the report can compare the duplication-trigger step against the
  immediately surrounding state,
- the study MUST preserve per-slot summaries that allow `coord_x1` /
  `coord_y1` escape from the previous/local neighborhood to be analyzed as the
  primary onset-local mechanism surface,
- the study MUST preserve enough layerwise detail to test whether visual-token
  access rises in late-middle layers but is overwritten by recent generated
  history or prior coord spans in the final decision layers,
- when full raw tensors are too large to keep as default outputs, the study MAY
  store summarized projections, but it MUST preserve enough detail to attribute
  the onset to specific layers, heads, or token competitions.

#### Scenario: First duplicated object onset is captured with internal state
- **WHEN** a selected rollout begins repeating the same object description and
  nearby geometry
- **THEN** the study records the exact onset step for that first repeated
  emission
- **AND** it preserves the onset object index, onset token span, onset
  field-phase marker, the surrounding decoded prefix, onset-step raw logits and
  decode-processed token competition, per-layer or per-head attention
  summaries, hidden-state or residual summaries, and the immediately preceding
  and following steps for comparison.

### Requirement: The study shall use rollout-first surgical probing for Qwen3-VL
The study SHALL reproduce duplication in rollout before performing a matched
re-forward surgery probe on the emitted prefix and onset window.

Normative behavior:

- each deep-probed case MUST first reproduce the failure under the
  authoritative baseline decode contract,
- each deep-probed case MUST then perform a deterministic re-forward or
  equivalent matched probe pass on the emitted prefix and onset window,
- the mandatory minimum probe surface for each deep-probed case MUST include:
  - LLM-side raw logits and decode-processed scores,
  - LLM-side per-layer or per-head attention summaries,
  - LLM-side hidden-state or residual summaries,
  - and LLM-to-visual-token attention summaries,
- the probe surface MUST preserve separable summaries for:
  - visual-token groups,
  - prior description-token groups when available,
  - prior coord-token groups when available,
  - and recent generated-history groups,
- when a concrete rollout continuation exists for the same onset-local prefix,
  the case bundle MUST preserve a `predicted_object` versus
  `exact_duplicate` comparison and SHOULD treat it as the primary causal
  ranking probe ahead of `gt_next` versus `exact_duplicate`,
- native vision-tower self-attention or feature summaries SHOULD be captured
  when the backend exposes them cleanly,
- when native vision-tower internals are unavailable, the case bundle MUST
  record that missingness explicitly,
- the study MAY inspect local upstream dependency sources when probe hook
  placement depends on upstream implementation details, especially:
  - `transformers` under `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`
  - `ms-swift` under `/data/ms-swift/swift/`,
- when upstream-local behavior informs a probe implementation, the manifest or
  case bundle MUST record that upstream dependency provenance,
- per-case deep-probe artifacts MUST be machine-readable and keyed by case id.

#### Scenario: Deep-probe case records both rollout and surgery outputs
- **WHEN** the study deep-probes a duplicated case
- **THEN** it stores the rollout reproduction metadata and a matched re-forward
  probe artifact for the same onset window
- **AND** the case bundle records which LLM-side, vision-facing, and native
  vision-tower probe surfaces were captured or unavailable
- **AND** it records any upstream-local dependency roots that were consulted for
  probe hook placement or instrumentation behavior.

### Requirement: The study shall support rollout perturbation and coord-split escape probes
The study SHALL support no-retraining counterfactuals that perturb the immediate
pre-collapse context and test whether rollout escapes the duplicate basin.

Normative behavior:

- for anchored same-desc collapse cases, the study MUST support at least one
  rollout-perturbation path that edits the immediate pre-collapse prefix under
  the same fixed decode contract,
- the perturbation set SHOULD include, when feasible:
  - dropping the immediate same-desc source object,
  - replacing that source object with a same-desc alternative target such as
    `gt_next`,
  - interpolating the source-object geometry toward that alternative,
  - and geometry-preserving desc-fixed jitter variants,
- when same-prefix candidate scoring is available, the study MUST support
  coord-split probes that compare:
  - `predicted_object` versus `exact_duplicate` when rollout produced a
    concrete continuation,
  - `x1/y1`-oracle alternatives versus `exact_duplicate` when a same-desc
    alternative target is available,
  - and `x2/y2`-oracle alternatives versus `exact_duplicate` when feasible,
- perturbation and coord-split outputs MUST record whether the model:
  - remained in the same duplicate basin,
  - escaped to a non-duplicate same-desc continuation,
  - escaped via semantic drift,
  - or collapsed into early stop / undercount,
- the report MUST treat these perturbation and coord-split probes as stronger
  evidence for local escape-barrier claims than attention summaries alone.

#### Scenario: Prefix perturbation tests whether the local duplicate basin is escapable
- **WHEN** the study edits the immediate pre-collapse prefix of an anchored
  same-desc duplication case and reruns rollout under the same fixed decode
  contract
- **THEN** it records the perturbation type, the resulting rollout branch, and
  whether the model escaped the duplicate basin
- **AND** when same-desc alternative targets are available it records
  coord-split oracle comparisons that isolate `x1/y1` versus `x2/y2`
  contributions.

### Requirement: The study shall test late-layer overwrite and re-grounding interventions
The study SHALL support targeted inference-time intervention probes that test
whether duplication is driven by late-layer overwrite of visual grounding by
recent text or coord history.

Normative behavior:

- the study MUST support intervention probes that do not modify checkpoint
  weights,
- when the relevant probe surface is available, the intervention set MUST
  support at least:
  - attenuation of prior-object history in late layers,
  - positive bias toward visual-token groups in late layers,
  - and phase-specific interventions restricted to coord emission steps,
- the study SHOULD compare "boost vision" against "suppress prior-object
  history" rather than assuming one intervention family is sufficient,
- intervention outputs MUST preserve both:
  - internal signal shifts such as layerwise attention mass changes,
  - and behavioral shifts such as candidate ranking, previous-box copy mass,
    and rollout escape or failure persistence,
- the study MUST NOT promote late history-overwrite to a primary-cause claim
  when healthy controls show similar overwrite but stronger early coord escape
  unless controlled interventions produce stronger contrary evidence,
- the study MUST NOT treat an intervention as successful only because
  duplication disappears if the rollout simultaneously collapses into obvious
  undercounting or semantic drift.

#### Scenario: Late-layer history attenuation is compared against visual biasing
- **WHEN** the study runs an intervention probe on a duplicated case
- **THEN** it can compare a late-layer visual-token bias intervention against a
  late-layer attenuation of prior-object history or prior coord spans
- **AND** it records whether either intervention changes final-layer group
  masses, candidate score margins, previous-box copy mass, or the rollout
  trajectory itself.

### Requirement: The study shall compare failure traces against controlled alternatives
The study SHALL not interpret a duplicated rollout in isolation.

Normative behavior:

- each failure case MUST be paired with one or more controlled comparison
  traces when available,
- whenever the onset case supports a same-desc control, the study MUST compare
  `gt_next` versus `exact_duplicate` under the same prefix,
- when same-desc controls are unavailable, the study MUST fall back to at least
  one of:
  - a non-duplicated trajectory for the same checkpoint,
  - a GT-conditioned or oracle prefix continuation,
  - or a nearby-jitter duplicate candidate,
- the study MUST preserve which comparison type was used for each case,
- same-desc candidate comparisons MUST be supported so coordinate scoring can be
  analyzed without semantic-token confounds,
- when coordinate logits are available, the study MUST record both expectation-
  style and argmax-style coordinate summaries,
- the report MUST preserve divergence points between the failure trace and the
  control trace rather than only reporting final aggregate differences.

#### Scenario: Same-desc control isolates coordinate behavior
- **WHEN** the prefix object and the GT-next object share the same description
- **THEN** the study compares `gt_next` against an exact-duplicate candidate
  under the same prefix
- **AND** it records coordinate-side differences such as target-bin probability,
  previous-box copy mass, entropy or sharpness, and score margin between those
  candidates
- **AND** it records expectation-versus-argmax differences for the same
  coordinate slots when logits are available.

### Requirement: The study shall report mechanism-level signals, not only surface metrics
The final study outputs SHALL prioritize mechanism-level diagnosis over
surface-only detection summaries.

Normative behavior:

- the final report MUST preserve separate evidence for:
  - continuation or stop calibration,
  - coordinate-distribution shape,
  - attention concentration,
  - late-layer visual-grounding retention or overwrite,
  - token-competition collapse,
  - inter-layer signal propagation,
  - and cross-step state accumulation,
- the report MUST identify the earliest observable precursor signals of
  duplication collapse for each probed case or checkpoint family,
- the report MUST distinguish:
  - hero-case findings,
  - checkpoint-cohort findings,
  - and multi-family findings,
- the report MUST NOT promote a mechanism claim from a single memorable case to
  a checkpoint-family conclusion without additional cohort support,
- when sample counts are sufficient, the report SHOULD include simple
  distributional or bootstrap-style uncertainty summaries for checkpoint-family
  effect patterns,
- the report MUST distinguish rollout evidence from teacher-forced or
  counterfactual evidence,
- the report MUST NOT present FN injection as the primary root cause unless a
  direct probe contradicts the current out-of-scope assumption,
- the report MUST NOT treat heuristic duplicate suppression or decode-policy
  mitigation as the primary deliverable of this study,
- the report MUST frame conclusions as observational mechanism diagnosis unless
  stronger intervention evidence is explicitly recorded,
- the report MUST state when evidence is mixed or insufficient rather than
  forcing one single causal explanation.

#### Scenario: Final report withholds surface-only explanation
- **WHEN** the study writes its final diagnosis for a duplicated checkpoint
- **THEN** it reports the internal signals observed at duplication onset,
  including continuation, coordinate, attention, late-layer overwrite, and
  token-competition evidence
- **AND** it does not reduce the explanation to only AP, duplicate counts, or
  other surface-level metrics.

### Requirement: The study shall support an isolated research workspace
The mechanism probe workflow SHALL support an isolated research workspace for
custom probing instrumentation and large intermediate artifacts.

Normative behavior:

- the study MUST allow a separate root-level workspace for deep-probe scripts,
  notebooks, manifests, and outputs,
- the workspace MUST remain checkpoint-read-only and analysis-only,
- the workspace SHOULD be the default home for invasive surgery-like probe
  helpers and large per-case bundles,
- the final canonical report MUST still point back to the authoritative source
  checkpoints and artifact roots in the main repository.

#### Scenario: Deep probe runs in an isolated workspace
- **WHEN** the user enables the deep mechanism probe path
- **THEN** the study can write custom instrumentation and artifacts under a
  separate root-level research workspace
- **AND** the resulting manifests still reference the authoritative checkpoint
  and artifact paths from the main repository.
