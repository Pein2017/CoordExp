# Unified Training Infrastructure Architecture Design

Status: super-power design/spec drafted from approved `grill-me` decisions.
Production implementation has not started.

Date: 2026-05-15

Owner: CoordExp training infrastructure

Source decision record:
`progress/explorations/2026-05-15_training_infrastructure_architecture_decisions.md`

Primary target: unified Stage-1 and Stage-2 detection training infrastructure
centered on `compact_full`, explicit supervision spans, entry-trie
multi-positive CE, optional coordinate/regression objectives, and typed
observability.

## Purpose

This design converts the 2026-05-15 architecture audit and `grill-me` decisions
into a Superpowers implementation contract. The goal is not to preserve every
historical training branch. The goal is to rebuild the training infrastructure
around the current research direction with smaller surfaces, explicit data
flow, clean module boundaries, and diagnosable behavior.

The new architecture treats Stage-1 and Stage-2 as first-class training
surfaces that share one supervision spine:

```text
source example or rollout
  -> semantic SupervisionPlan
  -> DetectionTemplateCodec
  -> EncodedDetectionView + ModelInputBundle + TrainingSidecars
  -> SpanAdapter
  -> SupervisionBatch[SupervisionSpan]
  -> TrainerLossBridge
  -> ModelForwardResult + PredictionCoordinateMapper
  -> ObjectiveRunner
  -> ObjectiveModule results
  -> ObservabilityService
```

`SupervisionPlan` is the current working name for the semantic per-example /
per-channel supervision intent. The name is still explicitly in discussion; the
architecture should keep the concept but allow a final naming pass before a
stable public API is committed.

## High-Level Diagnosis

The current repo contains useful training capabilities, but the engineering
shape reflects many experimental eras at once.

- `src/sft.py` is still too broad: it acts as a routing switchboard, runtime
  owner, compatibility layer, and trainer launcher instead of a thin launcher
  over resolved training surfaces.
- Training behavior is selected through legacy variants and `custom.*` knobs
  rather than a first-class `surface.id` and typed runtime plan.
- There are multiple incompatible supervision representations: legacy labels
  and token masks, compact recursive detection sidecars, Stage-2 rollout
  metadata, and flat metric dictionaries.
- Template structure is not authoritative enough downstream. Compact-full
  schema tokens, free-text descriptions, and coordinate tokens should be
  represented as token roles and spans, not rediscovered by string parsing.
- Loss modules are too entangled with trainer integration, sidecars, and legacy
  heuristics. Standard CE, trie CE, coordinate soft CE, and regression losses
  should all run through one explicit objective path.
- Stage-2 still carries Hungarian matching and rollout-aligned compatibility
  assumptions that should not shape the future architecture.
- Duplicate-negative training, adjacent repulsion, and training-time
  EOS/continuation hacks are rejected mechanisms. Keeping them as dormant
  abstractions would preserve the wrong research lessons.
- Metrics and diagnostics are spread across typed events, flat maps, monitor
  dumps, and historical artifact conventions. New code needs a canonical event
  surface with tolerant readers for old runs.
- `progress/` contains important evidence, but some notes record negative
  results or unhealthy directions and must be demoted so they do not remain
  active architectural guidance.

## Scope

This design covers the target architecture and staged refactor plan for:

- Stage-1 standard SFT / token CE.
- Stage-1 compact-full entry-trie multiple-positive / multiple-target CE.
- Optional coordinate soft-token and box/regression objectives.
- Stage-2 two-channel rollout-aware training, redesigned around clean
  assignment, duplicate filtering, false-negative insertion, and compact-full
  supervision planning.
- Shared template encoding, span adaptation, objective execution, metrics,
  diagnostics, and experiment records.
- Cleanup of rejected training mechanisms and active docs/config exposure.

The implementation plan produced from this design must start with cleanup and
then rebuild bottom-up. It must not introduce a top-level pipeline wrapper that
merely preserves the current messy sidecar world.

## Accepted Decisions

These decisions are binding for implementation unless the user explicitly
reopens them:

- The refactor is cleanup-first, then rebuild. Rejected mechanisms are removed
  before new abstractions are introduced.
- The rebuild is bottom-up. Start with semantic supervision, encoded views,
  spans, and objective contracts before introducing top-level pipeline
  orchestration.
- `SupervisionPlan` is the current working name for the semantic per-example /
  per-channel supervision object. The name is provisional and should remain
  localized until final vocabulary is approved.
- `TargetPlan` is not promoted as the canonical public name because it was too
  broad and ambiguous for this core contract.
- `TrainingPipeline` is a protocol, not a heavy shared base class.
- `TargetDistribution` is a registered dataclass family, not an arbitrary open
  duck-typed object.
- Duplicate handling is pre-target hygiene plus diagnostics, not a training
  loss.
- Assignment, duplicate filtering, false-negative insertion, and object
  ordering are separate Stage-2 components.
- Stage-2 rollout I/O is a first-class template-aware boundary. A
  compact-full checkpoint must not be routed through CoordJSON prompts,
  CoordJSON parsers, or CoordJSON false-negative append logic. The rollout
  prompt builder, decode policy, parser, append policy, supervision planner, and
  artifact writer must all agree on the resolved rollout template.
- Compact-full Stage-2 training rollouts use unconstrained decoding by default.
  Grammar-constrained compact decoding is diagnostic/control-only unless a
  future decision explicitly changes the research question.
- `MetricEvent` and `DiagnosticEvent` are separate streams coordinated by one
  observability service.
- Historical evidence is preserved, but old notes that promoted failed
  directions are demoted and cannot remain active guidance.
- Stable-contract contradictions are not ordinary cleanup. If current
  OpenSpec/docs require a removed mechanism, Stage-2 ordering default, or
  duplicate-control order, the implementation must include the corresponding
  OpenSpec/docs migration before deleting or flipping behavior.

## Non-Goals

- Do not preserve backward compatibility for rejected mechanisms.
- Do not keep duplicate-negative training as an experiment-only objective.
- Do not keep adjacent repulsion as a zero-weight or dormant objective.
- Do not keep training-time EOS loosen, forced-continuation, or stop-gate hacks
  in the canonical training path.
- Do not design the future Stage-2 stack around Hungarian matching.
- Do not enable compact-full packing until a tested `PackingSegmentMap` exists.
- Do not enable encoded-sample cache reuse until a complete
  `EncodedSampleFingerprint` exists.
- Do not mutate upstream Qwen3-VL or Hugging Face model files.
- Do not hide new behavior under broad `custom.extra`, arbitrary `extra`, or
  untyped sidecar dictionaries.
- Do not turn diagnostics JSONL into required training-cache input.
- Do not treat old `progress/` notes as current authority unless they are
  explicitly reclassified as current references.

## Hard Guardrails

| Area | Requirement |
|---|---|
| Template default | `compact_full` is the canonical default for Stage-1 compact training and future Stage-2 training. |
| JSON baseline | `stage1_json_pretty` remains an explicit baseline/debug template, not the default detection training target. |
| Model boundary | Qwen3-VL/ms-swift multimodal inputs are preserved through one normal model forward. Objective modules never call the model. |
| Logits | Keep raw full logits by default. Any `logits_to_keep`, packing, or projection path must be explicit and tested. |
| Causal shift | Label/token positions map to logits positions through `PredictionCoordinateMapper`; no objective may rely on hidden off-by-one assumptions. |
| Loss ownership | Standard CE is an explicit `TokenCEObjective`, not an implicit hidden base loss when the new runner owns training loss. |
| Objective precision | Each objective declares safe precision defaults and upcasts numerically sensitive math to `float32` when needed. |
| Removed mechanisms | Duplicate-negative, adjacent-repulsion, forced-continuation, and stop-gate distributions are rejected target types. |
| Config | Components receive typed runtime plans, not raw YAML dictionaries. |
| Diagnostics | Diagnostic artifact emission is bounded by profile and routed through `ObservabilityService`. |
| Stage-2 pairing | Greedy IoU assignment is the canonical future default; Hungarian is migration-only until replacement smoke passes. |
| Stage-2 rollout I/O | `compact_full` Stage-2 rollouts require a compact-full prompt/decode/parser/FN-append path. CoordJSON rollout parsing is legacy-only for CoordJSON surfaces. |
| Duplicate handling | Duplicate filtering happens before positive target realization and before forward pass. |
| Object ordering | Stage-1 and Stage-2 use shared `ObjectOrderingStrategy` with explicit provenance for nontrivial ordering. |

## Target Architecture

The rebuilt training stack uses a small professional pipeline/component
architecture. Pipeline implementations own stage-specific orchestration.
Components own narrow typed contracts. Shared reuse happens through dataclasses,
protocols, strategy registries, span projectors, objective modules, and
observability services, not inheritance-heavy trainer mixins.

```text
YAML config
  -> ConfigLoader / schema validation
  -> TrainingSurfaceResolver
  -> ResolvedTrainingRun
  -> TrainingPipeline
  -> SupervisionPlanner
  -> SupervisionPlan
  -> DetectionTemplateCodec
  -> EncodedTrainingExample
  -> ObjectiveRunner
  -> ObservabilityService
```

Initial pipelines:

- `Stage1JsonCePipeline`: standard JSON-based chat-template SFT with explicit
  token CE.
- `Stage1CompactTrieCePipeline`: compact-full Stage-1 training with entry-trie
  multi-positive CE and optional coordinate/regression objectives.
- `Stage2TwoChannelPipeline`: Stage-2 Channel-A / Channel-B training with
  assignment, duplicate filtering, false-negative insertion, compact-full
  supervision planning, and shared objective execution.

The shared protocol should stay small. A heavy base class with protected hooks
is explicitly rejected.

## Semantic Supervision Layer

`SupervisionPlan` is semantic-only. It represents the intent for one example or
one Stage-2 channel example. It does not contain rendered assistant text,
tokenized bytes, model tensors, metric accumulators, raw config dictionaries,
or trainer/model handles.

`SupervisionPlan` owns:

- stage and channel identity;
- semantic object entries;
- object ordering intent and provenance;
- accepted rollout objects when Stage-2 applies;
- false-negative insertions when Stage-2 Channel-B applies;
- ignored predictions with reason references;
- trie or multiple-positive object-entry structure when Stage-1 compact trie
  training applies;
- object-level provenance.

`SupervisionContext` is a small frozen semantic context object. It carries only
scalar identifiers and ownership/provenance metadata such as dataset id, split,
template id, stage, channel, run/experiment id, and diagnostic labels. It does
not carry tokenizer references, coordinate vocabularies, objective policy
objects, assignment services, duplicate-filter services, observability handles,
rendered text, tensors, or raw config dictionaries. Those policy and service
dependencies belong to later codec, resolver, objective, assignment, and runtime
bridge layers.

## Template And Encoding Layer

`DetectionTemplateCodec` is the sole owner of template rendering, chat-template
encoding, strict parsing support, token-role projection, and token/span
alignment. The codec explicitly models two template families:

- `stage1_json_pretty`: standard JSON-based chat-template baseline.
- `compact_full`: canonical compact object-entry target for Stage-1 and
  Stage-2.

For `compact_full`, the codec must represent the sequence as explicit token
roles:

- schema tokens such as `<|object_ref_start|>` and `<|box_start|>`;
- free-text description tokens;
- special coordinate tokens such as `<|coord_*|>`;
- stop / assistant-end tokens.

`EncodedDetectionView` is the authoritative tokenized semantic view. It stores
rendered assistant text only for diagnostics and provenance. Training authority
comes from token ids, labels, token roles, spans, coordinate slots, object
entries, and target-token label positions.

`ModelInputBundle` is separate from `EncodedDetectionView`. It owns the exact
ms-swift/Qwen3-VL bridge payload and uses a strict backend key registry.
Sidecars are forbidden inside the model-input bundle.

The initial `ms_swift_qwen3_vl` backend registry must explicitly cover the
fragile multimodal and position keys that are easy to drop accidentally:

- `input_ids`;
- `labels` when the bridge is not in runner-owned-loss mode;
- `attention_mask`;
- `token_type_ids`;
- `pixel_values`;
- `pixel_values_videos`;
- `image_grid_thw`;
- `video_grid_thw`;
- `second_per_grid_ts`;
- `position_ids`;
- `text_position_ids` as an auxiliary bridge input used to synthesize or
  augment Qwen mRoPE `position_ids`, not necessarily as a raw key forwarded to
  `model(**inputs)`;
- `cross_attention_mask`;
- `cache_position`;
- `past_key_values`;
- `use_cache`;
- `cu_seq_lens`;
- `cu_seq_lens_q`;
- `cu_seq_lens_k`;
- `max_length_q`;
- `max_length_k`;
- `logits_to_keep` only when an explicit projection contract is enabled;
- `output_router_logits`.

The bridge must preserve the current live detection/Qwen batch contract unless
an implementation task explicitly supersedes it with parity tests. Preservation
does not always mean raw forwarding: bridge-consumed keys such as
`text_position_ids` may be transformed into model-ready `position_ids`, and
batch-contract keys such as `max_length_q`, `max_length_k`, or
`pack_num_samples` must be classified as forwarded, consumed, or sidecar-only
instead of silently rejected. Labels and loss-related keys are intentionally
stripped in runner-owned-loss mode.

`EncodedTrainingExample` links:

```text
EncodedDetectionView
ModelInputBundle
TrainingSidecars
```

Typed objects are used internally. Backend-compatible dictionaries are created
only at the collator/trainer boundary.

## Span And Target Distribution Layer

`SpanAdapter` consumes `SupervisionPlan` plus `EncodedDetectionView`. It never
renders strings, calls the tokenizer, parses rendered text, or discovers schema
tokens independently.

`SupervisionSpan` represents a semantic region, not a single token by default.
Each span carries explicit label positions, not already-shifted logit
positions. Causal-shift behavior is visible and testable because
`PredictionCoordinateMapper` is the only component that maps a label position
`p` to a model-logit row `p - 1`.

Conceptual shape:

```python
@dataclass(frozen=True)
class SupervisionSpan:
    span_id: str
    sample_id: str
    stage: TrainingStage
    channel: str | None
    object_id: str | None
    role: SpanRole
    token_span: TokenSpan
    label_positions: tuple[int, ...]
    target: TargetDistribution
    weight: float
    mask_policy: SpanMaskPolicy
    provenance: SpanProvenance
```

`TargetDistribution` is a registered dataclass family. The initial accepted
family is:

- `HardTokenDistribution`;
- `MultiPositiveTokenDistribution`;
- `CoordinateSoftTokenDistribution`;
- `BoxRegressionDistribution`.

Coordinate and box objectives require a richer typed tensor contract than a
generic token distribution. The initial design must include a
`CoordinateSlotGroup` / `DecodedBoxTensor` contract for four-slot boxes,
coordinate token ids, target bins, decode mode, temperature, bbox format,
parameterization, group weights, and full-logit row handles. `coord_reg`-style
dependencies must be declarative; objective modules must not exchange hidden
mutable state.

`CoordinateSoftTokenDistribution` must encode its target family and loss mode,
for example `iou_gibbs_v0`, `ciou_gibbs_v0`, support masks, tau/sigma/truncation
policy, coordinate token id range, and whether it uses full-vocabulary
support/balance CE or a coordinate-vocabulary-only soft CE/W1 variant.

Rejected distribution types:

- `DuplicateNegativeDistribution`;
- `AdjacentRepulsionDistribution`;
- `ForcedContinuationDistribution`;
- `StopGateDistribution`.

## Objective And Forward Boundary

The Qwen3-VL/ms-swift forward path is fragile and must remain separate from
loss math. The trainer-facing bridge owns model-input preparation, sidecar
stripping, ms-swift template context, full-logit preservation, and forward
provenance.

```text
TrainerLossBridge
  -> strip sidecars
  -> call model once
  -> ModelForwardResult(raw logits)
  -> PredictionCoordinateMapper
  -> ObjectiveRunner
  -> ObjectiveModule(s)
  -> ObjectiveResult(s)
```

`ObjectiveRunner` groups spans, validates prediction coordinates, gathers
objective-ready tensors, invokes objective modules, combines normalized losses,
and emits metric/diagnostic events.

Initial objective modules:

- `TokenCEObjective`;
- `TrieCEObjective`;
- `CoordSoftCEObjective`;
- `BoxRegressionObjective`.

`TrieCEObjective` is the canonical multiple-positive primitive. It must cover
the current recursive detection CE semantics, not only a generic
multi-positive `logsumexp` toy loss. Required semantics include support mass,
balance loss, branch multiplicity weights, duplicate positive-token rejection,
fp32 log-softmax/logsumexp over selected rows, denominator policy, label-row to
logit-row mapping, and explicit preservation or removal of the current type-gate
term.

Coordinate and regression objective migration must classify every current
subterm before implementation. The required classification dimensions are:
`preserve`, `remove`, or `diagnostic-only`; future owner; typed tensor contract;
parity or absence test; metric-key policy; and required dtype/precision. The
initial classification set includes bbox SmoothL1, bbox CIoU, bbox size
auxiliary terms, coordinate-token CE, coordinate soft CE, coordinate W1,
coordinate gate, text gate, and adjacent repulsion.

Each objective computes and reports its own normalized loss. `ObjectiveRunner`
combines weighted objective losses. CE tokens, trie branch decisions,
coordinate slots, and bbox regression targets must not be forced into one
global denominator.

## Stage-2 Target Flow

Stage-2 uses template-aware rollout I/O and clean object-level planning before
span conversion. The future canonical path is:

```text
Stage2RolloutTemplatePolicy
  -> Stage2RolloutPromptBuilder
  -> DecodePolicy
  -> Stage2RolloutParser
  -> Stage2RolloutAppendPolicy
  -> RolloutViews
  -> DuplicateFilter
  -> DuplicateFilterResult
  -> accepted survivors
  -> AssignmentStrategy
  -> AssignmentResult
  -> Stage2ChannelBSupervisionPlanner
  -> Stage2ChannelBSupervisionPlan
  -> CompactFullTemplateCodec
  -> EncodedDetectionView
  -> Stage2CompactSpanAdapter
```

`Stage2RolloutTemplatePolicy` resolves the rollout sequence family for the
stage-2 surface. It is not inferred from a checkpoint path or from legacy
`custom.json_format` defaults. It must be recorded in the resolved config and
run artifacts.

Stage-2 rollout migration is dual-surface and explicit:

- `compact_full` is canonical for A2-style compact-full checkpoints and new
  Stage-2 work.
- `coordjson` is kept as an explicit legacy surface until compact-full Stage-2
  smoke parity is established.
- implicit fallback between the two surfaces is forbidden. A compact-full
  checkpoint or config must fail validation if it would otherwise be routed
  through CoordJSON rollout prompting, parsing, or false-negative append logic.

Initial rollout template families:

- `compact_full`: canonical target for A2-style compact-full checkpoints and
  future Stage-2 training. It uses compact-full prompting, unconstrained
  rollout decoding by default, strict compact-full parsing, compact-full
  false-negative append logic, compact-full artifact serialization, and
  `CompactFullTemplateCodec` for supervision conversion. Compact grammar
  decoding may be used only as an explicitly labeled diagnostic/control probe,
  not as the default training rollout path or readiness proof.
- `coordjson`: legacy compatibility surface. It may keep the current
  CoordJSON prompt/parser/append behavior only for checkpoints and configs that
  explicitly select CoordJSON.

The 2026-05-17 A2 smoke exposed this as a blocking boundary: the same
`checkpoint-3664` produced valid compact-full infer/eval outputs under the
compact-full infer pipeline, but the Stage-2 tiny smoke produced zero valid
predicted rollout objects because the training rollout path was still
CoordJSON-shaped. Therefore, a successful Stage-2 launch is not sufficient
evidence of compact-full Stage-2 readiness. Compact-full Stage-2 readiness
requires a rollout I/O smoke that proves valid generated compact-full objects
before assignment, duplicate filtering, and false-negative insertion are
interpreted as model-quality signals.

Malformed or empty model rollouts are training signals, not samples to discard.
For compact-full Stage-2, the default invalid/empty rollout policy is
`fallback_gt_fn_append_only`: ignore invalid predicted objects, construct a
clean compact-full Channel-B target from GT/FN append-only supervision, and
record the fallback reason. This fallback uses the same default Channel-B loss
weight as valid-rollout supervision (`fallback_loss_weight=1.0`) because empty
or malformed rollouts are correction-worthy model failures, not lower-priority
samples. Fallback spans must carry explicit provenance such as
`rollout_context=fallback_gt_fn_append_only`, and metrics must report fallback
loss share and fallback rate separately. If fallback samples exceed 30-40% of
Channel-B samples over a monitoring window, the run should be flagged as
rollout-distribution unhealthy. This fallback does not count as a valid rollout
and must not hide the model failure in metrics or artifacts. It is different
from a configuration/template mismatch, where a compact-full config is wired to
a CoordJSON parser or appender; that remains a hard validation failure.

Target direction is duplicate filtering before greedy-IoU assignment. Assignment
must operate on accepted survivors, not the raw predicted object list, so target
realization cannot accidentally reintroduce filtered duplicate predictions.
Changing this ordering remains compatibility-sensitive and requires a stable
contract/docs migration in the same change.

In all variants, duplicate filtering remains a separate component and must not
become a loss-side duplicate-negative objective. False-negative insertion
belongs to the Stage-2 Channel-B supervision planner. Object ordering runs
after filtering and false-negative insertion, on the final semantic target
object set.

Initial assignment strategies:

- `GreedyIoUAssignment`: canonical future default.
- `LegacyHungarianAssignment`: migration-only compatibility path, not a design
  anchor.

Initial object ordering strategies:

- `sorted`: future canonical target after default-migration approval.
- `seeded_random_shuffle`: explicit, seeded, provenance-recorded augmentation.
- `tail_append_legacy`: current Stage-2 compatibility/default option until an
  OpenSpec/default migration changes it.
- `source_order`: explicit source-order preservation.

Every Stage-2 run artifact must record the resolved ordering policy. Default
changes for Channel-B ordering are compatibility-sensitive and require stable
contract/docs migration, not only code changes.

## Config Architecture

The new config hierarchy uses small explicit top-level domains:

```yaml
run:
  name: ...
  output_dir: ...
  seed: ...

surface:
  id: stage1_compact_trie

data:
  train_jsonl: ...
  val_jsonl: ...
  geometry:
    do_resize: false
  packing:
    enabled: false
  encoded_sample_cache:
    enabled: false

template:
  id: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy

supervision:
  builder: stage1_compact_trie
  stage2:
    assignment:
      strategy: greedy_iou
    duplicate_filtering:
      strategy: deterministic_iou_cluster

objectives:
  profile: stage1_compact_trie
  entries:
    token_ce:
      enabled: true
      weight: 1.0
    trie_ce:
      enabled: true
      weight: 1.0
    coord_soft_ce:
      enabled: false

observability:
  metrics:
    typed_events: true
  diagnostics:
    level: standard
    sample_per_step: 8
    max_records_per_run: 5000

artifacts:
  manifest: true
  resolved_config: true
  supervision_plan_samples: true

runtime:
  trainer_backend: ms_swift
  precision:
    default: bf16
```

Stage-1 and Stage-2 share the same top-level hierarchy. Each `surface.id`
selects a surface-specific schema that rejects irrelevant sections and removed
mechanism keys.

Objective authoring uses keyed entries so inherited ablations can disable or
modify one objective without accidentally replacing the entire objective list.
The resolver produces a deterministic ordered runtime objective list.

Canonical configs are strict. Experimental knobs require an explicit
top-level `experimental` block with owner, expiry, notes, and explicit
surface/pipeline opt-in. Missing metadata must be a schema error, and production
surfaces must be able to reject the block unless they opt in.

The artifact contract is part of experiment management, not an optional UI
detail. The new hierarchy must preserve or deliberately migrate the current
rank-0 artifact set:

- `resolved_config.json`;
- `runtime_env.json`;
- `effective_runtime.json`;
- `pipeline_manifest.json` when the pipeline path applies;
- `experiment_manifest.json`;
- `run_metadata.json`;
- `train_data_provenance.json`;
- `eval_data_provenance.json`;
- source-config copies.

Stage-2 two-channel runs additionally preserve first-class
`stage2_policy_provenance` in rank-0 manifests for assignment strategy,
duplicate-filter strategy, object-ordering policy, and their effective
thresholds.

Existing bootstrap owners such as `src/bootstrap/experiment_manifest.py`,
`src/bootstrap/pipeline_manifest.py`, and `src/bootstrap/run_metadata.py` remain
the migration authority until a tested replacement exists.

## Packing And Cache Policy

Packing remains disabled for compact-full supervision in the initial clean
architecture:

```yaml
data:
  packing:
    enabled: false
```

Packing may be enabled only after `PackingSegmentMap` exists and rewrites:

- `EncodedDetectionView` spans;
- `SupervisionSpan.label_positions`;
- sidecar references;
- segment provenance;
- label/logit alignment checks.

Encoded-sample caching also remains disabled initially:

```yaml
data:
  encoded_sample_cache:
    enabled: false
```

Cache reuse requires an `EncodedSampleFingerprint` that covers all
supervision-shaping dependencies:

- template codec version;
- tokenizer version;
- ms-swift template backend;
- object ordering policy and seed;
- coordinate vocabulary;
- supervision planner version;
- span adapter version;
- source row hash;
- surface id;
- Stage-2 assignment, duplicate-filter, and false-negative policies.

## Observability Architecture

New code emits typed `MetricEvent` and `DiagnosticEvent` records through
`ObservabilityService`.

`MetricEvent` is for scalar or reduced quantities. The canonical event contract
must remain single-owned; in the current codebase that owner is
`src/metrics/events.py`. `ObservabilityService` is a sink/orchestration layer
over the canonical event type, not a second incompatible `MetricEvent`
definition.

- objective losses and weighted losses;
- denominators and objective weights;
- span counts by role;
- token-role statistics;
- trie support-size and branch statistics;
- coordinate accuracy and regression diagnostics;
- Stage-2 assignment IoU statistics;
- duplicate filter drop rates.

`DiagnosticEvent` is for structured debug payloads:

- supervision-plan samples;
- assignment pairs;
- duplicate clusters;
- ignored prediction reasons;
- rollout parse failures;
- EOS/continue probe traces;
- coordinate outlier samples;
- rendered-text references.

Diagnostic compatibility is also contractual. Cleanup must map current concrete
diagnostic artifacts to new bounded diagnostic writers rather than preserving
only scalar summaries. The compatibility map must cover `monitor_dumps/`,
`prepare_failures/`, Stage-2 `eval_detection/step_<global_step>/raw_rollouts.jsonl`,
`pred_token_trace.jsonl`, and guarded eval/post-op analysis artifacts.

Diagnostic artifact emission is profile-based:

- `off`: metrics only except hard failures.
- `standard`: bounded samples and summaries, default for normal runs.
- `debug`: richer traces for smoke/debug runs only.

New runs must not emit metric keys for removed training mechanisms. Historical
readers may tolerate old keys and label them as legacy or removed.

## Cleanup Contract

The first implementation wave is cleanup/deletion, not additive abstraction.

Remove from live training code/config/tests/docs:

- duplicate-negative training and `loss_duplicate_burst_unlikelihood`;
- adjacent repulsion;
- training-time EOS loosen;
- forced-continuation training losses;
- stop-gate training mechanisms;
- safe dead import shims and old aliases after confirming no live imports;
- active docs/config recommendations for removed mechanisms.

Keep:

- duplicate filtering and duplicate diagnostics;
- posthoc guarded eval as an explicitly labeled analysis view;
- EOS/continue diagnostic probes and monitoring metrics;
- historical run artifacts and minimal historical notes;
- old executable Stage-2/Hungarian path until greedy-IoU replacement smoke
  passes.

`progress/` notes should be preserved as evidence but demoted through the
repo's existing status style. Prefer extending existing hyphenated statuses
such as `concluded-negative`, `superseded`, `mechanism-evidence`, or
`archive-only` rather than introducing a parallel underscore taxonomy.

OpenSpec is not required by default for ordinary cleanup. However, the current
review found that some rejected mechanisms and Stage-2 defaults are likely
stable-contract surfaces. Before removing duplicate-burst UL from Stage-2
canonical paths, flipping Channel-B ordering defaults, or deleting/archiving
Hungarian executable paths, the implementation must audit and update or
supersede the relevant `openspec/specs/` contracts and current docs in the same
change.

The known duplicate-burst UL stable-contract surfaces are already identified:
`openspec/specs/stage2-ab-training/spec.md`,
`openspec/specs/teacher-forcing-objective-pipeline/spec.md`, and
`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`. They must be
migrated or superseded before code/config deletion makes
`loss_duplicate_burst_unlikelihood` invalid.

Stage-2 eval artifact preservation is a hard migration gate. Before any
Stage-2 executable-path migration, resolver default flip, or observability
rewrite, the current `rollout_matching.eval_detection.materialize_artifacts`
default and artifact set must be preserved or explicitly migrated. The required
artifact set includes `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`,
`infer_summary.json`, `metrics.json`, `per_image.json`, `raw_rollouts.jsonl`,
and `pred_token_trace.jsonl` when trace metadata is available. Stage-2 run
artifacts must also record the resolved assignment strategy, duplicate-filter
strategy, and object-ordering policy.

## Explicit Removals

Remove these from live training code, canonical configs, positive behavior
tests, and current docs:

- `loss_duplicate_burst_unlikelihood`;
- duplicate-negative objective modules;
- duplicate-negative target distributions;
- duplicate-negative training metrics for new runs;
- adjacent-repulsion objectives;
- adjacent-repulsion config knobs, including zero-weight clutter;
- training-time EOS-loosen mechanisms and production-looking presets;
- forced-continuation loss knobs and training hooks;
- stop-gate training mechanisms;
- old import/config shims that are proven dead and safe to delete.

Keep only as historical or diagnostic surfaces:

- old run artifacts and git history;
- historical `progress/` evidence with demoted status;
- duplicate filtering and duplicate diagnostics;
- EOS/continue probes when owned by analysis/diagnostics, not training losses;
- historical artifact readers that label old keys as legacy or removed;
- posthoc guarded eval duplicate views, explicitly labeled as analysis views;
- runnable legacy Stage-2/Hungarian path until greedy-IoU smoke passes.

If an existing stable spec requires one of the removed mechanisms, removal is
blocked until the stable contract is updated or superseded. The implementation
must not leave code, configs, docs, and OpenSpec in contradiction.

## Validation Strategy

The refactor must use a staged validation ladder before production runs:

```text
Level 0: config/schema validation
Level 1: semantic planning validation
Level 2: template/encoding validation
Level 3: span/coordinate validation
Level 4: objective math validation
Level 5: bridge/model-forward validation
Level 6: integrated tiny training smoke
Level 7: Stage-2 rollout-planning smoke
Level 7A: Stage-2 compact-full rollout I/O smoke with an A2 checkpoint
Level 7B: Stage-2 eval artifact preservation
```

Stage-2 compact-full rollout I/O uses two acceptance gates:

- Gate 1, launch/I/O wiring: tiny scope of 2-4 samples, unconstrained greedy
  decoding, compact-full parser path only, no CoordJSON fallback, preserved raw
  output artifacts, and at least one valid predicted compact-full object.
- Gate 2, readiness before real Stage-2 training: small scope of 16-32 samples,
  unconstrained greedy decoding, `sample_valid_pred_rate >= 0.75` as the
  initial threshold, zero parser-template mismatches, explicit reporting of
  parse truncation, empty-valid-object cases, and GT/FN fallback counts, and
  raw rollout plus parsed object artifacts available for manual inspection.

Gate 1 proves the template/codec/runtime seam is wired. Gate 2 proves the A2
rollout distribution is stable enough to drive Stage-2 learning. A Gate-1 pass
must not be presented as full Stage-2 readiness.

The first golden fixture is a compact-full multimodal Stage-1 example with
three objects, mixed descriptions, nearby non-duplicate boxes, boundary
coordinate tokens, expected `EncodedDetectionView`, expected
`SupervisionPlan`, expected `SupervisionSpan` list, mapper expectations, and a
small diagnostic snapshot.

The second golden fixture is a Stage-2 rollout-planning example covering greedy
assignment, deterministic duplicate filtering, false-negative insertion, final
object ordering, and Channel-B supervision planning.

## Acceptance Criteria

The design is successfully implemented when:

- rejected training mechanisms are absent from live code/config/tests/current
  docs and cannot be enabled by stale YAML;
- duplicate diagnostics, EOS/continue probes, and historical artifact readers
  remain available without shaping training losses;
- Stage-1 JSON CE, Stage-1 compact trie CE, and Stage-2 two-channel training
  resolve through `surface.id` and typed runtime plans;
- compact-full token roles are represented in `EncodedDetectionView` and used
  by span adapters instead of string reparsing;
- `SupervisionPlan` remains semantic-only and per-example/per-channel;
- objective modules consume objective-ready tensors and never call the model;
- raw logits and prediction coordinate mapping are explicit;
- standard CE and trie CE both run through `ObjectiveRunner`;
- per-objective precision policy, denominator, weight, and metric events are
  emitted;
- `compact_full` is canonical for Stage-1 compact training and Stage-2
  training, while JSON remains an explicit baseline;
- Stage-2 uses assignment, duplicate filtering, and false-negative insertion as
  separate components;
- Stage-2 rollout I/O is template-aware, and compact-full checkpoints are never
  evaluated through CoordJSON prompts/parsers/appenders;
- A2 compact-full Stage-2 smoke passes both rollout I/O gates before
  rollout-quality or training-readiness claims are made;
- malformed or empty compact-full rollouts fall back to GT/FN append-only
  Channel-B supervision by default with the same initial loss weight as normal
  Channel-B correction, while remaining visible as invalid/empty rollout
  diagnostics rather than valid rollout evidence;
- `progress/index.yaml`, relevant progress READMEs, and `docs/catalog.yaml` no
  longer route removed mechanisms as active current guidance;
- packing and encoded cache remain disabled until their contracts are complete;
- L0-L7 validation passes on targeted tests and tiny smokes before any
  production-scale training.

## Residual Risks

- Qwen3-VL multimodal input handling is fragile. The model-forward bridge must
  be tested with real multimodal payload keys before any production run.
- The `SupervisionPlan` name is still provisional. Implementation should keep
  naming localized until the user approves the final vocabulary.
- Removing rejected mechanisms may expose configs, docs, or tests that were
  silently relying on them. Absence tests should catch this intentionally.
- Stage-2/Hungarian code cannot be fully removed until the greedy-IoU path has
  a smoke-tested replacement.
- Stage-2 compact-full training can appear launch-healthy while producing no
  valid raw rollout objects if the rollout prompt/parser surface remains
  CoordJSON. The implementation must gate compact-full Stage-2 readiness on a
  compact-full rollout I/O smoke, not only on process exit or loss logging.
- YAML inheritance can still hide list replacement pitfalls until objective
  authoring moves to keyed profiles.
- Packing and cache are high-risk because they rewrite position-sensitive span
  and sidecar coordinates. They remain out of scope for the initial rebuild.
