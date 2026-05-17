# Unified Training Infrastructure Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up rejected training mechanisms and rebuild CoordExp Stage-1/Stage-2 training infrastructure around typed supervision planning, compact-full encoding, explicit token spans, objective modules, and typed observability.

**Architecture:** Start with deletion and absence tests, then rebuild bottom-up from semantic supervision plans through encoding, spans, objective execution, trainer loss bridging, observability, Stage-2 assignment/filtering, and finally pipeline resolution. Keep Qwen3-VL/ms-swift forward mechanics isolated in a trainer bridge and keep objective modules as logits-and-spans math components.

**Tech Stack:** Python, PyTorch, Hugging Face/Qwen3-VL, ms-swift trainer integration, CoordExp YAML config schemas, pytest under `conda run -n ms`, typed dataclasses/protocols.

---

Date: 2026-05-15

Spec: `docs/superpowers/specs/2026-05-15-unified-training-infrastructure-architecture-design.md`

Status: approved to start implementation after targeted readiness review.
Implementation has not started.

## Execution Policy

This is a staged architecture refactor. Do not try to implement all tasks in one
worker. Use fresh subagents or separate execution slices for each task group.

Implementation order is binding:

1. stable-contract, legacy Stage-2, Qwen bridge, and provenance gates first;
2. cleanup/deletion second, only after named gates pass;
3. bottom-up shared contracts third;
4. Stage-1 compact validation fourth;
5. Stage-2 assignment/planning fifth;
6. top-level pipeline resolver last.

The plan intentionally avoids production training until targeted tests and tiny
smokes prove each boundary.

## Review-Driven Refinements

The first critical review pass found no P0 issues, but it tightened several
implementation constraints. These refinements are binding for execution:

- Do not introduce a second canonical `MetricEvent`. Reuse the current
  `src/metrics/events.py` contract and build `ObservabilityService` around it.
- Do not create a parallel template registry that drifts from
  `src/detection/template.py` and `src/detection/tokenization.py`; wrap or
  migrate those owners with parity tests.
- Do not bypass `src/training_runtime/plan.py` while migrating to
  `TrainingSurfaceResolver`; the new resolver must preserve or explicitly
  replace its packing, collator, runtime, and Stage-2 namespace invariants.
- Do not flip `src/sft.py` routing/defaults until both legacy/current configs
  and new-surface shadow configs pass.
- Treat duplicate-burst UL removal, Stage-2 duplicate-control ordering, and
  Channel-B ordering defaults as stable-contract migrations when OpenSpec or
  current docs require them.
- Use `SupervisionSpan.label_positions` for target-token label positions. Only
  `PredictionCoordinateMapper` derives shifted logit rows.
- Ground every representative test helper/API against current imports before
  implementation; do not leave pseudocode fixtures as if they were real.

The final readiness review found blockers that must be resolved before code
implementation starts:

- Seed `ModelInputBundle` from the current live detection/Qwen batch boundary,
  including `token_type_ids`, `second_per_grid_ts`, `cross_attention_mask`,
  `past_key_values`, `use_cache`, `max_length_q`, and `max_length_k`.
- Treat `text_position_ids` as an auxiliary bridge input for Qwen mRoPE
  position construction, not necessarily as a raw key forwarded to
  `model(**inputs)`.
- Convert known duplicate-burst UL OpenSpec contradictions into named
  pre-Task-1 migration work before deleting code/config/schema support.
- Prove current Stage-2 runnable configs and eval artifacts before and after
  cleanup; do not wait until resolver migration to discover breakage.
- Preserve the existing experiment manifest/provenance artifact contract or
  migrate it deliberately through the current bootstrap owners.
- Make `experimental` an explicit top-level strict config domain with required
  owner, expiry, notes, and surface/pipeline opt-in.
- Reconcile the `SupervisionPlan` / `TargetPlan` vocabulary before public module
  names and tests are created.

The 2026-05-17 A2 Stage-2 launch smoke added one more binding refinement:

- Treat Stage-2 rollout I/O as a template-aware boundary. The A2 random
  ET-RMP-CE compact-full checkpoint loaded and trained for one tiny Stage-2 step
  through `model.adapters`, but the current Stage-2 rollout path still used
  CoordJSON-shaped prompting/parsing/false-negative append behavior and produced
  zero valid predicted rollout objects. The same checkpoint produced valid
  compact-full infer/eval outputs through the compact-full infer pipeline.
- Do not claim compact-full Stage-2 readiness from process exit, loss logging,
  or `invalid_rollout=0` alone. Compact-full Stage-2 readiness requires a
  rollout I/O smoke that proves valid generated compact-full objects before
  assignment, duplicate filtering, and false-negative insertion are interpreted
  as model-quality signals.
- The approved migration shape is dual-surface and explicit: `compact_full` is
  canonical for A2-style checkpoints and new Stage-2 work, while `coordjson`
  remains runnable only as an explicit legacy surface. Implicit fallback or
  mixed prompt/parser/appender selection between surfaces is forbidden.
- The approved compact-full Stage-2 decode policy is unconstrained by default.
  Do not hide rollout-format failures behind grammar-constrained decoding in
  the training path. Compact grammar decoding may be retained only as an
  explicitly labeled diagnostic/control probe.
- The approved invalid/empty rollout policy is GT/FN append-only fallback, not
  sample dropping. A malformed or empty compact-full rollout should construct
  clean Channel-B supervision from GT/FN append-only targets, while metrics and
  artifacts still record the rollout failure. Fallback supervision uses the same
  initial loss weight as normal Channel-B correction and must carry separate
  provenance plus dominance diagnostics. Configuration-level template mismatches
  remain hard failures.

## Hard Guardrails

| Guardrail | Requirement |
|---|---|
| No broad compatibility | Do not preserve rejected mechanisms behind aliases or zero weights. |
| No hidden loss | New canonical paths use explicit `ObjectiveRunner`; no hidden base CE plus additive extras. |
| No model mutation | Do not edit upstream HF/Qwen3-VL model files. |
| No sidecar forwarding | Sidecars must be stripped before model forward. |
| No raw config in components | Components consume typed runtime plans only. |
| No packing yet | Compact-full packing remains disabled until `PackingSegmentMap` exists. |
| No encoded cache yet | Cache remains disabled until `EncodedSampleFingerprint` exists. |
| No Stage-2 hard delete yet | Keep old runnable Stage-2/Hungarian path until greedy-IoU replacement smoke exists. |
| Template-aware Stage-2 rollout | Compact-full checkpoints must use compact-full rollout prompts, unconstrained default decoding, parsers, false-negative appenders, supervision conversion, and artifacts. CoordJSON rollout parsing is legacy-only for explicit CoordJSON surfaces. |
| Invalid compact-full rollouts | Malformed or empty model rollouts fall back to same-weight GT/FN append-only Channel-B supervision by default and remain visible in metrics/artifacts. Config/parser surface mismatches fail fast. |
| Diagnostics preserved | Keep duplicate diagnostics and EOS/continue probes, but not training hacks. |
| OpenSpec preconditions | If a stable spec requires a removed mechanism or current Stage-2 default, update/supersede the spec before code removal or default flips. |
| Current-owner migration | New modules must wrap, migrate, or retire existing owners explicitly; no parallel owner drift. |
| Live Qwen boundary first | `ModelInputBundle` must start from current detection/Qwen batch keys and classify forwarded, consumed, and sidecar-only keys. |
| Artifact contract preserved | Manifest/provenance and Stage-2 eval artifacts must be preserved or deliberately migrated before launcher/resolver/observability rewrites. |

## Planned File Map

New or heavily refactored owners:

| Path | Responsibility |
|---|---|
| `src/training/pipelines/base.py` | Small `TrainingPipeline` protocol and pipeline lifecycle types. |
| `src/training/pipelines/stage1_json_ce.py` | Stage-1 JSON CE pipeline. |
| `src/training/pipelines/stage1_compact_trie_ce.py` | Stage-1 compact-full trie CE pipeline. |
| `src/training/pipelines/stage2_two_channel.py` | Stage-2 two-channel pipeline after assignment/filtering seams exist. |
| `src/training/surfaces.py` | `TrainingSurfaceResolver`, surface registry, and `ResolvedTrainingRun`. |
| `src/training/supervision/plans.py` | `SupervisionPlan` family, provisional naming kept localized. |
| `src/training/supervision/context.py` | Frozen semantic `SupervisionContext` with scalar identifiers, ownership, and provenance metadata only. |
| `src/training/supervision/spans.py` | `SupervisionSpan`, roles, mask policies, provenance. |
| `src/training/supervision/distributions.py` | Registered `TargetDistribution` family. |
| `src/training/supervision/batch.py` | `SupervisionBatch` and validation helpers. |
| `src/training/templates/codec.py` | Adapter/facade over current `src/detection/template.py` and `src/detection/tokenization.py`, not an independent template authority. |
| `src/training/templates/compact_full.py` | Compact-full codec adapter with parity against current strict compact template owner. |
| `src/training/templates/json_chat.py` | JSON chat-template codec baseline adapter. |
| `src/training/encoding/view.py` | `EncodedDetectionView`, object entries, coordinate slots. |
| `src/training/encoding/model_inputs.py` | `ModelInputBundle`, backend allowed-key registry. |
| `src/training/encoding/example.py` | `EncodedTrainingExample` and typed sidecar linkage. |
| `src/training/sidecars.py` | Strict `TrainingSidecars` groups. |
| `src/training/span_adapters/compact_projector.py` | Shared compact-full token/span projection utilities. |
| `src/training/span_adapters/stage1_compact.py` | Stage-1 compact trie span adapter. |
| `src/training/span_adapters/stage2_compact.py` | Stage-2 compact span adapter. |
| `src/training/objectives/runner.py` | `ObjectiveRunner`, span grouping, mapper validation, loss combination. |
| `src/training/objectives/token_ce.py` | Explicit token CE objective. |
| `src/training/objectives/trie_ce.py` | Entry-trie multi-positive CE objective. |
| `src/training/objectives/coord_soft_ce.py` | Coordinate soft-token CE objective. |
| `src/training/objectives/box_regression.py` | Optional bbox/regression objective. |
| `src/training/bridge/loss_bridge.py` | ms-swift/Qwen3-VL forward bridge. |
| `src/training/bridge/coordinate_mapper.py` | `PredictionCoordinateMapper`. |
| `src/metrics/events.py` | Existing canonical `MetricEvent`; extend or import it rather than duplicating it. |
| `src/training/observability/events.py` | `DiagnosticEvent` and observability-local event helpers only if they import/reuse canonical `MetricEvent`. |
| `src/training/observability/service.py` | `ObservabilityService`, sinks, bounded diagnostics. |
| `src/training/stage2/assignment.py` | `AssignmentStrategy`, `GreedyIoUAssignment`, migration `LegacyHungarianAssignment`. |
| `src/training/stage2/rollout_codec.py` | `Stage2RolloutTemplatePolicy`, rollout parser/appender interfaces, compact-full and legacy CoordJSON implementations. |
| `src/training/stage2/rollout_prompting.py` | Template-aware Stage-2 rollout prompt construction and decode-policy wiring. |
| `src/training/stage2/rollout_artifacts.py` | Template-aware rollout artifact serialization for raw outputs and parsed objects. |
| `src/training/stage2/duplicate_filter.py` | Deterministic duplicate filtering and diagnostics. |
| `src/training/stage2/planners.py` | Channel-A and Channel-B supervision planners. |
| `src/training/ordering.py` | Shared `ObjectOrderingStrategy`. |
| `tests/fixtures/training_architecture/` | Static golden fixtures. |
| `tests/helpers/training_architecture_fixture_builder.py` | Small builders for golden fixtures. |

Existing owners to modify during migration:

| Path | Migration role |
|---|---|
| `src/sft.py` | Shrink to launcher over resolved training surfaces. |
| `src/config/schema.py` | Add strict surface-specific schemas and removed-key rejection. |
| `src/bootstrap/experiment_manifest.py` | Current experiment manifest owner; preserve or migrate with tests. |
| `src/bootstrap/pipeline_manifest.py` | Current pipeline manifest owner; preserve or migrate with tests. |
| `src/bootstrap/run_metadata.py` | Current run metadata owner; preserve or migrate with tests. |
| `src/training_runtime/plan.py` | Explicit migration seam for current runtime plan, packing owner, collator family, and Stage-2 namespace invariants. |
| `src/metrics/events.py` | Preserve canonical metric identity, reducer, alias, and flattening semantics. |
| `src/detection/template.py` | Current strict template authority; new codec must wrap/migrate it with parity tests. |
| `src/detection/tokenization.py` | Current chat-template/token-alignment owner; new encoding must preserve its contracts. |
| `src/detection/*` | Reuse or migrate current compact recursive CE pieces into the new owners without preserving old rejected surfaces. |
| `src/trainers/*` | Remove rejected objectives and route new losses through the bridge/runner. |
| `src/trainers/rollout_matching/matching.py` | Current Hungarian/matching owner; wrap behind assignment strategy before replacing. |
| `src/trainers/rollout_matching/parsing.py` | Current CoordJSON rollout parser/appender owner; wrap as legacy `coordjson` rollout codec and add compact-full parity before compact checkpoints use Stage-2. |
| `src/trainers/stage2_two_channel/target_builder.py` | Current Stage-2 target-building owner; bridge or retire through new planners with parity tests. |
| `src/infer/compact_grammar.py` | Current compact-full decode guard owner; reuse only for optional Stage-2 diagnostic/control probes, not the default compact-full training rollout path. |
| `src/infer/pipeline.py` and `src/infer/backends.py` | Current compact-full infer/eval evidence path; use as parity reference for Stage-2 compact-full rollout I/O. |
| `configs/**` | Remove rejected knobs and add strict surface configs. |
| `docs/training/**` | Update current guidance after implementation. |
| `progress/index.yaml` | Demote historical evidence statuses after cleanup. |

Exact paths may be adjusted during implementation if repo inspection shows a
nearby existing owner is cleaner. Any adjustment must preserve the same
boundaries.

## Task 0: Preflight Snapshot And Scope Guard

**Files:**

- Read: `docs/AGENT_INDEX.md`
- Read: `docs/PROJECT_CONTEXT.md`
- Read: `docs/SYSTEM_OVERVIEW.md`
- Read: `docs/IMPLEMENTATION_MAP.md`
- Read: `progress/explorations/2026-05-15_training_infrastructure_architecture_decisions.md`
- Read: `docs/superpowers/specs/2026-05-15-unified-training-infrastructure-architecture-design.md`

- [ ] **Step 1: Check dirty state**

Run:

```bash
git status --short
```

Expected: unrelated user changes may exist. Do not revert or stage unrelated
changes.

- [ ] **Step 2: Inventory rejected mechanisms**

Run:

```bash
rg -n "loss_duplicate_burst_unlikelihood|adjacent_repulsion|eos_loosen|force.*continu|stop_gate|stop_signal_damping" src configs docs tests --glob '!progress/**' --glob '!docs/superpowers/**'
```

Expected: all live references are classified into remove, preserve-diagnostic,
historical-reader, or staged-legacy categories before editing.

- [ ] **Step 3: Inventory Stage-2 matching surface**

Run:

```bash
rg -n "hungarian|linear_sum_assignment|rollout_aligned|stage2_two_channel|duplicate_filter|duplicate" src configs docs tests --glob '!progress/**' --glob '!docs/superpowers/**'
```

Expected: executable Stage-2/Hungarian paths are identified and marked
staged-legacy rather than deleted in Task 1.

- [ ] **Step 4: Record the classification table**

Update the implementation worklog or cleanup PR description with columns:

```text
symbol_or_path | category | action | verification
```

Expected: every live hit from Steps 2 and 3 has an action before code removal
starts.

## Task 0.5: Ground Tests In Current APIs And Stable Contracts

**Files:**

- Read: `src/config/loader.py`
- Read: `src/config/schema.py`
- Read: `src/trainers/teacher_forcing/module_registry.py`
- Read: `src/metrics/events.py`
- Read: `src/detection/template.py`
- Read: `src/detection/tokenization.py`
- Read: `src/training_runtime/plan.py`
- Read: `src/trainers/rollout_matching/matching.py`
- Read: `src/trainers/stage2_two_channel/target_builder.py`
- Read: `openspec/specs/stage2-ab-training/spec.md`
- Read: `openspec/specs/teacher-forcing-objective-pipeline/spec.md`
- Read: `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
- Read: `openspec/specs/trainer-metrics-components/spec.md`
- Read: `docs/ARTIFACTS.md`
- Read: `docs/IMPLEMENTATION_MAP.md`

- [ ] **Step 1: Map representative tests to real current APIs**

Before writing tests, create an implementation worklog table:

```text
planned assertion | current import/helper/config path | future owner | migration mode
```

Expected: no task relies on placeholder fixtures such as
`active_objective_registry` or `load_training_config` unless those helpers are
created deliberately in the same task.

- [ ] **Step 2: Check stable-contract contradictions**

Run:

```bash
rg -n "loss_duplicate_burst_unlikelihood|duplicate_burst|adjacent_repulsion|eos_loosen|force.*continu|forced_continuation|stop_gate|tail_append|insertion_order|hungarian|linear_sum_assignment" openspec/specs docs/training docs/ARTIFACTS.md docs/catalog.yaml
```

Expected: any stable spec or current doc requiring a removed mechanism,
ordering default, or Hungarian behavior is classified as an OpenSpec/docs
migration prerequisite, not ordinary cleanup.

Known duplicate-burst UL blockers must be recorded explicitly before Task 1:

```text
openspec/specs/stage2-ab-training/spec.md
openspec/specs/teacher-forcing-objective-pipeline/spec.md
openspec/specs/teacher-forcing-unified-loss-registry/spec.md
openspec/specs/trainer-metrics-components/spec.md
docs/training/STAGE2_RUNBOOK.md
docs/training/METRICS.md
src/config/schema.py
canonical Stage-2 configs
tests that currently require duplicate UL
```

Expected: no implementation slice may delete
`loss_duplicate_burst_unlikelihood` code/config/schema support until these
surfaces are migrated, superseded, or explicitly deferred without behavioral
contradiction.

- [ ] **Step 3: Define current-owner migration mode**

For each proposed new owner, record one of:

```text
wrap existing owner first
migrate existing owner with import-compatible facade
replace existing owner after parity tests
new owner because no current owner exists
```

Expected: template, tokenization, metrics, runtime plan, and Stage-2 target
building all have explicit migration modes before implementation starts.

## Task 0.75: Resolve Stable-Contract And Legacy Stage-2 Gates Before Cleanup

**Files:**

- Modify as needed: `openspec/specs/stage2-ab-training/spec.md`
- Modify as needed: `openspec/specs/teacher-forcing-objective-pipeline/spec.md`
- Modify as needed: `openspec/specs/teacher-forcing-unified-loss-registry/spec.md`
- Modify as needed: `openspec/specs/trainer-metrics-components/spec.md`
- Modify as needed: `docs/training/STAGE2_RUNBOOK.md`
- Modify as needed: `docs/training/METRICS.md`
- Read/Modify as needed: `src/config/schema.py`
- Read: `configs/stage2_two_channel/smoke/a_only.yaml`
- Read: representative `configs/stage2_two_channel/prod/*.yaml`
- Test: `tests/test_stage2_ab_config_contract.py`
- Test: `tests/test_stage2_two_channel_training.py`
- Test: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 1: Name the duplicate-burst UL migration**

Update or supersede every stable contract that still requires
`loss_duplicate_burst_unlikelihood` before Task 1 deletion starts.

Expected: stable specs, current docs, schema invariants, canonical configs, and
tests all agree whether duplicate-burst UL is active, legacy-only, removed, or
diagnostic-only.

- [ ] **Step 2: Prove legacy Stage-2 loads before cleanup**

Run representative current Stage-2 config parsing plus:

```bash
conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py tests/test_stage2_two_channel_training.py tests/test_training_runtime_sft_integration.py -q
```

Expected: current runnable Stage-2/Hungarian path passes before deletion. If a
test is intentionally changed by the stable-contract migration, record the old
assertion, new assertion, and OpenSpec/doc line that justifies it.

- [ ] **Step 3: Repeat legacy Stage-2 gate immediately after cleanup**

After Task 1 deletion, rerun the same config parse and test set, plus any new
absence tests.

Expected: cleanup either preserves runnable Stage-2 or fails with an explicit
migration error documented by the updated stable contracts.

## Task 0.8: Freeze Artifact, Provenance, And Diagnostic Contracts

**Files:**

- Read/Modify as needed: `docs/ARTIFACTS.md`
- Read/Modify as needed: `docs/IMPLEMENTATION_MAP.md`
- Read/Modify as needed: `src/bootstrap/experiment_manifest.py`
- Read/Modify as needed: `src/bootstrap/pipeline_manifest.py`
- Read/Modify as needed: `src/bootstrap/run_metadata.py`
- Test: `tests/test_experiment_manifest_file.py`
- Test: `tests/test_run_manifest_files.py`
- Test: `tests/test_run_metadata_file.py`
- Test: `tests/test_dependency_provenance.py`

- [ ] **Step 1: Preserve rank-0 experiment artifacts**

Record the current artifact contract and owner for:

```text
resolved_config.json
runtime_env.json
effective_runtime.json
pipeline_manifest.json
experiment_manifest.json
run_metadata.json
train_data_provenance.json
eval_data_provenance.json
source-config copies
```

Expected: new `run` / `surface` / `artifacts` / `runtime` config hierarchy maps
to the existing bootstrap owners or names an explicit replacement plus tests.

- [ ] **Step 2: Preserve Stage-2 eval artifact materialization**

Verify `rollout_matching.eval_detection.materialize_artifacts: true` remains
default-on for eval-enabled Stage-2 runs, and that
`eval_detection/step_<global_step>/` preserves:

```text
gt_vs_pred.jsonl
gt_vs_pred_scored.jsonl
infer_summary.json
metrics.json
per_image.json
raw_rollouts.jsonl
pred_token_trace.jsonl when trace metadata is available
```

Expected: no Stage-2 executable-path migration, resolver default flip, or
observability rewrite can pass review without preserving or deliberately
migrating these artifacts.

- [ ] **Step 3: Map diagnostic artifact compatibility**

Create a compatibility table from current diagnostics to future bounded
diagnostic writers:

```text
monitor_dumps/
prepare_failures/
raw_rollouts.jsonl
pred_token_trace.jsonl
guarded eval/post-op artifacts
duplicate/EOS diagnostic probes
```

Expected: cleanup keeps high-value structured diagnostics, not only scalar
metrics.

- [ ] **Step 4: Add catalog/progress parity check**

Check `docs/catalog.yaml`, `progress/index.yaml`, and relevant category READMEs
for status drift such as `running-ablation` versus `concluded-negative`.

Expected: removed or negative mechanisms are not advertised as active current
guidance.

## Task 1: Remove Rejected Training Mechanisms

**Prerequisite:** Task 0.75 and Task 0.8 are blocking gates. Do not start
duplicate-burst UL, adjacent-repulsion, Stage-2 config, or observability cleanup
until known stable-contract migrations, current Stage-2 legacy checks, and
artifact/provenance preservation checks are recorded.

**Files:**

- Modify/Delete: live source files containing duplicate-negative training.
- Modify/Delete: live source files containing adjacent-repulsion training.
- Modify/Delete: live training-time EOS loosen / forced-continuation / stop-gate code.
- Modify: configs that expose removed mechanisms.
- Modify/Delete: tests that positively preserve removed training behavior.
- Create/Modify: absence tests for removed mechanisms.
- Modify: current docs that advertise removed mechanisms.

- [ ] **Step 1: Add absence tests before deletion**

Create or update `tests/test_removed_training_mechanisms_absent.py` with tests
that assert removed objective ids are not present in current active registries,
schema allowlists, active configs, current docs/catalog routes, or new metric
writers.

Representative assertions should be grounded in current APIs such as
`OBJECTIVE_MODULE_CATALOG`, `TrainingConfig.from_mapping`, and real temp-YAML
`ConfigLoader` paths. Do not use placeholder fixtures unless the same task
creates them.

Representative shape:

```python
import pytest

REMOVED_KEYS = (
    "loss_duplicate_burst_unlikelihood",
    "adjacent_repulsion",
    "eos_loosen",
    "forced_continuation",
    "stop_gate",
)


def test_removed_objectives_are_not_registered():
    from src.trainers.teacher_forcing.module_registry import OBJECTIVE_MODULE_CATALOG

    for key in REMOVED_KEYS:
        assert key not in OBJECTIVE_MODULE_CATALOG


def test_stage2_objective_list_rejects_duplicate_burst():
    from src.config.schema import TrainingConfig

    payload = {
        "stage2_ab": {
            "pipeline": {
                "objective": [
                    {"name": "loss_duplicate_burst_unlikelihood", "weight": 1.0}
                ]
            }
        }
    }
    with pytest.raises(ValueError, match="loss_duplicate_burst_unlikelihood"):
        TrainingConfig.from_mapping(payload)
```

Expected before implementation: these tests should fail against the current
registry/config/doc surfaces that still expose removed mechanisms.

- [ ] **Step 1B: Add nested-path absence coverage**

Cover removed mechanisms at their real nested locations:

```text
stage2_ab.pipeline.objective[*].name
coord_reg.config.*
custom.coord_soft_ce_w1.*
docs/catalog.yaml current routes
docs/training/*.md current guidance
new metric writer keys
```

Expected: top-level removed-key tests are not considered sufficient.

- [ ] **Step 2: Delete duplicate-negative training support**

Remove live code/config/test/docs that implement or advertise
`loss_duplicate_burst_unlikelihood`.

Expected: duplicate filtering and duplicate diagnostics remain. No new training
run writes duplicate-negative loss metrics. This step is invalid until
`openspec/specs/stage2-ab-training/spec.md`,
`openspec/specs/teacher-forcing-objective-pipeline/spec.md`,
`openspec/specs/teacher-forcing-unified-loss-registry/spec.md`, schema
invariants, canonical Stage-2 configs, and tests no longer require the removed
module as active behavior.

- [ ] **Step 3: Delete adjacent-repulsion training support**

Remove live code/config/test/docs that implement or advertise adjacent
repulsion.

Expected: coordinate/regression objectives remain available only through the
approved target distribution family, not anti-duplication repulsion.

- [ ] **Step 4: Delete training-time EOS/continuation hacks**

Remove live training code/config/test/docs for EOS loosen,
forced-continuation, and stop-gate training mechanisms.

Expected: EOS/continue diagnostic probes and metrics remain under analysis or
diagnostics ownership and are not imported by training objectives.

- [ ] **Step 5: Run removal checks**

Run:

```bash
rg -n "loss_duplicate_burst_unlikelihood|adjacent_repulsion|eos_loosen|force.*continu|stop_gate|stop_signal_damping" src configs docs tests --glob '!progress/**' --glob '!docs/superpowers/**'
conda run -n ms python -m pytest tests/test_removed_training_mechanisms_absent.py tests/test_teacher_forcing_loss_catalog.py tests/test_training_config_strict_unknown_keys.py -q
```

Expected: remaining hits are only explicitly diagnostic, historical-reader, or
staged-legacy references. Absence tests pass.

## Task 2: Demote Historical Evidence And Clean Current Guidance

**Files:**

- Modify: `progress/index.yaml`
- Modify: `progress/explorations/README.md`
- Modify: `progress/diagnostics/README.md` if removed mechanisms are routed there.
- Modify: `docs/catalog.yaml` only if current routing advertises removed mechanisms.
- Modify: `docs/training/*.md` current guidance pages.

- [ ] **Step 1: Add explicit progress statuses**

Update progress routing using the repo's existing hyphenated status style.
Prefer extending current statuses instead of introducing a parallel underscore
taxonomy. Candidate statuses:

```text
active-reference
concluded-negative
mechanism-evidence
superseded
archive-only
```

Expected: duplicate-negative, adjacent-repulsion, and EOS loosen notes are not
current training guidance unless explicitly marked as diagnostic evidence.

- [ ] **Step 2: Label preserved diagnostics**

Update current docs so duplicate guards, EOS/continue probes, and historical
metric readers are described as diagnostics or analysis views.

Expected: no doc presents removed mechanisms as recommended training strategy.

- [ ] **Step 3: Run docs/index parse checks**

Run:

```bash
conda run -n ms python - <<'PY'
import yaml
from pathlib import Path
for path in [Path("progress/index.yaml"), Path("docs/catalog.yaml")]:
    if path.exists():
        yaml.safe_load(path.read_text())
        print(f"ok {path}")
PY
```

Expected: YAML parses successfully.

- [ ] **Step 4: Run current-guidance classification checks**

Run:

```bash
rg -n "loss_duplicate_burst_unlikelihood|adjacent_repulsion|eos_loosen|force.*continu|forced_continuation|stop_gate|stop_signal_damping" docs configs tests src --glob '!progress/**' --glob '!docs/superpowers/**'
```

Expected: remaining hits are classified in the Task 0 table as removed,
diagnostic-only, historical-reader, staged-legacy, or OpenSpec-migration
pending. No current doc should recommend a removed training mechanism as active
strategy.

## Task 3: Introduce SupervisionPlan And SupervisionContext

**Files:**

- Create: `src/training/supervision/plans.py`
- Create: `src/training/supervision/context.py`
- Create: `tests/test_supervision_plan_contract.py`

- [ ] **Step 1: Write semantic-only tests**

Add tests that construct Stage-1 and Stage-2 plans and assert they contain
semantic object entries and provenance but no rendered text, token ids, tensors,
or raw config dictionaries.

Representative test:

```python
def test_supervision_plan_is_semantic_only(stage1_compact_plan):
    assert stage1_compact_plan.sample_id == "fixture-stage1"
    assert stage1_compact_plan.stage == "stage1"
    assert stage1_compact_plan.template_id == "compact_full"
    assert not hasattr(stage1_compact_plan, "rendered_assistant_text")
    assert not hasattr(stage1_compact_plan, "input_ids")
    assert not hasattr(stage1_compact_plan, "model_inputs")
    assert not hasattr(stage1_compact_plan, "raw_config")
```

- [ ] **Step 2: Implement frozen dataclasses**

Implement provisional plan dataclasses and a small frozen context object. Keep
the naming localized because `SupervisionPlan` is still in discussion. Do not
create public `TargetPlan` APIs in this task. If Stage-2 needs a narrower
object-level helper, keep it private or name it as a `SupervisionPlan` subtype
until the naming decision is reopened.

Expected: tests prove per-example/per-channel ownership and context/plan split.

- [ ] **Step 3: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_supervision_plan_contract.py -q
```

Expected: pass.

## Task 4: Add TargetDistribution And Label-Position SupervisionSpan Skeleton

**Files:**

- Create: `src/training/supervision/distributions.py`
- Create: `src/training/supervision/spans.py`
- Create: `src/training/supervision/batch.py`
- Create: `tests/test_supervision_span_distribution_contract.py`

- [ ] **Step 1: Test accepted and rejected distributions**

Add tests for accepted kinds:

```text
hard_token
multi_positive_token
coordinate_soft_token
box_regression
```

Add tests that rejected kinds cannot be registered:

```text
duplicate_negative
adjacent_repulsion
forced_continuation
stop_gate
```

- [ ] **Step 2: Implement registered dataclass family**

Implement distribution dataclasses with explicit kind values and a registry
that validates objective support.

Expected: rejected kinds raise actionable errors.

- [ ] **Step 3: Implement region-level spans**

Implement `SupervisionSpan` with explicit target-token `label_positions` and
span roles.

Expected: spans store target-token `label_positions`, not shifted logit rows.
`PredictionCoordinateMapper` is the only component allowed to derive logit
positions. Full validation against encoded token roles is finalized after Task
5 adds `EncodedDetectionView`.

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_supervision_span_distribution_contract.py -q
```

Expected: pass.

## Task 5: Build Encoding Contracts

**Files:**

- Create: `src/training/templates/codec.py`
- Create: `src/training/templates/compact_full.py`
- Create: `src/training/templates/json_chat.py`
- Create: `src/training/encoding/view.py`
- Create: `src/training/encoding/model_inputs.py`
- Create: `src/training/encoding/example.py`
- Create: `src/training/sidecars.py`
- Create: `tests/test_compact_full_encoding_contract.py`
- Create: `tests/test_model_input_bundle_contract.py`

- [ ] **Step 1: Add compact-full golden fixture test**

Use a small fixture with three objects, mixed descriptions, nearby
non-duplicate boxes, and boundary coordinate tokens.

Expected fields:

```text
rendered_assistant_text
object_entries
schema_spans
description_spans
coordinate_slots
token_roles
label_positions
```

- [ ] **Step 2: Implement `EncodedDetectionView`**

Implement the authoritative tokenized view. Store rendered assistant text only
for diagnostics/provenance.

Expected: no objective/loss/assignment/duplicate/model-output state appears in
the view.

- [ ] **Step 3: Implement `ModelInputBundle` key registry**

Add strict backend allowed keys for ms-swift/Qwen3-VL and reject arbitrary
extras. Seed the registry from the current live detection/Qwen batch boundary;
do not start from a narrower invented list.

Initial allowed/preserved backend keys:

```text
input_ids
labels
attention_mask
token_type_ids
pixel_values
pixel_values_videos
image_grid_thw
video_grid_thw
second_per_grid_ts
position_ids
text_position_ids   # bridge-consumed auxiliary mRoPE input, not always raw-forwarded
cross_attention_mask
cache_position
past_key_values
use_cache
cu_seq_lens
cu_seq_lens_q
cu_seq_lens_k
max_length_q
max_length_k
logits_to_keep
output_router_logits
```

Also classify batch-contract keys that are not always model-forward keys:

```text
pack_num_samples
compute_loss_func
loss_scale
sidecar / supervision payloads
```

Expected: each key is explicitly classified as forwarded, bridge-consumed,
runner-owned-loss stripped, or sidecar-only.

Expected: sidecars such as `supervision_spans`, `assignment_result`, and
`duplicate_filter_result` cannot be placed in model inputs.

- [ ] **Step 4: Implement typed `TrainingSidecars`**

Group sidecars by supervision, diagnostics, dataset, and Stage-2 ownership.

Expected: sidecars are available to the bridge/runner but never forwarded to
the model.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_compact_full_encoding_contract.py tests/test_model_input_bundle_contract.py tests/test_detection_compact_full_template.py tests/test_chat_template_regression.py tests/test_compact_tokenizer_stop_contract.py -q
```

Expected: pass.

## Task 6: Add Span Adapters And CompactFullSpanProjector

**Files:**

- Create: `src/training/span_adapters/compact_projector.py`
- Create: `src/training/span_adapters/stage1_compact.py`
- Create: `src/training/span_adapters/stage2_compact.py`
- Create: `tests/test_compact_span_projector.py`
- Create: `tests/test_stage1_compact_span_adapter.py`
- Create: `tests/test_stage2_compact_span_adapter.py`

- [ ] **Step 1: Test shared compact projection**

Assert object-entry spans, schema token spans, description spans, coordinate
slots, stop positions, and label positions are projected from
`EncodedDetectionView`.

- [ ] **Step 2: Implement shared projector**

Implement compact-full syntax mechanics without stage-specific supervision
semantics.

- [ ] **Step 3: Implement Stage-1 compact adapter**

Map compact trie supervision plans to `SupervisionSpan` objects with
`MultiPositiveTokenDistribution` for object-entry trie regions and optional
coordinate distributions for coordinate slots.

- [ ] **Step 4: Implement Stage-2 compact adapter**

Map Channel-A/Channel-B supervision plans to spans while preserving assignment,
duplicate-filter, false-negative, and channel provenance.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_compact_span_projector.py tests/test_stage1_compact_span_adapter.py tests/test_stage2_compact_span_adapter.py -q
```

Expected: pass.

## Task 7: Add ObjectiveRunner And Objective Modules

**Files:**

- Create: `src/training/objectives/runner.py`
- Create: `src/training/objectives/types.py`
- Create: `src/training/objectives/token_ce.py`
- Create: `src/training/objectives/trie_ce.py`
- Create: `src/training/objectives/coord_soft_ce.py`
- Create: `src/training/objectives/box_regression.py`
- Create: `tests/test_objective_runner_math.py`
- Create: `tests/test_objective_precision_policy.py`

**Current-owner migration table:**

| Current owner | Target owner | Migration rule |
|---|---|---|
| `src/detection/loss.py` recursive CE / trie math | `src/training/objectives/trie_ce.py` plus runner adapter | Add numerical parity tests before switching ownership. |
| `src/detection/coord_soft_targets.py` IoU/CIoU-Gibbs targets | `CoordinateSoftTokenDistribution` and `CoordSoftCEObjective` | Preserve target family, full-vocab support/balance semantics, support masks, tau/sigma, and diagnostics. |
| `src/trainers/teacher_forcing/modules/token_ce.py` | `TokenCEObjective` | Preserve Stage-2 role masks, excluded coord positions, weights, and denominators before replacing. |
| `src/trainers/teacher_forcing/modules/bbox_geo.py` | `BoxRegressionObjective` plus `CoordinateSlotGroup` / `DecodedBoxTensor` | Preserve four-slot grouping, decode temperature, xyxy canonicalization, SmoothL1/CIoU terms, and group denominator. |
| `src/trainers/teacher_forcing/modules/bbox_size_aux.py` if present through catalog/config | Explicit optional subterm under `BoxRegressionObjective` or removed with absence tests | Do not silently drop active bbox-size losses; classify active configs first. |
| `src/trainers/teacher_forcing/modules/coord_reg.py` | Declarative dependency on coordinate slot / decoded-box tensors | Do not preserve hidden state sharing; make dependencies explicit. |

- [ ] **Step 1: Test `TokenCEObjective` against PyTorch CE**

Use fixed logits, target ids, masks, and weights.

Expected: objective loss matches the explicit PyTorch calculation.

- [ ] **Step 2: Test trie multi-positive CE**

Use fixed logits and sparse positive token sets.

Expected: loss covers the current recursive detection CE primitive, not only a
toy multi-positive support-mass loss. Tests must cover support plus balance
semantics, branch multiplicity weights, duplicate positive-token rejection,
fp32 `log_softmax` / `logsumexp`, label-position to logit-row mapping,
denominator policy, and explicit preservation or removal of the current
type-gate term.

- [ ] **Step 2B: Classify coordinate/regression subterms before migration**

Record the intended owner and metric policy for every current subterm:

| Current subterm | Initial decision | Future owner / policy |
|---|---|---|
| `bbox_smoothl1` | preserve | `BoxRegressionObjective`, fp32 geometry math. |
| `bbox_ciou` | preserve | `BoxRegressionObjective`, fp32 geometry math. |
| `bbox_size_aux` | preserve if any active config enables it; otherwise remove with absence tests | Explicit optional subterm, never hidden state. |
| `coord_token_ce` | preserve | Coordinate-token objective over explicit coordinate spans. |
| `coord_soft_ce` | preserve | `CoordSoftCEObjective` with target-family parity. |
| `coord_w1` | preserve if current configs use it; otherwise diagnostic-only | Explicit coordinate distribution diagnostic/loss mode. |
| `coord_gate` | diagnostic-only unless a stable spec requires active preservation | Metric/probe key, no hidden objective coupling. |
| `text_gate` | diagnostic-only unless a stable spec requires active preservation | Metric/probe key, no hidden objective coupling. |
| `adjacent_repulsion` | remove from training; keep diagnostic compatibility only | Absence tests plus optional historical-reader metric tolerance. |

Expected: each row has a parity test, absence test, or diagnostic writer test
before current `coord_reg` ownership is replaced.

- [ ] **Step 3: Test coordinate soft CE**

Use a normalized coordinate-token target distribution.

Expected: `iou_gibbs_v0` and `ciou_gibbs_v0` parity checks cover full-vocabulary
support/balance CE, support mass, balance loss, entropy, support bin count,
geometry-valid support masks, and bf16-logits-to-float32 behavior.

- [ ] **Step 3B: Test coordinate slot groups and box regression tensors**

Use one fixed four-slot box.

Expected: new runner output matches current bbox/regression behavior for
coord-logit gathering, decoded `pred_boxes_xyxy`, SmoothL1, CIoU, group
denominator, and group weights. The contract must expose `CoordinateSlotGroup`
or equivalent typed grouping rather than hidden objective-module state.

- [ ] **Step 4: Test objective-local normalization**

Combine token CE, trie CE, and regression objective results.

Expected: total loss is the sum of weighted normalized losses, not a global
mixed denominator.

- [ ] **Step 5: Implement runner and modules**

Implement objective-ready batch materialization, distribution-kind validation,
per-objective precision policy, metric event emission, and loss combination.

Precision requirements:

```text
input logits may be bf16
log-softmax/logsumexp/reductions run in float32 unless parity proves otherwise
IoU/CIoU-Gibbs target construction may use float64 where current behavior requires it
returned objective losses are float32 normalized tensors
gradients backpropagate to original logits
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_objective_runner_math.py tests/test_objective_precision_policy.py tests/test_teacher_forcing_token_ce.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_stage2_objective_atoms_projection.py -q
```

Expected: pass.

## Task 8: Add TrainerLossBridge And PredictionCoordinateMapper

**Files:**

- Create: `src/training/bridge/loss_bridge.py`
- Create: `src/training/bridge/coordinate_mapper.py`
- Create: `tests/test_prediction_coordinate_mapper.py`
- Create: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 1: Test causal coordinate mapping**

Assert label position `p` maps to logits position `p - 1`, index `0` is
invalid/masked, and span label positions validate against raw logits shape
through `PredictionCoordinateMapper`.

- [ ] **Step 2: Test sidecar stripping and model input preservation**

Use a fake model and a batch containing model inputs plus sidecars.

Expected:

```text
model called exactly once
sidecars not forwarded
input_ids preserved
pixel_values preserved
pixel_values_videos preserved when present
image_grid_thw preserved
video_grid_thw preserved when present
position_ids preserved when present
text_position_ids preserved or bridge-consumed when present
token_type_ids preserved when present
second_per_grid_ts preserved when present
cross_attention_mask preserved when present
past_key_values preserved when present
use_cache preserved when present
max_length_q preserved or bridge-consumed when present
max_length_k preserved or bridge-consumed when present
cache_position preserved when present
attention_mask preserved
full logits preserved
PredictionCoordinateMapper constructed
```

Expected: `text_position_ids` is allowed to be consumed to synthesize or augment
Qwen `position_ids`; the test must assert correct bridge behavior rather than
blind raw forwarding.

- [ ] **Step 2B: Test runner-owned loss mode**

Use a fake model that returns a deliberately wrong `outputs.loss = 999`.

Expected:

```text
labels stripped when runner owns loss
compute_loss_func stripped
loss_scale stripped
logits_to_keep rejected unless explicit projection is enabled
outputs.loss ignored
returned loss equals ObjectiveRunner total
```

- [ ] **Step 3: Implement bridge**

Implement the bridge around current ms-swift trainer seams. Remove labels from
model forward only when `ObjectiveRunner` owns canonical loss for the migrated
surface.

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_prediction_coordinate_mapper.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: pass.

## Task 9: Add ObservabilityService

**Files:**

- Modify: `src/metrics/events.py`
- Create: `src/training/observability/events.py` only for `DiagnosticEvent` or wrappers that import canonical `MetricEvent`
- Create: `src/training/observability/service.py`
- Create: `src/training/observability/legacy.py`
- Create: `tests/test_observability_events.py`
- Create: `tests/test_diagnostic_sampling_policy.py`

- [ ] **Step 1: Test typed metric events**

Assert metric events carry key, value, unit, reducer, stage, channel,
objective id, denominator, and provenance fields where applicable.

Expected: `rg -n "class MetricEvent" src tests` shows one canonical event
class. New observability code imports/reuses it instead of defining a second
incompatible metric type.

- [ ] **Step 2: Test diagnostic event sampling profiles**

Assert `off`, `standard`, and `debug` profiles emit the correct bounded
payloads.

- [ ] **Step 3: Test clean-write tolerant-read split**

Assert new metric writers reject removed mechanism keys, while the legacy
reader can parse and label old keys as legacy/removed.

Duplicate metrics must preserve reducer semantics:

```text
N_* duplicate counters sum across ranks
raw duplicate gauges use weighted mean or documented reducer
removed loss keys are rejected by new writers
legacy readers label old keys as legacy/removed
```

- [ ] **Step 4: Implement service and legacy adapter**

Implement event sinks, bounded diagnostic writers, metric flattening for
ms-swift reporting, and legacy Stage-2 flat metric adaptation where mapping is
known.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_observability_events.py tests/test_diagnostic_sampling_policy.py -q
```

Expected: pass.

## Task 10: Add Stage-2 Assignment, Duplicate Filtering, And Planning

**Files:**

- Create: `src/training/stage2/assignment.py`
- Create: `src/training/stage2/rollout_codec.py`
- Create: `src/training/stage2/rollout_prompting.py`
- Create: `src/training/stage2/rollout_artifacts.py`
- Create: `src/training/stage2/duplicate_filter.py`
- Create: `src/training/stage2/planners.py`
- Create: `src/training/ordering.py`
- Create: `tests/test_stage2_assignment_greedy_iou.py`
- Create: `tests/test_stage2_rollout_template_policy.py`
- Create: `tests/test_stage2_compact_full_rollout_io.py`
- Create: `tests/test_stage2_coordjson_rollout_legacy.py`
- Create: `tests/test_stage2_duplicate_filter.py`
- Create: `tests/test_stage2_supervision_planning_smoke.py`
- Create: `tests/test_object_ordering_strategy.py`

- [ ] **Step 0: Resolve Stage-2 stable-contract migration boundary**

Before changing executable Stage-2 behavior, audit current OpenSpec/docs for
duplicate-control ordering, duplicate-burst UL requirements, and Channel-B
ordering defaults.

Expected: if stable contracts still require duplicate-burst UL, pre-match
duplicate control, `tail_append` default, or Hungarian-backed behavior, create
or update the relevant OpenSpec/docs change before code removal/default flips.

- [ ] **Step 0B: Bridge current Stage-2 owners**

Wrap or extract current owners behind the new interfaces before building a
parallel synthetic-only path:

```text
src/trainers/rollout_matching/matching.py -> AssignmentStrategy adapter
src/trainers/rollout_matching/parsing.py -> legacy CoordJSON rollout codec adapter
src/trainers/stage2_two_channel/target_builder.py -> Stage2 supervision planner migration seam
src/infer/compact_grammar.py -> optional compact-full diagnostic/control decode guard
src/infer/pipeline.py / src/infer/backends.py -> compact-full infer parity reference
src/trainers/rollout_aligned_targets.py -> legacy compatibility adapter if still live
```

Expected: old executable trainers and new planning tests cannot silently
diverge.

- [x] **Step 0C: Add template-aware Stage-2 rollout I/O tests**

Add tests that prove Stage-2 rollout I/O cannot silently mix template families.
The tests must cover:

```text
compact_full prompt -> unconstrained compact_full decode policy -> compact_full parser -> compact_full append policy
coordjson prompt -> CoordJSON parser -> CoordJSON append policy
compact_full output rejected by CoordJSON-only parser path
CoordJSON output rejected by compact_full-only parser path
resolved rollout template recorded in diagnostics/artifacts
```

Expected: A compact-full checkpoint or config cannot be launched through the
legacy CoordJSON parser/appender without an explicit legacy compatibility
selection. `custom.json_format: standard` must not decide the rollout parser
for compact-full Stage-2. The accepted migration model is dual-surface:
`compact_full` canonical, `coordjson` explicit legacy, no implicit fallback.

- [x] **Step 0D: Implement `Stage2RolloutTemplatePolicy` and rollout codecs**

Implement a small policy object that resolves the rollout sequence family for
Stage-2 before rollout generation. Initial families:

```text
compact_full
coordjson
```

Expected: `compact_full` uses compact-full prompt construction, unconstrained
default rollout decoding, strict compact-full parsing, compact-full
false-negative append serialization, and compact-full artifact serialization.
Compact grammar decoding is allowed only as a labeled diagnostic/control probe.
`coordjson` wraps the current `src/trainers/rollout_matching/parsing.py`
behavior as an explicit legacy surface.

- [x] **Step 0E: Implement invalid/empty rollout fallback policy**

For compact-full Stage-2, implement the default policy:

```text
invalid_rollout_policy: fallback_gt_fn_append_only
```

Expected behavior:

```text
malformed model output -> no predicted survivors -> clean GT/FN append-only Channel-B target
empty valid object set -> clean GT/FN append-only Channel-B target
valid predicted objects -> normal duplicate filtering, assignment, and FN insertion
config/template parser mismatch -> hard validation failure, not fallback
fallback_loss_weight -> 1.0 by default, same as normal Channel-B correction
fallback provenance -> rollout_context=fallback_gt_fn_append_only
```

Expected metrics/artifacts:

```text
loss/B_fallback/*
rollout/invalid_fallback_gt_fn_count
rollout/invalid_fallback_gt_fn_rate
rollout/fallback_loss_share
rollout/fallback_dominance_warning
rollout/empty_valid_object_rate
rollout/parse_truncated_rate
rollout/parser_template_mismatch_rate
raw invalid rollout artifacts with fallback reason
```

Expected: fallback samples remain trainable correction signals, but they do not
count as valid rollouts and cannot make Gate 1 or Gate 2 pass. A monitoring
window with fallback samples above roughly 30-40% of Channel-B samples should
flag rollout-distribution health as degraded. `fallback_loss_weight` may exist
as an explicit ablation knob later, but the default is not weakened without
evidence.

Implementation note: the live Stage-2 compact-full path now uses the
template-aware codec and compact target builder directly. Malformed compact
output, empty compact output, and compact rows whose bboxes are dropped before
any valid survivor all route to `fallback_gt_fn_append_only`; compact explorer
fallback views remain in raw fallback metrics but are excluded from posterior
support denominators.

- [ ] **Step 1: Test greedy IoU assignment**

Use synthetic predictions and ground truth with known IoU ordering.

Expected: assigned pairs, unmatched predictions, unmatched GT, IoU scores, and
reason codes match the fixture.

- [ ] **Step 2: Test deterministic duplicate filtering**

Use duplicates around a stable survivor set.

Expected canonical survivor priority before a stable-contract migration:

```text
duplicate-policy evidence / explorer support / crowd exemption when available
higher confidence when available
stable input order tie-break
```

If the implementation intentionally uses assignment evidence such as “assigned
over unmatched,” that must be marked as a compatibility-sensitive Stage-2
contract change and covered by OpenSpec/docs plus golden parity tests.

- [ ] **Step 3: Test false-negative insertion and final ordering**

Use a fixture with accepted rollout survivors and one missing GT object.

Expected: Channel-B planner inserts the false negative, applies object ordering
after filtering/insertion, and records provenance.

Add two ordering snapshots:

```text
tail_append_legacy current/default compatibility
sorted future canonical target after migration approval
```

- [ ] **Step 4: Implement strategies and planners**

Implement `GreedyIoUAssignment`, `DuplicateFilter`, Channel-A/Channel-B
supervision planners, and shared object ordering.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_rollout_template_policy.py tests/test_stage2_compact_full_rollout_io.py tests/test_stage2_coordjson_rollout_legacy.py tests/test_stage2_assignment_greedy_iou.py tests/test_stage2_duplicate_filter.py tests/test_stage2_supervision_planning_smoke.py tests/test_object_ordering_strategy.py tests/test_stage2_two_channel_training.py tests/test_stage2_rollout_aligned.py -q
```

Expected: pass.

## Task 10.5: Stage-2 Legacy Protection Gate Before Resolver Migration

**Note:** Task 0.75 and Task 0.8 already run the pre-cleanup legacy and
artifact gates. This task repeats them before resolver/launcher migration, when
the chance of accidental artifact or routing drift is highest.

**Files:**

- Read/Modify only if needed: `src/sft.py`
- Read/Modify only if needed: `src/training_runtime/plan.py`
- Read: `configs/stage2_two_channel/smoke/a_only.yaml`
- Test: `tests/test_stage2_two_channel_training.py`
- Test: `tests/test_stage2_rollout_aligned.py`
- Test: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 1: Prove legacy Stage-2 still loads and runs**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_two_channel_training.py tests/test_stage2_rollout_aligned.py tests/test_training_runtime_sft_integration.py -q
```

Expected: current executable Stage-2/Hungarian path still passes before any
launcher/resolver default migration.

- [ ] **Step 2: Add greedy-IoU shadow planning smoke**

Add a shadow-mode config/test that exercises greedy-IoU planning without
changing the default runnable Stage-2 path.

Expected: greedy-IoU planning smoke passes and writes diagnostics, while the
legacy runnable path remains intact.

- [ ] **Step 3: Add A2 compact-full rollout I/O smoke**

Use the A2 random ET-RMP-CE compact-full adapter at `checkpoint-3664` as the
first real compact-full Stage-2 rollout I/O proof. Launch with the Qwen3-VL
base model and the adapter under `model.adapters`, not as a full model
checkpoint.

Required proof:

```text
Stage-2 launch exits successfully
resolved rollout template is compact_full
raw generated rollout contains at least one valid compact-full predicted object
raw output ends with <|im_end|> or another explicit compact-full stop contract
parser is compact_full, not CoordJSON
decode policy is unconstrained, not grammar-constrained
invalid/empty rollouts fall back to GT/FN append-only supervision but do not pass this gate
false-negative append policy serializes compact_full entries when needed
metrics distinguish launch health from valid-rollout health
```

Expected: this smoke is Gate 1 for compact-full Stage-2 rollout I/O wiring.
Use a tiny 2-4 sample scope. It must prove at least one valid predicted
compact-full object, no CoordJSON fallback, and preserved raw output artifacts.
If it fails with `valid_pred_objects_total=0` or a parser-template mismatch, do
not interpret assignment/duplicate/filter metrics as model quality. Gate 1
alone is not enough to claim training readiness.

2026-05-17 attempt:

- Added `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_1step.yaml`.
- Real HF Stage-2 launch exited successfully and wrote run manifests, train
  monitor dumps, eval monitor dumps, raw rollout artifacts, and COCO eval
  artifacts under
  `output/stage2_ab/smoke/compact_full_et_rmp_ce_ckpt3664_hf_1step/smoke_1step-compact_full-et_rmp_ce_ckpt3664-hf-unconstrained/v2-20260517-101433`.
- The run resolved to `compact_full`, `rollout_decode_policy=unconstrained`,
  `invalid_rollout_policy=fallback_gt_fn_append_only`, HF rollout/eval
  backends, and `checkpoint-3664` via `model.adapters`.
- The smoke did not pass Gate 1 readiness: tiny unconstrained samples produced
  compact object starts but no valid predicted objects, with malformed
  continuations such as tool-call/chat special tokens. Logged diagnostics
  surfaced `stage2/invalid_rollout=1.0`, `rollout/fn_appended_total=2.0`,
  `eval/runtime/coco_eval_ok=1.0`, and `eval/runtime/coco_counter_empty_pred=2.0`.
- Interpretation: launch/I/O wiring and fallback supervision are operational,
  but compact-full unconstrained rollout readiness remains unproven. Do not
  claim training-trajectory preservation from this smoke; run Gate 2 only after
  a Gate 1 smoke produces valid compact-full predictions.

- [ ] **Step 3B: Add A2 compact-full rollout readiness smoke**

After Gate 1 passes, run a small 16-32 sample readiness smoke with the same
unconstrained compact-full rollout path.

Required proof:

```text
sample_valid_pred_rate >= 0.75
parser_template_mismatch_rate = 0
parse_truncated_rate reported explicitly
empty_valid_object_rate reported explicitly
invalid_fallback_gt_fn_rate reported explicitly
raw rollouts and parsed objects materialized for manual inspection
compact grammar probes, if any, are diagnostic/control-only
fallback_loss_share reported separately
fallback dominance warning emitted if fallback exceeds 30-40% of Channel-B samples
```

Expected: Gate 2 is the minimum evidence required before real compact-full
Stage-2 training can treat rollout-quality, assignment, duplicate-filter, and
false-negative metrics as model-behavior signals.

- [ ] **Step 4: Re-check Stage-2 eval artifact materialization**

Run a unit writer test, artifact replay, or tiny eval-step smoke that proves
`rollout_matching.eval_detection.materialize_artifacts: true` is still
default-on and writes the required `eval_detection/step_<global_step>/` files.

Expected: `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `infer_summary.json`,
`metrics.json`, `per_image.json`, `raw_rollouts.jsonl`, and
`pred_token_trace.jsonl` when trace metadata is available.

- [ ] **Step 5: Block `src/sft.py` default flips until both paths pass**

Do not change `src/sft.py` routing/defaults or config discovery until the
legacy Stage-2 path and new shadow path both pass.

Expected: no implementation slice can pass review by only testing synthetic
planning while breaking current runnable Stage-2.

## Task 11: Add Config Resolver And Surface Pipelines

**Files:**

- Create: `src/training/pipelines/base.py`
- Create: `src/training/pipelines/stage1_json_ce.py`
- Create: `src/training/pipelines/stage1_compact_trie_ce.py`
- Create: `src/training/pipelines/stage2_two_channel.py`
- Create: `src/training/surfaces.py`
- Modify: `src/config/schema.py`
- Modify: `src/sft.py`
- Create: `tests/test_training_surface_resolver.py`
- Create: `tests/test_training_config_strict_unknown_keys.py`
- Create: `tests/test_objective_profile_resolution.py`

- [ ] **Step 0: Preserve current config loaders during migration**

Before adding strict `surface.id` schemas, prove representative current configs
still parse through current loaders:

```text
current Stage-1 compact smoke/prod config
configs/stage2_two_channel/smoke/a_only.yaml
representative Stage-2 prod config
```

Expected: current config parsing remains valid unless a config is intentionally
part of the rejected-mechanism removal and has an explicit migration error.

- [ ] **Step 1: Test top-level config hierarchy**

Assert canonical config domains are:

```text
run
surface
data
template
supervision
objectives
observability
artifacts
runtime
experimental
```

Expected: unknown top-level keys and removed mechanism keys fail fast.
`experimental` is allowed only as the strict escape hatch with required owner,
expiry, notes, and explicit surface/pipeline opt-in.

- [ ] **Step 2: Test surface-specific schema selection**

Assert `surface.id` selects Stage-1 JSON CE, Stage-1 compact trie CE, or Stage-2
two-channel schema and rejects irrelevant sections.

- [ ] **Step 3: Test keyed objective profile resolution**

Assert keyed objective entries resolve to a deterministic ordered objective
list and disabling one objective does not drop siblings.

- [ ] **Step 3B: Test experimental block contract**

Assert the top-level `experimental` block is rejected unless it includes:

```text
owner
expiry
notes
surface_or_pipeline_opt_in
```

Expected: production surfaces may reject `experimental` entirely unless they
explicitly opt in; one-off research knobs must not fall back to `custom.extra`
or unknown permissive keys.

- [ ] **Step 4: Implement resolver and pipeline protocol**

Implement `TrainingSurfaceResolver`, `ResolvedTrainingRun`, and the small
pipeline protocol in shadow mode first. The shadow resolver should opt into new
`surface.id` configs without becoming the only way current configs load.

- [ ] **Step 4B: Split launcher migration from resolver introduction**

Only after shadow resolver tests, legacy config passthrough tests, and new
surface smokes pass, migrate `src/sft.py` toward a thin launcher over the
resolved surface.

Expected: strict new schema work does not accidentally reject still-supported
current configs.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_training_surface_resolver.py tests/test_training_config_strict_unknown_keys.py tests/test_objective_profile_resolution.py tests/test_training_runtime_plan.py tests/test_training_runtime_profile.py tests/test_training_runtime_sft_integration.py tests/test_stage2_ab_config_contract.py tests/test_latest_training_config_contract.py -q
```

Expected: pass.

## Task 12: Add Golden Fixtures And Integrated Tiny Smokes

**Files:**

- Create: `tests/fixtures/training_architecture/compact_full_stage1_source.json`
- Create: `tests/fixtures/training_architecture/compact_full_stage1_expected.json`
- Create: `tests/fixtures/training_architecture/stage2_rollout_source.json`
- Create: `tests/fixtures/training_architecture/stage2_rollout_expected.json`
- Create: `tests/helpers/training_architecture_fixture_builder.py`
- Create: `tests/test_training_architecture_golden_thread.py`
- Create: `tests/test_training_architecture_tiny_smoke.py`

This task should be split into reviewable slices during execution:

```text
12A static fixtures only
12B builder helpers only
12C golden-thread assertions
12D tiny smoke wiring
```

Each slice should receive a main-session review before the next slice starts.

- [ ] **Step 1: Add static Stage-1 source and expected snapshots**

Snapshot only small semantic fields plus token ids where alignment matters.

Expected: fixture is human-reviewable and not a giant tensor archive.

- [ ] **Step 2: Add static Stage-2 rollout source and expected snapshots**

Include matched prediction, duplicate prediction, lower-quality duplicate, and
missing GT.

Expected: snapshot proves assignment, duplicate drop reasons, false-negative
insertion, and final object ordering.

- [ ] **Step 3: Implement builders through real code**

Builders should construct typed objects through the actual architecture rather
than hand-writing all internals.

- [ ] **Step 4: Add tiny integrated forward/backward smoke**

Use the smallest practical model/backend surface or a fake model where full
Qwen3-VL is too expensive for unit tests. If using a fake model, keep a
separate real-backend smoke command documented before production training.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_training_architecture_golden_thread.py tests/test_training_architecture_tiny_smoke.py -q
```

Expected: pass.

## Task 13: Update Current Documentation

**Files:**

- Modify: `docs/training/README.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/STAGE2_RUNBOOK.md`
- Modify: `docs/training/METRICS.md`
- Modify: `docs/ARTIFACTS.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/catalog.yaml`
- Modify: `docs/AGENT_INDEX.md`

- [ ] **Step 1: Update current training guidance**

Document `surface.id`, compact-full default, JSON baseline, objective profiles,
and cleanup decisions.

- [ ] **Step 2: Update Stage-2 runbook**

Document greedy-IoU assignment direction, duplicate filtering before target
realization, Channel-B false-negative insertion, and Hungarian as
migration-only until removal.

- [ ] **Step 3: Update metrics and artifacts docs**

Document `MetricEvent`, `DiagnosticEvent`, bounded diagnostic profiles,
resolved config artifacts, and clean-write/tolerant-read behavior.

- [ ] **Step 4: Run docs checks**

Run:

```bash
conda run -n ms python - <<'PY'
import yaml
from pathlib import Path
for path in [Path("docs/catalog.yaml"), Path("progress/index.yaml")]:
    if path.exists():
        yaml.safe_load(path.read_text())
        print(f"ok {path}")
PY
```

Expected: YAML parses successfully.

- [ ] **Step 5: Run current-guidance grep gate**

Run:

```bash
rg -n "loss_duplicate_burst_unlikelihood|adjacent_repulsion|eos_loosen|force.*continu|forced_continuation|stop_gate|stop_signal_damping" docs configs tests src --glob '!progress/**' --glob '!docs/superpowers/**'
```

Expected: remaining mentions are only explicit removal notes, absence tests,
historical-reader compatibility, diagnostic-only surfaces, or staged-legacy
Stage-2 references. Current docs must not present removed training mechanisms
as recommended active strategy.

## Validation Ladder

Run these levels in order before any production-scale training.

### Level 0: Config And Schema

Run:

```bash
conda run -n ms python -m pytest tests/test_training_config_strict_unknown_keys.py tests/test_training_surface_resolver.py tests/test_objective_profile_resolution.py tests/test_stage2_ab_config_contract.py -q
```

Expected: removed knobs fail, unknown keys fail, surface-specific schemas
reject irrelevant sections, objective profiles resolve deterministically,
`experimental` requires owner/expiry/notes/surface opt-in, and production
training surfaces preserve `data.geometry.do_resize: false` unless a stable
compatibility migration explicitly changes it.

### Level 0A: Stable Contracts And Artifact Provenance

Run:

```bash
conda run -n ms python -m pytest tests/test_experiment_manifest_file.py tests/test_run_manifest_files.py tests/test_run_metadata_file.py tests/test_dependency_provenance.py -q
```

Expected: resolved config, runtime env, effective runtime, pipeline manifest,
experiment manifest, run metadata, train/eval data provenance, and source
config copies remain present or are deliberately migrated with docs/specs.

### Level 1: Semantic Planning

Run:

```bash
conda run -n ms python -m pytest tests/test_supervision_plan_contract.py tests/test_stage2_supervision_planning_smoke.py tests/test_object_ordering_strategy.py -q
```

Expected: Stage-1 and Stage-2 semantic plans are correct without tokenizer or
loss dependencies.

### Level 2: Template And Encoding

Run:

```bash
conda run -n ms python -m pytest tests/test_compact_full_encoding_contract.py tests/test_model_input_bundle_contract.py -q
```

Expected: compact-full rendered bytes, token roles, coordinate slots, and
strict model-input keys are correct, including current Qwen/detection keys such
as `token_type_ids`, `second_per_grid_ts`, `cross_attention_mask`,
`past_key_values`, `use_cache`, `max_length_q`, and `max_length_k`.

### Level 3: Spans And Coordinates

Run:

```bash
conda run -n ms python -m pytest tests/test_supervision_span_distribution_contract.py tests/test_compact_span_projector.py tests/test_prediction_coordinate_mapper.py -q
```

Expected: span label positions, coordinate slots, and causal label-to-logit
mapping are valid.

### Level 4: Objective Math

Run:

```bash
conda run -n ms python -m pytest tests/test_objective_runner_math.py tests/test_objective_precision_policy.py tests/test_teacher_forcing_token_ce.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_stage2_objective_atoms_projection.py -q
```

Expected: CE, trie CE, coordinate soft CE, regression precision, denominators,
and weighted combination are correct.

### Level 5: Bridge And Forward

Run:

```bash
conda run -n ms python -m pytest tests/test_trainer_loss_bridge_qwen3vl_contract.py tests/test_training_runtime_sft_integration.py -q
```

Expected: sidecars are stripped, model inputs are preserved, full logits are
kept, and model forward is called once.

### Level 6: Tiny Training Smoke

Run:

```bash
conda run -n ms python -m pytest tests/test_training_architecture_tiny_smoke.py -q
```

Expected: one tiny compact-full Stage-1 batch runs forward/backward through the
new bridge and objective runner.

### Level 7: Stage-2 Rollout Planning Smoke

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_rollout_template_policy.py tests/test_stage2_compact_full_rollout_io.py tests/test_stage2_coordjson_rollout_legacy.py tests/test_stage2_assignment_greedy_iou.py tests/test_stage2_duplicate_filter.py tests/test_stage2_supervision_planning_smoke.py tests/test_stage2_two_channel_training.py tests/test_stage2_rollout_aligned.py -q
```

Expected: synthetic rollout predictions produce assignment results, duplicate
filter results, false-negative insertion, final object ordering, and Channel-B
supervision diagnostics. Template-family mismatches fail before assignment.

### Level 7A: Stage-2 Compact-Full Rollout I/O Smoke

Run a tiny real-backend Stage-2 smoke with the A2 random ET-RMP-CE compact-full
adapter at `checkpoint-3664`, loaded through `model.adapters` over the Qwen3-VL
base model.

Expected:

```text
process exits 0
resolved rollout template is compact_full
compact-full parser path is used
raw generated rollout contains at least one valid predicted object
raw generated rollout uses compact-full special tokens
CoordJSON parser/appender path is not used
default decode policy is unconstrained
invalid/empty rollout fallback policy is GT/FN append-only
false-negative append policy remains template-consistent
artifacts/logs distinguish valid-rollout health from launch health
```

If this smoke exits successfully but records `valid_pred_objects_total=0`,
`parse_truncated_rate=1.0`, or a parser-template mismatch, compact-full
Stage-2 readiness is not proven. Treat that as a rollout I/O failure, not a
greedy-IoU or duplicate-filter result.

Gate split:

```text
Gate 1 launch/I/O wiring:
  scope: 2-4 samples
  threshold: at least one valid predicted compact-full object
  required: no CoordJSON fallback, raw output artifacts preserved
  fallback: invalid/empty rollouts use GT/FN append-only supervision but do not pass the gate

Gate 2 rollout readiness:
  scope: 16-32 samples
  threshold: sample_valid_pred_rate >= 0.75
  required: parser_template_mismatch_rate = 0
  required: parse_truncated_rate, empty_valid_object_rate, and invalid_fallback_gt_fn_rate reported
  required: fallback_loss_share and fallback dominance warning reported
  required: raw rollouts and parsed objects available for manual inspection
```

Gate 1 proves wiring. Gate 2 is the minimum readiness evidence before real
compact-full Stage-2 training launch.

### Level 7B: Stage-2 Eval Artifact Preservation

Run a unit writer test, artifact replay, or tiny eval-step smoke for
`rollout_matching.eval_detection.materialize_artifacts`.

Expected: default-on eval materialization writes `gt_vs_pred.jsonl`,
`gt_vs_pred_scored.jsonl`, `infer_summary.json`, `metrics.json`,
`per_image.json`, `raw_rollouts.jsonl`, and `pred_token_trace.jsonl` when trace
metadata is available.

## Production-Run Gate

Before any production training launch, verify:

- resolved config artifact exists and contains `surface.id`, objective profile,
  precision policies, template id, packing/cache disabled status, and
  observability profile;
- image/geometry alignment still uses `do_resize=false`;
- compact-full token roles and coordinate slots are present in a golden
  encoded sample;
- no removed mechanism key appears in the resolved config;
- sidecars do not enter model forward;
- full Qwen3-VL/ms-swift multimodal keys are preserved, including video,
  grid, position, and cache-position keys when present;
- runner-owned loss mode strips hidden loss inputs and ignores `outputs.loss`;
- metrics include objective denominators and weighted losses;
- diagnostics are bounded under the selected profile;
- Stage-2 compact-full runs record the resolved rollout template, rollout
  parser, decode policy, and append policy in resolved config/artifacts;
- Stage-2 compact-full default decode policy is unconstrained; compact grammar
  probes are labeled diagnostic/control and cannot satisfy readiness alone;
- Stage-2 compact-full invalid/empty rollout policy is
  `fallback_gt_fn_append_only`; fallback rates are logged and fallback samples
  do not count as valid-rollout evidence;
- Stage-2 fallback supervision uses `fallback_loss_weight=1.0` by default,
  carries `rollout_context=fallback_gt_fn_append_only` provenance, logs
  `loss/B_fallback/*` and `rollout/fallback_loss_share`, and warns when
  fallback exceeds roughly 30-40% of Channel-B samples over a monitoring
  window;
- compact-full Stage-2 readiness has both real-backend A2 rollout gates:
  Gate 1 at 2-4 samples with at least one valid predicted compact-full object,
  and Gate 2 at 16-32 samples with `sample_valid_pred_rate >= 0.75` and zero
  parser-template mismatches;
- Stage-2 runs identify assignment strategy and duplicate-filter strategy in
  artifacts;
- Stage-2 runs identify object-ordering policy in resolved config/artifacts,
  including `tail_append_legacy` versus `sorted`;
- Stage-2 eval-enabled runs preserve `eval_detection/step_<global_step>/`
  artifact materialization;
- rank-0 experiment artifacts include resolved config, runtime env, effective
  runtime, pipeline manifest when applicable, experiment manifest, run metadata,
  train/eval provenance, and source-config copies;
- progress/docs/catalog routing no longer exposes removed mechanisms as active
  current guidance;
- OpenSpec/docs migrations are complete for any stable-contract changes;
- rollback path is the previous known runnable training config or commit.

## Residual Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Qwen3-VL multimodal forward breakage | Keep bridge tests focused on model-input preservation and run a tiny real-backend smoke before production. |
| Off-by-one loss coordinates | Centralize mapping in `PredictionCoordinateMapper` and validate spans before objective math. |
| YAML inheritance drops objectives | Use keyed objective authoring and resolved ordered runtime objective list. |
| Removed mechanisms reappear through old configs | Add removed-key config failure tests and registry absence tests. |
| OpenSpec/docs contradiction | Treat stable-contract changes as migration work before code removal/default flips. |
| Parallel owner drift | Wrap/migrate current owners before introducing replacement modules. |
| Diagnostics pollute training loop | Route all structured payloads through bounded `ObservabilityService` profiles. |
| Stage-2 loses runnable baseline too early | Keep old Hungarian path until greedy-IoU replacement passes smoke. |
| Stage-2 launch hides rollout-template mismatch | Gate compact-full Stage-2 on a template-aware A2 rollout I/O smoke with valid predicted objects, not only process exit or scalar losses. |
| Packing/cache silently corrupts spans | Keep both disabled until segment map and fingerprint contracts are implemented and tested. |

## Completion Definition

The refactor is complete when:

- cleanup tasks pass absence tests;
- all L0-L7 validation levels pass;
- current docs describe the new architecture and removed mechanisms correctly;
- `progress/index.yaml`, progress READMEs, and `docs/catalog.yaml` no longer
  route removed mechanisms as active current guidance;
- OpenSpec/docs contradictions for removed mechanisms, Stage-2 duplicate
  control, ordering defaults, and Hungarian removal are resolved or explicitly
  deferred before behavior changes;
- one canonical `MetricEvent` contract remains in use;
- no production config can enable removed training losses;
- Stage-1 JSON CE, Stage-1 compact trie CE, and Stage-2 two-channel surfaces
  resolve through typed runtime plans;
- one compact-full Stage-1 golden thread and one Stage-2 rollout-planning
  golden thread are stable;
- one compact-full Stage-2 rollout I/O smoke with the A2 checkpoint proves
  valid generated compact-full objects through the compact-full parser path;
- production-run gate checks pass for the first real training launch.

Plan complete and saved for review. Recommended execution mode after approval:
Subagent-Driven, one fresh worker per task group, with main-session review
between groups.
