## ADDED Requirements

### Requirement: Rollout-correction target construction is isolated from rollout execution

Stage-2 rollout-correction target construction SHALL be testable without
requiring rollout backend lifecycle, DDP phase orchestration, or full trainer
step execution.

Normative behavior:

- residual target construction MUST consume resolved correction policy, parsed
  rollout state, and GT state rather than directly controlling HF/vLLM backend
  lifecycle;
- duplicate filtering, triage, false-negative insertion, residual event
  construction, and target metadata assembly MUST be owned by trainer-side
  correction modules, not `src/infer`;
- rollout generation, DDP coordination, post-rollout packing, model forward,
  loss execution, and training metric projection MUST remain separate concerns;
- trainer compatibility adapters MAY remain during migration, but their
  contract MUST make target-construction inputs and outputs observable through
  targeted tests.

The target-construction contract MUST make the following boundary explicit:

- inputs include parsed rollout attempts, GT objects/source state, resolved
  correction policy, assignment decisions, duplicate decisions, and relevant
  rollout diagnostics;
- outputs include residual events, supervision metadata, target IR ingredients,
  target-construction metrics, and diagnostic records;
- excluded concerns include backend lifecycle, DDP barriers, optimizer state,
  model forward, template device movement, and pack scheduling.

#### Scenario: Target construction can be characterized without backend lifecycle

- **GIVEN** parsed rollout predictions, GT objects, and resolved correction
  policy
- **WHEN** rollout-correction target construction is exercised in a targeted
  test
- **THEN** the test does not need to initialize vLLM, a DDP phase monitor, or
  the full trainer step loop
- **AND** the produced supervision metadata remains compatible with the
  rollout-correction loss path.

### Requirement: Active Stage-2 surfaces do not expose retired A/B or Channel-B semantics

Unified Stage-2 SHALL keep old A/B and Channel-B naming out of public active
surfaces except where a rejection path or historical fixture explicitly proves
the names are removed.

Normative behavior:

- active config, manifest, metric, target metadata, and artifact surfaces MUST
  use `stage2_rollout_correction` or `rollout_correction` naming;
- old `stage2_ab`, `stage2_two_channel`, `channel_a`, `channel_b`, and
  `Stage2ChannelB*` names MUST NOT be part of new active public interfaces;
- tests that mention retired names MUST be absence, rejection, migration, or
  historical-compatibility tests rather than behavior-preservation tests;
- compatibility aliases MUST be quarantined or deleted after artifact
  consumers are checked.

The deletion taxonomy MUST distinguish:

- active public surfaces: configs, manifests, metrics, artifacts, docs, and
  non-archived specs;
- private implementation identifiers and filenames;
- rejection tests and absence tests;
- migration adapters;
- historical fixtures and archived docs/specs.

Remaining old-name matches after cleanup MUST be classified into one of those
categories and MUST NOT imply active Stage2-AB or Channel-B behavior.

#### Scenario: Retired channel naming cannot become a new public dependency

- **WHEN** active Stage-2 code, configs, docs, and tests are searched for
  public A/B or Channel-B surfaces
- **THEN** remaining matches are limited to explicit rejection,
  historical-compatibility, or deletion-gate contexts
- **AND** new rollout-correction artifacts and metrics use current names.

### Requirement: DDP and post-rollout packing remain trainer-owned explicit coordination contracts

Stage-2 DDP coordination and post-rollout packing SHALL remain trainer-owned
runtime concerns and SHALL be characterized before owner-shaped coordination is
narrowed.

Normative behavior:

- post-rollout packing is trainer-owned and MUST NOT be moved into
  target-construction or shared inference modules;
- coordination code MUST define the expected responsibilities for rank-local
  rollout segment buffers, pack scheduling, zero-pack behavior, shadow slots,
  final synchronization barriers, timeout/fail-fast behavior, and pack metrics;
- broad owner-shaped access MAY remain during migration only when covered by
  characterization tests for rank behavior and pack invariants;
- target construction may produce segments or metadata for packing, but MUST
  NOT own DDP barriers, model-forward execution, or optimizer-step behavior.

#### Scenario: Pack coordination owns rank behavior, not target construction

- **GIVEN** rollout-correction target construction has produced trainable
  segments and metadata
- **WHEN** post-rollout packing and DDP coordination consume those segments
- **THEN** rank-local buffer behavior, zero-pack handling, shadow slots, final
  sync barriers, and pack metrics are handled by trainer-owned coordination
  modules
- **AND** target construction remains independent of DDP barrier execution.

### Requirement: Stage-2 metric-bearing eval artifacts require exact source geometry and strict parser results

Stage-2 rollout-correction eval artifacts used for official metrics SHALL fail
before artifact materialization when source visual identity, geometry, parser
strictness, or provenance cannot be proven.

Normative behavior:

- metric-bearing Stage-2 eval rows MUST carry exact source image identity,
  image path when available, image ID when available, width, height, and source
  record identity;
- missing source dimensions MUST NOT be replaced by fabricated defaults;
- missing source image names MUST NOT be replaced by synthetic names such as
  `image_<idx>.jpg` for official eval;
- best-effort source lookup, post-hoc rescaling, or silent source-row recovery
  failures MUST NOT produce official `gt_vs_pred*.jsonl` rows;
- Stage-2 official eval MUST consume strict, `metric_bearing=true`,
  `salvage_recovered=false` parser results;
- diagnostic salvage and incomplete-geometry rows MAY be written only to
  diagnostic artifacts that declare `metric_bearing: false` and are rejected by
  official eval/comparison loaders.

#### Scenario: Missing Stage-2 eval geometry fails before official artifacts

- **GIVEN** Stage-2 eval is enabled
- **AND** an eval sample lacks exact source width, height, or image identity
- **WHEN** eval artifact materialization would write
  `eval_detection/step_<global_step>/gt_vs_pred_scored.jsonl`
- **THEN** materialization fails before official raw/scored artifacts or score
  sidecars are written
- **AND** no COCO/LVIS/mAP metric is computed from fabricated geometry.

#### Scenario: Salvage parser output stays diagnostic-only

- **GIVEN** a malformed Stage-2 eval rollout from which diagnostic salvage can
  recover an object
- **WHEN** official eval artifacts are materialized
- **THEN** the salvaged object is excluded from metric-bearing
  `gt_vs_pred*.jsonl` artifacts or the eval fails before materialization
- **AND** any diagnostic record that includes the salvage declares
  `metric_bearing: false`.
