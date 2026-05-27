## ADDED Requirements

### Requirement: Stage-2 rollout generation uses the shared inference runtime

Stage-2 rollout-correction SHALL call the shared inference runtime for rollout
prompt preparation, backend decode, generated-token tracing, parser policy, and
decode provenance while preserving Stage-2 residual target construction outside
the shared runtime.

Normative behavior:

- authored rollout runtime/backend/decode/eval config remains under
  `rollout_matching.*`;
- Stage-2 objectives remain under `stage2_rollout_correction.*`;
- Stage-2 rollout code MUST map `rollout_matching.*` into shared decode
  request objects;
- Stage-2 trainer code retains ownership of live rollout attempt selection,
  residual correction target construction, duplicate filtering, greedy IoU
  assignment, unmatched-GT recovery, DDP coordination, objective execution, and
  metric projection;
- shared inference runtime code MUST NOT import or own Stage-2 target builder,
  residual boundary, target IR, greedy IoU matching, or duplicate-control
  training symbols;
- Stage-2 rollout artifacts MUST record shared prompt, decode, model identity,
  parser, and backend-sync provenance.

#### Scenario: Stage-2 rollout decode preserves rollout namespace

- **GIVEN** a Stage-2 config authored with
  `rollout_matching.rollout_backend: vllm`
- **AND** optional eval rollout backend selection remains under
  `rollout_matching.eval_rollout_backend`
- **WHEN** the trainer prepares online rollout generation
- **THEN** it constructs a shared decode request from `rollout_matching.*`
- **AND** it does not require users to author a new `infer.*` namespace for
  training rollouts.

### Requirement: Trainable Stage-2 rollout requires verified prompt-token and visual parity

Stage-2 trainable rollout-correction segments SHALL require verified prompt
token and visual metadata parity between backend rollout prefixes and local
teacher-forced target construction.

Normative behavior:

- backend-provided or faithfully reconstructed prompt token IDs MUST match the
  local teacher-forced prefix for trainable rollout segments;
- visual metadata needed to preserve coordinate/vision-token alignment MUST
  match when available;
- samples with prompt/visual parity mismatch MUST be dropped or fail before
  residual target construction;
- backend paths that cannot expose required parity metadata MAY be used for
  eval-only or diagnostic decode but MUST NOT feed trainable Stage-2 residual
  correction segments.

#### Scenario: Rollout prefix mismatch does not train

- **GIVEN** the backend returns prompt token IDs that differ from the local
  teacher-forced prefix
- **WHEN** Stage-2 prepares a trainable residual correction segment
- **THEN** the sample is dropped or the backend path fails
- **AND** residual target positions are not constructed from the mismatched
  rollout prefix.

### Requirement: Stage-2 vLLM rollout trace and sync provenance are strict

Stage-2 vLLM rollout generation SHALL satisfy the shared trace contract and
record backend sync identity as part of model identity provenance.

Normative behavior:

- generated-token logprob tracing MUST be complete when Stage-2 rollout
  diagnostics, score policies, or downstream metrics require it;
- vLLM server/colocate/local adapters MUST hard-fail on missing or malformed
  trace payloads rather than continuing with partial diagnostics;
- active Stage-2 train vLLM server rollout requires
  `rollout_matching.vllm.mode=server`,
  `rollout_matching.vllm.sync.mode=adapter`, and
  `rollout_matching.vllm.enable_lora=true`;
- native `sync.mode=full` materialization is superseded for active unified
  Stage-2 rollout-correction server training unless a later OpenSpec revives it;
- Stage-2 train vLLM server rollout requires `sync_policy:
  per_global_step`;
- fixed-checkpoint Stage-2 eval may use `sync_policy: static`;
- LoRA tensor digest, coord-row digest/status, sync step, and server identity
  MUST be reflected in model identity or detailed backend-sync metadata;
- rank-0 vLLM sync failure under DDP MUST abort all ranks.

#### Scenario: Stage-2 train vLLM server sync identity changes with adapter step

- **GIVEN** Stage-2 online training updates adapter weights between global
  steps
- **WHEN** the next train rollout syncs vLLM servers
- **THEN** rollout artifacts identify the synced step and adapter/coord-row
  digest
- **AND** the model identity fingerprint changes when the effective rollout
  policy changes.

### Requirement: Stage-2 eval materialization artifact contract is preserved

Routing rollout generation through the shared inference runtime SHALL preserve
the current Stage-2 eval artifact materialization contract.

Normative behavior:

- `rollout_matching.eval_detection.materialize_artifacts` remains the authored
  Stage-2 eval materialization control;
- when enabled, training-time eval artifacts are written under
  `training.output_dir/eval_detection/step_<global_step>/`;
- official metric-bearing Stage-2 eval MUST fail fast when artifact
  materialization is disabled or `training.output_dir` is unavailable, because
  official metrics must consume the score-provenanced
  `gt_vs_pred_scored.jsonl` artifact;
- the materialized directory MUST preserve the frozen artifact family:
  `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `infer_summary.json`,
  `metrics.json`, `per_image.json`, `raw_rollouts.jsonl`, and
  `pred_token_trace.jsonl` when token traces are available;
- `raw_rollouts.jsonl` MUST retain rollout text, generated token IDs when
  available, logprob trace metadata when requested, parser diagnostics, score
  metadata, and pre/post-score prediction views needed by offline-compatible
  inspection.

#### Scenario: Shared runtime rollout preserves Stage-2 eval directory

- **GIVEN** `rollout_matching.eval_detection.materialize_artifacts: true`
- **WHEN** Stage-2 eval runs after rollout generation has moved through the
  shared inference runtime
- **THEN** artifacts are written under
  `eval_detection/step_<global_step>/`
- **AND** the frozen Stage-2 eval artifact family remains present and
  offline-compatible.

#### Scenario: Official Stage-2 eval rejects in-memory metric computation

- **GIVEN** `rollout_matching.eval_detection.materialize_artifacts: false`
- **WHEN** Stage-2 eval would compute official COCO/LVIS-style metrics
- **THEN** eval fails before metric computation
- **AND** the diagnostic requires `materialize_artifacts: true` and a
  `training.output_dir` so scored-artifact provenance can be checked.
