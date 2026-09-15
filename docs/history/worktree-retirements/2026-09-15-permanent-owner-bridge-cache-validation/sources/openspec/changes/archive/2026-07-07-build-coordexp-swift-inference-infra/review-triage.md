# Inference Infrastructure Review Triage

## Scope

Change: `build-coordexp-swift-inference-infra`

Mode at review time: docs/spec/plan only. Source implementation was blocked
until the user approved source-study review and implementation kickoff.

Current status update, 2026-07-03: implementation and real-smoke work have since
landed in this worktree. This triage remains review provenance, not a live
source-implementation blocker.

## Review Round 1

Completion evidence:

- Contract/spec auditor: `019f22bf-e094-70d0-b7b3-90149d0dd6be`,
  completed, no timeout.
- Qwen/upstream tracer: `019f22c0-1e42-7170-960f-d3c7428ba7aa`,
  completed, no timeout.
- Eval/artifact auditor: `019f22c0-58be-7a13-a737-58797dad9605`,
  completed, no timeout.
- Architecture/module-boundary auditor:
  `019f22c0-8822-70c3-ad11-cb6e82a8a847`, completed, no timeout.
- Smoke/benchmark auditor: `019f22c0-b5d3-7db1-b6f3-a4d71a87f73f`,
  completed, no timeout.

### Contract And Spec Auditor

Accepted P1 findings:

- Token scoring policy needed exact selected-token roles, span source,
  ambiguity behavior, selected count expectations, and replay evidence.
- Required input/image failure policy was ambiguous against one-row-per-input
  artifact contracts.

Accepted P2 finding:

- Production `generation.batch_size: 1` should fail validation rather than only
  mark a run non-production.

Resolution:

- Specs now pin valid compact-object scoring to exactly four schema wrappers
  plus four coordinate tokens, `n_selected == 8`, parser-span trace mapping,
  and persisted replay handles.
- Required input/image metadata failures are terminal before benchmark-eligible
  row artifacts.
- Production batch size one now fails validation.

### Architecture And Module Boundary Auditor

Accepted P1 finding:

- Reusing owner modules could still leak training-owned API shapes through
  `TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, and
  `src.training.*`.

Accepted P2 findings:

- Source-study gates should be split into owner-boundary, backend/adapter, and
  legacy/eval checkpoints.
- Evaluator ownership should default to rebuilt `src.eval`, not an ambiguous
  bridge.

Resolution:

- Config/runtime specs and design now require owner-neutral APIs for Qwen
  loading, resolved config writing, artifact primitives, adapter identity, and
  embedding-delta identity.
- Tasks now include residue tests against training-owned imports.
- Evaluator default is a minimal rebuilt `src.eval` detection consumer. A
  legacy bridge requires explicit user approval.

### Eval And Artifact Auditor

Accepted P1 findings:

- Raw/scored artifacts must preserve exact row cardinality and cannot include
  extra diagnostic rows inside the evaluator-facing JSONL files.
- mAP consumer/provenance acceptance needed stricter artifact and score
  binding.

Accepted P2 findings:

- Selected-token evidence needed concrete replay handles.
- Score source/version and numeric score range needed validation.

Resolution:

- Raw and scored artifacts now preserve exactly one row per input/raw row.
  Extra diagnostics are sidecars only.
- Scored provenance now binds source raw identity, scored identity when
  available, prompt/decode/model/processor/template/parser/score identities,
  row binding evidence, and detection template id.
- Scores must be finite and in `[0.0, 1.0]`; source is non-empty and version is
  integer.

### Upstream Qwen/PEFT Tracer

Accepted P1 findings:

- Processor/model vision field parity needed an explicit gate.
- Adapter and embedding-delta identity checks needed exact PEFT and tokenizer
  validation points.

Accepted P2 finding:

- `checkpoint_reload.py` should be treated as checkpoint-final metadata
  guidance, not a generic base-only inference loader.

Resolution:

- Prompt/image spec now requires processor `patch_size`, `merge_size`,
  `temporal_patch_size` to match model vision config fields.
- Config/runtime spec and design now require adapter base identity,
  missing/unexpected key checks, `set_adapter`, `get_model_status()` status,
  active adapter list, irregular field checks, and unexpected merged-state
  checks.
- Embedding-delta validation now requires base config SHA, tokenizer SHA, token
  strings, and token ids.

### Smoke And Benchmark Auditor

Accepted P2 finding:

- Approval wording could be read as authorizing implementation before source
  studies exist.

Resolution:

- Design, tasks, and roadmap now separate source-study/probe approval from
  later source-implementation approval.

## Rejected Findings

None in round 1.

## Review Round 2

Completion evidence:

- Contract/spec auditor: `019f22d0-dbc6-71f0-87bd-c29ac794c580`,
  completed, no timeout.
- Qwen/upstream/source-boundary tracer:
  `019f22d1-08cc-72b2-943f-fa2229a99cd8`, completed, no timeout.
- Eval/artifact auditor: `019f22d1-3b0b-79e0-baa6-dd4511d7bb2b`,
  completed, no timeout.
- Implementation-roadmap auditor: `019f22d1-6971-7452-958a-1f08b94cdfc5`,
  completed, no timeout.

Accepted P1 findings:

- Legacy tests importing `src.infer` or legacy inference configs must be
  triaged so they do not pressure implementation into recreating the forbidden
  `src/infer/` package.
- Adapter reload/status policy and tests must route through the existing
  `src/adapters/` owner rather than being inlined into inference runtime.
- Owner-boundary residue tests must reject all unallowlisted `src.training.*`
  imports, not only `src.training.pipeline`, and must block use of
  `load_train_config()` or `ResolvedTrainConfig` in inference config paths.

Accepted P2 findings:

- Review completion evidence should be recorded in this triage artifact.
- PEFT adapter loading must capture `load_result` or equivalent
  missing/unexpected-key evidence; warning-only load paths are not sufficient.

Resolution:

- Tasks and roadmap now require legacy `src.infer` test triage with explicit
  port/retain/remove decisions.
- Roadmap now names `src/adapters/` and `tests/adapters/` as owned source and
  test surfaces for adapter reload/status policy.
- Config/runtime spec, tasks, roadmap, and decisions now define the V1
  `src.training.*` allowlist as empty unless source study and OpenSpec patch a
  concrete exception.
- Config/runtime spec, tasks, roadmap, and decisions now require captured PEFT
  `load_result` or equivalent evidence for adapter missing/unexpected-key
  checks.

Rejected findings:

None in round 2.

## Residual Gates

- Source studies and required probes were completed before implementation.
- OpenSpec must still be patched if future upstream evidence contradicts current
  contracts.
- Source implementation approval was granted after source-study review.
- Full validation-dataset benchmark handles are optional, not a live V1 gate.
- V1 inference/eval readiness is accepted from the fixed val200 run with scored
  artifacts and mAP/mRecall output from the named Swift evaluator consumer.
