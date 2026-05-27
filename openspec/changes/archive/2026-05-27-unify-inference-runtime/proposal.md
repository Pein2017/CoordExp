## Why

CoordExp currently has two inference runtimes in practice:

- offline inference/evaluation under `src/infer`, driven by `scripts/run_infer.py`
  and `infer.*` YAML;
- online Stage-2 rollout generation under trainer-owned rollout modules, driven
  by `rollout_matching.*` YAML.

Both paths construct detection prompts, prepare one-image visual inputs, select
HF or vLLM backends, decode sequences, parse CoordJSON/compact outputs, and
materialize diagnostics. Keeping those behaviors in separate layout trees makes
offline/online prompt drift, preprocessing drift, backend trace drift, and
metric artifact ambiguity too easy to introduce.

This change defines a shared inference/decode/preprocessing root so offline
evaluation, online Stage-2 rollout, rollout diagnostics, and score-bearing
metric workflows all consume the same prompt, decode, backend, trace, parsing,
and provenance contracts.

## What Changes

- Define `src/infer` as the canonical shared inference runtime root for
  detection prompt rendering, one-image visual preparation, decode request
  normalization, backend adaptation, generated-token logprob tracing, strict
  parsing, and decode artifacts.
- Preserve public authored config namespaces:
  - offline workflows continue to author `infer.*`;
  - Stage-2 rollout workflows continue to author `rollout_matching.*` for
    runtime/backend/decode/eval settings.
- Map both config families into common internal request/result/policy objects.
- Require HF and vLLM paths to normalize generated outputs into one canonical
  result object, including generated-sequence logprob traces whenever
  `trace_logprobs=true`.
- Hard-fail trace-required paths when generated token IDs, token text, and
  token logprobs are missing, non-finite, silently clipped, padded, or shape
  mismatched after the allowed backend adaptation.
- Make metric-bearing parsing strict-only. Diagnostic salvage may exist only in
  explicitly non-metric diagnostic artifacts.
- Require prompt, decode, model-identity, and score-policy fingerprints where
  those axes affect caches, artifacts, comparison, Stage-2 rollout
  supervision, or score-aware evaluation.
- Preserve raw/scored artifact separation: `gt_vs_pred.jsonl` remains raw and
  unscored; `gt_vs_pred_scored.jsonl` and scored guarded companions carry
  score provenance.
- Preserve current CoordExp/ms-swift vLLM adapter sync semantics while moving
  lifecycle and provenance ownership into the shared backend adapter layer.
- Supersede older active full-sync materialization defaults for unified
  Stage-2 rollout-correction server training: the active server path uses
  official adapter sync plus CoordExp coord-row sync, not native full-sync
  materialization.
- Remove overlapping active layout/import surfaces during implementation:
  trainer-local rollout runtime modules and legacy infer backend/engine modules
  are not kept as compatibility shims.
- Require historical artifacts without the new provenance fingerprints to be
  explicit inspection-only data unless a migration/stamping tool can reconstruct
  exact provenance from existing metadata.

## Capabilities

### New Capabilities

- `shared-inference-runtime`: owns the common detection prompt, decode,
  backend, trace, parsing, provenance, and artifact contract used by offline
  inference/eval and online Stage-2 rollout.

### Modified Capabilities

- `inference-engine`: offline inference MUST call the shared runtime and emit
  shared fingerprints/results/artifacts.
- `inference-pipeline`: eval/viz pipeline stages MUST preserve raw/scored
  artifact separation and reject non-metric diagnostic salvage from official
  metric inputs.
- `stage2-rollout-correction`: online rollout generation MUST call the shared
  runtime while retaining `rollout_matching.*` as the authored runtime config
  namespace and requiring verified prompt-token/visual parity for trainable
  rollout-correction segments.
- `runtime-architecture-refactor-program`: architecture cleanup MUST delete
  overlapping active inference/rollout import surfaces rather than preserving
  redundant layouts.

## Impact

- Affected code:
  - `src/infer/`
  - `src/trainers/stage2_rollout_correction*.py`
  - `src/trainers/stage2_rollout_runtime.py`
  - `src/trainers/rollout_runtime/`
  - current caller-facing `src/infer/compact_grammar.py` and
    `src/infer/stop_pressure.py` import surfaces
  - `src/bootstrap/`
  - `src/training_runtime/`
  - `src/sft.py`
  - `src/datasets/` prompt/processor call sites
  - `src/eval/`
  - `scripts/`
  - analysis/callback utilities that currently import legacy inference or
    rollout runtime helpers
- Affected configs/docs/specs:
  - `configs/infer/**`
  - Stage-2 configs using `rollout_matching.*`
  - `docs/AGENT_INDEX.md`
  - `docs/IMPLEMENTATION_MAP.md`
  - `docs/ARTIFACTS.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/training/METRICS.md`
  - `docs/eval/WORKFLOW.md`
  - `docs/catalog.yaml`
  - relevant stable OpenSpec specs after this change is implemented
- Affected tests:
  - inference config/engine/pipeline tests
  - Stage-2 rollout runtime and rollout-correction tests
  - vLLM server/client/sync/logprob tests
  - artifact/manifest/provenance tests
  - parser strictness and confidence/scored-eval tests

## Non-Goals

- Do not rename public `infer.*` or `rollout_matching.*` config namespaces in
  this change.
- Do not rename `scripts/run_infer.py`.
- Do not move COCO/LVIS metric computation into `src/infer`.
- Do not move Stage-2 residual target construction, duplicate filtering,
  greedy IoU assignment, DDP coordination, objective execution, or metric
  projection into `src/infer`.
- Do not add active multi-image generation abstractions; the supported
  detection generation contract remains exactly one image per sample.
- Do not keep legacy import modules as final compatibility aliases.
- Do not treat benchmark improvement as an OpenSpec validity gate.
