# Review Triage

## Post-Implementation Closure

Implementation and verification closure is recorded in `tasks.md` and
`smoke-evidence.md`. All tasks in `tasks.md` are complete, targeted inference tests pass,
strict OpenSpec validation passes, `git diff --check` passes, and a real
two-GPU HF smoke completed with evaluator consumption of the merged scored
artifact.

Final implementation state: implemented and verified for the V1 HF
data-parallel inference scope. This remains smoke evidence only; production
benchmark claims still require a production-scope run that names the config,
dataset, checkpoint or adapter, visible GPU set, merged inference artifact root,
and evaluator metrics path.

## Scope

- Objective: converge and verify the
  `add-coordexp-swift-infer-data-parallelism` OpenSpec change through the V1
  source implementation.
- Mode: docs/spec/plan, source implementation, and post-implementation
  verification.
- Mutation scope: OpenSpec artifacts plus the inference/evaluator source and
  test files named by this change.
- Source of truth: this change directory plus the active baseline inference
  contract in `openspec/changes/build-coordexp-swift-inference-infra/`.
- Pre-implementation final state: approved to implement. This was not approval
  to make production benchmark claims before real multi-GPU smoke evidence.

## Review Lanes

Four independent read-only lanes reviewed the initial artifact set:

- Contract/spec auditor: OpenSpec format, baseline dependency, requirement
  coverage, and validation readiness.
- CUDA/runtime tracer: device discovery, CUDA binding, worker launch semantics,
  shard execution ownership, and direct-path behavior.
- Artifact/eval auditor: shard artifact families, strict merge, scorer
  provenance, trace replay, and evaluator compatibility.
- Roadmap mapper: task ordering, implementation readiness, and missing gates.

Two targeted re-review lanes then checked the revised contract:

- Runtime re-review: approved CUDA-required scope, dry-run exemption, fresh
  subprocess/spawn worker semantics, and per-worker logical `cuda:0` metadata.
- Artifact/contract re-review: approved baseline dependency, regenerated scored
  provenance sidecar, adapter and embedding-delta identity binding, trace replay
  completeness, rank/device metadata preservation, shard primitive boundaries,
  and evaluator compatibility checks.

## Accepted Findings

- P1: The baseline dependency on
  `openspec/changes/build-coordexp-swift-inference-infra/` was implicit.
  Resolution: proposal/design/tasks now require treating that change as the
  active baseline or syncing/archiving it before source implementation.
- P1: CUDA-required scope was ambiguous for direct single-rank inference.
  Resolution: every non-dry inference path requires CUDA before Qwen loading;
  `debug.dry_run: true` is exempt.
- P1: CUDA worker creation could be misread as permitting forked CUDA workers.
  Resolution: multi-rank execution requires fresh-interpreter subprocess/spawn
  semantics and forbids bare `multiprocessing.Process`, fork, or forkserver CUDA
  workers.
- P1: Shard execution ownership was underspecified.
  Resolution: workers receive resolved config plus assigned rows and a fixed
  shard output directory, process only assigned rows, and never publish root
  artifacts.
- P1: The merged scorer provenance sidecar could be confused with manifest
  metadata.
  Resolution: `gt_vs_pred_scored.jsonl.provenance.json` is regenerated and
  authoritative after merge; the manifest cannot substitute for it.
- P1: Strict merge identity needed adapter and embedding-delta identity.
  Resolution: the merge identity vector includes base, tokenizer, processor,
  template, dataset, generation, parser, scorer, adapter identity/status, and
  embedding-delta identity/status.
- P1: Token-trace merge completeness and rank/device metadata needed stronger
  contracts.
  Resolution: merged traces must preserve all rows needed to replay selected
  tokens and recompute scores, with rank/device metadata in trace and diagnostic
  sidecars while scored rows remain evaluator-compatible.
- P1: Failure publication semantics were ambiguous.
  Resolution: top-level artifacts are staged and atomically published only after
  validation; merge failure preserves shard directories and publishes only
  terminal failure evidence at the run root.
- P1: Empty input behavior was unspecified.
  Resolution: empty input JSONL fails fast before active-rank execution.

## Verification

Executed after revision:

```bash
openspec validate add-coordexp-swift-infer-data-parallelism --strict
```

Result: passed.

Additional hygiene:

```bash
git diff --check -- openspec/changes/add-coordexp-swift-infer-data-parallelism
```

Result: no whitespace errors.

Markdown fence and residue checks were also run against the change artifacts and
passed for the targeted planning surface.

## Gates At Pre-Implementation Triage Time

- Implement source changes with failing tests first.
- Run targeted inference pytest slices.
- Run a real 2-GPU HF smoke before any production data-parallel benchmark claim.
- Keep vLLM rejected in V1 while preserving backend-neutral shard data shapes.

These gates are now closed for implementation smoke scope in `tasks.md` and
`smoke-evidence.md`. The only remaining gate is production-scope evidence before
reporting a production data-parallel benchmark.

## Implementation Review Findings

Three independent post-implementation review lanes were run after the source
implementation and initial smoke. Accepted findings were patched before final
closure:

- P1: evaluator output under the smoke root was callable but did not carry a
  durable receipt binding it to the merged raw, scored, provenance, and
  run-manifest artifacts. Resolution: the detection evaluator now writes
  `evaluation_receipt.json` and embeds the same receipt in `metrics.json`;
  `benchmark_metric` follows run-manifest benchmark eligibility.
- P1: merge failure cleanup removed top-level inference artifacts but could
  leave stale `eval_detection` metrics from an earlier evaluator run.
  Resolution: merge failure cleanup removes known metric-bearing evaluator
  outputs under the run root.
- P1: controller-worker orchestration read rank-0 identity artifacts before
  entering strict merge, so a missing or corrupt rank-0 manifest could escape
  terminal-status publication. Resolution: strict merge now hydrates
  worker-owned runtime identity after shard validation, and the controller no
  longer pre-reads rank-0 artifacts.
- P2: direct single-process runs lacked post-load runtime device evidence.
  Resolution: direct manifests now record `parallelism.direct_runtime` with
  logical device, model first-parameter device when available, and CUDA probe
  evidence.
- P1: syntactically valid but semantically malformed rank-local JSONL rows could
  still escape as raw `ValueError`/`TypeError` from merge-side casts such as
  `row_index`, `generated_step_index`, prediction `score`, `object_span_id`, or
  selected-token replay fields. Resolution: strict merge now validates typed
  artifact fields with merge-local helpers and converts these failures into
  `ArtifactContractError`, preserving the terminal-status path.
- P2: the tiny real smoke had zero scoreable predictions, so it does not
  exercise real-run selected-token replay on non-empty predictions. Resolution:
  documented as residual coverage scope; unit fixtures cover replay
  recomputation, and production/val smokes should prefer at least one accepted
  prediction when validating score-replay behavior.
