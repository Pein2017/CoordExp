## 1. Spec And Review Gate

- [x] 1.1 Draft proposal, design, delta specs, and tasks for inference
  data-parallelism.
- [x] 1.2 Run OpenSpec validation and markdown/config hygiene checks.
- [x] 1.3 Run independent review lanes for contract/spec, CUDA/runtime,
  artifact/merge/eval, and implementation-roadmap risks.
- [x] 1.4 Treat `openspec/changes/build-coordexp-swift-inference-infra/` as
  the active baseline contract or archive/sync it into stable specs before
  source implementation.
- [x] 1.5 Patch all accepted P0/P1 findings before source implementation.
- [x] 1.6 Record final review triage and mark the change approved for
  implementation only after the review lanes converge.

## 2. Planning And CUDA Binding

- [x] 2.1 Add failing tests for visible CUDA device discovery from
  `CUDA_VISIBLE_DEVICES`, including integer tokens and non-integer tokens.
- [x] 2.2 Add failing tests for CUDA-required behavior when no devices are
  visible in any non-dry inference path.
- [x] 2.3 Add failing tests for active rank computation:
  `min(visible_device_count, num_decode_batches)`.
- [x] 2.4 Add failing tests proving `generation.batch_size` is per-device and
  not divided by world size.
- [x] 2.5 Implement data-parallel planning helpers and private test overrides
  without adding public GPU-id config or CLI knobs.
- [x] 2.6 Add failing tests that empty input JSONL fails fast before active-rank
  execution.
- [x] 2.7 Add failing tests that `debug.dry_run: true` does not require CUDA and
  does not load Qwen, while every non-dry path fails before Qwen loading when
  CUDA is unavailable.

## 3. Shard Execution Primitive

- [x] 3.1 Add failing tests that a shard execution primitive processes only
  assigned row ids/indices.
- [x] 3.2 Add failing tests that the shard primitive writes only to a fixed
  shard output directory and never publishes root `gt_vs_pred*.jsonl`.
- [x] 3.3 Add failing tests that the shard primitive does not re-resolve
  collision-policy run directories.
- [x] 3.4 Implement the shard execution primitive by deepening the existing
  single-process pipeline flow without changing prompt, image, parsing,
  scoring, or artifact row semantics.

## 4. Worker Launch And Local Runtime

- [x] 4.1 Add failing tests that worker environments narrow
  `CUDA_VISIBLE_DEVICES` to exactly one assigned token.
- [x] 4.2 Add failing tests that workers validate exactly one visible CUDA
  device before model loading.
- [x] 4.3 Add failing tests that worker metadata records parent visible token,
  worker `CUDA_VISIBLE_DEVICES`, `torch.cuda.device_count()==1`,
  `torch.cuda.current_device()==0`, normalized logical device `cuda:0`, model
  first-parameter device, rank, and world size.
- [x] 4.4 Add failing tests or static guards proving multi-rank workers use
  fresh-interpreter subprocess/spawn semantics and not bare
  `multiprocessing.Process`, fork, or forkserver CUDA workers.
- [x] 4.5 Implement subprocess worker launch through a private module
  entrypoint, not a stable user CLI.

## 5. Row Sharding And Worker Execution

- [x] 5.1 Add failing tests for decode-batch block construction from ordered
  rows.
- [x] 5.2 Add failing tests for round-robin batch-block assignment across
  active ranks.
- [x] 5.3 Add failing tests that final row order is restored by `row_index`,
  independent of worker completion order.
- [x] 5.4 Implement shard-plan materialization with row ids, row indices, batch
  ids, rank assignments, per-device batch size, and fingerprint.
- [x] 5.5 Use the shard execution primitive for each assigned shard without
  changing prompt/parsing/scoring semantics.

## 6. Shard Artifacts And Strict Merge

- [x] 6.1 Add failing tests that each rank writes the complete shard artifact
  set
  under `run_dir/shards/rank-XXX/`.
- [x] 6.2 Add failing tests for merge rejection on missing row, duplicate row,
  row-order mismatch, missing shard artifact, failed worker, and identity
  fingerprint mismatch.
- [x] 6.3 Add failing tests that merged top-level raw/scored artifacts preserve
  exactly one row per input row in original order.
- [x] 6.4 Add failing tests that merged scored provenance sidecar binds shard
  hashes, merged
  hashes, row coverage, generation policy, model identity, processor identity,
  tokenizer identity, adapter identity/status, embedding-delta identity/status,
  template identity, parser policy, score policy, raw/scored SHA binding,
  row-count/row-identity binding, and row-local score provenance.
- [x] 6.5 Add failing tests that merged `pred_token_trace.jsonl` preserves every
  generated-token row and selected-token replay row required to recompute every
  merged scored prediction.
- [x] 6.6 Add failing tests that rank/device metadata is preserved in merged
  token trace and diagnostic sidecars while `gt_vs_pred_scored.jsonl` remains
  evaluator-compatible.
- [x] 6.7 Add failing tests that a deterministic two-shard merged artifact root
  is accepted by `src.eval.detection_consumer.evaluate_scored_detection_artifacts`.
- [x] 6.8 Add failing tests that merge output is staged and atomically
  published; on injected merge failure, root contains only terminal status
  evidence while shard directories are preserved.
- [x] 6.9 Implement strict merge for raw, scored, token trace, parse
  diagnostics, image plan, summary, manifest, and scored provenance.
- [x] 6.10 Implement a single merge identity vector/fingerprint helper with
  field-specific mismatch diagnostics.

## 7. Pipeline Integration

- [x] 7.1 Add failing tests that `python -m src.infer --config` keeps the direct
  single-process path when `active_ranks == 1`.
- [x] 7.2 Add failing tests that `active_ranks > 1` uses controller/worker
  execution and writes canonical top-level artifacts only after strict merge.
- [x] 7.3 Add failing tests that direct non-dry single-rank inference requires
  CUDA before Qwen model loading.
- [x] 7.4 Add failing tests that no metric reduction is run inside `src.infer`.
- [x] 7.5 Add failing tests that `backend.type: vllm` remains rejected as not
  implemented even though shard data shapes stay backend-neutral.
- [x] 7.6 Implement pipeline orchestration without importing training runtime
  ownership surfaces.

## 8. Verification And Real Smokes

- [x] 8.1 Run targeted pytest slices for inference config/runtime, pipeline,
  artifact writer/merge, scoring provenance, and evaluator consumption.
- [x] 8.2 Run `openspec validate add-coordexp-swift-infer-data-parallelism --strict`.
- [x] 8.3 Run `git diff --check`.
- [x] 8.4 Run a real two-GPU HF smoke on a tiny or val subset and verify shard
  artifacts, merged artifacts, row coverage, generation policy, and evaluator
  consumption of the merged scored artifact.
- [x] 8.5 Document the smoke evidence scope before any production
  data-parallel benchmark claim.

## 9. Post-Implementation Review Closure

- [x] 9.1 Add evaluator receipt binding for detection consumer metrics so
  smoke eval output is tied to raw/scored/provenance/run-manifest hashes.
- [x] 9.2 Clean stale evaluator outputs on merge failure so failed runs do not
  retain benchmark-looking `eval_detection` artifacts.
- [x] 9.3 Route controller missing/corrupt rank-local artifacts through strict
  merge terminal failure status instead of pre-merge shard identity reads.
- [x] 9.4 Add direct single-process runtime device evidence after model load.
- [x] 9.5 Convert malformed, unreadable, or semantically malformed rank-local
  merge JSON/JSONL into terminal merge-failure evidence instead of raw parser,
  filesystem, or type-cast errors.
