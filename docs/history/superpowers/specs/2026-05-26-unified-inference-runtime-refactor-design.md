# Unified Inference Runtime Refactor Design

Status: proposed design approved for planning; implementation has not started.

Date: 2026-05-26

Owner: CoordExp inference, evaluation, and Stage-2 rollout infrastructure

Primary target: make `src/infer` the shared decode / inference / preprocessing
root for offline inference and online Stage-2 rollout, while deleting overlapping
layout modules and preserving stable public config and artifact contracts.

## Purpose

CoordExp currently has two inference implementations in practice:

- offline inference under `src/infer`, driven by `scripts/run_infer.py` and
  `infer.*` YAML;
- online Stage-2 rollout generation under `src/trainers/stage2_rollout_runtime.py`
  plus `src/trainers/rollout_runtime/`, driven by `rollout_matching.*` config.

Both paths build prompts, prepare visual inputs, select HF/vLLM backends, decode
sequences, parse model output, and materialize diagnostics. Because those
behaviors are owned in separate layout trees, prompt/template drift and backend
trace drift are easy to introduce.

This design collapses overlapping inference ownership into one shared root:
`src/infer`. Stage-2 rollout remains a training workflow, but it becomes a
client of shared inference runtime behavior instead of hosting a second online
decode stack.

## Approved Direction

The design uses a file-budgeted refactor:

- `src/infer` becomes the canonical shared inference/decode/preprocessing root.
- `src/trainers/rollout_runtime/` is deleted after its behavior is migrated.
- old inference/rollout import surfaces are hard-cut rather than preserved as
  long-lived compatibility shims.
- backend variants live behind one shared backend interface, not a directory
  tree of shallow variant files.
- public config namespaces remain stable:
  - offline inference continues to author `infer.*`;
  - Stage-2 continues to author `rollout_matching.*` for runtime/backend/decode/eval.
- internal code maps both config families into the same runtime request/result
  objects.
- all relevant callers are in refactor scope, including `src/`, `scripts/`,
  callbacks, analysis utilities, docs, configs, and tests.
- the shared prompt codec covers offline inference, online rollout, and
  teacher-forced dataset/training prompt-prefix encoding.
- stable artifact names and schemas remain contract-compatible unless a later
  OpenSpec delta explicitly changes them.

## Non-Goals

- Do not rename public `infer.*` or `rollout_matching.*` config namespaces.
- Do not rename `scripts/run_infer.py`.
- Do not move COCO/LVIS metric computation into the shared decode runtime.
- Do not move Stage-2 residual target construction, greedy-IoU assignment,
  duplicate filtering, DDP coordination, or objective execution into `src/infer`.
- Do not preserve a second backend/decode layout under
  `src/trainers/rollout_runtime/` as a compatibility comfort blanket.
- Do not keep `src.infer.engine`, `src.infer.backends`, or
  `src.trainers.rollout_runtime.*` as active import surfaces after migration.
- Do not preserve legacy imports merely to reduce churn in scripts, callbacks,
  tests, or analysis utilities; update those callers to the new shared runtime.
- Do not introduce a new top-level `src/decode/` or `src/inference_runtime/`
  package unless this design is explicitly reopened.
- Do not split backend variants into separate files unless the single-file
  backend module fails the complexity threshold described below.

## Stable Contract Guardrails

The refactor must preserve the stable contracts owned by:

- `openspec/specs/inference-engine/spec.md`;
- `openspec/specs/inference-pipeline/spec.md`;
- `openspec/specs/stage2-rollout-correction/spec.md`;
- `openspec/specs/runtime-architecture-refactor-program/spec.md`;
- `docs/eval/WORKFLOW.md`;
- `docs/training/STAGE2_RUNBOOK.md`;
- `docs/ARTIFACTS.md`.

In particular:

- `gt_vs_pred.jsonl`, `summary.json`, `resolved_config.json`, and
  `pred_token_trace.jsonl` remain the canonical offline inference artifact
  family.
- score-aware COCO/LVIS evaluation continues to consume
  `gt_vs_pred_scored.jsonl` when required.
- Stage-2 remains `stage2_rollout_correction`; old A/B or Channel-A/B semantics
  must not be reintroduced.
- `rollout_matching.*` remains the Stage-2 rollout runtime/backend/decode/eval
  config namespace even though implementation moves under `src/infer`.
- Qwen3-VL chat-template compatibility and `do_resize=false` geometry alignment
  remain non-negotiable.
- missing image dimensions must not be fabricated for metric-bearing artifacts.
- logprob-derived scoring, rollout diagnostics, or mAP-related score provenance
  must fail fast when a backend cannot return a complete generated-sequence
  logprob trace.
- metric-bearing parsing is strict-only. Diagnostic salvage parsing may exist
  only for debug artifacts and must never improve eval inputs.
- canonical eval input artifacts are metric-bearing-only. Non-metric diagnostic
  recoveries must be written to separate diagnostic artifacts.
- raw and scored eval artifacts remain strictly separate. `gt_vs_pred.jsonl`
  stays raw/unscored; score-bearing outputs are written to scored companions.
- semantic prompt/input rendering is single-owned. Dataset/training code may own
  sampling, caching, batching, and collation, but not an independent prompt
  template implementation.
- generation-time image normalization is single-owned by the shared prompt
  codec. Dataset modules retain storage, indexing, and cache ownership.
- this codebase's detection generation contract is exactly one image per
  sample. The refactor must not introduce active multi-image abstractions.
- semantic prompt/image ownership and backend transport serialization are
  separate: `prompt.py` builds normalized prompt bundles, `backend.py`
  serializes them for HF/Swift/vLLM transports.
- every prompt-bearing cache, resolved config, run manifest, and rollout/eval
  artifact must record the shared prompt policy fingerprint.
- every generation-bearing resolved config, run manifest, rollout/eval artifact,
  and decode diagnostic must record the shared decode policy fingerprint.
- every generation-bearing resolved config, run manifest, rollout/eval artifact,
  and diagnostic must record the model identity fingerprint separately from
  prompt and decode fingerprints.
- every score-bearing artifact must record score policy separately from decode
  policy through `score_policy_fingerprint`.
- comparable artifacts and caches fail fast when required fingerprints are
  missing. Diagnostic artifacts without fingerprints must declare
  `comparable: false`.
- old artifacts without fingerprints remain readable for historical inspection,
  but strict metric/comparison paths must reject them unless an explicit
  migration tool reconstructs and stamps the required provenance.
- strict offline-vs-online prompt parity requires exact prompt-token parity and
  visual-input metadata parity, not just prompt text/hash parity.
- `prompt_token_parity: unverifiable` may still be metric-bearing for ordinary
  eval, but it cannot support parity claims, template-alignment gates, or
  offline-vs-online equivalence assertions.
- Stage-2 trainable rollout-correction segments require
  `prompt_token_parity: verified`; unverifiable prompt parity is eval- or
  diagnostic-only for Stage-2.
- generated-sequence logprob tracing must support greedy and sampled decoding
  where the backend supports it, but metric score use is gated by explicit
  `score_policy`.

## vLLM Logprob Research Finding

As of 2026-05-26, official vLLM documentation supports treating vLLM as a
trace-capable backend:

- vLLM `SamplingParams` exposes `logprobs` for output-token logprobs and
  `prompt_logprobs` for prompt-token logprobs:
  `https://docs.vllm.ai/en/latest/api/vllm/sampling_params/`
- vLLM documents that OpenAI-style logprob behavior includes the chosen token's
  log probability in the returned logprob set.
- vLLM's logprob API defines per-position logprob containers, decoded-token
  metadata, and prompt/sample logprob containers:
  `https://docs.vllm.ai/en/latest/api/vllm/logprobs/`
- vLLM OpenAI-compatible serving has documented `--max-logprobs` support in the
  server configuration surface:
  `https://docs.vllm.ai/en/v0.7.0/serving/openai_compatible_server.html`

Design consequence: vLLM is not exempt from trace parity. HF, vLLM colocate,
vLLM local, and vLLM server adapters must all normalize their native response
formats into the same `DetectionDecodeResult` trace object whenever
`trace_logprobs=true`.

Because exact vLLM response schemas vary by client path, the backend adapter may
perform narrow conversion:

- convert token-id strings such as token-as-ID representations into integer
  token IDs;
- normalize backend choice/logprob payloads into canonical lists;
- normalize decoded-token text into the tokenizer's canonical token string
  representation;
- drop an explicit backend stop token only when that token is not part of the
  emitted generated text and the same stop policy is applied to token IDs,
  token text, and logprobs before validation.

The adapter must not silently clip or pad traces to hide shape mismatches.

## Target File Layout

The target shared inference root should be small:

```text
src/infer/
  __init__.py
  pipeline.py      # offline infer/eval/vis orchestration and resolved run dirs
  runtime.py       # shared InferenceRuntime: prompt -> backend -> parse
  prompt.py        # PromptPolicy, image loading, chat/template formatting
  backend.py       # HF/vLLM adapters and DecodeRequest/DecodeResult
  parsing.py       # strict/diagnostic CoordJSON + compact_full parser facade
  artifacts.py     # summaries, trace rows, resolved manifests
  checkpoints.py   # checkpoint / adapter resolution
  constraints.py   # compact grammar + stop-pressure constraints
  vis.py           # visualization helper
```

The target layout intentionally removes `src/infer/engine.py` and
`src/infer/backends.py` as active modules. If a migration slice needs temporary
files with those names, they must be deleted before the refactor is considered
complete.

### File Budget Rule

Keep this root flat unless a split clearly increases locality.

Default rule:

- one `backend.py`, not `backend/hf.py` plus `backend/vllm.py`;
- one `prompt.py`, not separate HF/Swift/vLLM prompt modules;
- one `parsing.py`, not separate offline and rollout parser files.

Split threshold:

- split only when one module crosses roughly 700-900 lines,
- and the split creates an independently testable adapter or policy,
- and the split does not force callers to learn another variant-specific path.

This is intentionally stricter than normal because the current problem is
overlapping layout ownership, not lack of files.

## Shared Runtime Interface

The shared seam is:

```text
DetectionPromptPolicy
  + DetectionDecodeRequest
  + input sample or prepared rollout sample
  -> InferenceRuntime.generate_many(...)
  -> DetectionDecodeResult[]
  -> DetectionParserResult[]
```

### DetectionPromptPolicy

Owns all prompt/input semantics a caller must keep aligned:

- `prompt_variant`
- `object_ordering`
- `object_field_order`
- `bbox_format`
- `detection_sequence_format`
- `coord_mode`
- `parser_policy`
- `resize_policy`, defaulting to `do_resize_false`
- `template_family`, such as CoordJSON or `compact_full`
- optional system-prompt behavior
- optional generation-prompt behavior
- `prompt_policy_fingerprint`

This policy is shared by:

- offline inference;
- online Stage-2 rollout;
- teacher-forced Stage-1/Stage-2 dataset prompt-prefix encoding;
- prompt parity tests and diagnostics.

Dataset and training modules may add loader-specific metadata, cache keys,
packing hints, and collator sidecars, but they should call this codec for the
semantic prompt and visual-input representation rather than reimplementing
message construction.

For the same sample, `prompt_policy_fingerprint`, and
`model_identity_fingerprint`, the prompt codec must make offline inference,
online rollout, and teacher-forced prompt-prefix encoding produce identical
prompt token IDs whenever the backend exposes prompt IDs. Matching rendered text
or prompt hashes is not sufficient.

For multimodal samples, strict parity checks must also compare visual-input
metadata when available:

- image count;
- image path or stable image identity;
- original width/height;
- post-preprocessing width/height;
- `do_resize=false` assertion;
- image placement in chat content;
- `image_grid_thw` or equivalent processor/template visual token metadata.

If a backend path cannot expose prompt IDs or visual metadata, artifacts must
record `prompt_token_parity: unverifiable` and cannot be used as evidence for a
strict offline-vs-online parity claim.

Metric-bearing eval may proceed with `prompt_token_parity: unverifiable` when:

- prompt/decode/model fingerprints are present;
- the generated-sequence trace contract is satisfied when required;
- parser policy is strict;
- the artifact does not claim offline-vs-online prompt-token equivalence.

Parity regression tests, template-alignment gates, and offline-vs-online
equivalence claims require `prompt_token_parity: verified`. If a backend can
expose prompt IDs but the runtime fails to capture them, that is a runtime/config
failure rather than an unverifiable-backend exemption.

Stage-2 trainable rollout-correction segments are stricter. They require
`prompt_token_parity: verified` because residual correction target offsets are
computed against the teacher-forced local encoding. If rollout prompt tokens or
visual-token metadata drift from the local teacher-forced prefix, target
positions can be corrupted. In that case the sample must be dropped or the
backend path must be rejected for trainable rollout construction.

Backend paths that cannot expose or faithfully reconstruct prompt IDs and
required visual metadata may still be used for eval-only or diagnostic decode,
but they are not valid for trainable Stage-2 rollout-correction segments.

The prompt codec must emit a required `prompt_policy_fingerprint` anywhere a
prompt-bearing artifact or cache could otherwise become ambiguous:

- dataset/cache fingerprints;
- resolved inference configs;
- run manifests;
- offline inference summaries;
- Stage-2 rollout artifacts;
- Stage-2 eval artifacts;
- prompt parity diagnostics.

The fingerprint includes semantic prompt/input policy:

- prompt variant;
- template family;
- system prompt text/hash;
- user prompt text/hash;
- object ordering policy;
- object field order;
- bbox format;
- detection sequence format;
- image placement convention;
- generation-prompt behavior;
- resize policy such as `do_resize=false`;
- parser policy when template family couples prompt and parse behavior.

The fingerprint excludes volatile runtime details:

- batch size;
- output directory;
- worker count;
- checkpoint path;
- decode temperature/top-p/top-k;
- seed;
- device/backend placement;
- rank/world-size.

### DetectionDecodeRequest

Owns backend-agnostic decode settings:

- backend: `hf` or `vllm`
- backend mode: local, colocate, or server when relevant
- decode mode: greedy, sampling, or beam
- `temperature`
- `top_p`
- `top_k`
- `max_new_tokens`
- `num_beams`
- `repetition_penalty`
- `seed`
- stop tokens / stop strings
- `trace_logprobs`
- optional `trace_prompt_logprobs` when prompt-token scores are explicitly
  needed; generated-sequence logprobs are the default trace requirement
- optional generation constraints such as compact grammar or stop pressure
- `decode_policy_fingerprint`

Offline `infer.generation.*` and online `rollout_matching.*` both map into this
object. Stage-2 per-attempt triage overrides also produce derived
`DetectionDecodeRequest` values.

`decode_policy_fingerprint` is separate from `prompt_policy_fingerprint`.
Prompt policy answers whether the model saw the same question in the same
format. Decode policy answers whether generation ran under the same sampling,
stopping, tracing, and constraint conditions.

The decode fingerprint includes decode semantics:

- backend family: HF or vLLM. This is required even when intended sampling
  settings are otherwise identical, because backend implementations can differ
  in tokenizer handling, stop behavior, numerical kernels, scheduling,
  determinism, and multimodal preprocessing details;
- backend mode: local, colocate, or server. This is required for provenance
  because colocate/server/local paths can expose different trace payloads and
  operational failure modes;
- decode mode: greedy, sampling, or beam;
- `max_new_tokens`;
- `temperature`;
- `top_p`;
- `top_k`;
- `num_beams`;
- `repetition_penalty`;
- stop tokens and stop strings;
- `trace_logprobs`;
- `trace_prompt_logprobs`;
- compact grammar settings;
- stop-pressure settings;
- other generation constraints that can change emitted tokens.

The decode fingerprint excludes operational placement details:

- checkpoint path;
- adapter path;
- output directory;
- batch size;
- worker count;
- server URL;
- rank/world-size;
- device IDs.

If future analysis needs to compare intended decode settings independent of
backend implementation, it may introduce an additional
`decode_semantics_fingerprint` that excludes backend family/mode. That optional
fingerprint must not replace the required provenance-bearing
`decode_policy_fingerprint`.

The fingerprint must be recorded in:

- resolved inference configs;
- run manifests;
- offline inference summaries;
- Stage-2 rollout artifacts;
- Stage-2 eval artifacts;
- decode diagnostics;
- prompt/decode parity reports.

### ScorePolicyFingerprint

Score policy is a separate provenance axis from decode policy:

- decode policy answers how tokens were generated;
- score policy answers how generated predictions received confidence scores.

The same generated output may be scored by different policies, so score
derivation must not be folded into `decode_policy_fingerprint`.

The required `score_policy_fingerprint` for score-bearing artifacts includes:

- score policy name, such as `generated_token_logprob`, `constant`, or a future
  calibrated policy;
- object-level aggregation rule;
- token span selection rule;
- coordinate-slot aggregation rule when applicable;
- calibration model identity when applicable;
- constant-score value and rationale when `score_policy: constant`;
- any filtering rule that changes which predictions receive scores.

The score policy fingerprint excludes:

- prompt text;
- model checkpoint identity;
- decode temperature/top-p/top-k;
- backend URL;
- output directory;
- batch size;
- rank/world-size.

Raw unscored artifacts may record `score_policy: none` and do not require a
score policy fingerprint. Score-bearing artifacts such as
`gt_vs_pred_scored.jsonl`, score-aware eval artifacts, confidence post-op
outputs, and score diagnostics require `score_policy_fingerprint`.

Raw/scored artifact separation is strict:

- `gt_vs_pred.jsonl` is the canonical strict standardized raw prediction
  artifact and must not carry confidence scores;
- `gt_vs_pred.jsonl` records or resolves `score_policy: none`;
- `gt_vs_pred_scored.jsonl` is the canonical score-bearing companion and must
  carry `score_policy_fingerprint`;
- guarded companions follow the same split:
  `gt_vs_pred_guarded.jsonl` remains raw/guarded, while
  `gt_vs_pred_scored_guarded.jsonl` is scored/guarded;
- re-scoring a raw artifact writes a new scored artifact and does not mutate the
  raw artifact.

### ModelIdentityFingerprint

Model identity is a separate provenance axis from prompt and decode policy:

- prompt policy answers whether the model saw the same question in the same
  format;
- decode policy answers whether generation ran under the same sampling,
  stopping, tracing, and constraint conditions;
- model identity answers whether generation used the same weights, adapters,
  tokenizer, and processor.

The required `model_identity_fingerprint` includes:

- resolved base model path or model ID;
- base model revision, commit, or content hash when available;
- checkpoint path or checkpoint identity;
- adapter path when present;
- adapter revision, commit, or content hash when available;
- tokenizer path and revision/hash;
- processor path and revision/hash;
- LoRA/server adapter sync identity for vLLM paths;
- vLLM backend sync identity, including adapter sync digest and coord-row offset
  digest/status when applicable;
- backend sync policy, such as `static` or `per_global_step`;
- key model/config fields that affect tokenization, multimodal processing, or
  generation compatibility.

The model identity fingerprint excludes:

- prompt text;
- object ordering;
- parser policy;
- decode temperature/top-p/top-k;
- stop policy;
- output directory;
- batch size;
- worker count;
- rank/world-size;
- device IDs.

The fingerprint must be recorded in:

- resolved inference configs;
- run manifests;
- offline inference summaries;
- Stage-2 rollout artifacts;
- Stage-2 eval artifacts;
- decode diagnostics;
- parity reports.

When vLLM server adapter sync is used, artifacts should also record detailed
debug metadata outside the compact fingerprint:

```text
backend_sync:
  mode
  sync_policy
  lora_tensors_digest
  coord_offset_digest
  coord_ids_digest
  synced_step
  server_ids
  status
```

The compact sync digest/status is part of `model_identity_fingerprint` because
the synced LoRA and coord-row state determine which model generated the rollout.
Detailed `backend_sync` metadata is for debugging and incident response.

Sync frequency stays explicit and lightweight:

- Stage-2 train vLLM server rollouts require `sync_policy: per_global_step`;
- offline inference and fixed-checkpoint eval use `sync_policy: static`;
- syncing less often than `per_global_step` for Stage-2 train changes training
  semantics and is not allowed silently;
- syncing more often than `per_global_step` is unnecessary unless a future
  measured issue justifies it;
- skipped sync under Stage-2 train is a hard error unless the decode path is
  explicitly eval-only or diagnostic-only.

### BackendAdapter Lifecycle

`src/infer/backend.py` owns backend lifecycle behind a small internal adapter
interface:

```text
BackendAdapter.prepare()
BackendAdapter.generate_many(PromptBundle[], DetectionDecodeRequest) -> DetectionDecodeResult[]
BackendAdapter.close()
```

Backend lifecycle ownership includes:

- HF model/processor/tokenizer loading;
- vLLM local/colocate engine creation and shutdown;
- vLLM server health/readiness checks;
- vLLM server client lifecycle and communicator lifecycle;
- request-config adaptation;
- backend trace response validation;
- adapter sync identity and provenance.

Launchers may still start external server processes, but client-side readiness,
identity checks, request adaptation, trace validation, and model/adapter sync
belong to the backend adapter layer.

The vLLM server adapter-sync path must preserve current CoordExp/ms-swift
semantics exactly:

- official adapter sync mode remains adapter-only for server rollouts;
- LoRA sync sends only vLLM-compatible LoRA tensors;
- PEFT `modules_to_save` payloads that vLLM cannot consume are filtered rather
  than sent as ordinary LoRA tensors;
- CoordExp `coord_offset_adapter` is synchronized separately as token-row
  offsets, not folded into the LoRA payload;
- the coord-row sync path requires the ms-swift VLLMClient/server patch that
  exposes `update_token_row_offsets`;
- missing coord-row sync support is a hard failure for checkpoints that declare
  or contain `coord_offset_adapter`;
- sync failures on rank 0 under DDP must abort all learner ranks rather than
  risking rollout/train deadlock;
- adapter sync must update `model_identity_fingerprint` or associated
  model-sync provenance so rollout artifacts identify the adapter/coord-row
  state used for generation.
- vLLM server artifacts should record detailed `backend_sync` metadata while
  keeping compact sync identity inside `model_identity_fingerprint`.
- Stage-2 train vLLM server rollouts require `sync_policy: per_global_step`;
  fixed-checkpoint inference/eval uses `sync_policy: static`.

This behavior currently lives in `src/trainers/rollout_runtime/vllm_server.py`
and `src/infer/backend_sync.py`; migration moves ownership without weakening
the functional contract.

## Comparability Contract

Artifacts and caches that can influence model comparison must carry the relevant
fingerprints:

- prompt-bearing caches and artifacts require `prompt_policy_fingerprint`;
- generation-bearing artifacts require `decode_policy_fingerprint`;
- generation-bearing artifacts require `model_identity_fingerprint`;
- metric-bearing eval inputs require all applicable prompt/decode/model
  fingerprints.

If a required fingerprint is missing:

- dataset caches must be invalidated or rejected;
- offline inference resolved configs and summaries must fail before claiming
  comparable output;
- Stage-2 rollout/eval artifacts must fail before metric-bearing emission;
- score-analysis and mAP pipelines must reject the artifact;
- diagnostic-only artifacts may still be written only if they explicitly record
  `comparable: false`.

This contract prevents stale caches, prompt drift, decode drift, or model-weight
drift from masquerading as comparable checkpoint differences.

Historical artifacts remain tolerant-read:

- old runs may be opened for debugging, qualitative inspection, or historical
  archaeology;
- old runs without required fingerprints are treated as `comparable: false`;
- new metric, score-analysis, mAP, checkpoint comparison, or promotion paths
  must reject them as non-comparable;
- fingerprint recovery must be explicit and audited if ever needed;
- migration must produce a new migrated artifact rather than mutating old
  outputs in place;
- best-effort reconstruction from partial old config fragments is not allowed
  for comparable evidence.

### DetectionDecodeResult

Every backend returns the same result shape:

- generated text;
- prompt token IDs when available;
- generated token IDs when available;
- generated token text when available;
- token logprobs when requested and available;
- optional prompt token logprobs when `trace_prompt_logprobs=true`;
- backend metadata;
- determinism note;
- trace availability state;
- structured error if generation failed.

If a metric-bearing workflow requests full sequence logprobs and the backend
cannot provide them, the runtime must fail fast. It must not silently emit
text-only predictions that later look score-ready.

When `trace_logprobs=true`, every successful `DetectionDecodeResult` must satisfy
the same post-adaptation invariants, regardless of backend:

- `generated_token_ids` is present and contains integers;
- `generated_token_text` is present and has the same length as
  `generated_token_ids`;
- `token_logprobs` is present, finite, and has the same length as
  `generated_token_ids`;
- `generated_text` is consistent with the decoded generated-token view under
  the active special-token and stop-token policy;
- the result records which backend-native fields were adapted into the
  canonical trace.

If any invariant fails, the runtime raises a structured trace error before the
result can feed confidence scoring, rollout diagnostics, or score-aware eval.

Constant-score compatibility remains possible only through an explicit
non-logprob score policy, such as `score_policy: constant`. Constant-score
artifacts must record `score_policy_fingerprint` and state that scores are
deterministic compatibility scores, not model logprob-derived scores.

Generated-sequence trace capture is not greedy-only. When a backend supports
sampled-token logprobs, `trace_logprobs=true` must produce the same canonical
trace object for greedy and sampled decoding. This is required for rollout
diagnostics, RL-style analysis, and sampled train-time investigations.

Metric score use is stricter than trace capture. Score-aware mAP/confidence
paths must declare an explicit `score_policy` and may only consume traces under
policies validated for that metric surface. Existing confidence scoring may
remain greedy-only until sampled score provenance is explicitly validated. A
sampled trace is therefore valid diagnostic evidence, but it is not
automatically valid mAP score provenance.

### DetectionParserResult

The parser facade returns:

- raw parsed payload;
- standardized valid objects;
- invalid object counters;
- truncation / terminal-token metadata;
- strict-vs-diagnostic parser policy;
- parser manifest metadata suitable for artifacts.

Metric-bearing parse mode must be strict-only. Salvage parsing can exist, but
only for diagnostic artifacts, monitor dumps, and ad hoc analysis. Any artifact
row produced or recovered through diagnostic salvage must carry
`metric_bearing: false` and must not be consumed by:

- `gt_vs_pred.jsonl`;
- `gt_vs_pred_scored.jsonl`;
- Stage-2 eval artifacts;
- confidence post-op;
- COCO/LVIS/mAP evaluation;
- rollout diagnostics that feed metrics.

If a prediction is only recoverable through wrong-field-order, malformed-row, or
compact-output salvage, metric-bearing paths record the parse diagnostics and
emit an empty or invalid prediction state rather than improving the metric with
tolerant parsing.

Canonical eval input artifacts are metric-bearing-only:

- `gt_vs_pred.jsonl`;
- `gt_vs_pred_scored.jsonl`;
- `gt_vs_pred_guarded.jsonl`;
- `gt_vs_pred_scored_guarded.jsonl`;
- Stage-2 eval detection artifacts consumed by official metrics.

Those artifacts may contain strict-parser error rows, but they must not contain
salvage-recovered predictions. Salvage/debug recoveries belong in separate
diagnostic artifacts such as monitor dumps or `diagnostics/*` JSONL files.

## Ownership After Refactor

### `src/infer` Owns

- prompt/input policy resolution;
- image/path loading for inference-style generation;
- visual input preparation and `do_resize=false` enforcement;
- chat/template formatting parity;
- decode config normalization;
- HF/vLLM backend selection and lifecycle;
- generated sequence token/logprob tracing;
- stop conditions and optional decode constraints;
- strict/diagnostic parse policy;
- inference artifact summaries and trace row construction.

### Stage-2 Owns

- rollout-correction sample selection;
- per-attempt decode override policy;
- duplicate filtering;
- greedy-IoU assignment;
- residual GT correction event construction;
- teacher-forced target IR creation;
- objective/loss execution;
- DDP and optimizer-step coordination;
- Stage-2 metric projection.

Stage-2 can keep trainer-facing adapter methods during migration, but their
implementation should delegate to `src/infer` and should not host independent
prompt, backend, or parser behavior.

### Eval Owns

- score-aware COCO/LVIS/F1ish metric computation;
- duplicate-control post-processing;
- official evaluator input selection;
- evaluator reports.

Eval consumes standardized inference artifacts and parser results. It does not
own model generation.

## Deletion And Folding Plan

### Delete `src/trainers/rollout_runtime/`

After migration, this package should disappear. There is no final compatibility
package under `src.trainers.rollout_runtime`.

Current files and target ownership:

- `dispatch.py` -> `src/infer/rollout_dispatch.py`
- `vllm_infer.py` -> `src/infer/backend_vllm_infer.py`
- `vllm_server.py` -> `src/infer/backend_vllm_server.py`
- `vllm_engine.py` -> `src/infer/backend_vllm_engine.py`
- `vllm_config.py` -> `src/infer/backend_vllm_config.py`
- `vllm_compat.py` -> `src/infer/backend.py` or a tiny `src/infer/vendor_compat.py`
- `swift_infer_compat.py` -> `src/infer/backend.py` or `src/infer/vendor_compat.py`
- `swift_coord_row_patch.py` -> `src/infer/backend_sync.py`

The deletion is complete only when active code and tests no longer import
`src.trainers.rollout_runtime`.

The search gate is:

```bash
rg -n "src\.trainers\.rollout_runtime|from \.rollout_runtime|rollout_runtime/" src tests scripts docs openspec configs
```

Expected result for active sources: no hits, except archived OpenSpec/history
records that are intentionally historical.

### Delete Old Offline Inference Import Surfaces

The new root uses `src/infer/runtime.py` and `src/infer/backend.py`; the old
offline names should not remain as active compatibility facades.

Deletion targets:

- `src/infer/engine.py`
- `src/infer/backends.py`

All active imports must move to the new modules. This includes:

- `scripts/run_infer.py`
- `scripts/tools/`
- `scripts/analysis/`
- `src/callbacks/`
- `src/analysis/`
- `src/eval/`
- tests
- docs and non-archived OpenSpec references

The deletion is complete only when:

```bash
rg -n "src\.infer\.engine|src\.infer\.backends|from src\.infer import GenerationConfig|InferenceEngine|InferenceConfig" src tests scripts docs openspec configs
```

returns no active references to the removed modules. Archived records may remain
if they are clearly historical.

### Collapse Parser Overlap

Move active decode-output parsing behavior behind `src/infer/parsing.py`:

- CoordJSON strict parser;
- CoordJSON diagnostic salvage parser;
- compact-full parser facade;
- terminal token and truncation metadata;
- parser manifests for artifacts.

The existing `src/trainers/rollout_matching/parsing.py` should keep only
matching-adjacent helpers if any are truly matching-specific. If most helpers
become parser-policy helpers, the file should be deleted. It should not remain
as a compatibility import after migration.

### Collapse Prompt/Input Overlap

Move prompt/input behavior behind `src/infer/prompt.py`:

- image-before-text message construction;
- generation-time image reference normalization;
- exactly-one-image validation for detection generation;
- PIL RGB loading when needed;
- image identity metadata;
- original and post-preprocessing dimension metadata;
- system prompt injection;
- generation prompt handling;
- teacher-forced prompt-prefix rendering for dataset/training records;
- Swift template encode wrapper;
- HF processor encode wrapper;
- vLLM OpenAI-style message serialization;
- actual processor/template kwargs recorded for provenance;
- `do_resize=false` assertion and metadata.

This replaces independent prompt builders in offline inference, dataset
encoding, and Stage-2 rollout preparation.

Active dataset/training callers are in scope, including:

- `src/datasets/builders/jsonlines.py`;
- `src/datasets/dense_caption.py`;
- `src/detection/dataset.py`;
- Stage-1 detection runtime/template encoding call sites;
- Stage-2 teacher-forced encoding call sites.

Those modules should retain ownership of data loading, row validation,
ordering-policy selection, cache metadata, packing, collation, assistant-target
rendering, residual slicing, loss masks, and target IR creation. They should not
own separate prompt/message/template rendering logic for prompt prefixes.

Image ownership is similarly split:

- `src/infer/prompt.py` owns generation-time image normalization for offline
  inference, online rollout, and teacher-forced prompt parity checks;
- dataset modules own storage layout, row iteration, dataset roots, cache keys,
  and dataset-level validation;
- the shared codec rejects zero or multiple images for detection generation,
  missing metric-bearing dimensions, or dimensions inconsistent with resolved
  image metadata;
- backend-specific transport forms, such as PIL images or vLLM base64/OpenAI
  image content, are derived from the same normalized image representation.

Do not add an active multi-image abstraction in this refactor. Multi-image
support would be a separate future contract change, not a latent list-shaped
extension in the unified runtime.

Backend transport serialization is not semantic prompt ownership:

- `src/infer/prompt.py` produces a `PromptBundle` containing canonical
  messages/text, the normalized single image, dimensions, image identity,
  prompt-token parity metadata, and `prompt_policy_fingerprint`;
- `src/infer/backend.py` converts that `PromptBundle` into HF processor inputs,
  Swift template inputs, or vLLM OpenAI/base64 payloads;
- backend adapters must not re-resolve image paths, reload dimensions, inject
  alternate system prompts, reorder image/text content, or construct their own
  semantic prompt;
- backend adapters may only perform transport-specific serialization required
  by the backend API.

### Rename Misleading Runtime Budget File

`src/trainers/rollout_correction/scheduler.py` no longer describes a scheduler.
It holds rollout-correction runtime budget helpers. Rename it to
`runtime_budget.py` or fold it into the executor/runtime-budget owner.

This is a small cleanup, but it removes legacy A/B scheduler vocabulary from the
active Stage-2 layout.

### Retire Dynamic Package Proxy When Tests Move

`src/trainers/rollout_correction/__init__.py` dynamically proxies implementation
attributes to preserve monkeypatch compatibility. That compatibility surface is
not a final-state requirement. Tests should be moved to stable seams directly:

- shared inference runtime for decode;
- rollout-correction target builder for target realization;
- objective runner for loss execution.

Once tests use those seams, remove the proxy behavior and keep a normal package
initializer. If this broadens the patch into old monkeypatch-heavy tests, that
work is in scope.

## Migration Phases

### Phase 1: Golden Contracts Before Movement

Add tests that describe the desired shared behavior before deleting files:

- offline HF and Stage-2 HF produce the same prompt text / prompt token IDs for
  a shared sample and prompt policy;
- teacher-forced dataset prompt-prefix encoding, offline inference, and online
  rollout share the same semantic prompt codec and prompt-policy fingerprint;
- strict offline-vs-online prompt parity checks compare prompt token IDs and
  visual-input metadata, not just rendered text/hash;
- metric-bearing eval may proceed with `prompt_token_parity: unverifiable`, but
  parity claims/gates require `prompt_token_parity: verified`;
- Stage-2 trainable rollout-correction segments require
  `prompt_token_parity: verified`;
- prompt-bearing caches, resolved configs, run manifests, and rollout/eval
  artifacts record `prompt_policy_fingerprint`;
- generation-bearing resolved configs, run manifests, rollout/eval artifacts,
  and diagnostics record `decode_policy_fingerprint`;
- generation-bearing resolved configs, run manifests, rollout/eval artifacts,
  and diagnostics record `model_identity_fingerprint`;
- score-bearing artifacts record `score_policy_fingerprint`; raw unscored
  artifacts may record `score_policy: none`;
- raw `gt_vs_pred.jsonl` remains unscored; score-bearing outputs are separate
  scored companion artifacts;
- comparable artifacts and caches fail fast when required fingerprints are
  missing; diagnostic-only artifacts without fingerprints record
  `comparable: false`;
- historical artifacts without fingerprints remain readable but are rejected by
  new metric/comparison paths unless explicitly migrated;
- all visual encode paths assert and record `do_resize=false`;
- offline inference, online rollout, and teacher-forced parity checks use the
  same generation-time image normalization path;
- offline and online parser policies agree for strict mode;
- metric-bearing parser policy is strict-only;
- diagnostic salvage mode is marked non-metric-bearing and cannot feed
  `gt_vs_pred.jsonl`, scored eval artifacts, confidence post-op, or mAP;
- `gt_vs_pred*.jsonl` canonical eval inputs remain metric-bearing-only and
  exclude salvage-recovered predictions;
- `DecodeResult` records logprob trace availability consistently;
- all trace-capable backends, including vLLM, either return the identical
  canonical trace object or fail fast before metric-bearing artifacts are
  written;
- sampled and greedy decode both use the same trace object contract, while
  score-aware metrics consume traces only through explicit validated
  `score_policy`;
- missing width/height cannot produce metric-bearing Stage-2 eval records.

### Phase 2: Introduce Shared Dataclasses And Runtime

Add `src/infer/runtime.py`, `src/infer/prompt.py`, `src/infer/backend.py`, and
`src/infer/parsing.py`. During the phase, temporary delegation to old
implementations is acceptable inside the new modules only as an implementation
bridge. Do not add new public compatibility wrappers.

At the end of this phase, new tests should pass and new code should import the
new modules directly.

### Phase 3: Migrate Offline Inference

Refactor offline inference callers to the new runtime shape and delete
`src/infer/engine.py` and `src/infer/backends.py` as active modules.

`scripts/run_infer.py` and `src/infer/pipeline.py` should continue to behave as
before from an operator perspective.

All active scripts, callbacks, analysis helpers, and tests that currently import
`InferenceEngine`, `GenerationConfig`, `InferenceConfig`, or
`generate_hf_batch()` from old modules must be updated in this phase or in the
same patch series before completion is claimed.

### Phase 4: Migrate Stage-2 Rollout

Replace Stage-2 local `_rollout_many_*`, `_prepare_samples_for_rollout`, and
vLLM dispatch logic with calls into shared `InferenceRuntime`.

Trainer-facing methods may exist only as internal implementation conveniences
while tests are migrated. They are not compatibility guarantees and should be
deleted or made private once callers use the shared runtime seam.

### Phase 5: Delete Overlapping Layout

Remove `src/trainers/rollout_runtime/` and update imports, docs, and tests.
Reduce `src/trainers/rollout_matching/parsing.py` to matching-only ownership or
delete it if no matching-specific logic remains.

Also remove old active offline modules and references:

- `src/infer/engine.py`
- `src/infer/backends.py`
- public imports of `InferenceEngine`, `GenerationConfig`, and
  `InferenceConfig` from old surfaces.

### Phase 6: Stable Docs And OpenSpec Updates

Update stable docs/specs only once the implementation actually changes runtime
ownership:

- `docs/IMPLEMENTATION_MAP.md`
- `docs/SYSTEM_OVERVIEW.md`
- `docs/eval/WORKFLOW.md`
- `docs/training/STAGE2_RUNBOOK.md`
- `docs/ARTIFACTS.md`
- `openspec/specs/inference-engine/spec.md`
- `openspec/specs/inference-pipeline/spec.md`
- `openspec/specs/runtime-architecture-refactor-program/spec.md`

The docs must state that `rollout_matching.*` remains a Stage-2 config namespace
while decode implementation is shared through `src/infer`.

## Verification Plan

Targeted tests to add or update:

- `tests/test_decode_config_contract.py`
- `tests/test_detection_prompt_input_codec.py`
- `tests/test_decode_backend_trace_contract.py`
- `tests/test_vllm_logprob_trace_contract.py`
- `tests/test_parser_policy_parity.py`
- `tests/test_unified_infer_pipeline.py`
- `tests/test_infer_batch_decoding.py`
- `tests/test_qwen_generation_contract.py`
- `tests/test_stage2_rollout_runtime.py`
- `tests/test_vllm_server_adapter_payload.py`
- `tests/test_swift_rollout_endpoints_contract.py`

Key search gates:

```bash
rg -n "src\\.trainers\\.rollout_runtime|from \\.rollout_runtime|rollout_runtime/" src tests scripts docs openspec configs
rg -n "src\\.infer\\.engine|src\\.infer\\.backends|from src\\.infer import GenerationConfig|InferenceEngine|InferenceConfig" src tests scripts docs openspec configs
rg -n "_generate_hf|_generate_vllm|_rollout_many_hf|_rollout_many_vllm|_prepare_samples_for_rollout" src tests
rg -n "load_prediction_dict|parse_rollout_for_matching|parse_compact_full_output_artifact|CompactFullRolloutCodec" src tests
```

Final verification should include narrow config/runtime tests first, then
offline inference tests, then Stage-2 rollout runtime tests. A full production
training run is not required for the refactor to land, but the implementation
must include a smoke path that exercises shared decode from both offline and
online callers.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Refactor silently changes prompt tokenization | Add golden prompt text, prompt IDs, and image-grid parity tests before movement. |
| vLLM logprob support differs by serving path | Treat vLLM as trace-capable but require every vLLM path to adapt into the same `DetectionDecodeResult`; fail fast on missing or mismatched trace fields. |
| File budget creates a large new god module | Use the split threshold only after the shared seam is stable and tests prove variant independence. |
| Compatibility shims become permanent | No compatibility shims are part of the final design; completion requires zero active imports of removed surfaces. |
| Parser salvage inflates offline metrics | Ban diagnostic salvage from metric-bearing paths; strict parsing is the only metric-bearing policy. |
| Stage-2 config docs imply implementation ownership | Docs must distinguish config namespace (`rollout_matching.*`) from implementation root (`src/infer`). |

## Audit Convergence Addendum

This addendum supersedes any earlier wording in this design that is looser or
more permissive.

### OpenSpec Review Outcomes

The OpenSpec audit loop found no P0 issues, but it tightened several boundaries
that the implementation plan must honor:

- `src/trainers/stage2_rollout_runtime.py` cannot remain an active decode owner.
  It must be deleted, renamed, or reduced to a thin Stage-2-owned facade that
  does not own prompt rendering, backend lifecycle, decode request conversion,
  trace normalization, or parser policy.
- Current caller-facing `src.infer.compact_grammar` and
  `src.infer.stop_pressure` imports are not final active surfaces. They must
  either fold behind `src/infer/constraints.py` or become private helper modules
  reached only through that facade.
- The shared prompt codec owns prompt-prefix/input rendering and parity
  metadata only. Assistant target rendering, residual slicing, target IR
  construction, duplicate control, greedy IoU matching, and loss masks remain
  training/trainer-owned.
- Final `src/infer` code must not import `ResidualBoundaryAdapter`,
  `TeacherForcingTargetIR`, `build_residual_set_target_ir`, rollout-correction
  `target_builder`, greedy IoU matching helpers, or duplicate-control training
  logic.
- The active Stage-2 vLLM server rollout sync contract is official adapter sync
  plus CoordExp coord-row sync:
  `rollout_matching.vllm.mode=server`,
  `rollout_matching.vllm.sync.mode=adapter`, and
  `rollout_matching.vllm.enable_lora=true`.
- Older native full-sync materialization is superseded for active unified
  Stage-2 rollout-correction server training unless a future OpenSpec revives
  it explicitly.
- vLLM trace sources must be path-specific: HF uses generation scores; vLLM
  local/colocate uses request-output token IDs/logprobs; ms-swift server uses
  `return_details=true`; OpenAI-compatible HTTP is trace-required only if token
  IDs are present or exactly reconstructable.
- Fire-and-forget coord-row sync acknowledgement is not worker verification.
  Artifacts must distinguish requested/learner-side sync from
  `worker_verified` sync.
- Provenance carriers are run-level `resolved_config.json` and `summary.json`,
  with optional colocated sidecars for copied artifacts. Standalone moved JSONL
  without metadata is inspection-only and fails comparable paths as
  `missing_provenance`.
- Historical artifacts may be stamped only when exact fingerprints can be
  reconstructed. Otherwise they remain `comparable: false`; migration must not
  invent prompt/decode/model/score provenance.
- Stage-2 eval materialization under
  `eval_detection/step_<global_step>/` remains frozen when
  `rollout_matching.eval_detection.materialize_artifacts: true`.
- Test ordering must avoid a giant red tree: write golden/parity/trace tests
  before movement, but reserve final import-ban/search-gate enforcement for the
  deletion phase.

### Updated Minimal Layout Policy

`src/infer/backend.py` remains the public adapter interface/factory, but private
helpers such as `backend_sync.py` are allowed if the module would otherwise
become a backend god file. Likewise `constraints.py` remains the public
constraint facade, while private compact/stop-pressure helpers are allowed if
needed. The rule is about minimizing caller-visible surfaces, not forcing every
implementation detail into one oversized file.

## Implementation Boundary

This document is a design contract, not an implementation checklist. The next
step is a separate implementation plan that orders patches, tests, and doc/spec
updates. Production code implementation starts only after explicit final user
approval, even if this design and the implementation plan are already written
and reviewed.
