# shared-inference-runtime Specification

## Purpose
Define the shared detection inference runtime contract for prompt rendering,
visual-input normalization, backend-agnostic decode requests, generated-token
trace validation, strict parser policy, and provenance/score comparability
across offline inference and online Stage-2 rollout callers.
## Requirements
### Requirement: Shared runtime owns prompt, decode, backend, trace, parse, and provenance seams

The system SHALL expose one shared detection inference runtime under `src/infer`
that mediates prompt/input rendering, backend-agnostic decode requests, backend
generation, generated-token trace validation, parser policy, and decode
provenance.

Normative behavior:

- offline `infer.*` YAML and Stage-2 rollout-correction runtime policy YAML MUST map into
  common internal prompt/decode/model policy objects;
- public authored `infer.*` stays stable; Stage-2 rollout-correction runtime policy may retain `rollout_matching.*` only as a classified migration handle until a schema-owned replacement is complete;
- callers MUST NOT implement independent semantic prompt construction,
  generation-time image normalization, backend trace conversion, or
  metric-bearing parser salvage outside the shared runtime;
- caller-specific downstream logic remains outside the runtime.

#### Scenario: Offline and online decode use the same runtime seam

- **GIVEN** an offline inference config and a Stage-2 rollout config with the
  same prompt, model, and decode policy
- **WHEN** each caller prepares decode work
- **THEN** both callers produce compatible internal prompt/decode request
  objects for the shared runtime
- **AND** downstream differences are limited to caller-owned eval or training
  logic.

### Requirement: Prompt and visual input rendering is single-owned

The shared prompt codec SHALL own semantic detection prompt-prefix rendering and
generation-time visual input preparation for offline inference, online rollout,
and teacher-forced training/dataset prompt-prefix encoding.

Normative behavior:

- the supported detection generation input contains exactly one image per
  sample;
- the prompt codec MUST validate exactly-one-image shape before generation or
  teacher-forced prompt parity checks;
- the codec MUST preserve image/geometry alignment and MUST NOT silently resize
  training or comparable inference images;
- `do_resize=false` MUST be asserted for training and comparable inference
  paths;
- datasets MAY own storage, indexing, sampling, cache layout, packing,
  batching, and collation, but MUST call the shared codec for semantic prompt
  and visual-input rendering;
- training-owned modules MUST retain ownership of assistant-target rendering,
  residual slicing, target IR construction, loss masks, duplicate control, and
  rollout-correction target semantics;
- backend adapters MUST serialize a normalized prompt bundle and MUST NOT
  re-resolve images, reload dimensions, inject semantic prompt text, reorder
  prompt content, or construct template-specific detection prompts.

#### Scenario: One-image validation fails before backend generation

- **GIVEN** a detection sample with zero images or more than one image
- **WHEN** the shared prompt codec prepares a prompt bundle
- **THEN** preparation fails before backend generation
- **AND** the error identifies the one-image contract violation.

#### Scenario: Teacher-forced and rollout prompts share tokenized semantics

- **GIVEN** the same sample, prompt policy, tokenizer, processor, and model
  identity
- **WHEN** teacher-forced encoding and online rollout prompt preparation run
- **THEN** the semantic prompt text, image placement, prompt token IDs, and
  visual metadata match whenever the backend exposes or can reconstruct those
  values.

### Requirement: Prompt parity is explicit and trainable Stage-2 requires verified parity

Prompt parity SHALL be recorded explicitly and SHALL distinguish verified
prompt-token/visual parity from unverifiable backend paths.

Normative behavior:

- strict parity requires exact prompt-token ID parity and visual-input metadata
  parity, not just rendered text or prompt hash equality;
- visual metadata parity SHOULD include image count, stable image identity,
  original width/height, post-preprocessing width/height, `do_resize=false`,
  image placement, and processor visual-token metadata such as
  `image_grid_thw` when available;
- if a backend cannot expose or faithfully reconstruct prompt IDs or required
  visual metadata, artifacts MUST record `prompt_token_parity:
  unverifiable`;
- ordinary metric eval MAY proceed with unverifiable prompt parity when
  fingerprints, strict parser policy, and trace requirements are satisfied;
- parity claims, template-alignment gates, and offline-vs-online equivalence
  assertions require `prompt_token_parity: verified`;
- Stage-2 trainable rollout-correction segments MUST require
  `prompt_token_parity: verified` and MUST reject or drop samples whose rollout
  prefix cannot be aligned to the teacher-forced local encoding.

#### Scenario: Stage-2 trainable rollout rejects unverifiable prompt parity

- **GIVEN** an online rollout backend cannot provide prompt token IDs or
  equivalent parity metadata
- **WHEN** the rollout would feed trainable Stage-2 residual correction target
  construction
- **THEN** the sample is rejected or the backend path fails before target
  construction
- **AND** the run does not train on unverifiable rollout offsets.

### Requirement: Decode request policy is backend-agnostic and fingerprinted

The shared runtime SHALL represent generation settings through a canonical
decode request and SHALL record `decode_policy_fingerprint` for
generation-bearing artifacts.

Normative behavior:

- decode request settings MUST cover backend family, backend mode, greedy or
  sampling mode, temperature, top-p, top-k, max generation length, beams,
  repetition penalty, seed, stop tokens, stop strings, trace flags, and active
  generation constraints;
- `decode_policy_fingerprint` MUST include backend family and backend mode
  because HF, vLLM local, vLLM colocate, and vLLM server paths can differ in
  tokenizer handling, stop behavior, trace payloads, determinism, scheduling,
  and failure modes;
- `decode_policy_fingerprint` MUST exclude operational placement details such
  as output directory, batch size, worker count, server URL, rank/world-size,
  and device IDs;
- if future analysis needs backend-independent comparison, it MAY introduce an
  additional fingerprint, but it MUST NOT replace the required provenance
  `decode_policy_fingerprint`.

#### Scenario: Backend switch changes decode provenance

- **GIVEN** the same temperature, top-p, max-new-tokens, and stop policy
- **WHEN** a run switches from HF to vLLM server backend
- **THEN** the `decode_policy_fingerprint` changes
- **AND** artifacts remain schema-compatible while not claiming backend-identical
  decode provenance.

### Requirement: Backend adapters emit one canonical detection decode result

All backend adapters SHALL normalize native generation outputs into one
canonical `DetectionDecodeResult`.

Normative behavior:

- supported backend families include HF and vLLM;
- supported vLLM modes MAY include local, colocate, and server paths;
- backend adapters own lifecycle, readiness, request adaptation, response
  conversion, trace validation, backend identity, and model-sync provenance;
- backend adapters MUST fail fast when a selected backend is unavailable,
  incompatible, or cannot satisfy the requested trace contract;
- backend adapters MUST NOT change downstream artifact schemas.

#### Scenario: Backend unavailable fails before partial artifacts

- **GIVEN** a config selects a vLLM backend that is not available in the
  runtime
- **WHEN** the backend adapter prepares generation
- **THEN** preparation fails before generation artifacts are written
- **AND** the diagnostic identifies the selected backend and how to switch or
  fix the config.

### Requirement: Generated-sequence logprob trace is complete when requested

When `trace_logprobs=true`, generated output traces SHALL contain complete
aligned generated-token IDs, generated-token text, and finite generated-token
logprobs after accepted backend adaptation.

Normative behavior:

- the chosen/generated token's logprob MUST be present for every emitted
  generated token;
- trace arrays MUST align 1:1 after accepted adaptation;
- accepted adaptation MAY normalize token-ID strings, backend logprob
  containers, tokenizer text representation, and explicit non-emitted backend
  stop-token handling;
- adapters MUST NOT silently clip, pad, default, or repair traces to hide shape
  mismatches;
- metric score derivation from logprobs MUST be gated by an explicit score
  policy and score fingerprint.

#### Scenario: Missing generated-token logprob fails trace-required decode

- **GIVEN** `trace_logprobs=true`
- **WHEN** the backend returns a generated token without a finite logprob
- **THEN** the shared runtime fails the result before metric-bearing artifacts
  are materialized.

#### Scenario: Trace shape mismatch is not clipped

- **GIVEN** a backend returns five generated token IDs and four generated-token
  logprobs
- **WHEN** the adapter normalizes the response
- **THEN** the shared runtime fails with a trace shape diagnostic
- **AND** it does not clip token IDs or pad logprobs to produce an apparently
  valid trace.

### Requirement: vLLM trace-capable paths must satisfy the same result contract

vLLM SHALL be treated as a trace-capable backend when configured for generated
token logprobs, and vLLM paths SHALL satisfy the same canonical result contract
as HF paths.

Normative behavior:

- vLLM local, colocate, and server adapters MUST request generated-token
  logprobs when `trace_logprobs=true`;
- HF adapters MUST derive the trace from generated sequences plus generation
  score tensors or equivalent chosen-token scores;
- vLLM Python/local/colocate adapters MUST derive the trace from
  `RequestOutput`-style token IDs and logprob payloads;
- ms-swift vLLM server rollout adapters MUST request `return_details=true` and
  require prompt token IDs, generated token IDs, and generated-token logprobs;
- OpenAI-compatible HTTP responses MUST fail `trace_logprobs=true` unless they
  provide generated token IDs or pass an exact tokenizer round-trip
  reconstruction check for every generated token;
- vLLM adapters MUST normalize vLLM native logprob payloads into canonical
  generated-token trace fields;
- vLLM adapters MUST fail fast when server configuration, sampling params, or
  response payloads cannot provide the requested trace;
- vLLM prompt logprobs remain optional and are required only when
  `trace_prompt_logprobs=true`.

#### Scenario: vLLM server omits logprobs under a trace-required request

- **GIVEN** a vLLM server decode request with `trace_logprobs=true`
- **WHEN** the server response omits generated-token logprobs
- **THEN** the adapter fails before the output can be used for rollout
  diagnostics, score-bearing eval, or metric-bearing artifacts.

### Requirement: Metric-bearing parsing is strict-only

The shared parsing facade SHALL prevent diagnostic salvage from entering
metric-bearing artifacts or rollout metrics.

Normative behavior:

- strict parser failures MAY produce structured error rows and counters;
- diagnostic salvage outputs MUST declare `metric_bearing: false`;
- diagnostic salvage predictions MUST NOT feed `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, guarded companions, Stage-2 official eval
  artifacts, confidence post-op, COCO/LVIS/mAP, or rollout metric computation;
- parser policy and metric-bearing status MUST be visible in artifacts or
  summaries that could otherwise be compared.

#### Scenario: Salvage parser output is rejected from official eval input

- **GIVEN** diagnostic salvage recovers an object from malformed model text
- **WHEN** the run materializes official evaluation input
- **THEN** the recovered object is excluded from metric-bearing artifacts
- **AND** the salvage artifact, if written, declares `metric_bearing: false`.

### Requirement: Provenance fingerprints are orthogonal and required for comparable artifacts

The shared runtime SHALL record orthogonal prompt, decode, model-identity, and
score-policy provenance fingerprints.

Normative behavior:

- prompt-bearing caches/artifacts require `prompt_policy_fingerprint`;
- generation-bearing artifacts require `decode_policy_fingerprint` and
  `model_identity_fingerprint`;
- score-bearing artifacts require `score_policy_fingerprint`;
- raw unscored artifacts MUST record or resolve `score_policy: none`;
- comparable artifacts and caches MUST fail fast when required fingerprints are
  missing;
- historical artifacts without required fingerprints MAY remain readable for
  inspection, but strict comparison, official eval, and reporting paths MUST
  reject them unless explicitly migrated and stamped.

Provenance carrier contract:

- `resolved_config.json` is the required run-level carrier for resolved
  policies, artifact paths, and fingerprint values;
- `summary.json` is the required compact run-level carrier for prompt, decode,
  model, parser, score, comparability, and backend-sync status;
- score-bearing artifacts MUST be resolvable to score provenance through the
  run metadata or a colocated sidecar;
- `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` line schemas remain
  compatible with the existing inference-engine output schema;
- standalone moved JSONL files without their metadata or sidecar MUST load only
  as inspection data and MUST fail comparable paths with `missing_provenance`;
- `score_policy_fingerprint` MUST include policy name, score source,
  aggregation rule, token/span selection rule when applicable, constant score
  value when applicable, source raw artifact identity, parser policy, and
  `metric_bearing` status.

Historical artifact migration contract:

- an explicit migration/stamping tool MAY stamp exact fingerprints only when
  old `summary.json`, `resolved_config.json`, raw/scored JSONL, and required
  sidecars contain enough evidence to reconstruct them;
- when exact reconstruction is impossible, the tool MUST mark or report the
  artifact as `comparable: false` with a reason;
- migration MUST NOT invent missing prompt, decode, model, or score provenance
  to make historical artifacts official-eval eligible.

#### Scenario: Score-bearing artifact without score fingerprint is rejected

- **GIVEN** a `gt_vs_pred_scored.jsonl` candidate missing
  `score_policy_fingerprint`
- **WHEN** score-aware evaluation or comparison loads it
- **THEN** loading fails with a provenance diagnostic
- **AND** the raw unscored artifact is not mutated as a fallback.

### Requirement: Backend sync identity is part of model identity

The shared backend layer SHALL include vLLM adapter/token-row sync identity in
model identity provenance whenever server or colocate sync affects generation.

Normative behavior:

- model identity MUST include base/checkpoint/adapter/tokenizer/processor
  identity and backend sync identity when applicable;
- vLLM server adapter sync MUST preserve current CoordExp/ms-swift behavior:
  LoRA-compatible tensors sync through the LoRA path and token_embeddings_adapter
  offsets sync through the patched token-row update path;
- active Stage-2 vLLM server rollout MUST use
  `rollout_matching.vllm.mode=server`,
  `rollout_matching.vllm.sync.mode=adapter`, and
  `rollout_matching.vllm.enable_lora=true`;
- native `sync.mode=full` materialization is superseded for active unified
  Stage-2 rollout-correction server training unless a later OpenSpec revives it;
- PEFT `modules_to_save` and `token_embeddings_adapter` MUST NOT be sent as
  ordinary LoRA tensors;
- missing token-row sync support MUST hard-fail for token_embeddings_adapter checkpoints;
- Stage-2 train vLLM server rollouts require `sync_policy:
  per_global_step`;
- offline inference and fixed eval use `sync_policy: static`;
- sync failures visible to rank 0 before rollout MUST abort all ranks rather
  than allowing rollout or training deadlock;
- backend sync metadata MUST distinguish request acknowledgement from
  worker-side verification; without a worker acknowledgement, sync status MUST
  be recorded as requested/learner-side rather than `worker_verified`.

#### Scenario: Token-row sync support is missing

- **GIVEN** a checkpoint contains `token_embeddings_adapter`
- **AND** the selected vLLM server path lacks the patched token-row sync
  endpoint
- **WHEN** backend sync prepares generation
- **THEN** the run fails before rollout generation
- **AND** no artifact claims a model identity that omitted token-row state.

#### Scenario: Worker verification is not claimed from fire-and-forget sync

- **GIVEN** the token-row sync endpoint acknowledges request receipt before
  worker-side application is proven
- **WHEN** the backend records sync provenance
- **THEN** it records requested/learner-side sync status
- **AND** it does not stamp `worker_verified` model identity.

### Requirement: Shared runtime provenance carries compact row axes
The shared inference runtime SHALL include resolved compact template id and
object field order in prompt, parser, backend request, and artifact provenance.

Parser policy metadata MAY contain internal parser ids, but those ids MUST be
derived from the two resolved axes rather than independently authored config
knobs.

#### Scenario: Runtime parser policy includes field order
- **GIVEN** `detection_template.id: compact_object_box_closed`
- **AND** object field order `geometry_first`
- **WHEN** the shared runtime builds parser policy metadata
- **THEN** the metadata records the compact template id and `geometry_first`
- **AND** no independent compact parse-mode or separator knob is accepted as the
  source of truth.

#### Scenario: Comparable artifacts require both axes
- **GIVEN** a compact artifact family missing object field order provenance
- **WHEN** a comparable evaluation path loads the artifact
- **THEN** loading fails before metrics are compared
- **AND** the diagnostic names the missing object field order metadata.
