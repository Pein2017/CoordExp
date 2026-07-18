## Context

The current CoordExp-Swift inference path is correct for HF Qwen3-VL but is
not genuinely backend-neutral. `InferenceRuntime` always loads an HF model,
`DecodeRequest.model_inputs` carries HF tensor fields, the pipeline manually
chunks requests, and backend identity is stamped before the backend executes.
The schema and stable specs reserve `backend.type: vllm` but require it to
fail.

Current production checkpoints compose three identities: a base Qwen3-VL
model, a PEFT DoRA adapter, and an additive selected-token embedding delta over
tied input/output weights. Installed vLLM 0.14.1 supports Qwen3-VL and generated
or prompt logprobs, but rejects DoRA and has no CoordExp embedding-delta loader.
Direct adapter loading would therefore execute different weights.

The current artifact field `logprob` is produced from normalized HF generation
scores after decode processors. It drives the selected eight-token confidence
and evaluator ranking. Raw model likelihood before repetition penalty is a
different research quantity and must not silently replace that contract.

## Goals / Non-Goals

**Goals:**

- Execute the existing single-image inference and evaluator flow through HF or
  offline vLLM without backend objects leaking into pipeline or artifacts.
- Preserve exact base/DoRA/embedding-delta model semantics through a reusable,
  content-addressed execution model.
- Expose aligned policy and opt-in raw-model generated-token likelihoods.
- Reuse the existing one-visible-GPU-per-worker data-parallel controller.
- Keep current raw/scored rows, confidence semantics, and evaluator behavior.
- Make unsupported config combinations fail before expensive model loading.

**Non-Goals:**

- vLLM server or async APIs, native vLLM DoRA, tensor/pipeline parallelism,
  multi-image/video requests, stochastic sampling, grammar constraints,
  hidden-state traces, or top-k/full-vocabulary traces.
- Changing `pred[*].score`, `pred_score_version: 1`, the parser, bbox units, or
  metric reduction.
- Exporting a materialized model as a new canonical training checkpoint.

## Decisions

### Backend-owned sessions replace pipeline-composed adapters

Two designs were considered. A shallow adapter design would keep model
loading, native input projection, fixed batches, raw replay, and cleanup in the
pipeline. It minimizes initial movement but makes valid combinations the
caller's responsibility and preserves the existing HF leakage. The selected
design uses one backend session that owns those operations:

```python
launch = prepare_backend_launch(config, execution_model=execution_model)
with open_backend_session(launch, worker_context) as session:
    results = session.decode(prepared_requests)
```

The shared request contains request id, authoritative executable `chat_text`,
expected prompt token ids, image path/dimensions/content SHA-256, and shared
generation/trace policy. Human-facing task prompt text remains diagnostic and
is never a second execution input. Each backend reopens and hashes the image
immediately before decoding. HF privately builds Qwen tensors and chunks
requests. vLLM privately builds multimodal prompts and submits the rank-local
request set to its scheduler. Both return the same validated `DecodeResult`
and session receipt; vLLM prompt parity uses ids returned by `RequestOutput`,
not self-attestation.

Generated token ids are authoritative for backend normalization. vLLM 0.14.1
retains a stop-token id but omits it from native `CompletionOutput.text`.
Therefore the session reconstructs raw generated text and per-token text from
the shared tokenizer with special tokens preserved, while recording native
text separately as upstream evidence. Parser stripping is applied only after
this backend-neutral raw text exists.

When vLLM reports `finish_reason="stop"`, native `stop_reason=None`, and the
retained final token id is exactly `<|im_end|>`, backend normalization records
the semantic stop reason as `im_end` while preserving the native fields in the
session receipt. No other missing native stop reason is inferred.

`src/inference/backend.py` owns records and the protocol;
`hf_backend.py` and `vllm_backend.py` own implementations;
`execution_model.py` owns derived model identity and publication; and
`runtime.py` owns processor-only frontend and session construction. The old
`InferenceRuntime`, model-input request field, backend factories, and pipeline
batch loop are deleted after the backend-session HF implementation proves
canonical artifact equivalence. HF remains a supported first-class backend.
The backend session accepts semantic multimodal decode requests and has no
dependency on evaluator or inference-artifact orchestration. Future GRPO or
other post-training rollout owners are expected to reuse this session,
execution-model, likelihood, and worker-lifecycle boundary rather than create a
second vLLM wrapper. Building a GRPO trainer is outside this change.

The accepted support matrix has four explicit roles. Dynamic HF is the
first-class compatibility backend and loads the configured base, DoRA adapter,
and selected-token embedding delta directly. Materialized HF is the execution-
model composition oracle. vLLM FP32 is the strict cross-backend parity mode.
vLLM BF16 is a supported throughput mode, but observed BF16 token/likelihood
and val200 metric drift means it MUST NOT claim strict HF parity.

### Semantic image planning stays shared; native projection is private

Prompt rendering, expected prompt ids, decoded image dimensions and content
hash, no-resize geometry, and expected image grid remain shared CoordExp
evidence. A request also carries the validated logical geometry transform
(`identity`, `hflip`, `vflip`, or `hvflip`). Each backend reopens and
hashes the source bytes, applies that transform exactly once in memory, and
records a canonical RGB8 pixel hash for the executed media. HF materializes
pixel tensors lazily per batch. vLLM receives the same transformed,
hash-validated in-memory single-image media and `do_resize=False`.
`image_plan.jsonl` distinguishes a shared reference plan from backend-observed
evidence and never labels locally generated HF tensors as the tensors executed
by vLLM. HF records executed `image_grid_thw`; vLLM records the exact executed
media hash, actual prompt ids, multimodal placeholder ranges/count, processor
identity, and no-resize kwargs. Returned vLLM prompt ids and visual-placeholder
count must exactly equal the shared expected values.

### Existing score uses policy likelihood; raw likelihood is auxiliary

Internal `TokenTrace` stores `policy_logprob` and nullable
`raw_model_logprob`. Artifact `logprob` remains the policy value for backward
compatibility and `selected_logprobs` remains unchanged. Provenance names the
score-owned channel explicitly.

HF gathers policy likelihood from generation scores and raw likelihood from
`output_logits=True`, with FP32 `log_softmax` in both cases. A source probe
compares HF raw generation logits against a teacher-forced forward reference.
vLLM first runs with `logprobs_mode="processed_logprobs"` and treats that
completion as the only generation and policy-score authority. When raw tracing
is enabled, it closes that engine and opens a fresh rank-local engine over the
same execution snapshot with `logprobs_mode="raw_logprobs"`. A version-pinned
CoordExp logits processor forces the authoritative generated token at every
incremental decode step. vLLM computes raw FP32 log-softmax before applying
that processor, so the forced choice does not alter the reported raw
distribution; it only guarantees that the second decode follows the same
conditioning prefix. Returned prompt ids, generated ids, stop semantics, and
length MUST match before the two channels are combined. Prompt-logprob replay
is retained only as superseded qualification evidence because multimodal
prefill is not the same executed continuation state after token zero.

### Current checkpoint composition is materialized once

Base-only vLLM may load the base directory directly, but only through an
immutable execution-model receipt that exhaustively hashes every regular file
in the snapshot, including weights, config, tokenizer, processor, token
metadata, and chat templates. Every worker revalidates that receipt before
engine construction. Any adapter or embedding delta uses
`model_cache/coordexp_swift/vllm_materialized/<composition-key>/snapshot/`.
The composition key binds source model shards/config/tokenizer, every adapter
config/tensor payload, embedding metadata/tensor plus selected token identity,
target dtype, composition algorithm, and relevant library versions. The saved
snapshot receives a separate fingerprint over its actual executable bytes.
The source-determinant composition key and executable snapshot fingerprint MUST
NOT be conflated.

Materialization loads the base directly in the target dtype on CPU, validates
the DoRA payload through the adapter owner, constructs
`PeftModel.from_pretrained(..., is_trainable=False,
autocast_adapter_dtype=False)`, and calls
`merge_and_unload(safe_merge=True, adapter_names=["default"])`. It then folds
the FP32 selected-token delta exactly once into target-dtype rows of the tied
embedding/lm-head weight without a later whole-model cast, removes adapter and
parametrization residue, and saves a standard HF snapshot. Adapter and delta
identity inspection and mutation remain owned by `src.adapters.dora` and
`src.qwen.special_token_embeddings`; the execution-model module orchestrates
them rather than re-parsing their formats.

A lock at `.locks/<composition-key>.lock` serializes builders; a unique staging
directory is atomically renamed only after all hashes and the manifest
validate. `coordexp_materialization.json` lives beside, not inside, `snapshot/`
so the snapshot can be exhaustively hashed without self-reference. Existing
corrupt published entries fail rather than being silently repaired. A
checkpoint-handoff manifest may provide source paths, but it is not required;
explicit base/adapter/delta config remains a supported research composition.

The controller prepares this execution model before launching rank workers and
passes a serializable execution-model receipt. This applies to one-GPU and
multi-GPU vLLM runs. Workers validate the same fingerprint before engine
construction.

### Outer data parallelism remains authoritative

CoordExp continues to shard deterministic decode-batch blocks across fresh
workers. Unlike HF, vLLM also uses a fresh worker when only one rank is active,
because process exit is part of its qualified cleanup contract. Each worker
sees one logical `cuda:0` and opens one vLLM engine with
`tensor_parallel_size=1`, `data_parallel_size=1`, and
`max_num_seqs=generation.batch_size`. `generation.batch_size` remains the
per-device shard-block and concurrency contract; it is not divided by world
size. vLLM output order is normalized by request id before artifact writing.

### Keep the public config narrow and truthful

HF remains the shared default. Shared `model` fields contain only base identity,
dtype, and processor policy. HF-only attention implementation and patch policy
live in a required `backend.hf` block; vLLM requires only its strict
`backend.vllm` block with `gpu_memory_utilization`. The opposite backend block
is rejected, and arbitrary engine kwargs are not exposed. The initial
candidate is 0.14.1. The accepted version set is derived only from passed
runtime-qualification receipts committed under the stable
`src/inference/qualification_receipts/` runtime surface. Version changes require
updating the source-study receipt and rerunning real probes.

A runtime qualification receipt binds the probe implementation hash, exhaustive
source-base snapshot identity, image/fixture identity, installed dependency
versions, every source module loaded from the qualified vLLM, Transformers,
PEFT, and Qwen utility package boundary after generation and replay, plus named
required owners for execution, loading, sampling, multimodal processing, and
private cleanup paths, CUDA binding, engine process mode,
effective engine arguments, prompt and generated ids, stop evidence,
processed-logprob evidence, raw-replay alignment, and post-close process-tree
evidence. It qualifies a runtime and compatible model family, not only one
output checkpoint fingerprint. A base run must match the exact qualified base
snapshot. A composed execution model may differ only when its receipt points
to that qualified source base, preserves the qualified architecture,
tokenizer, processor, and template identities, and carries a passed
composition-fidelity receipt. The receipt proves exact merged target weights,
selected-token rows, and tied storage while retaining dynamic-HF behavioral
diagnostics. An unrelated model or source drift
fails before engine construction.

The receipt distinguishes semantic invariants from run-varying engine values.
TP=1, DP=1, processed-logprob mode, deterministic decoding, single-image
no-resize projection, process mode, and model-family identities are invariant.
`max_num_seqs`, maximum model length, memory sizing, and production automatic
KV-cache sizing are accepted only for values covered by an executed receipt;
actual effective arguments are always recorded. Wave 0 qualifies the narrow
single-sequence harness. Wave 3 must add a real concurrency receipt for the
largest initially supported `generation.batch_size` before production vLLM
configuration is accepted. A version without a matching passed receipt remains
unsupported even if its version string appears in source code.

The vLLM session owns a version-pinned lifecycle adapter. For qualified
uniprocess mode it explicitly calls the 0.14.1 in-process engine-core shutdown
surface, releases references, clears CUDA caches, and verifies that no child
process appeared or survived. vLLM 0.14.1 may retain compiled model tensors
until interpreter exit, so the fresh rank worker remains the final resource
boundary: its parent requires worker exit and no surviving owned process group.
Parent-observed GPU memory remains receipt diagnostics rather than a publication
gate because shared-device occupancy can change independently. Multiprocess mode, if later
qualified, requires the corresponding explicit core-client shutdown plus the
same post-worker proof. Garbage collection alone is never the cleanup contract.

Canonical scored inference requires temperature 0, top-p 1, scoring enabled,
token traces enabled, and parse diagnostics enabled. Existing false or
stochastic values fail instead of acting as no-ops. Raw likelihood is the only
new optional behavior.

## Risks / Trade-offs

- **Materialized models consume disk and CPU build time** -> content-addressed
  reuse, one controller-side build, explicit hashes, and no rebuild on a valid
  hit.
- **DoRA or tied-delta folding can subtly change weights** -> require exact
  merged-target identities, selected-token rows, and tied storage after reload;
  record dynamic-HF/materialized-HF logits and generated ids as BF16 behavioral
  diagnostics. Materialized HF is the vLLM oracle and canonical dynamic HF
  remains the val200 behavioral baseline. Use FP32 for strict cross-backend
  parity; label BF16 vLLM as throughput-only evidence.
- **vLLM processor behavior can drift from HF** -> require exact prompt ids,
  no-resize receipts, one-image limits, fixed-fixture token parity, and matched
  val200 metrics.
- **Dual vLLM likelihood requires a second engine/decode when enabled** -> keep
  raw tracing opt-in, close the processed engine before opening the raw engine,
  bind the forced-replay processor source, require exact token/stop replay, and
  fail closed. Policy-only vLLM retains one engine and one generation pass.
- **One engine per rank adds lifecycle complexity** -> distinguish the offline
  API caller from engine-owned child processes, pin the process mode,
  context-manage close, bound worker waits, inspect child processes, preserve
  terminal failure evidence, and publish no top-level artifacts after failure.
- **Deterministic-only inference narrows accepted configs** -> this converts
  existing nonfunctional knobs into explicit failures; stochastic inference
  can be added later with stable per-row seeds.

## Migration Plan

1. Add config and session contracts while preserving HF output behavior.
2. Move HF implementation behind the session and delete old factories/batching.
3. Add and verify execution-model materialization independently of vLLM.
4. Add a base-only vLLM tracer bullet, then raw likelihood replay.
5. Connect materialized checkpoints, outer data parallelism, strict merge, and
   additive artifact provenance.
6. Run fixed tiny, one-GPU, two-GPU, eight-GPU, and matched val200 gates.
7. Update canonical docs/specs only after review convergence; rollback remains
   selecting `backend.type: hf` and ignoring derived cache entries.

## Open Questions

No user-owned decisions remain. The real upstream, composition, concurrency,
forced-replay, distributed, and matched-val200 probes have executed. Final
promotion remains gated on exact-source receipt revalidation, independent
review convergence, stable-spec sync, and archival. Any future version or
engine-argument envelope requires its own executed qualification before use.
