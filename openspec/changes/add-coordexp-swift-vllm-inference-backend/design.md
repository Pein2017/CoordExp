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
batch loop are deleted after HF parity is proven.

### Semantic image planning stays shared; native projection is private

Prompt rendering, expected prompt ids, decoded image dimensions and content
hash, no-resize geometry, and expected image grid remain shared CoordExp
evidence. HF materializes pixel tensors lazily per batch. vLLM receives the
same hash-validated in-memory single-image media and `do_resize=False`.
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
vLLM runs with `logprobs_mode="processed_logprobs"`; when raw tracing is
enabled it submits `input_prompt_token_ids + generated_ids` with the same image
using prompt logprobs, verifies the returned executed prefix equals
`expected_executed_prompt_token_ids + generated_ids`, and extracts continuation
positions. It MUST NOT replay the already-expanded visual-placeholder prefix.
Replay-generated throwaway tokens are never included in artifacts.

### Current checkpoint composition is materialized once

Base-only vLLM may load the base directory directly, but only through an
immutable execution-model receipt that exhaustively hashes every regular file
in the snapshot, including weights, config, tokenizer, processor, token
metadata, and chat templates. Every worker revalidates that receipt before
engine construction. Any adapter or embedding delta uses
`model_cache/coordexp_swift/vllm_materialized/<fingerprint>/`. The fingerprint
binds source model shards/config/tokenizer, adapter config/tensor, embedding
metadata/tensor, dtype, composition algorithm, and relevant library versions.

Materialization loads the base on CPU, validates and merges DoRA through PEFT,
folds the selected-token delta exactly once into the tied embedding/lm-head
weight, removes adapter/parametrization residue, and saves a standard HF
snapshot. A lock serializes builders; a unique staging directory is atomically
renamed only after all hashes and the manifest validate. Existing corrupt
published entries fail rather than being silently repaired.

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
qualification receipts committed under this change. Version changes require
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
dynamic-HF/materialized-HF parity receipt. An unrelated model or source drift
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
boundary: its parent requires worker exit, no surviving descendants, and GPU
memory returned to the pre-worker baseline. Multiprocess mode, if later
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
- **DoRA or tied-delta folding can subtly change weights** -> compare dynamic
  HF and materialized HF logits, selected-token logits, generated ids, and tied
  storage before vLLM is accepted.
- **vLLM processor behavior can drift from HF** -> require exact prompt ids,
  no-resize receipts, one-image limits, fixed-fixture token parity, and matched
  val200 metrics.
- **Raw prompt-logprob behavior is upstream-version-sensitive** -> pin one
  version, record source handles, use repeated-token probes, and fail closed.
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

No user-owned decisions remain. Wave 0 has executed the real upstream probe;
its corrected exhaustive-identity and executed-media receipt must pass the
second independent review gate before Wave 1 begins. Any later probe failure
stops at its wave gate and revises the change before another version or engine
argument envelope is qualified.
