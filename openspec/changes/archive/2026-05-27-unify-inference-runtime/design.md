## Context

The current training/inference split creates a subtle research risk: offline
evaluation and online Stage-2 rollout can differ in prompt construction,
chat-template formatting, image preprocessing, decode settings, trace capture,
and parser behavior while still producing superficially comparable artifacts.
That can make evaluation results misleading and can corrupt Stage-2 residual
correction target offsets.

The desired endpoint is not a large framework. It is a small shared runtime
root with a strict contract:

```text
sample + DetectionPromptPolicy
  -> PromptBundle
  -> DetectionDecodeRequest
  -> BackendAdapter.generate_many(...)
  -> DetectionDecodeResult
  -> DetectionParserResult / artifacts / caller-specific downstream logic
```

Offline inference, offline evaluation, online rollout generation, rollout
diagnostics, confidence/score materialization, and Stage-2 trainable rollout
segments all share the root prompt/decode/backend/parse behavior. Their
downstream branches stay separate.

## Key Decisions

### Keep `src/infer` As The Shared Root

The shared root will be flat and small:

```text
src/infer/
  __init__.py
  pipeline.py
  runtime.py
  prompt.py
  backend.py
  backend_sync.py  # private helper allowed if backend.py crosses size threshold
  parsing.py
  artifacts.py
  checkpoints.py
  constraints.py
  _compact_constraints.py       # private helper allowed if needed
  _stop_pressure_constraints.py # private helper allowed if needed
  vis.py
```

`pipeline.py` remains responsible for offline infer/eval/vis orchestration and
resolved run directories. `runtime.py` owns the prompt -> backend -> parser
sequence. `prompt.py` owns detection prompt semantics and one-image visual
input normalization. `backend.py` owns HF/vLLM adapter lifecycle, request
adaptation, trace validation, and vLLM sync provenance. `parsing.py` owns
strict/diagnostic parser policy. `artifacts.py` owns decode summaries,
trace rows, and resolved manifests.

This intentionally avoids parallel directories such as `src/decode/`,
`src/inference_runtime/`, `src/trainers/rollout_runtime/`, `src/infer/engine.py`,
or `src/infer/backends.py` as final active surfaces. Current constraint modules
such as `src.infer.compact_grammar` and `src.infer.stop_pressure` are not final
caller-facing import surfaces. They must either fold behind `constraints.py` or
become private helper modules with all callers importing through the shared
constraints facade.

### Preserve Public Config Namespaces

The refactor is internal for authoring:

- offline users continue to configure `infer.*`;
- Stage-2 users continue to configure `rollout_matching.*` for rollout
  runtime/backend/decode/eval;
- Stage-2 objectives remain under `stage2_rollout_correction.*`.

Both config families are converted into shared internal policies:

- `DetectionPromptPolicy`
- `DetectionDecodeRequest`
- `ModelIdentity`
- optional `ScorePolicy`

### Make Prompt/Input Rendering Single-Owned

`src/infer/prompt.py` owns semantic prompt-prefix and visual input rendering
for:

- offline inference;
- online Stage-2 rollout;
- teacher-forced Stage-1/Stage-2 dataset prompt-prefix encoding;
- prompt parity diagnostics and tests.

Dataset modules retain storage, indexing, cache, sampling, packing, batching,
and collation responsibilities. They do not retain an independent semantic
prompt template. Training-owned code remains responsible for assistant-target
rendering, residual slicing, target IR construction, loss masks, duplicate
control, and rollout-correction target semantics.

The supported detection generation input is exactly one image per sample. The
prompt codec validates that shape, resolves/loads the image when needed,
normalizes to PIL RGB or the agreed processor input, records original
width/height and visual metadata, and asserts `do_resize=false` for training
and comparable inference paths.

### Separate Prompt Semantics From Backend Transport

`prompt.py` produces a normalized `PromptBundle`; `backend.py` serializes that
bundle into HF processor inputs, Swift template inputs, vLLM colocate payloads,
or vLLM OpenAI/server payloads.

Backend adapters must not reload images, re-resolve dimensions, inject
semantic prompt text, reorder content, or construct caller-specific detection
templates.

### Require Canonical Decode Results And Logprob Trace Parity

All backends normalize native outputs into one `DetectionDecodeResult`.

When `trace_logprobs=true`, each generated sequence result requires:

- generated token IDs;
- generated token text;
- finite generated-token logprobs;
- stop reason and stop metadata;
- raw generated text;
- backend identity and request identity;
- prompt-token IDs and visual metadata when the backend exposes or can
  faithfully reconstruct them for parity checks.

The trace arrays must align 1:1 after allowed backend adaptation. Allowed
adaptation includes converting backend token-ID strings to ints, normalizing
per-position logprob containers, normalizing token text through the tokenizer,
and consistently dropping an explicit backend stop token only when that token
is not part of emitted text. Silent clipping, padding, defaulting missing
logprobs, or hiding shape mismatches is forbidden.

vLLM is trace-capable and is not exempt. Official vLLM documentation exposes
`SamplingParams.logprobs`, `prompt_logprobs`, logprob containers, and
OpenAI-compatible `--max-logprobs`; implementation must hard-fail when the
selected vLLM path cannot return the required trace for a trace-required
workflow.

Trace sources are backend-specific:

- HF uses generated sequences plus generation score tensors/logits processors
  to reconstruct chosen generated-token IDs and logprobs;
- vLLM Python/local/colocate uses `RequestOutput`-style token IDs and logprob
  payloads;
- ms-swift vLLM server rollout uses `return_details=true` and requires
  `prompt_token_ids`, generated `token_ids`, and generated-token logprobs;
- OpenAI-compatible HTTP responses are accepted for `trace_logprobs=true` only
  when the response provides generated token IDs and generated-token logprobs
  (or a future adapter proves exact tokenizer round-trip reconstruction for
  every generated token); otherwise they hard-fail before artifacts or metrics.

### Keep Parsing Strict For Metric-Bearing Artifacts

Metric-bearing artifacts use strict parser policy only. Strict parser errors
may produce error rows and counters, but they must not be repaired into
predictions for official metric inputs.

Diagnostic salvage parsing may exist for debug/monitor/ad hoc artifacts only
when those outputs declare `metric_bearing: false`. Salvage-recovered
predictions must not feed `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`,
guarded companions, Stage-2 official eval artifacts, confidence post-op,
COCO/LVIS/mAP, or rollout metric computation.

### Preserve Raw/Scored Artifact Separation

`gt_vs_pred.jsonl` remains the canonical strict standardized raw artifact and
does not carry confidence scores. `gt_vs_pred_scored.jsonl` is the canonical
score-bearing companion and requires `score_policy_fingerprint`.

Guarded companions preserve that split:

- `gt_vs_pred_guarded.jsonl` remains raw/guarded;
- `gt_vs_pred_scored_guarded.jsonl` remains scored/guarded.

Re-scoring a raw artifact writes a new scored artifact. It must not mutate the
raw artifact in place.

### Record Orthogonal Provenance Fingerprints

The runtime records separate fingerprints:

- `prompt_policy_fingerprint`: semantic prompt/input policy;
- `decode_policy_fingerprint`: backend family/mode, sampling/greedy settings,
  length, stop policy, tracing, and constraints;
- `model_identity_fingerprint`: weights, adapters, tokenizer, processor, and
  backend sync identity;
- `score_policy_fingerprint`: required only for score-bearing artifacts.

Comparable artifacts and caches fail fast when required fingerprints are
missing. Historical artifacts without fingerprints remain readable for
inspection but are `comparable: false` and rejected by strict comparison,
official eval, or reporting paths unless explicitly migrated.

The provenance carrier is deliberately run-level, not a per-line JSONL schema
expansion:

- `resolved_config.json` records resolved policy objects, artifact paths, and
  fingerprint values;
- `summary.json` records the same compact fingerprint values and parser/score
  policy status needed by readers;
- score-bearing artifact sidecars MAY be written for portability, but standalone
  moved JSONL files without their resolved metadata or sidecars load only as
  inspection data and must fail as `missing_provenance` in comparable paths;
- `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` line schemas remain
  compatible with the existing inference-engine contract.

Implementation must provide a lightweight provenance stamping/checking path for
historical artifacts. It may stamp exact fingerprints only when the old
`summary.json`, `resolved_config.json`, raw/scored JSONL, and required sidecars
contain enough information to reconstruct them. Otherwise it must stamp or
report `comparable: false` with an auditable reason; it must not invent
fingerprints to make old artifacts official-eval eligible.

### Stage-2 Trainable Rollout Requires Verified Prompt Parity

Stage-2 trainable rollout-correction segments require
`prompt_token_parity: verified`, including prompt-token IDs and visual metadata
parity where available. If rollout prompt IDs or visual-token metadata drift
from the teacher-forced local prefix, residual correction target offsets can be
wrong. Such samples must be dropped or the backend path rejected for trainable
rollout construction.

Ordinary metric evaluation may proceed with
`prompt_token_parity: unverifiable` only when fingerprints and strict parser
contracts are satisfied and the artifact does not claim offline-vs-online
prompt-token equivalence.

### Preserve CoordExp/ms-swift vLLM Adapter Sync Semantics

Backend lifecycle ownership moves to `src/infer/backend.py`, but the current
functional sync contract must stay intact:

- official server rollout adapter sync remains adapter-only;
- active Stage-2 vLLM server rollout requires
  `rollout_matching.vllm.mode=server`,
  `rollout_matching.vllm.sync.mode=adapter`, and
  `rollout_matching.vllm.enable_lora=true`;
- older native full-sync materialization semantics are superseded for active
  unified Stage-2 rollout-correction server training and remain historical or
  future/deferred unless a later OpenSpec revives them explicitly;
- LoRA sync sends only vLLM-compatible LoRA tensors;
- PEFT `modules_to_save` and `coord_offset_adapter` are not sent as ordinary
  LoRA tensors;
- `coord_offset_adapter` is synchronized separately as token-row offsets via
  the patched ms-swift `update_token_row_offsets` path;
- missing coord-row sync support hard-fails for coord-adapter checkpoints;
- rank-0 sync failure under DDP aborts all learner ranks;
- Stage-2 train vLLM server rollouts use `sync_policy: per_global_step`;
- offline inference/fixed eval use `sync_policy: static`.

vLLM sync identity contributes to `model_identity_fingerprint`; detailed
debug metadata is recorded under `backend_sync`.

The current ms-swift coord-row endpoint is request-acknowledged rather than a
full worker-side verification barrier. Until the server provides a real worker
acknowledgement or health proof, artifacts must distinguish
`sync_status: requested` from `sync_status: worker_verified`. The runtime may
only claim worker-verified model identity after such an acknowledgement exists.

## Proposed Architecture

### Request/Result Objects

`DetectionPromptPolicy` includes prompt variant, object ordering, object field
order, bbox format, detection sequence format, coord mode, parser policy,
template family, image placement, system/user prompt behavior, generation
prompt behavior, resize policy, and `prompt_policy_fingerprint`.

`DetectionDecodeRequest` includes backend family, backend mode, decode mode,
temperature, top-p, top-k, max-new-tokens, beams, repetition penalty, seed,
stop strings/tokens, trace flags, compact grammar/stop-pressure constraints,
and `decode_policy_fingerprint`.

`DetectionDecodeResult` contains raw generated text, generated token trace,
prompt-token/visual metadata when available, backend metadata, model identity,
stop reason, and structured errors.

`DetectionParserResult` separates strict metric-bearing predictions from
diagnostic salvage output. A result cannot be both metric-bearing and salvage
recovered.

### Caller Ownership

Shared runtime owns:

- prompt/input rendering;
- one-image generation-time normalization;
- backend lifecycle and request adaptation;
- generated sequence trace validation;
- strict/diagnostic parsing policy;
- decode/provenance artifacts.

Offline pipeline owns:

- resolved run directories;
- stage orchestration (`infer`, `eval`, `vis`);
- canonical raw/scored/guarded artifact family;
- invoking evaluators and visualizers.

Stage-2 trainer owns:

- sample selection and rollout attempt scheduling;
- learner/peer/current policy context;
- residual target construction;
- duplicate filtering and greedy IoU assignment;
- DDP coordination;
- objective execution and metric projection.

Dataset/training encoding owns:

- storage/indexing/cache layout;
- sampling;
- packing/collation;
- supervision labels and loss masks;
- calling the shared prompt codec for semantic rendering.

The shared runtime must not import or own Stage-2 target-construction symbols
such as `ResidualBoundaryAdapter`, `TeacherForcingTargetIR`,
`build_residual_set_target_ir`, rollout-correction `target_builder`, greedy IoU
matching helpers, or duplicate-control training logic.

## Migration Strategy

1. Add golden behavior, prompt parity, strict parser, provenance, trace-shape,
   and Stage-2 eval-materialization tests before moving code.
2. Keep legacy import/search-ban tests phase-scoped until the deletion phase so
   incremental slices can run useful targeted tests while migration is in
   progress.
3. Introduce shared runtime request/result/policy objects under `src/infer`.
4. Move offline inference to consume the shared runtime without changing
   artifact schemas.
5. Move Stage-2 rollout generation to consume the shared runtime while keeping
   residual target construction trainer-owned.
6. Move vLLM lifecycle/sync/client adaptation into the backend adapter layer
   while preserving existing CoordExp/ms-swift sync behavior.
7. Invert old permissive trace tests so short traces, arbitrary long-trace
   trimming, or unverified multi-token stop tails fail.
8. Delete overlapping active layout/import surfaces once callers are migrated.
9. Update docs/configs/tests and run search gates to ensure legacy public or
   import surfaces are not active.

## Alternatives Considered

### Keep Thin Compatibility Shims

Rejected for final state. The user explicitly approved refactoring without
compatibility consideration across relevant files, including `scripts/`.
Keeping alias modules would preserve the very ambiguity this refactor is meant
to remove.

### Add A New `src/decode` Package

Rejected. The repo already has `src/infer` as the operator-facing inference
root. A second decode root would make ownership harder to discover.

### Split Backend Variants Into Many Files Immediately

Rejected for now. The primary problem is overlapping ownership and too many
variant surfaces, not insufficient files. A single `backend.py` is preferred
unless the implementation crosses a concrete complexity threshold.

### Allow Diagnostic Salvage Into Eval Inputs With A Flag

Rejected. This would make metric claims dependent on repair behavior and would
blur parser failures with model quality.

## Verification Strategy

- OpenSpec review by independent subagents before implementation.
- Superpowers implementation plan review by independent subagents before code.
- Targeted tests for config mapping, prompt parity, one-image preprocessing,
  backend trace result contract, vLLM logprob hard-fail behavior, strict vs
  diagnostic parser policy, raw/scored artifact separation, fingerprints, and
  Stage-2 trainable prompt parity.
- Search gates for deleted active import surfaces:
  - `src.trainers.rollout_runtime`
  - `src.infer.engine`
  - `src.infer.backends`
  - old rollout helper entrypoints after migration.
- Targeted Stage-2 and inference/eval test families before broad runs.

## Approval Gate

This OpenSpec change and the repo-local Superpowers implementation plan are
planning artifacts. Production code implementation starts only after explicit
final user approval.
