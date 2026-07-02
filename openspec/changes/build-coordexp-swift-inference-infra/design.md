# CoordExp-Swift Inference Infrastructure Design

## Context

CoordExp-swift has rebuilt the training infrastructure around local ownership
of data, templates, Qwen encoding, packing, supervision, losses, runtime,
artifacts, metrics, checkpointing, and forward eval. The repository still needs
a self-contained offline inference path so trained checkpoints can be decoded,
scored, and evaluated without depending on legacy `src/infer/*` behavior or
MS-Swift runtime wiring.

The design baseline is the proposal-scoped inference section in
`docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`. Legacy
`configs/infer/*` and old `/data/CoordExp/src/infer/*` remain reference-only
material. They can inform source studies, but they are not current schema,
artifact, or implementation authority.

The V1 user goal is offline benchmark inference for Qwen3-VL single-image
detection checkpoints. The first backend is Hugging Face Transformers
`model.generate`. vLLM support is required in the future, but V1 only reserves
backend-neutral result fields and response-family metadata.

## Goals / Non-Goals

**Goals:**

- provide a strict inference config and thin `python -m src.infer --config CONFIG.yaml`
  entrypoint;
- reuse CoordExp-swift Qwen/template/adapter/artifact ownership instead of
  importing `TrainConfig` or calling `src.training.pipeline`;
- run batched HF/Qwen generation with deterministic default decoding and full
  trace evidence;
- preserve prompt-token parity, no-resize image-plan evidence, and Qwen
  `<|im_end|>` stop-token semantics;
- parse compact object-box-closed output into canonical detection rows while
  preserving raw decode and invalid-span diagnostics;
- compute score-bearing predictions from selected schema and coordinate token
  logprobs with a fixed length-invariant formula;
- write raw, scored, trace, image-plan, diagnostics, resolved-config, manifest,
  and provenance artifacts;
- provide a concrete mAP acceptance path through a minimal `src.eval` detection
  consumer or explicitly documented compatibility bridge;
- keep implementation test-first and review-gated by public contract surface.

**Non-Goals:**

- no implementation during this OpenSpec/superpower drafting loop;
- no interactive demo surface;
- no Stage-2 rollout decoding backend;
- no vLLM implementation in V1;
- no grammar-constrained decoding default;
- no multi-image, video, or resize-enabled inference;
- no parsing of old JSON assistant response formats in V1;
- no merged full-model export requirement;
- no reliance on mocked backend tests as acceptance evidence.

## Decisions

### Entry And Module Layout

Use a thin public entry file:

```text
src/infer.py
```

and implementation modules under:

```text
src/inference/
  __init__.py
  runtime.py
  backend.py
  prompt.py
  parsing.py
  scoring.py
  artifacts.py
  pipeline.py
```

Rationale: this preserves the familiar role-named entry style of
`python -m src.train` while avoiding the Python filesystem collision of trying
to create both `src/infer.py` and `src/infer/`.

Alternative considered: make `src/infer/` a package with `__main__.py`.
Rejected for V1 because it diverges from the existing training entry style and
does not improve the first implementation.

### Config And Runtime

`InferConfig` is a strict schema, not a subset of `TrainConfig`. It includes
`schema_version`, `run` / `artifact_root`, shared `model`, optional `adapter`,
optional special-token embedding delta, `data`, `template`, `backend`,
`generation`, `scoring`, `artifacts`, and `debug`.

Inference configs live under `configs/coordexp_swift/infer/`. Legacy
`configs/infer/*` is reference-only. Production leaves must make dataset,
adapter/checkpoint, generation, and artifact fields explicit even if they
inherit shared model/template defaults from a shallow base.

Runtime setup assembles existing deeper owner surfaces:

- `src/config` owns strict config loading and resolved-config artifacts through
  inference-safe APIs that do not require `ResolvedTrainConfig`;
- `src/qwen` owns Qwen model/processor/tokenizer loading and image planning
  through config-neutral options, not `TrainConfig`;
- adapter and special-token delta identity checks stay with adapter/Qwen and
  artifact helpers, but their inference-facing functions must not require
  training schedule or checkpoint classes;
- shared artifact helpers expose type-neutral primitives, while inference
  artifact writers avoid `ResolvedStepSchedule` and training checkpoint
  semantics.
- `src/inference/runtime.py` coordinates those seams for inference.

`src/inference/runtime.py` must not import `TrainConfig` or
`src.training.pipeline`. The implementation plan must include an owner-boundary
residue check that prevents inference-facing modules from depending on
`TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, or
`src.training.*` except for explicitly approved type-free utilities. The V1
default is an empty `src.training.*` allowlist; any exception must be named in
the source-study note and patched into this OpenSpec before source
implementation continues.

### Model, Adapter, And Delta Reload

Inference loads a base model from `model_cache/<model-id>`, then optionally applies an
explicit adapter checkpoint and optional special-token embedding delta. V1
supports `checkpoint-final` metadata and explicit concrete checkpoint paths.
`best_acc_top1` aliases are deferred.

Adapter/delta reload must validate identity instead of trusting path shape:

- adapter base identity matches the resolved base model;
- PEFT load results have no missing adapter keys or unexpected keys;
- warning-only PEFT load paths are insufficient unless equivalent
  missing/unexpected-key evidence is captured;
- `set_adapter` activates the expected adapter name;
- `get_model_status()` reports enabled adapters, the expected active adapter
  list, no irregular fields, and no unexpected merged state;
- special-token delta metadata validates expected base config fingerprint,
  tokenizer fingerprint, token strings, and token ids;
- artifacts record resolved adapter and delta payload identities.

`src/artifacts/checkpoint_reload.py` may guide checkpoint-final metadata
resolution, but it must not be treated as a generic base-only runtime loader.
Base-only, explicit adapter, and base-plus-delta inference paths need their own
runtime identity checks.

Rationale: mAP artifacts are only meaningful if the actual composed model
matches the recorded model identity.

### Backend Trace

The backend contract is backend-neutral even though V1 implements only HF. It
defines `DecodeRequest`, `DecodeResult`, and `TokenTrace`, with reserved fields
for future `backend`, `backend_mode`, and `response_family`.

HF scored generation must request `return_dict_in_generate=True` and
`output_scores=True`. The implementation must pin prompt-width alignment,
generated-token indexing, padding-after-stop exclusion, and the score gathering
method. V1 should use `compute_transition_scores(SEQUENCES, SCORES, normalize_logits=True)`
or a proven equivalent `log_softmax(scores[t])` gather. Every generated token
trace item records step index, token id, token text, logprob, stop/pad flags,
and backend source.

Raw trace decoding preserves special tokens. Parser-facing text can strip the
terminal `<|im_end|>` only under a recorded policy.

### Prompt And Image Parity

Prompt construction stays aligned with training templates. Inference prompt
helpers live under `src/inference/prompt.py` but delegate to shared template
semantics rather than copying a new prompt language.

Tiny and smoke runs must compare local rendered prompt token ids with backend
prompt token ids. Production artifacts record prompt/template fingerprints.

Image processing is no-resize. V1 must call Qwen image processing with
`do_resize=False` and write mandatory `image_plan.jsonl` rows with declared
dimensions, decoded dimensions, patch/merge sizes, expected and observed
`image_grid_thw`, raw patch rows, merged visual-token counts, and batch-order
indices. Runtime must also verify processor/model vision identity: processor
`patch_size`, `merge_size`, and `temporal_patch_size` must match the model
vision config fields used by Qwen3-VL before inference can be benchmark
eligible.

### Parser And Geometry

V1 parser scope is compact object-box-closed output only. It preserves model
prediction order and does not geo-sort decoded predictions after generation.

`src/inference/parsing.py` owns generated-text parsing, span salvage, and
invalid-span diagnostics. Coordinate-token recognition, bbox validation, and
norm1000-to-pixel conversion should reuse or deepen shared `src/templates`,
`src/qwen`, and `src/data/geometry` semantics.

Every parsed row carries inline parser status and metric-eligibility fields.
`parse_diagnostics.jsonl` is a detailed sidecar keyed by stable row id or line
index; it does not replace inline row diagnostics.

### Scoring And Artifacts

V1 score is fixed as:

```text
pred[*].score = exp(sum(selected_token_logprobs) / n_selected)
```

Selected tokens are exactly the schema wrapper tokens
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
`<|box_end|>` plus four coordinate tokens for the parsed object span. A valid
V1 compact object therefore has `n_selected == 8`. Free-text
description/category text is excluded in V1. The parser object span is the
source of truth for trace alignment. It must map to one contiguous generated
token interval; selected tokens may have gaps only where description/category
tokens sit inside that interval. Missing, duplicated, or ambiguous wrapper or
coordinate alignment invalidates the affected object for scored output.
Scoring artifacts must persist replay handles: row id, object span id,
generated-step indices, token ids/text, selected logprobs, and score-policy
fingerprint.

`gt_vs_pred.jsonl` is diagnostic-capable but still preserves exactly one row per
input row. Additional diagnostics belong in sidecars. `gt_vs_pred_scored.jsonl`
preserves exactly one row per raw row with identical image identity, image
dimensions, GT payload, row order, and record index. Invalid or unscoreable
predictions are excluded from scored `pred`; rows with no scoreable predictions
remain present with `pred: []`.

Scored artifacts require evaluator-readable provenance. `run_manifest.json`
records broad run identity, but score-bearing portability comes from row-local
`pred_score_source`, row-local `pred_score_version`, bounded numeric scores,
and `gt_vs_pred_scored.jsonl.provenance.json` binding the scored artifact to
the source raw artifact, prompt, decode, model, processor, template, parser, and
score policy identities.

### Evaluation Ownership

`src.infer` produces inference artifacts. It does not own metric reduction.
The V1 default mAP consumer is a minimal rebuilt `src.eval` detection consumer
for `gt_vs_pred_scored.jsonl`. A legacy/current evaluator bridge is not a V1
default; it requires explicit user approval before implementation and must name
the command, row schema, score fields, output metrics, and compatibility
boundary.

### Review And Approval Gates

Implementation remains blocked after this drafting loop. The implementation
roadmap must require approval gates for:

- config/runtime;
- backend trace;
- prompt/image parity;
- parser/geometry;
- scoring/artifacts;
- eval-consumer compatibility;
- production benchmark acceptance.

## Risks / Trade-offs

- HF `generate` scores can be misaligned by one step in batched decode →
  mitigate with explicit prompt-width, transition-score, token-text, stop/pad,
  and shape assertions.
- No-resize image processing can drift silently from training →
  mitigate with mandatory `image_plan.jsonl` and real HF/Qwen smoke evidence.
- Adapter directories can load with warning-level PEFT irregularities →
  mitigate with fatal identity/status checks and adapter-enabled reload smoke.
- Scored artifact row drops can inflate or corrupt mAP →
  mitigate by preserving one scored row per input row and dropping only
  unscoreable prediction objects.
- Parser salvage can accidentally feed non-metric rows to mAP →
  mitigate with explicit `metric_bearing` and parser-policy fields.
- Adding an eval consumer may broaden V1 scope →
  mitigate by implementing only the minimal detection artifact consumer needed
  for `gt_vs_pred_scored.jsonl` mAP acceptance.
- Reserving vLLM too aggressively could overdesign V1 →
  mitigate by reserving only backend-neutral fields and response-family names,
  not building a registry or vLLM adapter.

## Migration Plan

1. Draft and validate this OpenSpec change and the matching superpower plan.
2. Run isolated review lanes and patch accepted P0/P1 findings.
3. Wait for explicit user approval to run source studies and probes.
4. Patch this OpenSpec if source studies contradict a contract, then wait for
   explicit user approval before source implementation.
5. Implement contract slices test-first in the order listed in `tasks.md`.
6. Run tiny real HF/Qwen smoke, adapter-enabled real smoke, evaluator fixture,
   and final full benchmark acceptance before claiming inference correctness.

Rollback is straightforward before implementation: archive or remove this
change and continue using existing legacy inference workflows outside
CoordExp-swift. After implementation, rollback means disabling the new
`configs/coordexp_swift/infer/*` entrypoints and keeping legacy inference
reference workflows unchanged.

## Open Questions

- The first production benchmark leaf still needs the exact dataset path,
  adapter checkpoint path, artifact root, and mAP command named before launch.
- Any request to use a legacy evaluator bridge instead of the rebuilt `src.eval`
  consumer needs an explicit user decision because it changes V1 ownership.
