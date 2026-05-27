# Tasks

These tasks are approval gates. Production-code implementation resumed after
the OpenSpec and Superpowers implementation plan were reviewed and the user
gave explicit final approval.

Current implementation status (2026-05-26): the worktree already contained
partial implementation scaffold under `src/infer/*` and related tests when
implementation resumed. Preserve and audit that scaffold before marking slices
complete.

## 1. Spec And Plan Review

- [x] 1.1 Draft the OpenSpec change for shared inference runtime behavior.
- [x] 1.2 Review the OpenSpec change with subagents for contract clarity,
  implementation feasibility, vLLM/logprob/sync risk, and eval-validity risk.
- [x] 1.3 Revise proposal/design/spec/tasks from review findings.
- [x] 1.4 Create or update the Superpowers design/implementation plan from the
  converged OpenSpec contract.
- [x] 1.5 Review the Superpowers plan with subagents for missing tests,
  over-complexity, deletion risk, and implementation sequencing.
- [x] 1.6 Revise the Superpowers plan from review findings.
- [x] 1.7 Wait for explicit user approval before editing production code.

## 2. Contract Tests

- [x] 2.1 Add golden behavior/parity tests before movement; keep final
  import-ban/search-gate tests phase-scoped until deletion.
- [x] 2.2 Add tests for mapping offline `infer.*` and Stage-2
  `rollout_matching.*` config families into shared decode request objects.
- [x] 2.3 Add prompt parity tests covering offline inference, online rollout,
  and teacher-forced encoding for the same sample and prompt policy.
- [x] 2.4 Add one-image validation tests and `do_resize=false` visual metadata
  tests.
- [x] 2.5 Add backend result trace tests requiring aligned token IDs, token
  text, and finite generated-token logprobs when `trace_logprobs=true`.
- [x] 2.6 Add vLLM-specific logprob tests that hard-fail when the selected vLLM
  path omits logprobs, clips traces, pads traces, or returns shape mismatches.
- [x] 2.7 Add strict parser tests proving diagnostic salvage cannot enter
  metric-bearing artifacts.
- [x] 2.8 Add raw/scored artifact tests proving `gt_vs_pred.jsonl` remains
  unscored and score-bearing artifacts require `score_policy_fingerprint`.
- [x] 2.9 Add Stage-2 tests requiring verified prompt-token/visual parity for
  trainable rollout-correction segments.
- [x] 2.10 Add vLLM adapter-sync provenance tests covering LoRA digest,
  coord-row digest/status, requested-vs-worker-verified sync status, sync
  policy, and rank-symmetric failure semantics.
- [x] 2.11 Add legacy artifact provenance tests covering old-style
  `gt_vs_pred*.jsonl` inspection-only loading, `missing_provenance` failure for
  official eval, and exact-stamp-or-refuse migration behavior.
- [x] 2.12 Add Stage-2 eval materialization tests preserving
  `eval_detection/step_<global_step>/` and the frozen artifact family.

## 3. Shared Runtime Core

- [x] 3.1 Add shared policy/request/result dataclasses or typed objects under
  `src/infer`.
- [x] 3.2 Add the shared detection prompt codec and one-image visual input
  normalizer.
- [x] 3.3 Add backend adapter lifecycle for HF and vLLM local/colocate/server
  paths.
- [x] 3.4 Add generated-token logprob trace normalization and validation.
- [x] 3.5 Add strict/diagnostic parser facade with metric-bearing guards.
- [x] 3.6 Add artifact/provenance helpers for prompt, decode, model, score, and
  backend-sync metadata.
- [x] 3.7 Add a provenance stamping/checking helper for historical artifacts
  that never invents missing fingerprints.
- [x] 3.8 Move compact grammar and stop-pressure behavior behind the shared
  constraints facade or private helper modules.

## 4. Offline Inference And Eval Integration

- [x] 4.1 Route `scripts/run_infer.py` and `src/infer/pipeline.py` through the
  shared runtime.
- [x] 4.2 Preserve `gt_vs_pred.jsonl`, `summary.json`,
  `resolved_config.json`, and token-trace artifact schemas.
- [x] 4.3 Preserve score-aware evaluation through `gt_vs_pred_scored.jsonl`
  and guarded scored companions.
- [x] 4.4 Update offline docs/config examples only after behavior is wired and
  tests pass.

## 5. Stage-2 Rollout Integration

- [x] 5.1 Route Stage-2 rollout generation through the shared runtime while
  preserving `rollout_matching.*` as the authored runtime namespace.
- [x] 5.2 Preserve residual target construction, duplicate filtering, greedy
  IoU assignment, DDP coordination, objective execution, and metric projection
  outside `src/infer`.
- [x] 5.3 Preserve current CoordExp/ms-swift vLLM adapter sync and coord-row
  update behavior under the shared backend adapter.
- [x] 5.4 Ensure wrong live rollout tokens remain roll-in context only and do
  not enter oracle valid sets.
- [x] 5.5 Preserve Stage-2 eval materialization artifacts under
  `eval_detection/step_<global_step>/`.
- [x] 5.6 Update Stage-2 docs/config examples only after behavior is wired and
  tests pass.

## 6. Delete Overlapping Layout

- [x] 6.1 Migrate callers from `src.trainers.rollout_runtime.*`.
- [x] 6.2 Migrate or reduce `src.trainers.stage2_rollout_runtime` so it no
  longer owns prompt/backend/decode/trace/parser behavior.
- [x] 6.3 Migrate callers from `src.infer.engine`; `src.infer.backends` has
  been folded into `src.infer.backend` and removed.
- [x] 6.4 Migrate callers from `src.infer.compact_grammar` and
  `src.infer.stop_pressure` to the shared constraints facade.
- [x] 6.5 Delete trainer-local rollout runtime modules after all callers are
  migrated.
- [x] 6.6 Delete legacy infer engine/backend modules after all callers are
  migrated.
- [x] 6.7 Update `scripts/`, callbacks, analysis helpers, tests, docs, configs,
  and non-archived OpenSpec references in the same cleanup.
- [x] 6.8 Invert or delete old permissive trace tests that accept short traces,
  arbitrary clamping, or unverified multi-token stop-tail trimming.

Status update (2026-05-26): incremental deletion-phase cuts are verified but
not complete enough to close 6.1/6.3. `src.infer.runtime` no longer imports
`src.infer.engine` for legacy generation result construction. Stop-pressure
policy constants moved from `src.infer.engine` to the shared
`src.infer.constraints` facade, and `src.infer.pipeline` now imports those
policy literals from the facade. vLLM EngineArgs compatibility validation moved
from `src.trainers.rollout_runtime.vllm_compat` into `src.infer.backend`, and
the old trainer-local `vllm_compat.py` module was deleted with an import gate.
Stage-2 rollout metrics also no longer use the duplicate
`_decoding_cfg()`/`_decoding_params()` readers; decode metric fields are
derived from the canonical rollout `DetectionDecodeRequest`. The heavier
ms-swift server adapter-sync and coord-row patch paths remain trainer/launcher-owned
for now. ms-swift/vLLM `RequestConfig` base kwargs now also come from a shared
`src.infer.backend` projection over `DetectionDecodeRequest`; call-site-specific
seed and logprob overlays remain in the trainer rollout modules. Stage-2 eval
score-provenance decode fingerprints are now derived from the canonical
rollout decode request, with only eval backend/mode and trace-logprob policy
overridden for the eval artifact surface.
HF `GenerationConfig` sampling projection now also lives in
`src.infer.backend` and is consumed from the canonical rollout decode request.
ms-swift/vLLM choice-response trace normalization now routes through
`src.infer.backend.normalize_vllm_trace_response`, and the old server/colocate
short-trace and long-trace clipping paths were removed. Active tests now
fail-fast on arbitrary trace shape mismatches instead of accepting clipped,
padded, or fallback-sized token-logprob traces.
The tiny trainer-local ms-swift infer helper was also moved into
`src.infer.backend` and deleted, with an import gate ensuring
`src.trainers.rollout_runtime.swift_infer_compat` stays removed.
Reviewer follow-up: confidence-postop Stage-2 eval now treats generated-token
trace violations as fatal for metric-bearing eval. The vLLM trace-required eval
path no longer retries/skips samples after trace exceptions, and malformed or
missing confidence traces no longer downgrade to constant-score artifacts.
ms-swift `RequestConfig` construction now also lives in `src.infer.backend`;
trainer rollout modules provide only explicit overlays such as deterministic
seed and `trace_logprobs`.
Code-quality follow-up: stale confidence trace fallback counters and effective
fallback score-mode plumbing were removed after the fail-fast conversion.
Malformed colocate vLLM outputs now raise instead of becoming empty rollouts,
and current canonical docs route inference backend ownership to
`src.infer.backend` / `src.infer.runtime` instead of deleted
`src.infer.backends`.
Server-mode vLLM response parsing now also routes directly through
`src.infer.backend.normalize_vllm_trace_response`; the Stage-2
`_parse_vllm_server_output*` wrappers were removed.
Reviewer follow-up: `backend_mode="ms-swift"` choice responses now fail fast
instead of falling through to OpenAI-compatible parsing when server
`return_details` fields are missing. Legacy Stage-2 tests were updated to
construct canonical `rollout_matching_cfg` decode inputs instead of patching the
deleted `_decoding_params` helper, and canonical `channel="rollout_correction"`
now receives the telemetry and FN-desc weighting previously gated on internal
`channel == "B"`.
Canonical-channel cleanup: active Stage-2 execution paths now use
`rollout_correction` for post-rollout packing buffers, timing logs,
teacher-forcing masks, FN-desc weighting, and residual target provenance. The
remaining Channel-A/B literals in active source are rejection/legacy-key
validation surfaces.
Offline pipeline caller migration: `src.infer.pipeline._run_infer_stage` no
longer imports or constructs `src.infer.engine` directly. It now delegates
canonical inference/decode kwargs through `src.infer.runtime.run_offline_inference`.
This is still an intentional compatibility seam internally backed by the legacy
engine until Stage-1 callback, analysis/debug helpers, and legacy engine tests
move in later deletion slices.
Stage-1 eval callback migration: `src.callbacks.stage1_detection_eval` now uses
the same runtime-owned offline inference seam for live-model eval, preserving
model train/eval restoration and cached processor handoff without importing
legacy engine classes directly.
Reviewer follow-up: offline `infer.backend.type=vllm` with
`infer.generation.trace_logprobs=true` now fails fast before backend execution or
artifact writes until the unified offline vLLM trace adapter is implemented.
This prevents comparable artifacts from carrying a trace-required decode
fingerprint without enforced generated-token logprob traces.
Scripts/debug helper cleanup: `scripts/tools/dump_rollout_text.py` and
`scripts/analysis/debug_rollout_collection_parity.py` now call the runtime-owned
offline debug generation seam instead of importing `src.infer.engine`
directly. The debug seam preserves one output row per requested input row and
fails fast on helper row-count mismatches so image-load or generation failures
cannot silently truncate diagnostic comparisons.
Analysis-source cleanup: active `src/analysis` diagnostic harnesses now acquire
legacy debug-only inference helpers through `src.infer.runtime` factories rather
than importing `src.infer.engine` directly. This keeps the current diagnostic
behavior available while centralizing the remaining compatibility bridge under
the shared runtime. The active `src`/`scripts` direct-import search gate is now
clean except for the intentional bridge in `src.infer.runtime` and the legacy
engine module itself.
Low-risk test cleanup: prompt parity, compact-full policy, and stop-pressure
tests now validate through prompt/runtime seams instead of importing the legacy
engine directly. Remaining direct test bindings are isolated to the dedicated
legacy engine batch/facade tests and must be migrated, inverted, or deleted
before closing the final engine-removal tasks.
Legacy batch-decoding test containment: `tests/test_infer_batch_decoding.py`
no longer imports `src.infer.engine` directly. It uses runtime-owned
test/compatibility factories while preserving still-valuable coverage for
artifact row emission, token-trace sidecars, loader behavior, and distributed
merge behavior until those checks are split into final shared-runtime tests.
Backend-sync ownership cleanup: CoordExp's ms-swift coord-row patch moved from
`src.trainers.rollout_runtime.swift_coord_row_patch` to
`src.infer.backend_sync`, and launcher/server/test imports now use the shared
inference backend-sync namespace. The constant worker-extension class path was
updated to `src.infer.backend_sync.CoordExpWeightSyncWorkerExtension`, preserving
the existing adapter sync and coord-row update behavior while reducing trainer
runtime ownership.
Trainer rollout-runtime package deletion: the remaining trainer-local rollout
backend modules were moved to shared inference ownership:
`dispatch.py` -> `src.infer.rollout_dispatch`,
`vllm_config.py` -> `src.infer.backend_vllm_config`,
`vllm_engine.py` -> `src.infer.backend_vllm_engine`,
`vllm_infer.py` -> `src.infer.backend_vllm_infer`, and
`vllm_server.py` -> `src.infer.backend_vllm_server`. Active `src`/`scripts`
callers no longer import `src.trainers.rollout_runtime.*`, and the old package
files are deleted. Stage-2 still owns residual rollout-correction target
construction and orchestration through `src.trainers.stage2_rollout_runtime`
until that facade is reduced or deleted in a later slice.
The layout import gate now also scans active Python under `src`, `scripts`, and
`tests` to prevent new `src.trainers.rollout_runtime` imports.
Stage-2 trainable residual rollout-correction now requires verified prompt-token
parity for nonempty teacher-forced prompt spans and requires verified one-image
visual metadata before residual target IR construction, regardless of backend.
Missing backend prompt IDs, missing prepared visual metadata, or visual metadata
drift drop the sample and emit dedicated prompt parity counters instead of
constructing trainable residual targets.
vLLM adapter-sync provenance tests now cover stable LoRA payload digests,
coord-row digest/status, requested-vs-worker-verified status, per-step adapter
sync policy, endpoint call order, and DDP rank-symmetric failure broadcast.
The active sync path stores the identity on
`_vllm_server_last_backend_sync_identity` while preserving the existing
ms-swift adapter endpoint followed by CoordExp coord-row endpoint behavior.
Reviewer follow-up: active coord adapters now have direct old-client hard-fail
coverage when `update_token_row_offsets` is missing, and Stage-2 eval
score-provenance model identity now incorporates the stored backend-sync
identity so LoRA/coord-row/sync-step changes alter metric-bearing model
fingerprints.
Code-quality follow-up: Stage-2 eval now includes backend-sync identity only
for server-mode vLLM eval surfaces, preventing stale vLLM sync state from
perturbing HF or colocate eval model identities. vLLM sync provenance also records a stable
CoordExp server/worker implementation identity rather than only endpoint names.
Duplicate-control diagnostic metrics now read and emit the documented
`stage2_rollout_correction/correction/dup/*` namespace consistently.
Active `src`/`scripts` direct-import gates are now clean for both deleted
`src.trainers.rollout_runtime.*` modules and legacy `src.infer.engine` callers.
The only remaining legacy engine dependency is the intentionally centralized
compatibility bridge under `src.infer.runtime`, so engine module deletion itself
remains open under task 6.6.
Docs/spec contract checks covering artifact docs, legacy-surface absence, and
removed-mechanism docs/catalog guards pass in the `ms` environment after the
namespace/provenance follow-ups.
Offline runtime routing checks pass for shared decode request propagation,
run-infer legacy CLI wiring, Stage-1 eval callback routing, batch artifact
schema preservation, prompt/input codec parity, and runtime backend facade
normalization. This closes the implemented offline routing and artifact schema
preservation tasks while leaving final legacy engine deletion open.
Offline vLLM server trace now flows through the shared result normalizer instead
of being rejected before runtime. Trace-required OpenAI-compatible vLLM requests
ask for logprobs and token IDs, normalize strict generated-token result fields,
and fail fast if the server omits token IDs/logprobs; local vLLM trace remains
strictly validated before use. The broader offline/trace batch passed in `ms`.
Stage-2 objective/planning coverage passes after restoring the
`stage2_rollout_correction` wrapper helper exports. This verifies the shared
rollout runtime routing, residual target construction, duplicate filtering,
greedy IoU assignment, supervision planning, pending metric aggregation, and
roll-in-only handling for wrong live rollout tokens.
Current docs now route offline inference and Stage-2 rollout backend/decode/trace
ownership through `src.infer.runtime` and `src.infer.backend`, while preserving
`rollout_matching.*` as the authored Stage-2 runtime namespace.
Reviewer follow-up: Stage-2 score-policy config metrics now map to
`eval/config/*` instead of polluting `eval/runtime/*`, and active search gates
are clean for deleted trainer rollout-runtime imports, legacy engine direct
imports outside the runtime bridge, and drifted duplicate-control metric
namespaces.
Stage-2 runtime ownership reduction progressed but remains open. Rollout prompt
message normalization, dense prompt rebuild, vLLM system prompt injection,
single-image side-channel normalization, and prompt visual metadata stamping now
live in `src.infer.prompt.prepare_rollout_prompt_samples`, with trainer runtime
providing only policy/config values. HF rollout generation, max-position guard,
compact-grammar logits-processor assembly, and greedy generated-token logprob
trace capture now live in `src.infer.backend`, while trainer methods remain
narrow orchestration wrappers. vLLM colocate/server decode-request resolution no
longer calls trainer-private `_resolve_rollout_decode_request`; backend adapters
use shared runtime helpers or explicit decode requests. Reviewer findings fixed
in this slice: absent `rollout_matching.decode_mode` now infers sampling from
nonzero `rollout_matching.decoding.temperature`, `src.infer.prompt` and
`src.infer.runtime` import without heavy backend dependencies, and `src.infer`
modules no longer import `src.trainers.*`. Task 6.2 is still not closeable:
Stage-2 runtime still owns orchestration wrappers, vLLM lifecycle owner methods,
and eval parser/metric-artifact assembly pending a later parser/context split.
Reviewer follow-up (2026-05-27): 6.2 is explicitly reopened until
`src.trainers.stage2_rollout_runtime` is reduced to trainer-owned orchestration
only or removed/renamed. Active scripts and canonical docs now route shared
inference ownership to `src.infer.*`, but this final Stage-2 runtime reduction
gate remains incomplete.
Additional reduction progress (2026-05-27): colocated vLLM lazy engine
initialization moved to `src.infer.backend_vllm_engine.ensure_vllm_engine`, and
Stage-2 eval confidence trace validation/scoring plus raw rollout artifact
payload shaping moved to `src.infer.artifacts`. The trainer runtime now
delegates those slices, but 6.2 remains open because Stage-2 eval orchestration
and vLLM lifecycle wrappers still live in `src.trainers.stage2_rollout_runtime`.
Follow-up reduction (2026-05-27): vLLM server config normalization, server-list
validation, timeout policy, world-size discovery, adapter-sync mode validation,
and per-rank server chunk sizing now live in `src.infer.backend_vllm_server`.
`Stage2RolloutRuntime` retains thin wrappers for existing callers/tests. This
shrinks backend lifecycle ownership, but 6.2 remains open until the remaining
wrappers/eval orchestration are reduced or moved.
Policy-reader reduction (2026-05-27): rollout backend selection, current
rollout context detection, vLLM mode normalization, and context-specific decode
batch-size validation now live in `src.infer.runtime`. Active `src.infer`
callers use those shared helpers instead of trainer policy methods; the trainer
keeps thin wrappers for compatibility with existing internal call sites.
Final Stage-2 runtime reduction (2026-05-27): colocated vLLM eval lifecycle
normalization, sleep/wake validation, raw-engine access, sleep/wake calls, and
best-effort allocator/sleep-mode cleanup now live in
`src.infer.backend_vllm_engine`. `Stage2RolloutRuntime` retains only trainer
orchestration and thin compatibility wrappers for existing tests/callers. With
the earlier prompt-prep, decode-request, normal/traced rollout dispatch,
parser-adapter, confidence-trace scoring, artifact-shaping, and backend
lifecycle moves, `src.trainers.stage2_rollout_runtime` no longer owns shared
prompt/backend/decode/trace/parser behavior.
Earlier engine deletion groundwork moved runtime-owned `GenerationConfig`,
`InferenceConfig`, `GenerationResult`, and `RunCounters` into
`src.infer.runtime`, with guard coverage proving the factories worked while
`sys.modules["src.infer.engine"]` was blocked. `src.infer.runtime` keeps only
local lightweight type/default aliases for offline config fields and imports
stop-pressure policy names from the shared lightweight constraints facade, so
prompt/runtime/constraints facades remain importable without `torch`,
`transformers`, `swift`, or `vllm`.
Reviewer follow-up: added a fresh-interpreter import guard asserting
`src.infer.runtime` does not load `src.infer.engine` or heavyweight backend
dependencies, plus focused `detect_mode_from_gt` coverage for nested polygon
points, empty-record skipping before coord-token detection, and non-list object
container rejection.
Offline artifact runner, preflight, distributed output handling, row helpers,
debug generation, prompt construction, and HF/vLLM offline decode behavior are
now runtime/backend-owned. The artifact loop lives in
`src.infer.runtime.run_offline_artifact_inference`, input preflight lives in
`src.infer.runtime.preflight_offline_inference_inputs`, distributed helpers live
in `src.infer.runtime`, row helpers live in `src.infer.runtime`, prompt message
construction lives in `src.infer.prompt.build_offline_detection_chat_messages`,
and offline HF/vLLM decode implementations live in `src.infer.backend`. Guard
coverage proves preflight happens before model load, malformed inputs write no
artifacts, JSONL `width`/`height` mismatches fail before side effects,
post-preflight image-load failures fail fast, one-image prompt payloads retain
image-before-text ordering, and generated-token trace fields are preserved.
HF trace-required generation fails fast when the model cannot provide aligned
generated-token scores, OpenAI-compatible vLLM server responses fail fast on
empty choices or missing trace fields, and local vLLM selected-token logprobs
must match the generated token ID instead of falling back to an arbitrary map
entry.
Legacy offline engine deletion completed for this slice. The remaining offline
owner/model lifecycle moved to `src.infer.runtime.OfflineInferenceEngine`,
including checkpoint resolution, prompt/config normalization, HF/vLLM model
loading, Swift adapter shorthand loading, coord-offset adapter reattachment,
seeding, tokenizer left-padding setup, and batch decode bridging through the
shared `InferenceRuntime`. `src/infer/engine.py` was deleted, the import gate
now asserts `src.infer.engine` is absent, and tests patch runtime-owned seams
directly for prompt construction, loader injection, progress bars, and trace
preservation. Focused migration coverage passed in `ms` for
`tests/test_inference_runtime_backend_facade.py`,
`tests/test_infer_batch_decoding.py`, and `tests/test_infer_layout_import_gates.py`.
Stage-2 runtime ownership reduction continued. Eval detection record builders,
confidence-postop input row construction, and eval GT extraction moved from
`src.trainers.stage2_rollout_runtime` into
`src.trainers.rollout_aligned_evaluator`, where Stage-2 eval materialization
and artifact-backed metric computation already live. Stage-2 eval rollout
prediction parsing now routes through
`src.infer.parsing.parse_stage2_detection_rollout_predictions`, which owns the
compact-full vs coord-token parser adapter logic while accepting trainer-local
factories/functions so `src.infer` still does not import `src.trainers.*`.
Focused parser/eval coverage passed in `ms` for `tests/test_parser_policy_parity.py`
and the Stage-2 eval/confidence subset of `tests/test_stage2_rollout_runtime.py`;
`tests/test_infer_layout_import_gates.py` also passed.
The remaining trainer-local `_build_vllm_server_infer_requests` pass-through
wrapper was removed; direct tests now exercise
`src.infer.backend_vllm_server.build_vllm_server_infer_requests`, including the
current ms-swift `RolloutInferRequest` import path.
Stage-2 runtime ownership reduction continued again: shared
`src.infer.rollout_dispatch` no longer calls trainer-private rollout backend
wrappers, the vLLM colocate backend no longer depends on
`owner._vllm_infer_tp_group` or `owner.__class__._strip_left_padding_token_ids`,
and vLLM server rollout dispatch plus strict server chunk/cap allocation helpers
now live in `src.infer.backend_vllm_server`. Stage-2 keeps thin aliases and
compatibility wrappers for existing call sites/tests while backend/decode
ownership moves into `src.infer`.
Reviewer follow-up: direct `_rollout_many` server-mode coverage now patches the
shared `src.infer.rollout_dispatch.rollout_many_vllm_server` binding and asserts
prepared samples, original debug samples, offsets, and decode overrides flow
through the new production call chain. The stale
`Stage2RolloutRuntime._vllm_infer_tp_group` wrapper was removed; colocate
rollout tests now patch `src.infer.backend_vllm_infer.vllm_infer_tp_group`,
which is the active shared-runtime seam.
Stage-2 eval confidence rollout dispatch now also routes through
`src.infer.rollout_dispatch.rollout_many_traced`, so prompt preparation and
HF/vLLM traced backend selection are shared-runtime-owned. The old
`Stage2RolloutRuntime._rollout_many_hf_traced` and
`Stage2RolloutRuntime._rollout_many_vllm_traced` wrappers were removed, and
confidence eval tests patch the shared traced dispatch seam instead.
Stage-2 rollout prompt preparation now lives entirely under
`src.infer.prompt.prepare_rollout_prompt_samples_from_owner`, and both train-step
rollout correction plus normal/traced dispatch call that shared helper directly.
The old `Stage2RolloutRuntime._prepare_samples_for_rollout` method was removed.
Trainable residual target construction computes expected visual metadata from
the original input sample. If source metadata was already present, that source
metadata must match and cannot be silently overwritten; otherwise the guard
accepts the visual metadata stamped by shared prompt preparation on the
backend-normalized copy. The remaining HF/vLLM backend wrapper methods on
`Stage2RolloutRuntime` were also removed; active tests now call
`src.infer.backend.rollout_many_hf`,
`src.infer.backend.build_hf_rollout_logits_processor`,
`src.infer.backend.enforce_hf_rollout_max_position_embeddings`,
`src.infer.backend_vllm_infer.rollout_many_vllm_colocate`, and
`src.infer.backend_vllm_server.rollout_many_vllm_server` directly. Static gates
show no remaining active definitions or calls for those trainer-owned prompt or
backend wrappers.
Reviewer follow-up: the visual parity guard now distinguishes conflicting
source metadata from absent source metadata. Conflicting source metadata still
drops the trainable sample, while ordinary raw visual samples can rely on the
shared prompt helper's prepared metadata stamp. The old broad
`unexpected keyword argument` retry around `_rollout_many` was removed so inner
backend `TypeError`s cannot be masked by retrying without `decode_override` or
`request_index_offset`; rollout-correction tests now use the canonical kwargs
and cover propagation of nested TypeErrors.
The pass-through `_default_rollout_decode_request`,
`_resolve_rollout_decode_request`, and `_strip_left_padding_token_ids` helpers
were removed from `Stage2RolloutRuntime`. Active call sites now use
`src.infer.runtime.build_decode_request_from_rollout_owner` or
`build_decode_request_from_rollout_matching_config` directly; left-padding
token cleanup remains only in `src.infer.backend`.
Active source/script/config/docs route gates are clean for deleted inference
layout modules and trainer-local rollout runtime imports, excluding historical
`docs/superpowers/**` planning notes and active tests that intentionally assert
the deleted surfaces remain absent.

## 7. Verification

- [x] 7.1 Run targeted inference/config/runtime tests.
- [x] 7.2 Run targeted Stage-2 rollout-correction and vLLM server tests.
- [x] 7.3 Run artifact/manifest/provenance tests.
- [x] 7.4 Run docs/spec contract tests.
- [x] 7.5 Run search gates for removed active surfaces and parser/score
  ambiguity terms.
- [x] 7.6 Report skipped expensive checks and the narrowest production-like
  smoke needed before training-scale use.

Completion note (2026-05-27): golden/parity coverage now exists across the
contract-specific tests listed in 2.2-2.12, including decode-request mapping,
prompt/visual parity, trace-logprob shape validation, strict parser guards,
artifact provenance, Stage-2 rollout prompt parity, adapter-sync provenance, and
Stage-2 eval materialization. Deletion-phase import/search gates are active for
non-archived docs/specs/configs/scripts/source plus dedicated tests that assert
legacy surfaces stay absent.

Skipped expensive checks (2026-05-27): no production-scale Stage-2 training run,
no full-dataset offline inference/eval, no live multi-process GPU vLLM server
smoke, and no colocated vLLM GPU smoke were run in this implementation loop.
Before training-scale use, run the narrowest real runtime smoke available on the
target machine: the already-tested contract suite plus one actual Stage-2
rollout-correction smoke config, preferring
`configs/stage2_rollout_correction/smoke/compact_full_hf_1step.yaml` for a
single-process HF sanity check or
`configs/stage2_rollout_correction/smoke/compact_full_vllm_train64_val32_base_control_lr0_1step.yaml`
when the vLLM server/GPU launch shape is available. Treat those as smoke-only
evidence, not full validation.
