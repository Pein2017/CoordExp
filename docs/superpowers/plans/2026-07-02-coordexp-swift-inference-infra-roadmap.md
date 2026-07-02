# CoordExp-Swift Inference Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CoordExp-swift offline inference and benchmark-evaluation infrastructure from the approved OpenSpec baseline, ending in real HF/Qwen smokes and a full scored mAP benchmark gate.

**Architecture:** Keep `openspec/changes/build-coordexp-swift-inference-infra/` as the contract authority and use the patched inference decisions as the design rationale. The public entry is thin `src/infer.py`; implementation lives under `src/inference/` and reuses existing `src/qwen`, `src/templates`, `src/artifacts`, `src/config`, and `src/eval` ownership rather than copying training internals or legacy inference code.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL `generate`, PEFT, safetensors, Pydantic or dataclasses consistent with existing config code, pytest, OpenSpec, existing CoordExp-swift Qwen/template/artifact helpers.

---

## Source Of Truth

- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`
- OpenSpec change: `openspec/changes/build-coordexp-swift-inference-infra/`
- Proposal: `openspec/changes/build-coordexp-swift-inference-infra/proposal.md`
- Design: `openspec/changes/build-coordexp-swift-inference-infra/design.md`
- Tasks: `openspec/changes/build-coordexp-swift-inference-infra/tasks.md`
- Wave 1 source-study gate note:
  `openspec/changes/build-coordexp-swift-inference-infra/source-studies/2026-07-02-wave1-gate-resolution.md`
- Specs:
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-config-runtime/spec.md`
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-backend-trace/spec.md`
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-prompt-parsing/spec.md`
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-scoring-artifacts/spec.md`
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-pipeline/spec.md`
  - `openspec/changes/build-coordexp-swift-inference-infra/specs/coordexp-swift-infer-benchmark-smoke/spec.md`
- Decisions: `docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- Training-source reference: current rebuilt `src/`
- Legacy inference reference only: `/data/CoordExp/src/infer/*` and `configs/infer/*`

## Execution Policy

- Wave 1 source studies are resolved in the gate note above. Source
  implementation remains blocked until the user explicitly approves kickoff
  after reviewing that note and the active OpenSpec change.
- An initial approval may authorize only source studies, probes, and contract
  patching. It does not authorize source implementation.
- Do not mark OpenSpec tasks complete until corresponding implementation and verification evidence exists.
- Do not create `src/infer/`; use `src/infer.py` plus `src/inference/`.
- Do not import or call `TrainConfig`, `ResolvedTrainConfig`,
  `ResolvedStepSchedule`, `load_train_config()`, or unallowlisted
  `src.training.*` from inference-facing modules.
- Treat the V1 `src.training.*` import allowlist as empty unless the source
  study names a concrete type-free exception and OpenSpec is patched to approve
  it.
- Do not use mocked backend tests as acceptance evidence; mocks are allowed only for narrow unit tests.
- Do not claim benchmark correctness until full val or benchmark inference writes scored artifacts and the named evaluator consumer writes mAP metrics.
- After each wave, run isolated review before continuing.

## Intended Source And Test Files

Create or modify:

- `src/infer.py`: thin CLI entrypoint.
- `src/inference/__init__.py`: package marker only.
- `src/inference/runtime.py`: inference setup assembler.
- `src/inference/backend.py`: backend-neutral decode contracts and HF generate adapter.
- `src/inference/prompt.py`: training-aligned prompt construction and parity evidence.
- `src/inference/parsing.py`: compact object-box-closed parser and diagnostics.
- `src/inference/scoring.py`: selected-token alignment, score calculation, score policy fingerprint.
- `src/inference/artifacts.py`: inference artifact row writers and provenance.
- `src/inference/pipeline.py`: orchestration.
- `src/config/`: add or extend inference config schema/loading without weakening training config.
- `src/qwen/`: reuse or deepen Qwen loading, no-resize image planning, tokenizer, and embedding-delta helpers.
- `src/adapters/`: own adapter setup/reload/status policy, including PEFT load-result checks and active-adapter status.
- `src/data/geometry.py`: add shared coordinate-token to pixel conversion only if not already present.
- `src/artifacts/`: reuse or deepen resolved config, manifest, checkpoint alias, and write-once helpers.
- `src/eval/`: minimal detection consumer for `gt_vs_pred_scored.jsonl` or explicitly approved bridge.
- `configs/coordexp_swift/infer/base.yaml`: shallow inference base.
- `configs/coordexp_swift/infer/<production-leaf>.yaml`: production-like benchmark leaf.
- `tests/inference/`: config, backend trace, prompt, parser, scoring, artifacts, pipeline tests.
- `tests/adapters/`: adapter-owner tests for inference reload/status checks and existing DoRA/source-gate behavior.
- `tests/eval/`: evaluator consumer tests.
- `tests/fixtures/smoke/`: tiny single-image and two-row batched inference fixtures.

## Wave 0: Approval And State Check

- [ ] Confirm whether current user approval is for source studies/probes only or for source implementation after source-study review.
- [ ] Run `git status --short` and record unrelated dirty files to preserve.
- [ ] Run `openspec validate build-coordexp-swift-inference-infra --strict`.
- [ ] Run `python - <<'PY'` checks that `src/infer/` does not exist and `src/inference/` does not yet conflict with unrelated user work.
- [ ] Stop before source implementation if OpenSpec validation fails, if source studies contradict the contract, or if the user has not approved implementation kickoff after source-study review.

## Wave 1: Source Studies And Probes

- [ ] Create a source-study note under `openspec/changes/build-coordexp-swift-inference-infra/source-studies/`.
- [ ] Study `src/qwen/images.py`, `src/qwen/loading.py`, `src/qwen/tokens.py`, `src/qwen/special_token_embeddings.py`, `src/adapters/`, `src/templates/renderer.py`, `src/config/`, `src/artifacts/`, and current tests that prove no-resize/tokenizer/adapter behavior.
- [ ] Define owner-neutral seams for Qwen loading, resolved config writing, manifest/provenance writing, adapter identity, and embedding-delta identity so inference does not depend on training-owned config, schedule types, `load_train_config()`, or unallowlisted `src.training.*`.
- [ ] Study installed Transformers generation source for `return_dict_in_generate`, `output_scores`, `scores`, `compute_transition_scores`, processor vision fields, and model vision config fields.
- [ ] Study installed PEFT adapter loading/status behavior and define fatal checks for adapter base identity, captured PEFT `load_result.missing_keys`, captured PEFT `load_result.unexpected_keys`, `set_adapter`, `get_model_status().enabled`, active adapter list, irregular fields, and unexpected merged state. Warning-only PEFT load paths are not sufficient.
- [ ] Study special-token embedding delta identity validation for base config SHA, tokenizer SHA, token strings, and token ids.
- [ ] Study legacy `/data/CoordExp/src/infer/*`, `/data/CoordExp/src/eval/*`, and `/data/CoordExp/docs/eval/*` only for artifact/eval lessons. The default V1 evaluator remains a minimal rebuilt `src.eval` consumer unless the user explicitly approves a bridge.
- [ ] Triage existing tests that import `src.infer` or legacy inference configs. Classify each as port-to-`tests/inference/`, retained with an explicit allowlist reason, or removed/archived with rationale. Do not recreate a `src/infer/` package to satisfy stale tests.
- [ ] Write probe scripts only under `scripts/probes/coordexp_swift/` if source reading is not enough to prove trace alignment or adapter status behavior.
- [ ] Run probes with sample-limited inputs and save outputs under `temp/coordexp_swift_infer_probes/`.
- [ ] Review source-study findings and patch OpenSpec before coding if upstream evidence contradicts the current contract.
- [ ] Stop for explicit user approval before Wave 2 source implementation.

## Wave 2: Config Runtime And Entry

- [ ] Add failing tests in `tests/inference/test_config_runtime.py` for valid production config, rejected training-only keys, rejected unknown keys, rejected legacy config path, vLLM not implemented, debug `batch_size: 1`, resolved config artifacts, and `src/infer.py` entry resolution.
- [ ] Implement `InferConfig` without importing `TrainConfig`.
- [ ] Implement inference config loading under `configs/coordexp_swift/infer/`.
- [ ] Add `src/infer.py` as a thin entry that parses `--config` and delegates to `src.inference.pipeline`.
- [ ] Add minimal `src/inference/__init__.py`.
- [ ] Implement resolved config artifacts as `configs/resolved.json` and `configs/resolved.yaml`.
- [ ] Implement runtime assembly that records base-only, base-plus-adapter, and base-plus-adapter-plus-delta identity. Adapter reload/status policy stays in `src/adapters/`; `src/inference/runtime.py` coordinates validated adapter-owner results.
- [ ] Add residue tests that fail if inference-facing modules import or call `TrainConfig`, `ResolvedTrainConfig`, `ResolvedStepSchedule`, `load_train_config()`, or any unallowlisted `src.training.*` module.
- [ ] Add or extend `tests/adapters/` coverage for inference adapter reload/status checks, and keep existing adapter tests passing.
- [ ] Run `pytest tests/inference/test_config_runtime.py -q`.
- [ ] Run `pytest tests/adapters/test_dora_setup.py tests/adapters/test_source_gates.py -q` or the current adapter-owner test slice after confirming exact names.
- [ ] Run `python -m src.infer --help`.
- [ ] Run `git diff --check`.
- [ ] Request isolated review for config/runtime before continuing.

## Wave 3: Backend Trace

- [ ] Add failing tests in `tests/inference/test_backend_trace.py` for backend-neutral records, HF scored-generation arguments, missing scores, missing trace fields, prompt-width alignment, post-stop padding exclusion, special-token raw trace preservation, and vLLM not implemented.
- [ ] Implement `DecodeRequest`, `DecodeResult`, and `TokenTrace`.
- [ ] Implement HF `generate_batch` with deterministic greedy defaults, Qwen `<|im_end|>` stop, `return_dict_in_generate=True`, `output_scores=True`, and normalized transition-score extraction.
- [ ] Store per-token `step_index`, `token_id`, `token_text`, `logprob`, `is_stop`, `is_pad`, backend, backend mode, and response family.
- [ ] Run `pytest tests/inference/test_backend_trace.py -q`.
- [ ] Run a real tiny HF/Qwen trace probe with one image.
- [ ] Run a real two-row HF/Qwen trace probe with different prompt lengths.
- [ ] Request isolated review for backend trace before continuing.

## Wave 4: Prompt Image And Parser

- [ ] Add failing tests in `tests/inference/test_prompt_image.py` for prompt-token parity, template identity, mandatory image-plan fields, processor-model vision parity, valid no-resize image, invalid no-resize dimensions as terminal input failure, and image-plan row count.
- [ ] Add failing tests in `tests/inference/test_parsing.py` for valid compact object, unsupported JSON assistant response, generated-order preservation, accepted-with-drops, all-spans-dropped, degenerate bbox, and out-of-range coordinates.
- [ ] Implement prompt helpers that reuse training template semantics.
- [ ] Implement mandatory `image_plan.jsonl` materialization through existing Qwen no-resize helpers.
- [ ] Implement compact object-box-closed parser with inline parser status and detailed sidecar diagnostics.
- [ ] Implement or reuse shared coordinate-token to pixel `xyxy` conversion in the shared geometry owner.
- [ ] Run `pytest tests/inference/test_prompt_image.py tests/inference/test_parsing.py -q`.
- [ ] Run a tiny real no-resize image smoke.
- [ ] Request isolated review for prompt/image/parser before continuing.

## Wave 5: Scoring And Artifacts

- [ ] Add failing tests in `tests/inference/test_scoring.py` for score formula, exact selected-token count `n_selected == 8`, selected-token replay evidence, description exclusion, empty selected set, non-finite logprob, duplicate ambiguity, object-span contiguity, and score-policy fingerprint.
- [ ] Add failing tests in `tests/inference/test_artifacts.py` for raw/scored row parity, no extra diagnostic JSONL rows, `pred: []` rows, GT preservation, row-local non-empty score source, integer score version, finite score range `[0.0, 1.0]`, provenance sidecar, raw/scored SHA binding, trace-based score recomputation, and manifest status.
- [ ] Implement selected-token alignment and scoring.
- [ ] Implement `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, and inference manifest writers.
- [ ] Verify scored artifact production refuses missing trace or missing provenance.
- [ ] Run `pytest tests/inference/test_scoring.py tests/inference/test_artifacts.py -q`.
- [ ] Inspect a tiny scored artifact set with a script or JSON parser and record artifact handles.
- [ ] Request isolated review for scoring/artifacts before continuing.

## Wave 6: Pipeline And Eval Consumer

- [ ] Add failing tests in `tests/inference/test_pipeline.py` for end-to-end orchestration, batching, counters, terminal failure status, artifact write order, and no metric reduction inside `src.infer`.
- [ ] Add failing tests in `tests/eval/test_detection_consumer.py` for consuming a tiny `gt_vs_pred_scored.jsonl` fixture, refusing missing or mismatched provenance, and writing `metrics.json` or an explicitly named metric artifact.
- [ ] Implement `src/inference/pipeline.py`.
- [ ] Implement the minimal rebuilt `src.eval` detection consumer. Implement a legacy bridge only if the user explicitly changes the V1 default before this wave.
- [ ] Run `pytest tests/inference/test_pipeline.py tests/eval/test_detection_consumer.py -q`.
- [ ] Run a tiny scored fixture through the named evaluator consumer.
- [ ] Request isolated review for pipeline/eval consumer before continuing.

## Wave 7: Real Smokes

- [ ] Run tiny real HF/Qwen inference smoke with `generation.batch_size: 1` under debug/smoke policy.
- [ ] Run two-row batched trace smoke with `generation.batch_size: 2`.
- [ ] Run adapter-enabled real smoke with base model, adapter checkpoint, and any configured special-token embedding delta.
- [ ] Verify each smoke writes resolved configs, manifest, summary, raw artifact, scored artifact, provenance, token trace, parse diagnostics, and image plan.
- [ ] Verify adapter smoke records resolved base, adapter, and delta identity.
- [ ] Run `openspec validate build-coordexp-swift-inference-infra --strict`.
- [ ] Run targeted inference/eval pytest suites.
- [ ] Run adapter-owner pytest slices and legacy inference-test triage residue checks.
- [ ] Run `git diff --check`.
- [ ] Request isolated smoke/benchmark-readiness review.

## Wave 8: Production Benchmark Gate

- [ ] Prepare a benchmark launch packet naming production config path, dataset path, base model path, adapter checkpoint path, artifact root, evaluator command, expected runtime scope, and rollback path.
- [ ] Ask the user for explicit benchmark-launch approval.
- [ ] Launch full val or benchmark inference only after approval.
- [ ] Verify batched decode, scored artifacts, and mAP artifact output.
- [ ] Label metrics with evidence scope and exact artifact root.
- [ ] Do not claim final inference correctness if the run is sample-limited, debug-only, or missing scored provenance.

## Review Requirements

- [ ] After drafting docs, dispatch isolated reviewers for OpenSpec contract coverage, upstream HF/Qwen/PEFT correctness, mAP artifact validity, module boundary quality, and smoke/benchmark adequacy.
- [ ] After each implementation wave, dispatch an isolated reviewer scoped to the files and requirements changed by that wave.
- [ ] Treat reviewer timeout as unresolved, not approval.
- [ ] Fix accepted P0/P1 findings before moving to the next wave.
- [ ] Record rejected findings with concrete evidence.

## Verification Commands

Run during planning:

```bash
openspec status --change build-coordexp-swift-inference-infra --json
openspec validate build-coordexp-swift-inference-infra --strict
python - <<'PY'
from pathlib import Path
assert not Path("src/infer").exists()
assert Path("openspec/changes/build-coordexp-swift-inference-infra/tasks.md").exists()
print("planning checks: ok")
PY
git diff --check
```

Run during implementation waves as applicable:

```bash
pytest tests/inference/test_config_runtime.py -q
pytest tests/adapters/test_dora_setup.py tests/adapters/test_source_gates.py -q
pytest tests/inference/test_backend_trace.py -q
pytest tests/inference/test_prompt_image.py tests/inference/test_parsing.py -q
pytest tests/inference/test_scoring.py tests/inference/test_artifacts.py -q
pytest tests/inference/test_pipeline.py tests/eval/test_detection_consumer.py -q
openspec validate build-coordexp-swift-inference-infra --strict
git diff --check
```

## Stop Conditions

- User has not approved implementation kickoff.
- User approval only covers source studies/probes, not source implementation.
- OpenSpec validation fails.
- Source study contradicts the current score, trace, no-resize, adapter, or eval contract.
- HF/Qwen real trace alignment cannot be proven on a tiny batch.
- Adapter identity checks cannot be made fatal with available PEFT surfaces.
- Adapter reload/status policy cannot be kept in `src/adapters/` without
  duplicating PEFT policy in inference runtime.
- Legacy `src.infer` tests remain untriaged and pressure the implementation
  toward recreating the forbidden `src/infer/` package.
- Scored artifacts cannot be consumed by a named evaluator without weakening provenance.
- Any production benchmark handle is missing.

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-07-02-coordexp-swift-inference-infra-roadmap.md`.

Source implementation is blocked until explicit user approval after Wave 1
source-study review. Recommended execution mode after approval:
subagent-driven development, one wave at a time, with isolated review after
each wave.
