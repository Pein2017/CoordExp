## Context

See proposal.md for the user-approved outcome, isolation and resource boundaries. The new worktree is clean apart from this change's planning files. The existing native execution, renderer, image planning, target encoder and diagnostic reducers are the implementation base.

Four consumers repeat equivalent processor/template conversion: DORA runtime, Logit Lens runtime, Human13 runtime and inference pipeline. DORA and Human13/Logit Lens additionally repeat image-plan → generation-prompt → request assembly. DORA `_native_ce_group` independently renders/encodes a target and its preflight builds the native request again. Existing `PromptRecord` intentionally omits assistant messages; its metadata is not a supervised target.

Baseline focused CPU check: 18 passed, exit 0, on the unchanged fork (`tests/qwen/test_encoding.py::test_rendered_example_encodes_token_spans_and_visual_expansion`, `tests/qwen/test_native.py`, `tests/inference/test_prompt_image.py`). Raw log: `/tmp/research-probes-upgrade-input-baseline-20260912.log`.

## Goals / Non-Goals

**Goals:** one owner for input composition, fewer repeated image/render operations, explicit shared/profile separation, preserved literal tokens and physical spans, and a useful small inspection entry. A real caller should be simpler after extraction; retaining every old internal wrapper is not a goal.

**Non-Goals:** model-agnostic runtime, global ProbeContext, training executor, adapter topology/schema expansion, generic trajectory storage, universal reducer, research-result migration, arbitrary historical trainer cleanup, lower-precision numerics or model-quality improvements.

## Decisions

### 1. Shared input planning, explicit native materialization

Add `src/inference/inputs.py` as the composition owner. Its inputs are existing `InferConfig`, `RawExample` rows, caller-owned Qwen components, optional row indices, and an optional positive target maximum length. Its small public surface is:

```python
processor_config(config) -> ProcessorConfig
template_config(config) -> TemplateConfig
plan_examples(rows, *, config, components, row_indices=None,
              target_max_length=None) -> tuple[PlannedExample, ...]
```

`PlannedExample` groups existing typed values: rendered row, image plan, prompt, native request, and optional `EncodedExample`. It has no model, optimizer, output directory, scientific rule, mutable registry or lifecycle method. Existing `prepare_native_inputs` remains the materialization API; its result already feeds exact-history replay and owns device movement. Request-only callers keep their existing lazy materialization.

The call hides repeated config adaptation, render/image/prompt assembly and consistency checks. The caller retains row selection, object seed, profile, native generation policy, full target length, objective/mask/reduction and model lifetime.

Three alternatives were compared:

| Interface | Benefit | Cost / decision |
| --- | --- | --- |
| Only share two config converters | Minimal migration | Leaves the image/prompt/target coherence burden duplicated; insufficient for the accepted workflow |
| Plan explicit typed values, materialize with existing native API | Removes real repeated caller knowledge and preserves lazy/eager distinction | Selected; requires targeted encoder/prompt factoring |
| Stateful profile session that selects, loads, runs and reduces | Short common-case call | Would own scientific policy and duplicate current executors; rejected |

### 2. Reuse render/image plans inside their current owners

Refactor prompt construction to accept the already rendered row internally while preserving the existing `build_prompt_record` entry where it has callers. Refactor image planning/target encoding so the same `QwenImageEncoding` can supply `ImagePlanRow` and annotated encoding without another plan or pixel tensor allocation. Keep reused-plan arguments internal when possible; if an externally reachable argument accepts a plan, validate row identity/path/dimensions/transform before using it.

The target path uses the existing encoder and its physical span objects. It verifies full-prefix agreement; Source256 retains its additional contiguity, ignored-post-EOS and fixed-EOS assertions in its artifact consumer. Generation and full-target tokenization are different sequences and may each run once. Planning does not allocate pixel tensors; the subsequent native call materializes them once.

Do not add a persistent cache or key schema. Passing a result already computed for the same row is sufficient and avoids invalidation machinery. Native media/grid validation remains intact.

### 3. Preserve actual profile boundaries during migration

- DORA `runtime.build_request` preserves its return contract using the shared plan. `_native_ce_group` consumes the optional encoded target and retains its exact artifact projection. Its strict preflight reuses each prepared row for target checks and native materialization.
- Logit Lens base/causal request assembly consumes shared plans; its DecodeRequest policy, historical stage prerequisites and output bindings remain local.
- Human13 request construction consumes shared plans; conditional histories, intervention sites and launch policy remain local.
- Inference pipeline imports the shared config converters only; its scheduling, chunking, production resource policies and artifact pipeline are outside this refactor.
- Delete repeated adapters and assembly after reference analysis. Keep a narrow old internal entry only when retained callers or source-bound execution require it.

New code dependencies must enter the existing current-execution source snapshot lists. Old receipts and producer hashes are never rewritten. Historical packets continue to refer to their original effective sources; a new execution must be freshly prepared. Do not claim old-packet resume.

### 4. Direction-local inspection and cohort choice

Add `probes/dora_owner_learning/inspect.py` as a CPU convenience entry, separate from the historical `preflight`. It accepts an existing profile plus explicit IDs, or count and seed, and optional annotated targets with explicit maximum length. Its selection helper is local until another real consumer requires sharing. It reads/validates the source through the existing JSONL reader, freezes the ordered selected rows, loads one processor-only frontend, plans them and explicitly materializes once. It emits a JSON inspection result with source/config identity, selected order, prompt/media/grid and optional target/spans. It does not emit a Source256 plan or claim scientific validation.

The existing Source256 and Logit Lens preflights retain their CLI and gates. Document the ordinary two-call Python path as well as this convenience entry; a caller is not required to use the CLI or create a receipt to run a probe.

### 5. Frozen Source256 binding and existing diagnostics

The second refactor consolidates only the identical freeze → select language DoRA → fixed count/name checks → enable → frozen-complement operation at the existing DORA runtime owner. Fixed accepted Source256 surface is 588 tensors and 18,006,016 scalars. It does not add layer, regex, dtype or adapter-topology choices. The admitted callers are `selective_preservation.py` and its `_wide`, `_dense`, `_strong`, `_seven` and `_stable` profiles. Inspection of their current input packets found no Python paths in inherited `source_files`; the only repository-local binding is the unchanged Source256 config. Their historical code snapshots retain staged original sources (25 relevant staged entries verified against recorded hashes), and new preparation snapshots current executable sources. No historical manifest refresh is needed.

The shared binding returns ordered selected and frozen `(name, parameter)` tuples. It accepts the profile's expected tensor/scalar counts and adapter name, verifies the existing fixed contract and model selection, then preserves the current trainability transition. Invalid selections remain fatal before any optimizer or update. Keep it in `runtime.py`, which every admitted profile already snapshots; the input package owns that file first, and the binding package follows sequentially to avoid concurrent writers.

Per-profile optimizers, scientific assertions, tensor-version snapshots and DDP placement remain local; notably seven/stable take frozen tensor version snapshots after DDP construction. Byte-bound `repeat_recovery_train`, `margin_preserved_train` and `positive_progress_matched_train` remain unchanged. Existing public parameter selection and Human13 magnitude transactions remain available for experiment-owned small surfaces.

Use existing owner-change/aggregate/branch reducers in the workflow example and acceptance. Remaining reducer differences encode scientific comparator meanings. No new trajectory format or broad projection is justified.

### 6. Acceptance and efficiency evidence

Use the two committed real image rows in `tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl` with the retained Source256 and Logit Lens profile values. Capture old outputs before source changes. Compare exact prompt IDs, grids, RGB/media identities, target tokens, physical spans, ignored spans and EOS. Validate prompt edits and target-order edits independently. Failures need caller-visible sensitivity, including wrong media and invalid selection; no implementation-mirroring test suite.

First implement a generation+target vertical slice with one real processor before migrating all consumers. Verify it avoids weights and matches the frozen old outputs. Count render, image-plan and native materialization work. If this fails, fix the slice before broad migration.

For final model acceptance, use one real image with Source FP32/SDPA, the existing adapter and embedding composition, prepared native inputs, exact annotated replay and one profile-owned update. Bound generated continuation to a small explicit diagnostic budget when a downstream reducer requires generated output. Compare old/new zero-update tokens/logits and score; confirm selected tensor updates and frozen tensor identity. Save/reload when testing a changed save-adjacent surface; no checkpoint format change is planned. Capture model/image forward counts, elapsed GPU time, peak memory and source identity. At most one GPU runs, all model invocations count toward 60 GPU minutes; technical failure or budget exhaustion stops model work rather than opening another experiment.

CPU benchmark uses fixed selected rows and retained profile values, explicit cold versus warm phases, repeated runs with median/range and operation counts. Model time, CPU preparation time and code-removal counts are separate. No hard speedup threshold is assumed; a material slowdown caused by the changed path must be explained or fixed before acceptance.

## Risks / Trade-offs

- Reusing a stale image plan → keep ownership local and validate identity at the accepting boundary; exercise a different-row/media counterexample.
- Mixing prompt and target spans → preserve distinct values and exact expanded-prefix assertion.
- Eager materialization in old lazy callers → test request-only behavior and count work in the real CPU vertical slice.
- New dependency missing from a source receipt → inspect each migrated profile's existing source-closure mechanism and validate a changed helper is covered by fresh execution identity.
- Hash-bound historical producers look highly redundant → preserve their bytes; archive identity is an obligation, not a line-count target.
- Active task changes research main during development → all writes stay in the temporary worktree; integrate the latest committed target into an isolated candidate before considering merge.
- No process currently holds a file does not prove the active task released it → require actual task completion/release evidence for affected dependencies before writing the canonical worktree.

## Migration Plan

Freeze old caller outputs and source bindings; complete this proposal/design/spec/tasks and one independent Astra contract review; implement the early real CPU slice; migrate accepted consumers and remove obsolete code; run focused regression, real model acceptance and the paired benchmark; inspect the final diff and record standards/intent verdicts. Commit explicit logical batches locally. Reconcile latest research main in isolation and replay affected acceptance before merging at the already agreed release boundary. Keep the canonical lock, unrelated dirt and historical output roots unchanged. Retire the temporary worktree only after actual merge and preservation checks.
