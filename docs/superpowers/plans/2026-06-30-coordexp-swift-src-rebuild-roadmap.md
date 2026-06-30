# CoordExp-Swift Src Rebuild Roadmap And Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the CoordExp-swift `src/` training infrastructure from the approved OpenSpec baseline into a professional, inspectable, five-step-smoke-verified Qwen3-VL supervised training stack.

**Architecture:** Keep OpenSpec requirements as the contract authority, keep `BLUEPRINT.md` cards as the manual approval surface, and execute implementation in waves that each produce a working, testable slice. Source studies and probes run before fragile Qwen/dLoRA/embedding code; the new `src/` tree is rebuilt around strict config, validated examples, Qwen-owned encoding/forward boundaries, no-padding packing, token-wise supervision, protected losses, explicit optimizer groups, and artifact-backed training.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL/Qwen2-VL processor paths, PEFT, Accelerate, DeepSpeed schema handling, flash-attn branch checks, PyYAML or Pydantic/dataclasses for config, safetensors, pytest, OpenSpec.

---

## Source Of Truth

- Active worktree: `/data/CoordExp/.worktrees/CoordExp-swift`
- OpenSpec change: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/`
- OpenSpec tasks: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md`
- OpenSpec design: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/design.md`
- OpenSpec specs:
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-config-runtime/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-data-template-encoding/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-packing-forward/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-supervision-losses/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-adapters-embeddings-optim/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-training-artifacts/spec.md`
  - `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-vertical-smoke/spec.md`
- Design decisions: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- Approval cards: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`
- Review triage: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/review-triage.md`
- Current legacy source reference: `/data/CoordExp/.worktrees/CoordExp-swift/src/`
- Future legacy source archive: `/data/CoordExp/.worktrees/CoordExp-swift/reference/legacy_src/`
- Archived legacy OpenSpec reference: `/data/CoordExp/.worktrees/CoordExp-swift/reference/legacy_openspec_2026-06-29/`
- Local base model for first smoke: `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`

## Roadmap Vocabulary

- **Wave:** a large implementation milestone with its own review gate.
- **Slice:** a self-contained unit inside a wave that can be tested independently.
- **Wave-specific plan:** a later, granular Superpowers implementation plan for
  one wave. This roadmap is not detailed enough to authorize writing all V1
  code in one pass.
- **Approval card:** a `BLUEPRINT.md` module/functionality card that the user approves before coding that public surface.
- **Probe:** executable source-study script proving delicate upstream behavior before production code depends on it.
- **Receipt:** compact JSON/YAML artifact emitted by runtime to prove what happened.
- **Acceptance gate:** checks that must pass before moving to the next wave.
- **Stop condition:** a condition that requires pausing and asking the user before continuing.

## Superpowers Execution Protocol

This document is a top-level roadmap. Before coding a source wave, the
implementation agent MUST create or request a wave-specific Superpowers plan
with exact files, exact test names, red/green order, and review checkpoints.

Allowed direct execution from this roadmap:

- Wave 0 state checks.
- Wave 1 read-only source studies and isolated probes after explicit user
  approval.

Not allowed direct execution from this roadmap:

- Moving old `src/`.
- Creating the new production `src/` implementation.
- Marking OpenSpec implementation tasks complete.
- Running the five-step smoke as an acceptance claim without the Wave 8
  artifact audit.

Each wave-specific plan MUST state:

- exact approved `BLUEPRINT.md` card(s);
- exact OpenSpec tasks it intends to mark complete;
- exact files to create/modify;
- exact tests and smoke commands;
- exact review gate before the next wave;
- stop conditions that require user clarification.

Review after each wave SHOULD use the Superpowers review style: independent
reviewer, precise files/requirements, severity labels, and a clear verdict.
Critical or Important findings must be fixed before proceeding.

## Global Non-Negotiables

- Do not implement source code before explicit user approval for the relevant wave/card.
- Do not import old `src/` modules from the new `src/`; old code moves to `reference/legacy_src/` and becomes reference-only.
- Do not edit upstream Transformers model files.
- Do not silently resize images; Qwen processor calls for training use `do_resize=False`.
- Do not create a padded training batch for the standard supervised forward path.
- Do not pass `inputs_embeds` in V1 Qwen forward.
- Do not use model-side CE; repo-owned losses consume logits and `TokenSequence`.
- Do not claim dLoRA support before the dLoRA source-study gate and round-trip probe pass.
- Do not claim selected special-token embedding support before the embedding source-study gate passes.
- Do not claim DeepSpeed production support until a later systems smoke proves it.
- Do not expand V1 into rollout training, hidden-state losses, persistent caches, video, multi-image, vLLM, exact resume, or old production coordinate-soft-CE parity.
- Do not mark OpenSpec tasks complete until their implementation evidence exists.
- Do not let exploratory probes become production modules by accident; probes
  live under `scripts/probes/coordexp_swift/` and must be promoted deliberately.
- Do not let the roadmap override OpenSpec. If implementation evidence
  contradicts OpenSpec, stop and patch/review OpenSpec before coding through the
  contradiction.

## Intended Final Source Topology

The rebuilt V1 `src/` should be small and direct:

```text
src/
  __init__.py
  train.py
  common/
    __init__.py
    errors.py
  config/
    __init__.py
    schema.py
    loader.py
    schedule.py
    fingerprints.py
  artifacts/
    __init__.py
    manager.py
    manifest.py
    checkpoints.py
    receipts.py
  data/
    __init__.py
    examples.py
    jsonl.py
    images.py
    validation.py
  templates/
    __init__.py
    renderer.py
    spans.py
  qwen/
    __init__.py
    loading.py
    tokenizer.py
    encoding.py
    mrope.py
    forward.py
    adapters.py
    special_token_embeddings.py
  packing/
    __init__.py
    sequence.py
    planner.py
    mapping.py
  supervision/
    __init__.py
    tokens.py
    alignment.py
  losses/
    __init__.py
    context.py
    ce.py
    token_type_gate.py
    normalizers.py
    runner.py
    finite.py
  metrics/
    __init__.py
    accuracy.py
    events.py
  optim/
    __init__.py
    groups.py
    builder.py
  runtime/
    __init__.py
    accelerate_runtime.py
    deepspeed_status.py
    distributed.py
  training/
    __init__.py
    stream.py
    trainer.py
    step.py
  eval/
    __init__.py
    forward.py
```

Supporting files:

```text
configs/coordexp_swift/
  base.yaml
  research_base.yaml
  smoke_qwen3_vl_single_image_pack.yaml

tests/coordexp_swift/
  test_config_runtime.py
  test_data_template_encoding.py
  test_packing_forward.py
  test_supervision_losses.py
  test_adapters_embeddings_optim.py
  test_training_artifacts.py
  test_vertical_smoke_contract.py

tests/fixtures/smoke/qwen3_vl_single_image_pack/
  README.md
  config.yaml
  examples.jsonl
  expected_rendered.json
  checksums.json
  image.<ext>

docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/
  dlora.md
  special-token-embeddings.md
  qwen-noresize-mrope-fa2.md

scripts/probes/coordexp_swift/
  dlora_roundtrip.py
  special_token_embedding_roundtrip.py
  qwen_processor_forward_probe.py
  fa2_varlen_probe.py
```

## Wave 0: Implementation Authorization And State Guard

**Goal:** Make the starting state explicit before any new source implementation.

**OpenSpec tasks covered:** none directly; this wave protects the transition from planning to coding.

**Files:**
- Read: `/data/CoordExp/.worktrees/CoordExp-swift/AGENTS.md`
- Read: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md`
- Read: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`
- Read: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md`
- Modify only after approval: no source files.

### Slice 0.1: Preflight State Snapshot

- [ ] Run `git status --short --branch` from `/data/CoordExp/.worktrees/CoordExp-swift`.
- [ ] Run `openspec instructions apply --change rebuild-coordexp-swift-training-infra --json`.
- [ ] Confirm OpenSpec apply progress is `total=55`, `complete=10`, `remaining=45` before implementation begins.
- [ ] Record in the implementation chat that unrelated dirty files must be ignored and only touched files should be staged if a commit is requested.

### Slice 0.2: User Authorization Gate

- [ ] Ask the user to approve Wave 1 source-study work.
- [ ] Ask separately before Wave 2 moves old `src/` to `reference/legacy_src/`.
- [ ] Do not interpret this roadmap as authorization to rewrite `src/`.

**Acceptance gate:** explicit user approval for Wave 1.

**Stop condition:** if the worktree path is not `/data/CoordExp/.worktrees/CoordExp-swift`, stop and ask.

**Wave-specific plan requirement:** not required for Wave 0 because it is
state inspection only.

## Wave 1: Source Studies And Probes

**Goal:** Resolve the delicate external behavior before implementing production code.

**OpenSpec tasks covered:** 2.1, 2.2, 2.3, 2.4, 2.5.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/dlora.md`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/special-token-embeddings.md`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/qwen-noresize-mrope-fa2.md`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/dlora_roundtrip.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/special_token_embedding_roundtrip.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/qwen_processor_forward_probe.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/scripts/probes/coordexp_swift/fa2_varlen_probe.py`

### Slice 1.1: dLoRA Source Study

- [ ] Discover installed package paths with:

```bash
python - <<'PY'
import inspect
import peft
import transformers
print("peft", peft.__version__, inspect.getfile(peft))
print("transformers", transformers.__version__, inspect.getfile(transformers))
try:
    import swift
    print("swift", inspect.getfile(swift))
except Exception as exc:
    print("swift_import_error", type(exc).__name__, str(exc))
PY
```

- [ ] Search PEFT for `use_dora`, `LoraConfig`, `trainable_token_indices`, and `TrainableTokens`.
- [ ] Search MS-Swift for DoRA/dLoRA naming and target-module discovery behavior.
- [ ] Search local legacy source for prior LoRA/dLoRA-like config surfaces.
- [ ] Document the exact chosen definition of `adapter.type: dlora` in `source-studies/dlora.md`.
- [ ] State whether dLoRA is PEFT DoRA-backed, a CoordExp-owned mechanism, or rejected until a later design.

**Acceptance gate:** `dlora.md` names the exact implementation mechanism and why it satisfies or does not satisfy the OpenSpec dLoRA gate.

**Stop condition:** if dLoRA cannot be defined without inventing semantics, stop and ask the user whether to use standard LoRA for a pre-smoke or keep dLoRA-first blocked.

### Slice 1.2: dLoRA Round-Trip Probe

- [ ] Implement `scripts/probes/coordexp_swift/dlora_roundtrip.py` to load the local base model with `local_files_only=True`, initialize the selected dLoRA mechanism on a minimal target set, run one tiny forward pass, save adapter payloads to `outputs/probes/coordexp_swift/dlora_roundtrip/`, reload base-plus-adapter, and compare output shape plus finite logits.
- [ ] Keep probe outputs under ignored `outputs/probes/`.
- [ ] Run:

```bash
python scripts/probes/coordexp_swift/dlora_roundtrip.py
```

- [ ] Record command, environment, model path, PEFT version, target modules, output artifact path, and result in `source-studies/dlora.md`.

**Acceptance gate:** probe exits 0 and produces a compact JSON receipt with adapter init/save/load/forward success.

**Stop condition:** if the probe requires more than one GPU or a long run, reduce the target set and sequence length before asking for heavier resources.

### Slice 1.3: Special-Token Embedding Mechanism Study

- [ ] Compare custom Qwen wrapper, PEFT `TrainableTokens`, and LoRA `trainable_token_indices`.
- [ ] Verify how the local Qwen3-VL model ties or does not tie input embeddings and `lm_head`.
- [ ] Verify the selected token ids for `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`, and `<|coord_0|>` through `<|coord_999|>`.
- [ ] Document the exact selected mechanism in `source-studies/special-token-embeddings.md`.
- [ ] Include checkpoint payload shape, metadata fields, load order, and base-plus-adapter-plus-embedding-delta composition.

**Acceptance gate:** the study proves how selected embedding deltas affect input lookup and selected output-logit columns without training all embedding/head rows.

**Stop condition:** if tied/untied behavior is ambiguous for Qwen3-VL, stop and ask whether to prefer a custom wrapper or a full selected-row state-dict delta.

### Slice 1.4: Special-Token Embedding Round-Trip Probe

- [ ] Implement `scripts/probes/coordexp_swift/special_token_embedding_roundtrip.py` to select a small subset of the final token group, apply deterministic nonzero deltas, save compact delta metadata and tensor payload, reload into a fresh base model, and verify that only selected ids changed.
- [ ] Run:

```bash
python scripts/probes/coordexp_swift/special_token_embedding_roundtrip.py
```

- [ ] Record result in `source-studies/special-token-embeddings.md`.

**Acceptance gate:** probe proves selected-token deltas can save/load without full embedding/head export.

### Slice 1.5: Qwen No-Resize, MRoPE, And FA2 Source Study

- [ ] Implement `scripts/probes/coordexp_swift/qwen_processor_forward_probe.py` to load `AutoProcessor` and model from `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
- [ ] Verify loaded `patch_size`, `merge_size`, `temporal_patch_size`, and processor class.
- [ ] Verify valid no-resize dimensions pass and invalid dimensions fail before or at processor reshape.
- [ ] Verify `image_grid_thw`, placeholder expansion, and Qwen forward output shape.
- [ ] Verify MRoPE position id row count and meaning in the installed Transformers version.
- [ ] Implement `scripts/probes/coordexp_swift/fa2_varlen_probe.py` to capture the branch-level evidence required by the packing-forward spec.
- [ ] Record findings in `source-studies/qwen-noresize-mrope-fa2.md`.

**Acceptance gate:** source study explains the exact no-resize admissibility rule, the MRoPE row-shape contract, placeholder count formula, and FA2 varlen evidence.

**Stop condition:** if installed Qwen behavior differs from the current OpenSpec wording, patch OpenSpec before implementation.

### Slice 1.6: Wave 1 Review And Gate Decision

- [ ] Request an independent review of `source-studies/*.md` and
  `scripts/probes/coordexp_swift/*.py`.
- [ ] Confirm whether dLoRA is approved for Wave 6 implementation.
- [ ] Confirm whether selected special-token embedding deltas are approved for
  Wave 6 implementation.
- [ ] Patch OpenSpec or `BLUEPRINT.md` if the studies invalidate any current
  contract.
- [ ] Mark OpenSpec tasks 2.1 through 2.5 complete only after the studies and
  probes have passing evidence.

**Acceptance gate:** reviewer reports no unresolved P0/P1 findings and the user
explicitly approves moving to Wave 2.

## Wave 2: Fixture, Archive, Skeleton, Config, And Artifact Seed

**Goal:** Create the new repo surface and the first stable smoke input/output skeleton.

**OpenSpec tasks covered:** 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5.

**Files:**
- Move: `/data/CoordExp/.worktrees/CoordExp-swift/src/` to `/data/CoordExp/.worktrees/CoordExp-swift/reference/legacy_src/`
- Create: all package markers in the final topology.
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/base.yaml`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/research_base.yaml`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke_qwen3_vl_single_image_pack.yaml`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/common/errors.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/config/schema.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/config/loader.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/config/schedule.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/config/fingerprints.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/artifacts/manager.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/artifacts/manifest.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/train.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/fixtures/smoke/qwen3_vl_single_image_pack/`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_config_runtime.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_vertical_smoke_contract.py`

### Slice 2.1: Archive Old Source And Create Empty New Package

- [ ] Ask user approval for moving old `src/`.
- [ ] Confirm `reference/legacy_src/` does not already exist.
- [ ] Run:

```bash
test ! -e reference/legacy_src
mkdir -p reference
mv src reference/legacy_src
mkdir -p src/{common,config,artifacts,data,templates,qwen,packing,supervision,losses,metrics,optim,runtime,training,eval}
find src -type d -exec touch {}/__init__.py \;
```

- [ ] Confirm no new code imports from `reference/legacy_src/`.
- [ ] Do not stage generated `__pycache__` files if they move with the legacy
  source tree.
- [ ] Add a README or package-level note only if needed to explain that `reference/legacy_src/` is read-only reference.

**Acceptance gate:** `python -c "import src"` succeeds and no old module import shim exists.

### Slice 2.2: Contract Errors

- [ ] Implement `src/common/errors.py` with small domain-specific exception classes: `CoordExpError`, `ConfigError`, `DataValidationError`, `TemplateError`, `QwenContractError`, `PackingError`, `LossError`, `OptimizerConfigError`, `RuntimeContractError`, and `ArtifactError`.
- [ ] Add `tests/coordexp_swift/test_config_runtime.py` coverage that errors carry stable class names and concise messages.

**Acceptance gate:** `pytest tests/coordexp_swift/test_config_runtime.py -q` passes.

### Slice 2.3: Config Loader And Resolved Schedule

- [ ] Implement strict YAML loading with inheritance, unknown-field rejection, path-origin metadata, source fingerprints, `training.effective_batch_size`, `epochs`, debug `max_steps`, cadence fields, and derived accumulation.
- [ ] Implement `resolved_step_schedule.json` materialization from planned steps.
- [ ] Implement `src.train --dry-run` through config resolution and artifact initialization only.
- [ ] Add tests for unknown field failure, inherited config preservation, `max_steps` priority, fractional cadence, forbidden aliases, and effective-batch/world-size divisibility.

**Acceptance gate:** `pytest tests/coordexp_swift/test_config_runtime.py -q` passes and `python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml --dry-run` writes resolved config and manifest without model mutation once the fixture exists.

### Slice 2.4: Smoke Fixture Pinning

- [ ] Locate current `len12000` JSONL sources and candidate single-image rows.
- [ ] Select exactly one short valid two-object row or a deterministic first-two-valid-objects reduction.
- [ ] Copy one real image into `tests/fixtures/smoke/qwen3_vl_single_image_pack/`.
- [ ] Create `examples.jsonl`, `checksums.json`, `expected_rendered.json`, `config.yaml`, and `README.md`.
- [ ] Ensure fixture config writes outputs under an ignored artifact root such as `outputs/smoke/qwen3_vl_single_image_pack/`.

**Acceptance gate:** fixture metadata records source path, row id or row number, image checksum, selected object ids, reduction rule when used, no-resize dimensions, and expected rendered text.

**Stop condition:** if no no-resize-compatible source image is available, stop and ask whether to use a copied/resized fixture image as a controlled exception for smoke only.

**Wave-specific plan requirement:** required before Slice 2.1 because moving
`src/` is high impact and must be reviewed before execution.

## Wave 3: Data, Template, And Qwen Encoding

**Goal:** Turn fixture JSONL into validated encoded examples with exact supervision spans and visual metadata.

**OpenSpec tasks covered:** 5.1, 5.2, 5.3, 5.4, 5.5, 5.6.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/data/examples.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/data/jsonl.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/data/images.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/data/validation.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/templates/renderer.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/templates/spans.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/loading.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/tokenizer.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/encoding.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_data_template_encoding.py`

### Slice 3.1: RawExample And JSONL Loader

- [ ] Implement `RawExample` with `example_id`, image path, image metadata, and object boxes in `x1,y1,x2,y2`.
- [ ] Reject video, multi-image, missing image path, missing object list, invalid coordinate bins, and silent object reorder.
- [ ] Preserve `example_id` through receipts and errors.

**Acceptance gate:** tests prove valid fixture loads and malformed rows fail before rendering.

### Slice 3.2: Template Renderer And Span Algebra

- [ ] Implement English-only rendering with `source_order` and deterministic `random`.
- [ ] Reject legacy `sorted`.
- [ ] Render `supervised_response_text`, typed character spans, and expected object order.
- [ ] Enforce half-open spans, no crossing spans, whole special-token literal spans, and leaf-span coverage for all loss-bearing characters.
- [ ] Supervise answer content plus `<|im_end|>`; ignore the trailing newline in `<|im_end|>\n`.

**Acceptance gate:** fixture rendered output equals `expected_rendered.json`; boundary tests prove first assistant answer token and `<|im_end|>` handling.

### Slice 3.3: Qwen Setup And Token Identity

- [ ] Load processor/tokenizer/model identity from the local base model path.
- [ ] Verify all required wrapper and coordinate tokens exist.
- [ ] Reject invalid aliases such as `<|object_start|>`.
- [ ] Emit a Qwen setup receipt with tokenizer vocab size, special token ids, processor class, `patch_size`, `merge_size`, and `temporal_patch_size`.

**Acceptance gate:** `pytest tests/coordexp_swift/test_data_template_encoding.py -q` proves token preflight and receipt fields.

### Slice 3.4: No-Resize Encoding And EncodedExample

- [ ] Implement no-resize image validation before processor/model forward.
- [ ] Enforce processor-derived admissible dimensions and raw-pixel/merged-visual-token caps.
- [ ] Encode rendered text and visual payloads.
- [ ] Convert character spans to token atoms/spans without losing boundaries.
- [ ] Reject an `EncodedExample` longer than `packing.global_max_length`.

**Acceptance gate:** fixture encodes with recorded `image_grid_thw`; invalid dimension and over-budget fixture variants fail with typed diagnostics.

### Slice 3.5: Wave 3 Review

- [ ] Request focused review of raw example validation, template boundary
  handling, token identity preflight, no-resize validation, and span-to-token
  alignment.
- [ ] Patch any P0/P1 findings before packing work begins.

**Acceptance gate:** no unresolved P0/P1 findings in data/template/Qwen
encoding.

## Wave 4: Packing And Qwen Forward

**Goal:** Build no-padding packed sequences and prove Qwen forward boundaries before loss/backward work.

**OpenSpec tasks covered:** 6.1, 6.2, 6.3, 6.4, 6.5.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/packing/sequence.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/packing/planner.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/packing/mapping.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/mrope.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/forward.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_packing_forward.py`

### Slice 4.1: Pack Planning

- [ ] Implement overflow-commit packing with one physical packed row per rank/step.
- [ ] Preserve segment ids and example ids.
- [ ] Emit `reports/pack_plan.json` for smoke/debug.
- [ ] Exclude incomplete final optimizer-step windows during run-length resolution.

**Acceptance gate:** tests prove fit/overflow behavior, no padding, and deterministic pack plan receipt.

### Slice 4.2: Supervision Remapping

- [ ] Map logical target positions to physical target positions.
- [ ] Derive `logits_position = target_position - 1`.
- [ ] Preserve example id, segment id, token type, and span provenance.
- [ ] Provide invertible debug traces.

**Acceptance gate:** tests reconstruct original encoded example positions from packed atoms.

### Slice 4.3: Qwen MRoPE Inputs

- [ ] Implement or call installed Qwen helper behavior for MRoPE position ids.
- [ ] Validate row count and row meaning against the source-study result.
- [ ] Reset per packed segment where required.

**Acceptance gate:** source-study fixture and tests prove row-shape parity against installed Transformers behavior.

### Slice 4.4: Qwen Forward Wrapper

- [ ] Implement forward wrapper with `labels=None`, `use_cache=False`, no `inputs_embeds`, full logits, output-shape validation, and model-side loss ignored.
- [ ] Validate placeholder/grid agreement before model forward.
- [ ] Emit `debug/qwen_forward_contract.json`.

**Acceptance gate:** a tiny forward contract run returns finite-shaped logits and emits a receipt without invoking repo losses.

### Slice 4.5: FlashAttention Varlen Proof

- [ ] Implement FA2 branch validation from the source-study proof.
- [ ] Reject packed isolation that relies only on a 2D padding mask.

**Acceptance gate:** tests or probe receipts prove explicit varlen segment isolation evidence when FA2 is enabled.

**Stop condition:** if installed Qwen/FA2 does not expose enough branch evidence, stop and patch the spec with a proven equivalent mechanism.

### Slice 4.6: Wave 4 Review

- [ ] Request focused review of pack isolation, position mapping, MRoPE parity,
  placeholder/grid validation, and FA2 branch evidence.
- [ ] Patch any P0/P1 findings before loss/backward work begins.

**Acceptance gate:** no unresolved P0/P1 findings in packing/Qwen forward.

## Wave 5: Supervision, Losses, Metrics, And Finite Gates

**Goal:** Compute repo-owned protected losses and top-level metrics over packed token-wise supervision.

**OpenSpec tasks covered:** 7.1, 7.2, 7.3, 7.4, 7.5.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/supervision/tokens.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/supervision/alignment.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/context.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/ce.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/token_type_gate.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/normalizers.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/runner.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/losses/finite.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/metrics/accuracy.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/metrics/events.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_supervision_losses.py`

### Slice 5.1: TokenSequence Core

- [ ] Implement `TokenAtom`, `TokenSpan`, and `TokenSequence`.
- [ ] Reject `target_position == 0` for causal losses.
- [ ] Derive dense label parity artifacts only from `TokenSequence`.

**Acceptance gate:** tests prove dense labels are reproducible from token atoms and cannot carry independent semantics.

### Slice 5.2: LossContext And FP32 Selected Logits

- [ ] Implement selected logits upcast to fp32 for objective math.
- [ ] Return selected logits, target ids, and metadata together to avoid caller mismatch.
- [ ] Keep metrics detached where appropriate.

**Acceptance gate:** bf16 input logits are upcast for CE/gate math while metrics can avoid gradient retention.

### Slice 5.3: BaseTokenCE And TokenTypeGateLoss

- [ ] Implement full-vocabulary `BaseTokenCE`.
- [ ] Implement closed V1 token-type vocabulary groups: `desc_text`, `schema`, `coordinate`, and `eos`.
- [ ] Exclude Qwen chat/control, image/video/pad, tool/FIM/repo, think, reserved CoordExp, and other non-target special tokens from free-text allowance.
- [ ] Reject claims of old coordinate soft-CE/object-balanced parity.

**Acceptance gate:** tests prove coordinate targets allow only `<|coord_0|>` through `<|coord_999|>`, schema targets allow only wrappers, and base CE remains full-vocabulary.

### Slice 5.4: Planned-Step Normalizers

- [ ] Compute denominators over the complete planned optimizer-step window across accumulation and ranks.
- [ ] Avoid equally averaging already-normalized micro-step losses.
- [ ] Avoid backend double scaling.

**Acceptance gate:** tests with unequal supervised-token counts prove length-invariant loss behavior.

### Slice 5.5: LossBundle, Accuracy, And Finite Gates

- [ ] Implement `LossRunner` returning total weighted loss, weighted per-term losses, top-level `acc_top1`, top-level `acc_top5`, selected counts, and finite diagnostics.
- [ ] Implement pre-backward scalar finite gate with all-rank consensus interface.
- [ ] Implement post-backward gradient/overflow gate with all-rank consensus interface.

**Acceptance gate:** tests prove non-finite scalar loss skips backward/update and unsafe gradients skip update without mutating planned-step ids.

### Slice 5.6: Wave 5 Review

- [ ] Request focused review of causal target/logits positions, fp32 objective
  logits, token-type vocabulary groups, planned-step normalizers, top-level
  accuracy metrics, and finite gates.
- [ ] Patch any P0/P1 findings before adapter/optimizer work begins.

**Acceptance gate:** no unresolved P0/P1 findings in supervision/losses.

## Wave 6: Adapters, Special-Token Embeddings, Optimizer Groups

**Goal:** Make trainable surfaces explicit, source-study-gated, and receipt-backed.

**OpenSpec tasks covered:** 8.1, 8.2, 8.3, 8.4, 8.5.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/adapters.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/qwen/special_token_embeddings.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/optim/groups.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/optim/builder.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_adapters_embeddings_optim.py`

### Slice 6.1: Adapter Gate And Loading

- [ ] Reject `adapter.type: dlora` until Wave 1 dLoRA gate has passed.
- [ ] Support base-only, base plus existing adapter, and base plus initialized adapter.
- [ ] Enumerate `all_linear` target modules for vision, aligner, and language towers.
- [ ] Exclude `lm_head` from adapter targets.

**Acceptance gate:** tests prove dLoRA fails before the gate and succeeds after a recorded gate receipt.

### Slice 6.2: dLoRA Setup

- [ ] Implement the mechanism selected in `source-studies/dlora.md`.
- [ ] Emit adapter target and initialized adapter receipts.
- [ ] Verify save/load compatibility with the probe and smoke setup.

**Acceptance gate:** dLoRA adapter-enabled setup reaches trainable-surface receipt generation.

### Slice 6.3: Selected Special-Token Embedding Deltas

- [ ] Implement the selected mechanism from `source-studies/special-token-embeddings.md`.
- [ ] Fully train selected wrapper and coordinate-token embeddings.
- [ ] Prevent or mask gradients outside the selected token set.
- [ ] Save compact `special_token_embeddings.safetensors` and `special_token_embeddings.json` or approved equivalents.

**Acceptance gate:** tests prove selected ids change, unselected ids remain frozen, and compact payload reloads.

### Slice 6.4: Explicit Optimizer Groups

- [ ] Require every trainable parameter to match exactly one group.
- [ ] Support vision, aligner, language, adapter parameters, and selected embedding deltas.
- [ ] Fail on missing or duplicate group matches.
- [ ] Emit optimizer group receipt before the first backward pass.

**Acceptance gate:** tests prove unmatched trainable parameters fail fast and intended groups produce exact counts.

### Slice 6.5: Wave 6 Review

- [ ] Request focused review of dLoRA gate usage, adapter target discovery,
  selected-token embedding deltas, compact checkpoint payloads, and optimizer
  group matching.
- [ ] Patch any P0/P1 findings before trainer/runtime integration begins.

**Acceptance gate:** no unresolved P0/P1 findings in adapters, embeddings, and
optimizer setup.

## Wave 7: Trainer, Runtime, Artifacts, Checkpoints, And Eval.Forward

**Goal:** Wire the approved components into a small, professional training loop with artifact-backed observability.

**OpenSpec tasks covered:** 9.1, 9.2, 9.3, 9.4, 9.5.

**Files:**
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/training/stream.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/training/trainer.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/training/step.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/runtime/accelerate_runtime.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/runtime/deepspeed_status.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/runtime/distributed.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/artifacts/checkpoints.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/artifacts/receipts.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/src/eval/forward.py`
- Create: `/data/CoordExp/.worktrees/CoordExp-swift/tests/coordexp_swift/test_training_artifacts.py`

### Slice 7.1: TrainRuntime

- [ ] Implement Accelerate-first prepare/backward/clip/step/save helpers.
- [ ] Implement DeepSpeed status labels: `schema_accepted`, `conflict_validation_implemented`, `systems_smoke_verified`, and `production_supported`.
- [ ] Do not claim DeepSpeed production support in V1.
- [ ] Provide rank guards and all-rank decision helpers.

**Acceptance gate:** tests prove backend accumulation conflicts fail and rank-safe artifact writes are centralized.

### Slice 7.2: Artifact Manager And Metric Events

- [ ] Implement run directory creation and collision policy.
- [ ] Write `run_manifest.json` before training mutation.
- [ ] Preserve minimum manifest top-level keys: `run_id`, `artifact_root`, `run_dir`, `resolved_config`, `resolved_step_schedule`, `artifacts`, `receipts`, `metrics`, `checkpoints`, `eval`, and `backend_status`.
- [ ] Emit metric events with required fields: `event_type`, `planned_step_id`, `split`, `name`, `value`, `trigger_reasons`, `optimizer_update_status`, `finite_status`, and `warning_status`.

**Acceptance gate:** dry-run writes a valid manifest and resolved config without checkpoints, train metrics, or model mutation.

### Slice 7.3: Checkpoint Writer

- [ ] Save adapter payloads when enabled.
- [ ] Save selected special-token embedding deltas when enabled.
- [ ] Save metadata with `checkpoint_id`, `planned_step_id`, `checkpoint_path`, `adapter`, `special_token_embeddings`, `processor_identity`, `resolved_config_fingerprint`, `schedule_identity`, `metric_status`, `trainable_surface`, and `optimizer_update_status`.
- [ ] Use unpadded planned-step ids.
- [ ] Always write `checkpoints/checkpoint-final.json` after completed planned steps.

**Acceptance gate:** checkpoint metadata test rejects `step-000005` style paths and verifies final alias.

### Slice 7.4: Eval.Forward

- [ ] Implement packed forward-only eval using the same render, encode, pack, Qwen forward, loss, and metric stack as training.
- [ ] Do not perform backward or optimizer update.
- [ ] Write `eval/forward/step-<planned_step_id>.json` with `planned_step_id`, `split`, `trigger_reasons`, `example_count`, `pack_count`, `loss_summary`, `metric_summary`, and `artifact_links`.

**Acceptance gate:** scheduled step 4 writes `eval/forward/step-4.json` and records the trigger reason.

### Slice 7.5: SupervisedTrainer

- [ ] Implement loop order: fetch pack, move tensors, Qwen forward, loss context, loss bundle, pre-backward finite check, backward, post-backward global decision, clip, optimizer step when safe, scheduler step on planned-step clock, zero gradients, metrics/artifacts.
- [ ] Keep objective math outside the trainer.
- [ ] Keep artifact schemas outside the trainer.

**Acceptance gate:** trainer-level test with tiny fake components proves event order and update-skipped behavior without invoking Qwen.

### Slice 7.6: Wave 7 Review

- [ ] Request focused review of trainer ownership boundaries, runtime backend
  mechanics, artifact schemas, checkpoint metadata, metric event shape, and
  eval.forward artifact contracts.
- [ ] Patch any P0/P1 findings before the full five-step smoke.

**Acceptance gate:** no unresolved P0/P1 findings in trainer/runtime/artifacts.

## Wave 8: Five-Step Vertical Smoke

**Goal:** Prove the first real end-to-end training path.

**OpenSpec tasks covered:** 10.1, 10.2, 10.3, 10.4, 10.5.

**Files:**
- Read/write runtime artifacts under ignored `/data/CoordExp/.worktrees/CoordExp-swift/outputs/smoke/qwen3_vl_single_image_pack/`
- Update only if evidence requires it: smoke fixture files under `/data/CoordExp/.worktrees/CoordExp-swift/tests/fixtures/smoke/qwen3_vl_single_image_pack/`
- Mark completed OpenSpec tasks only after evidence exists.

### Slice 8.1: Full Dry Run

- [ ] Run:

```bash
python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml --dry-run
```

- [ ] Verify run dir, `run_manifest.json`, `configs/resolved.yaml`, `configs/resolved.json`, and cheap receipts.
- [ ] Verify no checkpoint, train metric event, adapter weights, or embedding deltas are written by dry-run.

**Acceptance gate:** dry-run exits 0 and artifact contract is correct.

### Slice 8.2: Five-Planned-Step dLoRA Smoke

- [ ] Run:

```bash
python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml
```

- [ ] Use real `packing.global_max_length`, fixture-local sample limit, `training.effective_batch_size: 1`, `max_steps: 5`, and explicit `eval.forward.steps: [2, 4]`.
- [ ] Verify five planned steps, two eval.forward summaries, train/eval metric events, final checkpoint, no old production parity claims, and no DeepSpeed production-support claim.

**Acceptance gate:** `checkpoints/checkpoint-final.json` exists and links to final run state.

### Slice 8.3: Artifact Audit

- [ ] Verify resolved config, manifest, Qwen setup receipt, pack plan, loss plan, trainable-surface receipt, optimizer receipt, metrics, eval-forward summaries, checkpoint metadata, and final alias.
- [ ] Verify metric events include top-level `acc_top1` and `acc_top5`.
- [ ] Verify protected weighted losses are stored.
- [ ] Verify non-finite and warning fields exist even when values are safe/empty.

**Acceptance gate:** audit produces no P0/P1 findings and no untriaged P2 that affects accuracy, efficiency, or simplicity.

### Slice 8.4: Wave 8 Review

- [ ] Request focused review of the complete smoke artifact tree.
- [ ] Confirm no hidden base-only or standard-LoRA pre-smoke replaced the
  intended dLoRA adapter-enabled smoke unless the user approved that change.
- [ ] Confirm no V1 non-goals leaked into implementation.

**Acceptance gate:** user accepts the five-step smoke evidence as the first V1
implementation milestone.

## Wave 9: Post-Smoke Consolidation And Handoff

**Goal:** Make the new `src/` implementation easy for the next agent and future user to continue.

**Files:**
- Modify: `/data/CoordExp/.worktrees/CoordExp-swift/openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md`
- Modify: `/data/CoordExp/.worktrees/CoordExp-swift/docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md`
- Create or modify: `/data/CoordExp/.worktrees/CoordExp-swift/docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md`

### Slice 9.1: Task Completion Evidence

- [ ] Mark OpenSpec tasks complete only for implemented, verified work.
- [ ] Keep future rollout, hidden-state, cache, video, multi-image, vLLM, and exact-resume items out of V1.
- [ ] Update `BLUEPRINT.md` card statuses from `approved` to `implemented` or `verified` only when evidence exists.

### Slice 9.2: Handoff Packet

- [ ] Write a compact handoff with objective, current task progress, exact commands, smoke artifact root, unresolved risks, and next recommended step.
- [ ] Include the residual risk text:

```text
Residual risk is now the intended kind: dLoRA definition/probe, special-token embedding mechanism study, smoke fixture materialization, implementation, and the five-step vertical smoke are still pending tasks.
```

**Acceptance gate:** a fresh Codex worker can read the handoff, run the stated commands, and know whether to continue source studies, implementation, or verification.

## Cross-Wave Test Matrix

| Area | Earliest Wave | Required Command |
| --- | --- | --- |
| OpenSpec validity | Wave 0 | `openspec validate rebuild-coordexp-swift-training-infra --strict` |
| Apply progress | Wave 0 and after each wave | `openspec instructions apply --change rebuild-coordexp-swift-training-infra --json` |
| Config runtime | Wave 2 | `pytest tests/coordexp_swift/test_config_runtime.py -q` |
| Fixture/rendering | Wave 2 and Wave 3 | `pytest tests/coordexp_swift/test_data_template_encoding.py -q` |
| Qwen encode/forward | Wave 3 and Wave 4 | `pytest tests/coordexp_swift/test_packing_forward.py -q` |
| Loss math | Wave 5 | `pytest tests/coordexp_swift/test_supervision_losses.py -q` |
| Adapter/embedding/optimizer | Wave 6 | `pytest tests/coordexp_swift/test_adapters_embeddings_optim.py -q` |
| Trainer/artifacts/eval | Wave 7 | `pytest tests/coordexp_swift/test_training_artifacts.py -q` |
| Full smoke contract | Wave 8 | `pytest tests/coordexp_swift/test_vertical_smoke_contract.py -q` |
| Dry run | Wave 8 | `python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml --dry-run` |
| Real smoke | Wave 8 | `python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml` |
| Diff hygiene | Every edited wave | `git diff --check -- <touched paths>` |

## Parallelization Map

Safe parallel lanes:

- Wave 1 source-study lanes can run in parallel: dLoRA, special-token
  embeddings, and Qwen no-resize/MRoPE/FA2.
- Wave 3 raw data validation and template rendering can be developed in
  parallel after the fixture contract is pinned.
- Wave 5 CE/token-type-gate implementation and accuracy metric implementation
  can be developed in parallel after `TokenSequence` and `LossContext` exist.
- Wave 7 checkpoint writer and eval.forward artifacts can be developed in
  parallel after artifact manager interfaces are fixed.

Unsafe parallel lanes:

- Do not move old `src/` while another agent is editing files under `src/`.
- Do not implement dLoRA and selected embedding deltas before their Wave 1
  studies converge.
- Do not implement trainer integration before packing/Qwen forward and losses
  have passed review.
- Do not run the five-step smoke while source studies or checkpoint/artifact
  schemas are still changing.

## Risk Register

| Risk | Wave | Guard |
| --- | --- | --- |
| dLoRA name has no precise upstream meaning | 1, 6 | Source-study gate and round-trip probe before config validation |
| Selected-token embedding deltas break tied input/output behavior | 1, 6 | Mechanism study plus save/load probe before optimizer work |
| No-resize images fail late in Qwen reshape | 1, 3 | Processor-derived dimension validation before processor/model forward |
| FA2 packed isolation is assumed from a 2D mask | 1, 4 | Branch-level varlen evidence required |
| Old `src/` move loses useful reference context | 2 | Separate user approval, move intact to `reference/legacy_src/`, no import shims |
| Roadmap becomes over-broad implementation permission | all | Wave-specific plans and user approval gates |
| Loss normalization regresses to pack-local averaging | 5, 8 | Planned-step denominator tests with unequal token counts |
| Artifact files exist but schemas drift | 7, 8 | Minimum key tests and smoke artifact audit |
| DeepSpeed is accidentally advertised as production-supported | 7, 8 | Status-label tests and smoke artifact review |

## Review And Audit Cadence

- Use a narrow review after Wave 1 because dLoRA and special-token embeddings determine whether Wave 6 can proceed.
- Use a module-card review before Wave 2 archive/skeleton work because moving `src/` is high impact.
- Use focused reviews after Waves 4, 5, 6, and 8.
- Review severity policy:
  - P0: unsafe or impossible implementation; must stop.
  - P1: blocks correctness, reproducibility, or smoke acceptance; must patch before proceeding.
  - P2: patch when cheap and aligned with accuracy, efficiency, and simplicity.
  - Wrong/duplicate: record and close.

## Pre-Kickoff Roadmap Audit

Before real kickoff, ask an independent agent to audit this roadmap in
read-only mode. Use the prompt saved at:

`docs/superpowers/plans/2026-06-30-coordexp-swift-roadmap-audit-prompt.md`

Kickoff is allowed only after the audit verdict is either:

- `READY FOR WAVE 1` with no unresolved P0/P1 findings; or
- `READY WITH PATCHES` and every accepted P0/P1 patch is applied and verified.

## Implementation Agent Start Prompt

Use this prompt when handing the work to a fresh Codex implementation agent:

```text
You are implementing CoordExp-swift in /data/CoordExp/.worktrees/CoordExp-swift.

Read these first:
- AGENTS.md
- docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md
- openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md
- openspec/changes/rebuild-coordexp-swift-training-infra/design.md
- openspec/changes/rebuild-coordexp-swift-training-infra/specs/**/*.md
- docs/architecture/proposals/2026-06-27-coordexp-swift/DECISIONS.md
- docs/architecture/proposals/2026-06-27-coordexp-swift/BLUEPRINT.md

Use superpowers:subagent-driven-development or superpowers:executing-plans.
Start at Wave 0. Do not rewrite src before explicit user approval for the archive/skeleton move.
Do not mark OpenSpec tasks complete without evidence.
Preserve unrelated dirty work.
The current implementation priorities are accuracy/precision first, efficiency second, simplicity third, extensibility fourth.
```

## Self-Audit

- Spec coverage: every remaining OpenSpec task from 2.1 through 10.5 maps to at least one wave and slice.
- Source-study gates: dLoRA, special-token embeddings, Qwen no-resize/MRoPE/FA2 are before production code.
- Approval discipline: old `src/` move and public modules are gated.
- Simplicity guard: no rollout, hidden-state, persistent cache, video, multi-image, vLLM, exact resume, or old objective parity enters V1.
- Artifact guard: smoke acceptance requires resolved config, manifest, receipts, metrics, eval summaries, checkpoint metadata, and final checkpoint alias.
- Handoff guard: a fresh Codex worker has exact paths, commands, stop conditions, and priority order.
- Execution guard: every source-code wave now requires a wave-specific
  Superpowers plan before coding.
- Audit guard: an independent pre-kickoff roadmap audit prompt is saved beside
  this roadmap.
