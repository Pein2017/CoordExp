# Qwen3-VL Painted-GT Transcription Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **2026-07-06 revision note:** This plan remains historical authority for the
> completed painted-GT substrate, warm-start, paint-all, teacher-prefix, and
> counterfactual work. The active human-annotation-style follow-up is
> `next_object_steering`; use
> `docs/superpowers/specs/2026-07-06-next-object-steering-design.md` and
> `docs/superpowers/plans/2026-07-06-next-object-steering-plan.md` for the new
> target grammar, `text_image` / `image_only` policies, and rollout contract.

**Goal:** Build the branch-local painted-GT transcription probe end to end, from strict warm-start expansion and painted materialization through free raw decode, debug metrics, tiny gates, and either a blocked report or a larger two-epoch launch. The mechanistic variable under test is current-pointer identity: whether an obvious visual mark can supply stable "this object now" information strongly enough to overcome language-prior drift during Qwen3-VL autoregressive row generation.

**Architecture:** Keep the original pretrained Qwen base plus adapter philosophy. Add narrow branch-local probe infrastructure around CoordExp-Swift instead of replacing the existing training, inference, and evaluator backbone. The riskiest seams are separated into receipts and gates: all-tower DoRA warm-start, repaired special-token embedding load, geometry-accurate painting, free raw decoding, denominator-bearing debug F1, and tiny overfit eligibility.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL, PEFT DoRA, safetensors, PIL/Pillow, pytest, OpenSpec, CodeGraph, existing CoordExp-Swift config/training/inference/eval modules.

---

## Source Of Truth

- Worktree: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`
- Branch: `codex/qwen3-vl-painted-gt-transcription-probe`
- Research overview: `research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md`
- Experiment plan: `research/ideas/qwen3-vl-painted-gt-transcription-probe/experiment-plan.md`
- Review log: `research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md`
- OpenSpec change: `openspec/changes/add-painted-gt-transcription-probe/`
- OpenSpec task ledger: `openspec/changes/add-painted-gt-transcription-probe/tasks.md`

## Fixed External Inputs

- Base model:
  `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Source LLM-only step-917 DoRA adapter:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917/adapter`
- Repaired selected-token embedding payload:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_support/repaired_special_token_embeddings_step917`
  - `special_token_embeddings.safetensors` SHA256:
    `db95b6e489a73c4314fa1f2562b51ad21b471d4fd683fe669cc0108fb8e4f474`
  - `special_token_embeddings.json` SHA256:
    `6c810e81312ef8e294d5ac0a64e282dbb398e4d193a3c512487bc509892fcb24`
  - `repair_receipt.json` SHA256:
    `1fe802d1e3cc9c9121f55cb781fc7c31a7c45cb766fcb9f815898a78ad6ba276`
  - `shared_embed_delta` tensor SHA256 over contiguous `int16` storage view:
    `af18475fd419822681e23702b47f88a19b5d5d8023b3e3cc8a27ed5c544a1934`
  - tensor summary: shape `[1004, 2048]`, dtype `bfloat16`, nonzero count `2056192`, float32 abs sum approximately `2502.4614`
- Accepted unpainted val200 metric:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z/eval_coco_fixed_gt_scale/metrics.json`
- Accepted unpainted val200 run root:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z`
- Accepted val200 input JSONL:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
  - SHA256:
    `9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4`
- Accepted val200 row-binding identities:
  - `gt_vs_pred.jsonl` SHA256:
    `065ac3dd3a3f8093de97570cfa20bcee3e449802066dec8b1fc19d0c0d64d9d5`
  - `gt_vs_pred_scored.jsonl.provenance.json` row-binding SHA256:
    `217d90415e9ba6f471b16803bc3ca066382949a8feceb00f34b733ae3a45e2e9`
  - `image_plan.jsonl` SHA256:
    `3b4bdb0be78f2cb21164970ae7e0d84e21252c4cbe9d4f5de961f4638fb41398`
  - first row id:
    `coco2017_val_000000000139`
  - first-32 row-id SHA256:
    `81cd5e3bf6218d47b9e3f895204522dede0bdc9ac8dd8bd11a432f6bcb6cbbb9`
- Accepted unpainted val200 values:
  `mAP=0.4111788135144427`, `row_count=200`, `metric_family=coordexp_swift_detection_coco_bbox_v1`

## Fixed V1 Constants

- Paint style:
  - bbox outline: opaque magenta `rgb(255, 0, 255)`
  - center point: opaque yellow `rgb(255, 255, 0)`
  - outline thickness: `max(3, round(min(width, height) * 0.004))`
  - center radius: `max(4, 2 * thickness)`
  - no class text, object id, number, or visible ordering cue
- Geometry audit tolerance: `1` pixel for planned bbox corners and center coordinates.
- Debug metric IoU threshold: `0.50`.
- Primary decode surface: `free_raw_generation`.
- Primary effective generation kwargs: `do_sample=False`, per-mode `max_new_tokens`, `repetition_penalty=1.10`, Qwen `<|im_end|>` `eos_token_id`, tokenizer `pad_token_id`, `return_dict_in_generate=True`, and `output_scores=True`.
- Primary deterministic config invariants: `temperature=0` and `top_p=1.0` may be stored as requested config fields, but they are not effective HF greedy kwargs while `do_sample=False`. They must be rejected when non-default or recorded under `ineffective_config_fields` outside the primary decode hash.
- Primary stop/EOS policy: pass Qwen `<|im_end|>` token id as `eos_token_id`, pass tokenizer `pad_token_id`, preserve raw `<|im_end|>` in token trace, strip exactly one terminal `<|im_end|>` only for parser text with `strip_policy=terminal_im_end`, record `stop_reason` as `im_end` or `length`, and include this policy in decode identity.
- Paint-all max-new-token budget: `512`.
- Stepwise teacher-prefix and self-prefix max-new-token budget: `96` per target step.
- Expanded-seed sanity sample: first `32` rows from the accepted val200 input artifact in artifact row order.
- Expanded-seed block thresholds: row-validity drop `> 0.10` absolute or `debug_detection_f1_v1` drop `> 0.05` absolute.
- Tiny gate slice size: `256` training images.
- Tiny planned budget: `2` epochs, extendable to `8` total epochs only under recorded improving behavior.
- Tiny gate hard-fail thresholds: row validity below `0.50`, malformed/unparsed rate above `0.50`, correct-mark gain over best negative below `0.02`, missing frozen baseline, failed visual audit, failed unpainted baseline, failed expanded-seed sanity, or no primary-metric improvement over paired frozen baseline.
- Tiny gate pass thresholds: row validity at least `0.80`, malformed/unparsed rate at most `0.20`, primary metric gain at least `0.05` absolute for both paint-all and teacher-prefix, and correct-mark advantage at least `0.10` over every required negative control.
- Tiny gate gray-zone rule: any non-self-prefix gray zone blocks larger training.
- Bounded self-prefix diagnostic: `64` images or `512` object steps, whichever is smaller.
- Self-prefix block thresholds: row validity below `0.50`, malformed/unparsed rate above `0.50`, correct-mark advantage at most `0.02`, previous-prefix or next-by-order errors above `0.30`, or raw-prefix contamination preventing more than `25%` of schedules from completing.

## Module Boundaries

Create these branch-local modules unless the implementation discovers a clearer existing owner during Task 1 and records the deviation in the plan review log:

- `src/painted_gt/constants.py`: fixed V1 constants and enum-like literal names.
- `src/painted_gt/slice.py`: deterministic 256-image slice selection and manifest validation.
- `src/painted_gt/painting.py`: painted-plan construction, raster drawing, and plan ids.
- `src/painted_gt/materialize.py`: paint-all and stepwise example materialization as an artifact writer; it must call the existing template owner for prompt text, row text, spans, and label-bearing boundaries.
- `src/painted_gt/schedules.py`: object-id schedule manifests only; it must not independently render compact rows or own final row ordering semantics.
- `src/painted_gt/audit.py`: geometry, raster, visual-gallery, and Qwen no-resize preflight audits.
- `src/painted_gt/decode.py`: paint-all, teacher-prefix, and bounded self-prefix controllers over existing inference backend primitives.
- `src/painted_gt/metrics.py`: `debug_detection_f1_v1` and `per_target_step_debug_f1_v1` over parser-normalized records; it must not implement another compact-row parser, category registry, or official COCO evaluator.
- `src/painted_gt/reports.py`: comparison reports and launch-gate report validation.
- `src/painted_gt/config.py`: branch-local painted-probe config models if painted-specific fields grow beyond generic adapter seed-mode schema.
- `src/painted_gt/cli.py`: thin internal dispatch helpers if an entrypoint needs a small stable surface.

Modify these existing owners narrowly:

- `src/config/models.py`: add only generic, reusable `warm_start_expand_dora` seed schema fields; keep painted-specific condition, slice, and gate config under `src/painted_gt/config.py` when needed.
- `src/adapters/source_gates.py`: add branch-local all-tower DoRA source-gate validation.
- `src/adapters/dora.py`: add exact-key warm-start expansion while preserving existing `initialize_new` and `load_existing`.
- `src/qwen/special_token_embeddings.py`: expose payload identity validation and load-before-optimizer hook if not already sufficient.
- `src/optim/parameter_groups.py`: ensure all expanded adapter and selected-token trainables are explicitly grouped exactly once.
- `src/optim/trainable_surface.py`: receipt dense-base freeze and expanded trainable surface.
- `src/training/pipeline.py`: call warm-start and selected-token payload loading before optimizer grouping; route materialized JSONL through existing packing.
- `src/inference/backend.py`: expose effective generation kwargs and decode hash.
- `src/inference/pipeline.py`: support painted-probe materialized input rows or delegate through `src/painted_gt/decode.py`.
- `src/eval/detection_consumer.py`: keep official mAP/mRecall path stable; add only bridge hooks needed by painted reports.
- `configs/coordexp_swift/painted_gt/`: add tiny and smoke configs.
- `tests/painted_gt/`: add focused unit and integration tests.

Do not edit upstream Transformers or HF model files. Do not merge step-917 into a new full base model. Do not make compact grammar or trie-constrained decoding part of primary evidence.

`src/painted_gt` is an orchestration and artifact package, not a parallel training, template, parser, or evaluator stack. It must call existing owners:

- raw examples and geometry from `src.data`;
- prompt, compact row text, realized object order, and spans from `src.templates`;
- packing/cache and label tensor construction from the existing training path;
- Qwen image and token identity from `src.qwen`;
- generation from `src.inference.backend`;
- parsing and category normalization from `src.inference.parsing` and `src.eval.detection_categories`;
- official mAP/mRecall from `src.eval.detection_consumer`.

## Execution Policy

- Start each implementation task with a narrow dirty-state check.
- Use CodeGraph for source exploration before editing code in that task. If CodeGraph remains stale or empty for this worktree, record the tooling limitation and use exact file reads as fallback.
- Follow TDD for code changes: failing test, minimal implementation, passing test.
- After each coherent task, run targeted tests, `git diff --check`, and make a narrow commit.
- After each important phase, run isolated review using `review-convergence-loop`.
- Preserve unrelated `.codex/agents/*.toml` modifications.
- Stop only for research semantic drift, destructive cleanup, unsafe GPU use, impossible source artifact identity, or a real implementation blocker.

## Task 0: Planning Gate Before Source Work

**Files:**
- Modify: `docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md`
- Modify: `openspec/changes/add-painted-gt-transcription-probe/tasks.md`
- Modify: `research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md`

- [ ] **Step 1: Close plan review findings before source work**

Record the Superpowers plan review lanes, accepted findings, rejected findings, and patch resolutions. Any timed-out, disconnected, vague, or missing reviewer lane is unresolved until either it returns or a focused replacement review covers the same risk.

- [ ] **Step 2: Run planning-only verification**

Run:

```bash
openspec validate add-painted-gt-transcription-probe --strict
openspec instructions apply --change add-painted-gt-transcription-probe --json > temp/painted_gt_openspec_instructions.json
python - <<'PY'
from pathlib import Path
plan = Path("docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md")
text = plan.read_text()
for bad in ["TO" + "DO", "TB" + "D"]:
    assert bad not in text, bad
assert text.count(chr(96) * 3) % 2 == 0
print("planning gate hygiene: ok")
PY
git diff --check -- docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md openspec/changes/add-painted-gt-transcription-probe research/ideas/qwen3-vl-painted-gt-transcription-probe
```

- [ ] **Step 3: Mark only governance tasks backed by evidence**

Before any `src/`, `scripts/`, `configs/`, or `tests/` implementation change, update only OpenSpec governance tasks `1.1` through `1.5` when their evidence exists. Do not mark source-study or implementation tasks complete during this planning-only commit.

- [ ] **Step 4: Commit the converged planning artifacts**

Run:

```bash
git add docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md openspec/changes/add-painted-gt-transcription-probe/tasks.md research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md
git diff --cached --check
git commit -m "docs: add painted gt implementation roadmap"
```

## Task 1: Source Preflight And Implementation Notes

**Files:**
- Create: `openspec/changes/add-painted-gt-transcription-probe/implementation-notes.md`
- Create: `scripts/probes/painted_gt/probe_warm_start_inputs.py`
- Create: `scripts/probes/painted_gt/probe_dora_targets.py`
- Create: `scripts/probes/painted_gt/probe_val200_sanity_source.py`
- Modify: `openspec/changes/add-painted-gt-transcription-probe/tasks.md`
- Test: command-level probes, no pytest required

- [ ] **Step 1: Verify worktree state and CodeGraph**

Run:

```bash
git status --short --branch
codegraph init -i || true
codegraph index || true
codegraph status || true
python - <<'PY'
from pathlib import Path
print("codegraph_dir", Path(".codegraph").exists())
PY
```

Expected:

- branch is `codex/qwen3-vl-painted-gt-transcription-probe`;
- only unrelated `.codex/agents/*.toml` files may be dirty before this task;
- CodeGraph status either names this exact worktree with nonzero source nodes, or implementation notes record that exact file reads are authoritative for this task.

- [ ] **Step 2: Write source-input probe script**

Create `scripts/probes/painted_gt/probe_warm_start_inputs.py` with a script that:

- verifies the fixed source adapter path exists;
- verifies `adapter_config.json` exists and contains DoRA semantics;
- loads all safetensors in the source adapter directory;
- counts language `lora_A`, `lora_B`, and DoRA magnitude tensors;
- verifies the repaired embedding payload path exists;
- verifies repaired payload files `special_token_embeddings.json`, `special_token_embeddings.safetensors`, and `repair_receipt.json`;
- computes the repaired payload safetensors SHA256;
- computes `special_token_embeddings.json` and `repair_receipt.json` SHA256;
- computes the `shared_embed_delta` tensor SHA256 over a contiguous `int16` storage view;
- records shape, dtype, nonzero count, and float32 abs sum;
- fails when the selected-token tensor is all zero unless an explicit later contract approves that;
- prints JSON to stdout with absolute paths, tensor counts, tensor key samples, payload files, and payload content hashes.

Minimum output schema:

```json
{
  "source_adapter_path": "/absolute/path",
  "adapter_config_exists": true,
  "safetensor_files": ["adapter_model.safetensors"],
  "language_lora_a_count": 196,
  "language_lora_b_count": 196,
  "language_dora_magnitude_count": 196,
  "repaired_embedding_payload_path": "/absolute/path",
  "embedding_payload_files": [
    "repair_receipt.json",
    "special_token_embeddings.json",
    "special_token_embeddings.safetensors"
  ],
  "embedding_safetensors_sha256": "db95b6e489a73c4314fa1f2562b51ad21b471d4fd683fe669cc0108fb8e4f474",
  "embedding_json_sha256": "6c810e81312ef8e294d5ac0a64e282dbb398e4d193a3c512487bc509892fcb24",
  "repair_receipt_sha256": "1fe802d1e3cc9c9121f55cb781fc7c31a7c45cb766fcb9f815898a78ad6ba276",
  "shared_embed_delta_sha256": "af18475fd419822681e23702b47f88a19b5d5d8023b3e3cc8a27ed5c544a1934",
  "shared_embed_delta_nonzero_count": 2056192
}
```

- [ ] **Step 3: Write all-tower target probe script**

Create `scripts/probes/painted_gt/probe_dora_targets.py` with a script that loads the approved base model metadata without training, runs the existing DoRA target-discovery helper, and prints JSON with per-tower target counts.

Expected JSON values for the approved 2B base:

```json
{
  "target_policy": "all_linear",
  "target_towers": ["language", "vision", "aligner"],
  "language_count": 196,
  "vision_count": 96,
  "aligner_count": 8,
  "total_count": 300
}
```

If loading the full model is too expensive for the probe, use the same source path and symbol route that production setup will use, then record the reason in `implementation-notes.md`.

- [ ] **Step 4: Write accepted-val200 sanity-source probe script**

Create `scripts/probes/painted_gt/probe_val200_sanity_source.py` with a script that:

- verifies the accepted val200 run root exists;
- verifies `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `image_plan.jsonl`, and `configs/resolved.json` exist;
- asserts `gt_vs_pred.jsonl` SHA256 is `065ac3dd3a3f8093de97570cfa20bcee3e449802066dec8b1fc19d0c0d64d9d5`;
- asserts provenance `row_binding.row_ids_sha256` is `217d90415e9ba6f471b16803bc3ca066382949a8feceb00f34b733ae3a45e2e9`;
- asserts resolved config `data.input_jsonl` is `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`;
- asserts the resolved input JSONL SHA256 is `9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4`;
- asserts `image_plan.jsonl` SHA256 is `3b4bdb0be78f2cb21164970ae7e0d84e21252c4cbe9d4f5de961f4638fb41398`;
- asserts the first `32` row ids have SHA256 `81cd5e3bf6218d47b9e3f895204522dede0bdc9ac8dd8bd11a432f6bcb6cbbb9`;
- asserts the first row id is `coco2017_val_000000000139`;
- asserts the first `32` `image_plan.jsonl` rows have `status="ok"` and `do_resize=false`.

The expanded-seed sanity probe must use the resolved input JSONL for re-inference and the accepted artifact row ids for row-order binding.

- [ ] **Step 5: Run probes and save notes**

Run:

```bash
mkdir -p temp/painted_gt/source_probes
python scripts/probes/painted_gt/probe_warm_start_inputs.py | tee temp/painted_gt/source_probes/warm_start_inputs.json
python scripts/probes/painted_gt/probe_dora_targets.py | tee temp/painted_gt/source_probes/dora_targets.json
python scripts/probes/painted_gt/probe_val200_sanity_source.py | tee temp/painted_gt/source_probes/val200_sanity_source.json
python - <<'PY'
import json
from pathlib import Path
warm = json.loads(Path("temp/painted_gt/source_probes/warm_start_inputs.json").read_text())
targets = json.loads(Path("temp/painted_gt/source_probes/dora_targets.json").read_text())
val200 = json.loads(Path("temp/painted_gt/source_probes/val200_sanity_source.json").read_text())
assert warm["language_lora_a_count"] == 196
assert warm["language_lora_b_count"] == 196
assert warm["language_dora_magnitude_count"] == 196
assert warm["embedding_safetensors_sha256"] == "db95b6e489a73c4314fa1f2562b51ad21b471d4fd683fe669cc0108fb8e4f474"
assert warm["embedding_json_sha256"] == "6c810e81312ef8e294d5ac0a64e282dbb398e4d193a3c512487bc509892fcb24"
assert warm["repair_receipt_sha256"] == "1fe802d1e3cc9c9121f55cb781fc7c31a7c45cb766fcb9f815898a78ad6ba276"
assert warm["shared_embed_delta_sha256"] == "af18475fd419822681e23702b47f88a19b5d5d8023b3e3cc8a27ed5c544a1934"
assert warm["shared_embed_delta_nonzero_count"] == 2056192
assert targets["language_count"] == 196
assert targets["vision_count"] == 96
assert targets["aligner_count"] == 8
assert targets["total_count"] == 300
assert val200["first_row_id"] == "coco2017_val_000000000139"
assert val200["input_jsonl_sha256"] == "9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4"
assert val200["first_32_row_ids_sha256"] == "81cd5e3bf6218d47b9e3f895204522dede0bdc9ac8dd8bd11a432f6bcb6cbbb9"
print("source probes: ok")
PY
```

- [ ] **Step 6: Record implementation notes and task progress**

Create `implementation-notes.md` with:

- CodeGraph status for this exact worktree;
- exact probe commands and output paths;
- source adapter tensor counts;
- all-tower target counts;
- repaired embedding payload identity;
- repaired embedding safetensors SHA, JSON SHA, repair-receipt SHA, tensor hash, nonzero count, and abs-sum summary;
- fixed V1 constants;
- accepted val200 input JSONL path and hash used for the first-32 sanity sample;
- accepted val200 raw artifact SHA, row-binding SHA, image-plan SHA, first row id, and first-32 row-id SHA;
- any fallback or deviation.

Update `openspec/changes/add-painted-gt-transcription-probe/tasks.md` by checking off completed items in sections 1 and 2 only when evidence exists.

- [ ] **Step 7: Verify and commit**

Run:

```bash
openspec validate add-painted-gt-transcription-probe --strict
git diff --check -- openspec/changes/add-painted-gt-transcription-probe scripts/probes/painted_gt
git add openspec/changes/add-painted-gt-transcription-probe scripts/probes/painted_gt
git commit -m "chore: record painted gt source probes"
```

## Task 2: Warm-Start Expand DoRA Seed Mode

**Files:**
- Modify: `src/config/models.py`
- Modify: `src/adapters/dora.py`
- Modify: `src/adapters/source_gates.py`
- Modify: `src/training/pipeline.py`
- Create: `tests/adapters/test_warm_start_expand_dora.py`
- Modify: `tests/training/test_pipeline_assembly.py`

- [ ] **Step 1: Write failing config and source-gate tests**

Add tests proving:

- `adapter.seed_mode: warm_start_expand_dora` validates only with a source adapter path, all-tower target towers, and repaired embedding payload path;
- direct all-tower `load_existing` from the LLM-only source is rejected;
- branch-local source-gate evidence must cover `language`, `vision`, and `aligner`, not language-only evidence.

Minimum test names:

```python
def test_warm_start_expand_dora_requires_source_adapter_and_embedding_payload() -> None: ...
def test_warm_start_expand_dora_rejects_language_only_source_gate() -> None: ...
def test_direct_all_tower_load_existing_from_llm_only_source_is_rejected() -> None: ...
```

Run:

```bash
pytest tests/adapters/test_warm_start_expand_dora.py -q
```

Expected: tests fail because the seed mode and validators do not exist yet.

- [ ] **Step 2: Implement strict schema**

Extend `AdapterConfig` without weakening current modes:

- accepted `seed_mode` values include `initialize_new`, `load_existing`, and `warm_start_expand_dora`;
- `warm_start_expand_dora` requires source adapter path, repaired embedding payload path, target towers, and target policy;
- legacy `adapter.type: dora` behavior remains unchanged when `seed_mode` is absent or set to an existing mode.

Reject unknown seed modes with a clear runtime/config error.

- [ ] **Step 3: Implement branch-local source-gate validation**

Add a source-gate validator that requires:

- base model path equals the approved base path or is recorded as an explicit config identity match;
- target policy `all_linear`;
- target towers exactly include `language`, `vision`, and `aligner`;
- discovered counts `196`, `96`, `8`, and `300` for the approved base;
- source embedding payload evidence exists.

- [ ] **Step 4: Write failing key-map tests**

Add tests with small fake state dicts proving:

- language `lora_A`, `lora_B`, and DoRA magnitude tensors are copied by exact canonical key;
- shape-only ambiguous candidates are rejected;
- missing language magnitude vector is rejected;
- missing vision/aligner source tensors are allowed only when recorded as initialized.

Minimum test names:

```python
def test_warm_start_copies_language_tensors_by_exact_key() -> None: ...
def test_warm_start_rejects_shape_only_ambiguity() -> None: ...
def test_warm_start_requires_language_dora_magnitude() -> None: ...
def test_warm_start_records_initialized_vision_and_aligner_tensors() -> None: ...
```

- [ ] **Step 5: Implement exact-key copy and report**

In `src/adapters/dora.py`, add a warm-start helper that:

- constructs the fresh all-tower DoRA adapter first;
- loads source adapter tensors from safetensors;
- normalizes adapter-name and DoRA magnitude key spelling in one explicit function;
- builds a deterministic `source_key -> target_key` map for language tensors;
- checks tensor shapes after key match;
- copies tensor data;
- verifies post-copy equality;
- records missing-source tensors for vision/aligner as initialized;
- writes `warm_start_report.json` before optimizer construction.

Minimum report fields:

```json
{
  "seed_mode": "warm_start_expand_dora",
  "source_adapter_path": "/absolute/path",
  "target_towers": ["language", "vision", "aligner"],
  "target_policy": "all_linear",
  "copied": {"lora_A": 196, "lora_B": 196, "dora_magnitude": 196},
  "initialized": {"vision": 288, "aligner": 24},
  "missing_unexpected": [],
  "post_copy_equality": "pass"
}
```

Use exact actual initialized tensor counts from the implementation, but preserve per-tower separation.

- [ ] **Step 6: Wire training setup order**

Update `src/training/pipeline.py` so warm-start order is:

1. load base Qwen model;
2. construct expanded DoRA adapter;
3. copy source language tensors;
4. load repaired selected-token embedding payload;
5. build optimizer groups;
6. build trainable-surface receipt;
7. start first forward/backward.

Add a pipeline assembly test that logs calls and asserts this order.

- [ ] **Step 7: Verify and commit**

Run:

```bash
pytest tests/adapters/test_warm_start_expand_dora.py tests/adapters/test_dora_setup.py tests/adapters/test_source_gates.py -q
pytest tests/training/test_pipeline_assembly.py -q
git diff --check -- src/config src/adapters src/training tests/adapters tests/training
git add src/config/models.py src/adapters/dora.py src/adapters/source_gates.py src/training/pipeline.py tests/adapters tests/training/test_pipeline_assembly.py
git commit -m "feat: add painted gt warm start dora seed"
```

## Task 3: Selected-Token Embedding And Optimizer Coverage

**Files:**
- Modify: `src/qwen/special_token_embeddings.py`
- Modify: `src/optim/parameter_groups.py`
- Modify: `src/optim/trainable_surface.py`
- Modify: `src/training/pipeline.py`
- Create: `tests/qwen/test_painted_gt_embedding_payload.py`
- Modify: `tests/optim/test_parameter_groups.py`
- Modify: `tests/optim/test_trainable_surface_receipts.py`

- [ ] **Step 1: Write failing embedding payload tests**

Add tests proving:

- the repaired payload is required for `warm_start_expand_dora`;
- payload identity checks cover tensor key, shape, dtype, token strings, token ids, base config hash, and tokenizer hash;
- payload content checks cover `special_token_embeddings.safetensors` SHA256, `shared_embed_delta` tensor SHA256, nonzero count, and a small numeric summary;
- zero-delta fallback is rejected for this branch;
- metadata copied onto an all-zero or perturbed safetensors payload is rejected before optimizer grouping;
- load happens before optimizer group construction.

Minimum test names:

```python
def test_repaired_step917_embedding_payload_identity_is_required() -> None: ...
def test_repaired_step917_embedding_payload_content_hash_is_required() -> None: ...
def test_zero_delta_selected_embedding_fallback_rejected_for_warm_start() -> None: ...
def test_embedding_payload_load_precedes_optimizer_grouping() -> None: ...
```

- [ ] **Step 2: Implement identity validation**

Expose or add a helper in `src/qwen/special_token_embeddings.py` that validates:

- tensor key `shared_embed_delta` or the current default key;
- selected token count `1004`;
- coordinate tokens and wrapper tokens are present;
- token strings match token ids;
- dtype and shape are recorded;
- base config and tokenizer identities match the current runtime.
- the payload safetensors SHA256 matches `db95b6e489a73c4314fa1f2562b51ad21b471d4fd683fe669cc0108fb8e4f474` for the canonical repaired step-917 payload;
- the `shared_embed_delta` tensor SHA256 over contiguous `int16` storage view matches `af18475fd419822681e23702b47f88a19b5d5d8023b3e3cc8a27ed5c544a1934`;
- the tensor is not all zero and reports nonzero count `2056192` for the canonical payload.

- [ ] **Step 3: Write failing optimizer coverage tests**

Add tests proving:

- every trainable adapter parameter belongs to exactly one explicit LR group;
- selected coordinate and wrapper embedding deltas belong to exactly one explicit LR group;
- dense base parameters are frozen;
- unmatched trainable parameters fail fast.

Minimum test names:

```python
def test_expanded_adapter_and_selected_embeddings_have_exact_optimizer_coverage() -> None: ...
def test_dense_base_trainable_parameter_fails_warm_start_surface_check() -> None: ...
def test_unmatched_trainable_parameter_fails_optimizer_grouping() -> None: ...
```

- [ ] **Step 4: Implement receipts**

Extend optimizer and trainable-surface receipts to record:

- language, vision, and aligner adapter trainable counts;
- coordinate-token embedding delta count;
- wrapper-token embedding delta count;
- optimizer group names and learning rates;
- dense-base frozen status.

Fail before first backward if any trainable parameter is unmatched or multiply matched.

- [ ] **Step 5: Verify and commit**

Run:

```bash
pytest tests/qwen/test_painted_gt_embedding_payload.py tests/qwen/test_special_token_embeddings.py -q
pytest tests/optim/test_parameter_groups.py tests/optim/test_trainable_surface_receipts.py -q
pytest tests/training/test_pipeline_assembly.py -q
git diff --check -- src/qwen src/optim src/training tests/qwen tests/optim tests/training
git add src/qwen/special_token_embeddings.py src/optim/parameter_groups.py src/optim/trainable_surface.py src/training/pipeline.py tests/qwen tests/optim tests/training/test_pipeline_assembly.py
git commit -m "feat: validate painted gt trainable surface"
```

## Task 4: Painted Slice, Plans, And Visual Audit

**Files:**
- Create: `src/painted_gt/__init__.py`
- Create: `src/painted_gt/constants.py`
- Create: `src/painted_gt/slice.py`
- Create: `src/painted_gt/painting.py`
- Create: `src/painted_gt/schedules.py`
- Create: `src/painted_gt/audit.py`
- Create: `tests/painted_gt/test_slice.py`
- Create: `tests/painted_gt/test_painting.py`
- Create: `tests/painted_gt/test_visual_audit.py`

- [ ] **Step 1: Write failing slice tests**

Add tests proving:

- deterministic selection returns exactly 256 image rows;
- manifest records dataset identity, seed, image ids, source row ids, object counts, bucket labels, builder identity, and `slice_id`;
- `slice_manifest.json` is the canonical slice filename;
- changing image ids while claiming the same `slice_id` is rejected.

Minimum test names:

```python
def test_slice_manifest_records_256_deterministic_images() -> None: ...
def test_slice_manifest_rejects_changed_image_ids_for_same_slice_id() -> None: ...
```

- [ ] **Step 2: Implement slice manifest builder**

Implement a coverage-aware deterministic builder with a fixed seed. It may start with simple bucket quotas based on object count and repeated-class flags; keep the code small and record bucket labels explicitly. Use source JSONL identities already present in the dataset rows.

- [ ] **Step 3: Write failing painting tests**

Add tests proving:

- paint-all and stepwise plans share style fields;
- bbox and center pixels come from `src/data/geometry.py::coord_bins_to_pixel_xyxy`;
- no class text, object id, number, or visible ordering cue is drawn;
- `painted_plan_id` is stable for the same source image, object set, style, and target object.

Minimum test names:

```python
def test_paint_all_and_stepwise_share_v1_style() -> None: ...
def test_painted_plan_uses_coord_bins_to_pixel_geometry() -> None: ...
def test_painted_plan_id_is_stable() -> None: ...
```

- [ ] **Step 4: Implement painting primitive**

Implement plan construction and raster drawing with PIL. The plan object must record:

- source image identity and dimensions;
- model-input dimensions;
- painter coordinate space;
- painted object ids;
- target object id when applicable;
- bbox rectangle geometry;
- center point geometry;
- painting style;
- stable `painted_plan_id`.

The canonical plan writer must write `painted_plan.jsonl`. This filename is not replaceable; only materialized training/eval payload filenames may use a documented equivalent format.

- [ ] **Step 5: Write failing audit tests**

Add tests proving:

- planned geometry mismatch greater than 1 pixel fails;
- forced off-by-two mismatch fails;
- raster checks find painted bbox edges and center disk;
- reopened painted images preserve dimensions;
- Qwen no-resize preflight rejects changed dimensions.

Minimum test names:

```python
def test_geometry_audit_accepts_one_pixel_tolerance() -> None: ...
def test_geometry_audit_rejects_forced_off_by_two() -> None: ...
def test_qwen_no_resize_preflight_rejects_resized_painted_image() -> None: ...
```

- [ ] **Step 6: Implement visual audit artifacts**

Write `visual_audit/index.json` with entries for sampled paint-all and stepwise examples. Each entry must include source image id, painted-plan id, target object id when applicable, geometry status, raster status, image path, and failure reasons.

Task 4 must leave the artifact writer able to produce the canonical manifest set used by Task 5: `slice_manifest.json`, `painted_plan.jsonl`, `schedule_manifest.<schedule_id>.json`, `condition_manifest.json`, and `visual_audit/index.json`.

- [ ] **Step 7: Verify and commit**

Run:

```bash
pytest tests/painted_gt/test_slice.py tests/painted_gt/test_painting.py tests/painted_gt/test_visual_audit.py -q
git diff --check -- src/painted_gt tests/painted_gt
git add src/painted_gt tests/painted_gt
git commit -m "feat: add painted gt materialization primitives"
```

## Task 5: Paint-All And Stepwise Materialized Examples

**Files:**
- Create: `src/painted_gt/materialize.py`
- Modify: `src/painted_gt/schedules.py`
- Modify: `src/painted_gt/audit.py`
- Modify: `src/training/pack_cache.py`
- Modify: `src/training/pipeline.py`
- Create: `tests/painted_gt/test_materialize.py`
- Create: `tests/painted_gt/test_stepwise_labels.py`
- Modify: `tests/training/test_pack_cache.py`
- Modify: `tests/training/test_pipeline_assembly.py`

- [ ] **Step 1: Write failing paint-all materialization tests**

Add tests proving:

- all GT objects are painted;
- assistant targets use full compact rows in canonical `geo_sorted` order;
- materialized payload records `slice_id`, `condition_name`, `painted_plan_id`, source image identity, geometry audit status, and resolved artifact root.
- materialization writes exact canonical manifest filenames: `slice_manifest.json`, `painted_plan.jsonl`, `condition_manifest.json`, and `visual_audit/index.json`.

Minimum test name:

```python
def test_paint_all_materialization_writes_geo_sorted_full_rows() -> None: ...
```

- [ ] **Step 2: Write failing stepwise schedule and label tests**

Add tests proving:

- `geo_sorted` schedule and one frozen random schedule are deterministic and reusable;
- canonical schedule files use `schedule_manifest.<schedule_id>.json`;
- step index maps to the same target object across frozen, trained, teacher-prefix, self-prefix, and controls;
- stepwise schedule object ids match `RenderedExample.realized_object_order` from `src.templates`;
- compact row text, prompt bytes, and spans come from `src.templates`, not from a painted-GT copy of the renderer;
- previous GT prefix tokens have `ignore_index` labels;
- previous-prefix `ignore_index` spans are derived from renderer spans rather than ad hoc string offsets;
- only the current row is label-bearing.

Minimum test names:

```python
def test_stepwise_schedule_manifest_reuses_geo_sorted_and_random_orders() -> None: ...
def test_stepwise_teacher_prefix_ignores_previous_row_labels() -> None: ...
def test_stepwise_first_step_has_empty_prefix_and_one_target_row() -> None: ...
```

- [ ] **Step 3: Implement materialization**

Write materialized examples as JSONL or the documented equivalent already consumed by CoordExp-Swift training. The files must preserve:

- prompt bytes;
- target text;
- image path;
- source row id;
- object ids;
- condition id;
- slice id;
- schedule id when applicable;
- painted plan id;
- label-mask intent for stepwise examples.

Materialization must call the existing template renderer or a small public helper extracted from it for prompt text, compact row text, realized object order, and span boundaries. `src/painted_gt/materialize.py` must not implement a second compact-row renderer.

- [ ] **Step 4: Implement counterfactual controls**

Implement required controls:

- `painted_correct`;
- `unpainted_same_prompt`;
- stepwise `wrong_object_mark`;
- stepwise `shuffled_or_offset_mark`;
- paint-all `offset_all_marks`;
- paint-all `style_matched_false_marks`.

Record negative selection reason and nearest-GT IoU. If a strong negative is unavailable, record `no_hard_negative_available` and mark the example scoped for gate reporting.

Weak or unavailable negatives must be counted in report coverage. They must not contribute easy correct-mark wins. If required mark-margin evidence is dominated by `no_hard_negative_available`, weak fallback, off-frame, size-mismatched, style-mismatched, or GT-overlapping negatives, the mark-dependence gate must report `gray_zone` or scoped claims rather than `pass`.

- [ ] **Step 5: Keep packing deterministic**

Ensure existing pack-cache and training pipeline consume materialized examples without changing label masks or object-step order. Cache identity should include semantic materialization identity, not ephemeral artifact timestamps.

`src/painted_gt` must emit a stable materialization fingerprint or equivalent identity field derived from slice id, condition id, painted-plan ids, schedule ids when applicable, prompt/template identity, and materialized example content. `src/training/pack_cache.py` may consume that value as a generic dataset/materialization determinant, but it must not parse `slice_manifest.json`, `painted_plan.jsonl`, `schedule_manifest.<schedule_id>.json`, or import `src.painted_gt`.

Add tests proving two materialized JSONLs with different painted-plan or schedule identity do not share a cache fingerprint, and that the pack-cache module has no `src.painted_gt` dependency.

- [ ] **Step 6: Verify and commit**

Run:

```bash
pytest tests/painted_gt/test_materialize.py tests/painted_gt/test_stepwise_labels.py -q
pytest tests/training/test_pack_cache.py tests/training/test_pipeline_assembly.py -q
git diff --check -- src/painted_gt src/training tests/painted_gt tests/training
git add src/painted_gt src/training tests/painted_gt tests/training/test_pack_cache.py tests/training/test_pipeline_assembly.py
git commit -m "feat: materialize painted gt training examples"
```

## Task 6: Free Raw Decode Surface And Painted Controllers

**Files:**
- Create: `src/painted_gt/decode.py`
- Modify: `src/inference/backend.py`
- Modify: `src/inference/pipeline.py`
- Modify: `src/config/inference.py`
- Create: `tests/painted_gt/test_decode_controllers.py`
- Modify: `tests/inference/test_backend_trace.py`
- Modify: `tests/inference/test_config_runtime.py`
- Modify: `tests/inference/test_pipeline.py`

- [ ] **Step 1: Write failing decode-surface tests**

Add tests proving:

- primary decode config requires `decode_surface=free_raw_generation`;
- primary paint-all uses `max_new_tokens=512`;
- primary stepwise uses `max_new_tokens=96`;
- effective generation kwargs include `do_sample=False`, `repetition_penalty=1.10`, Qwen `<|im_end|>` `eos_token_id`, tokenizer `pad_token_id`, `return_dict_in_generate=True`, and `output_scores=True`;
- `temperature=0` and `top_p=1.0` are treated as requested deterministic config invariants but excluded from the primary effective decode hash unless the backend actually passes and uses them;
- non-default ineffective config-only generation fields are rejected or recorded under `ineffective_config_fields` outside the primary decode hash;
- stop/EOS policy, strip policy, stop reason, and truncation counts are recorded in artifacts;
- compact grammar or trie-constrained decoding cannot feed primary gate metrics;
- config-only knobs not passed to backend are rejected or recorded as ineffective outside the decode hash.

Minimum test names:

```python
def test_primary_decode_surface_requires_free_raw_generation() -> None: ...
def test_decode_hash_uses_effective_backend_kwargs() -> None: ...
def test_primary_decode_hash_includes_stop_eos_policy() -> None: ...
def test_ineffective_temperature_top_p_are_outside_primary_decode_hash() -> None: ...
def test_constrained_decode_marked_sensitivity_not_primary() -> None: ...
```

- [ ] **Step 2: Expose effective generation kwargs**

In `src/inference/backend.py`, expose a deterministic method or receipt field that records exactly the kwargs passed to `generate`. Use that receipt to compute decode hash. Do not hash config-only fields that are not passed to the backend.

The effective receipt must include stop/EOS policy:

- Qwen `<|im_end|>` token id used as `eos_token_id`;
- tokenizer `pad_token_id`;
- `strip_policy=terminal_im_end`;
- raw token trace preserves `<|im_end|>`;
- parser text strips exactly one terminal `<|im_end|>`;
- per-row `stop_reason` and truncation counters.

- [ ] **Step 3: Write failing controller tests**

Add tests proving:

- paint-all emits one full-response artifact per image;
- teacher-prefix emits exactly one step artifact per scheduled object;
- self-prefix appends raw generated text, never normalized or salvaged rows;
- malformed self-prefix text remains in the next prefix and parser salvage is stored separately;
- comparison reports reject rows with different stop/EOS policy even when all other decode fields match.
- painted controllers consume materialized prompt records from the existing template/inference prompt owner, or call a public helper from `src.inference.prompt`, and every decode artifact records prompt-token parity evidence.

Minimum test names:

```python
def test_paint_all_decode_writes_raw_suffix_even_when_parse_fails() -> None: ...
def test_teacher_prefix_decode_emits_one_row_per_scheduled_step() -> None: ...
def test_self_prefix_controller_appends_raw_text_not_salvaged_row() -> None: ...
def test_painted_controller_uses_inference_prompt_parity_owner() -> None: ...
def test_comparison_report_rejects_stop_policy_mismatch() -> None: ...
```

- [ ] **Step 4: Implement controllers**

Implement controllers over existing inference runtime/backend primitives:

- paint-all: one prompt/image to one full compact-row response;
- teacher-prefix: current painted target image plus previous GT prefix to one current-row generation;
- self-prefix: current painted target image plus accumulated raw model prefix to one current-row generation, advancing the oracle target schedule until schedule exhaustion or block condition.

Each artifact row must preserve condition, slice id, schedule id when applicable, painted-plan id, source image id, target object id when applicable, raw prompt, prefix text, raw generated suffix, parser status, scoreability flags, effective generation kwargs, stop/EOS policy id, strip policy, stop reason, and truncation status.

Controllers must not hand-roll chat-template text or prompt tokenization. They must consume materialized prompt records built by the existing template/inference prompt owner, or use a small public helper extracted from `src.inference.prompt` that performs the same prompt-token parity check as the standard inference path.

- [ ] **Step 5: Verify and commit**

Run:

```bash
pytest tests/painted_gt/test_decode_controllers.py -q
pytest tests/inference/test_backend_trace.py tests/inference/test_config_runtime.py tests/inference/test_pipeline.py -q
git diff --check -- src/painted_gt src/inference src/config tests/painted_gt tests/inference
git add src/painted_gt/decode.py src/inference src/config/inference.py tests/painted_gt tests/inference
git commit -m "feat: add painted gt free decode controllers"
```

## Task 7: Debug F1 Metrics, Comparisons, And Gate Reports

**Files:**
- Create: `src/painted_gt/metrics.py`
- Create: `src/painted_gt/reports.py`
- Modify: `src/eval/detection_consumer.py`
- Create: `tests/painted_gt/test_debug_metrics.py`
- Create: `tests/painted_gt/test_reports.py`
- Modify: `tests/eval/test_detection_consumer.py`

- [ ] **Step 1: Write failing debug metric tests**

Add tests proving:

- exact normalized COCO-80 description match is required;
- IoU threshold is `0.50`;
- matching is one-to-one greedy by score, or emission order when score is absent;
- scoreless predictions remain eligible through emission-order fallback when the report records the scoreability policy;
- unknown categories, malformed rows, invalid geometry, and unparseable attempts count as false positives;
- for paint-all, a full-response parse failure with no identifiable row attempts counts as one invalid generation-attempt false positive, while unmatched GT objects remain false negatives;
- unmatched GT objects or target steps count as false negatives;
- stepwise invalid generation counts as both a missed target and false positive attempt;
- inputs are parser-normalized records from `src.inference.parsing` or a narrow adapter over that output;
- category normalization uses `src.eval.detection_categories`, not a painted-GT-local category registry.

Minimum test names:

```python
def test_debug_f1_requires_exact_category_and_iou_050() -> None: ...
def test_debug_f1_counts_malformed_attempt_in_precision_denominator() -> None: ...
def test_paint_all_full_response_parse_failure_counts_one_invalid_attempt() -> None: ...
def test_scoreless_debug_f1_uses_emission_order_with_scoreability_policy() -> None: ...
def test_stepwise_invalid_attempt_counts_false_positive_and_false_negative() -> None: ...
def test_debug_f1_handles_duplicate_predictions_one_to_one() -> None: ...
```

- [ ] **Step 2: Implement metrics**

Implement `debug_detection_f1_v1` and `per_target_step_debug_f1_v1` as pure functions that accept normalized prediction and GT records. Return a structured object with:

- precision;
- recall;
- f1;
- true positives;
- false positives;
- false negatives;
- precision denominator;
- recall denominator;
- target-step denominator when applicable;
- `invalid_retained_in_denominator`;
- metric scope;
- invalid/malformed/unparseable counts;
- truncation counts;
- scoreability policy and the effective matching-order source, either `score_descending` or `emission_order`;
- matching rule id;
- IoU threshold;
- category normalization id.

Public report metric names and implementation rules must be explicit:

- report key `debug_detection_f1` uses matching rule `debug_f1_v1`;
- report key `per_target_step_debug_f1` uses matching rule `debug_f1_v1`;
- the versioned function names must not leak as incompatible public metric keys.

- [ ] **Step 3: Write failing report tests**

Add tests proving:

- comparison reports reject mismatched slice, schedule, source adapter, base/tokenizer/processor identity, decode identity, or painted condition identity;
- comparison reports reject mismatched stop/EOS policy;
- comparison reports reject mismatched scoreability policy;
- a gate threshold cannot use an unregistered metric;
- best-only checkpoint report is rejected;
- reports missing any required checkpoint provenance field are rejected, including planned epochs, actual epochs, eval cadence, full checkpoint table, loss trajectory, selected-checkpoint rule, selected-checkpoint reason, and selected-checkpoint provenance;
- correct marks must beat required negatives by the configured margin;
- weak or unavailable negative controls cannot produce a primary mark-dependence pass;
- all tiny-gate hard-fail/pass/gray-zone thresholds from Fixed V1 Constants are enforced exactly;
- all self-prefix catastrophe thresholds from Fixed V1 Constants block larger-run eligibility;
- frozen self-prefix skipped disables comparative language.

Minimum test names:

```python
def test_comparison_report_rejects_decode_identity_mismatch() -> None: ...
def test_comparison_report_rejects_stop_policy_mismatch() -> None: ...
def test_gate_report_rejects_unregistered_primary_metric() -> None: ...
def test_gate_report_rejects_best_only_checkpoint_table() -> None: ...
def test_gate_report_requires_eval_cadence_and_selected_checkpoint_rule() -> None: ...
def test_tiny_gate_enforces_open_spec_thresholds() -> None: ...
def test_mark_dependence_cannot_pass_on_weak_or_missing_negatives() -> None: ...
def test_self_prefix_block_thresholds_block_larger_launch() -> None: ...
def test_self_prefix_without_frozen_baseline_disables_comparative_claims() -> None: ...
```

- [ ] **Step 4: Implement reports**

Implement report builders for:

- frozen versus trained paint-all;
- frozen versus trained teacher-prefix;
- mark-dependence controls;
- bounded self-prefix;
- larger-run eligibility.

Report statuses are `pass`, `gray_zone`, `hard_fail`, `blocked`, or `narrow_proceed` for bounded self-prefix only. Any non-self-prefix `gray_zone` blocks larger training.

Report builders must include machine-readable threshold constants and evidence fields:

- row validity hard-fail `<0.50` and pass `>=0.80`;
- malformed/unparsed hard-fail `>0.50` and pass `<=0.20`;
- primary metric gain pass `>=0.05` absolute for both paint-all and teacher-prefix;
- mark-margin hard-fail `<0.02` and pass `>=0.10`;
- previous-prefix and next-by-order self-prefix block threshold `>0.30`;
- self-prefix schedule-completion contamination block threshold `>25%`;
- planned epochs, actual epochs, eval cadence, all evaluated checkpoints, loss trajectory, full checkpoint metrics, selected-checkpoint rule, selected-checkpoint reason, selected-checkpoint provenance, and scoreability policy.

Reports must also include interpretation safeguards:

- oracle painted GT is not an inference-time detector method;
- paint-all is a `geo_sorted` compact-row transcription probe, not order-free set prediction;
- wrong-description failures are separate from bad-box and coordinate-slot failures;
- self-prefix improvement/degradation language is disabled without paired frozen and trained self-prefix rows.

- [ ] **Step 5: Verify and commit**

Run:

```bash
pytest tests/painted_gt/test_debug_metrics.py tests/painted_gt/test_reports.py -q
pytest tests/eval/test_detection_consumer.py -q
git diff --check -- src/painted_gt src/eval tests/painted_gt tests/eval
git add src/painted_gt src/eval/detection_consumer.py tests/painted_gt tests/eval/test_detection_consumer.py
git commit -m "feat: add painted gt debug metrics and gates"
```

## Task 8: Configs, Dry Runs, And Minimal GPU Smoke

**Files:**
- Create: `configs/coordexp_swift/painted_gt/base.yaml`
- Create: `configs/coordexp_swift/painted_gt/tiny_paint_all_warm_start_dora.yaml`
- Create: `configs/coordexp_swift/painted_gt/tiny_stepwise_teacher_prefix_warm_start_dora.yaml`
- Create: `configs/coordexp_swift/painted_gt/smoke_warm_start_dora_one_step.yaml`
- Create: `tests/painted_gt/test_config_integration.py`
- Modify: `openspec/changes/add-painted-gt-transcription-probe/tasks.md`

- [ ] **Step 1: Write failing config integration tests**

Add tests proving:

- configs resolve to the approved base model, source adapter, repaired embedding payload, and `warm_start_expand_dora`;
- tiny paint-all and teacher-prefix configs use the same `slice_id` after materialization;
- primary decode fields match fixed V1 constants, including effective generation kwargs and stop/EOS policy;
- LR groups explicitly cover language, vision, aligner, and selected-token embeddings;
- configs do not expose dense base training.

Minimum test names:

```python
def test_painted_gt_tiny_configs_resolve_warm_start_inputs() -> None: ...
def test_painted_gt_configs_pin_primary_decode_surface() -> None: ...
def test_painted_gt_configs_require_explicit_lr_groups() -> None: ...
```

- [ ] **Step 2: Implement configs**

Use concise YAML inheritance:

- shared runtime in `base.yaml`;
- tiny paint-all leaf;
- tiny stepwise teacher-prefix leaf;
- one-step smoke leaf.

Set production-scale knobs only where needed. Keep `max_steps` available for smoke, and use epochs for tiny training.

- [ ] **Step 3: Run dry materialization and packing checks**

Run:

```bash
python -m src.painted_gt.cli materialize --config configs/coordexp_swift/painted_gt/smoke_warm_start_dora_one_step.yaml --limit-images 4
python -m src.train --config configs/coordexp_swift/painted_gt/smoke_warm_start_dora_one_step.yaml --dry-run
```

Expected:

- materialization writes manifests and visual audit;
- dry run resolves packing and label-mask receipts;
- no GPU training begins during dry run.

- [ ] **Step 4: Run minimal GPU smoke**

Run on available GPUs without interrupting other jobs:

```bash
python -m src.train --config configs/coordexp_swift/painted_gt/smoke_warm_start_dora_one_step.yaml
```

Expected:

- model setup completes;
- warm-start report is written;
- selected-token embedding payload is loaded;
- one forward/backward path completes;
- optimizer grouping receipt exists;
- checkpoint-final or configured smoke checkpoint exists.

- [ ] **Step 5: Update OpenSpec tasks and commit**

Check off only tasks backed by evidence. Then run:

```bash
pytest tests/painted_gt/test_config_integration.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check -- configs/coordexp_swift/painted_gt tests/painted_gt openspec/changes/add-painted-gt-transcription-probe
git add configs/coordexp_swift/painted_gt tests/painted_gt openspec/changes/add-painted-gt-transcription-probe
git commit -m "feat: add painted gt smoke configs"
```

## Task 9: Tiny Gate Execution And Larger-Run Decision

**Files:**
- Modify: `research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md`
- Modify: `openspec/changes/add-painted-gt-transcription-probe/tasks.md`
- Create runtime artifacts under `outputs/` only

- [ ] **Step 1: Verify accepted unpainted baseline identity**

Run a JSON check against the accepted metric file:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z/eval_coco_fixed_gt_scale/metrics.json")
data = json.loads(path.read_text())
assert data["row_count"] == 200
assert data["metric_family"] == "coordexp_swift_detection_coco_bbox_v1"
assert data["mAP"] >= 0.40
print("accepted baseline ok", data["mAP"])
PY
```

Also verify accepted artifact row binding:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z")
raw = root / "gt_vs_pred.jsonl"
prov = root / "gt_vs_pred_scored.jsonl.provenance.json"
image_plan = root / "image_plan.jsonl"
resolved = root / "configs/resolved.json"
assert hashlib.sha256(raw.read_bytes()).hexdigest() == "065ac3dd3a3f8093de97570cfa20bcee3e449802066dec8b1fc19d0c0d64d9d5"
assert json.loads(prov.read_text())["row_binding"]["row_ids_sha256"] == "217d90415e9ba6f471b16803bc3ca066382949a8feceb00f34b733ae3a45e2e9"
assert hashlib.sha256(image_plan.read_bytes()).hexdigest() == "3b4bdb0be78f2cb21164970ae7e0d84e21252c4cbe9d4f5de961f4638fb41398"
assert json.loads(resolved.read_text())["config"]["data"]["input_jsonl"] == "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
rows = [json.loads(line) for line in raw.open()]
first_ids = [str(row.get("id") or row.get("row_id") or row.get("image_id") or row.get("source_id")) for row in rows[:32]]
assert first_ids[0] == "coco2017_val_000000000139"
assert hashlib.sha256("\n".join(first_ids).encode()).hexdigest() == "81cd5e3bf6218d47b9e3f895204522dede0bdc9ac8dd8bd11a432f6bcb6cbbb9"
plans = [json.loads(line) for line in image_plan.open()]
assert all(row["status"] == "ok" and row["do_resize"] is False for row in plans[:32])
print("accepted val200 row binding ok")
PY
```

- [x] **Step 2: Run expanded-seed sanity probe**

Run source LLM-only seed and expanded all-tower seed on the first 32 accepted val200 rows with identical effective decode identity and stop/EOS policy. The sample is bound by accepted `gt_vs_pred.jsonl` row order, the resolved input JSONL path and SHA256 `9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4`, `row_binding.row_ids_sha256`, `image_plan.jsonl` no-resize status, first row id `coco2017_val_000000000139`, and first-32 row-id hash `81cd5e3bf6218d47b9e3f895204522dede0bdc9ac8dd8bd11a432f6bcb6cbbb9`. Block if thresholds are exceeded. Write a report under this worktree's `outputs/painted_gt/sanity/`.

- [ ] **Step 3: Run frozen painted baselines**

Run frozen paint-all and teacher-prefix inference on the fixed 256-image slice before training. Write primary metric and row-validity reports.

- [ ] **Step 4: Run tiny paint-all training**

Run the predeclared 2-epoch tiny paint-all training. Evaluate all configured checkpoints under primary free raw decode. Extend to at most 8 total epochs only if reports show improving but not passing behavior.

- [ ] **Step 5: Run tiny teacher-prefix training**

Run the predeclared 2-epoch tiny teacher-prefix training. Evaluate all configured checkpoints under primary free raw decode. Extend to at most 8 total epochs only if reports show improving but not passing behavior.

- [ ] **Step 6: Run required counterfactual panels**

Run no-mark, wrong-mark, shuffled/offset, offset-all, and false-mark controls with matching slice, source, trained checkpoint, and decode identity.

The counterfactual report must include hard-negative coverage counts, weak-negative counts, `no_hard_negative_available` counts, off-frame/overlap/size/style validity counters, and scoped-example counts. Weak negatives must not create primary mark-margin credit.

- [ ] **Step 7: Run bounded self-prefix diagnostic**

Run `64` images or `512` object steps, whichever is smaller. Preserve raw prefixes. Record row validity, malformed/unparsed rate, correct-mark advantage, previous-prefix errors, next-by-order errors, mark-insensitive errors, prefix-contamination counters, schedule-completion rate, and block/pass/narrow-proceed status.

- [ ] **Step 8: Produce final gate report**

The report must include:

- all evaluated checkpoints;
- planned epochs and actual epochs;
- eval cadence;
- full checkpoint metrics table;
- loss trajectory;
- selected-checkpoint rule and selected-checkpoint reason;
- selected-checkpoint provenance;
- primary metrics;
- public metric names and matching-rule ids;
- precision, recall, and target-step denominators;
- scoreability policy and matching-order source;
- secondary official mAP/mRecall when available;
- row validity;
- invalid/malformed/unparseable rates;
- mark margins;
- hard-negative coverage and weak-negative scoped-example counts;
- visual-audit status;
- expanded-seed sanity status;
- self-prefix status;
- stop/EOS policy id and effective decode hash;
- failure-axis tables for wrong-description, bad-box, coordinate-slot, duplicate, previous-prefix, next-by-order, mark-insensitive, and prefix-contamination errors;
- interpretation caveats that oracle painted GT is not an inference-time detector, paint-all is not order-free set prediction, and category-text failures are not pure coordinate failures;
- pass, gray-zone, hard-fail, blocked, or narrow-proceed classification.

- [ ] **Step 9: Review and decide launch**

Run isolated review-convergence on the final gate report. If all non-self-prefix prerequisite gates pass and self-prefix passes or is an allowed scoped `narrow_proceed`, prepare and launch the larger two-epoch painted-input training run in tmux. If any hard gate fails, stop with a blocked report and do not launch larger training.

- [ ] **Step 10: Commit docs and task ledger**

Run:

```bash
openspec validate add-painted-gt-transcription-probe --strict
git diff --check -- research/ideas/qwen3-vl-painted-gt-transcription-probe openspec/changes/add-painted-gt-transcription-probe
git add research/ideas/qwen3-vl-painted-gt-transcription-probe openspec/changes/add-painted-gt-transcription-probe
git commit -m "docs: record painted gt gate outcome"
```

## Review Requirements

After this plan is drafted, run four independent read-only review lanes:

- contract/spec auditor: OpenSpec coverage, requirement-to-task mapping, launch-gate semantics, artifact contracts;
- upstream relation tracer: Qwen3-VL image processing, PEFT/DoRA key surfaces, selected-token embedding identity, generation kwargs;
- implementation mapper: module boundaries, file ownership, testability, likely conflict surfaces;
- model-behavior auditor: metric validity, counterfactual strength, tiny-gate interpretation, self-prefix failure modes.

For each review:

- classify every finding as P0, P1, P2, wrong, duplicate, or non-blocking;
- patch all accepted P0/P1 before source implementation;
- patch P2 only when cheap and aligned with accuracy, efficiency, and simplicity;
- record the review and resolutions in this file or `research/ideas/qwen3-vl-painted-gt-transcription-probe/review-log.md`;
- do not start source implementation until the plan has no unresolved P0/P1 findings.

## Final Verification Commands

Run before claiming the planning gate is ready:

```bash
openspec validate add-painted-gt-transcription-probe --strict
python - <<'PY'
from pathlib import Path
plan = Path("docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md")
text = plan.read_text()
for bad in ["TO" + "DO", "TB" + "D"]:
    assert bad not in text, bad
assert text.count(chr(96) * 3) % 2 == 0
print("plan hygiene: ok")
PY
git diff --check -- docs/superpowers/plans/2026-07-04-qwen3-vl-painted-gt-transcription-probe.md
```

Run before larger two-epoch launch:

```bash
pytest tests/adapters/test_warm_start_expand_dora.py tests/qwen/test_painted_gt_embedding_payload.py -q
pytest tests/painted_gt -q
pytest tests/inference/test_backend_trace.py tests/inference/test_pipeline.py -q
pytest tests/eval/test_detection_consumer.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

## Stop Conditions

Stop and report instead of improvising if:

- source adapter tensor counts or all-tower target counts disagree with fixed OpenSpec expectations;
- repaired embedding payload identity cannot be validated;
- expanded-seed sanity fails beyond the predeclared thresholds;
- visual audit detects geometry or Qwen no-resize mismatch;
- primary free raw decode cannot be made stable enough for denominator-bearing metrics;
- tiny paint-all or teacher-prefix gates hard-fail;
- a fix would require dense base-model fine-tuning, merged full-model export as the primary path, replacing the research question, deleting major artifacts, or interrupting unrelated GPU jobs.
