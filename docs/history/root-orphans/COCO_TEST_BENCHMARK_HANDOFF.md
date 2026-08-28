# COCO Test Benchmark Handoff

## Objective

Benchmark the current CoordExp model on the **official COCO 2017 test-dev / test2017** benchmark, generate a valid submission artifact, submit it to the official evaluation server, and report the resulting metrics.

This is a separate execution task. Treat this document as the working brief.

---

## Mission

Determine the model's official COCO-test performance in a reproducible, paper-ready way.

Deliverables must include:

- a reproducible infer config for the chosen model on COCO test,
- a validated submission JSON in official COCO detection format,
- the official test-dev metrics after submission,
- and a short progress note documenting what was run, what was submitted, and what score came back.

If official submission is blocked by credentials, account access, missing images, or network issues, stop at a **ready-to-submit artifact** plus exact handoff instructions.

---

## Important Context

This repo is **config-first** and already has a stable infer/eval workflow for COCO/LVIS-style JSONL datasets.

Relevant canonical docs:

- `docs/PROJECT_CONTEXT.md`
- `docs/eval/WORKFLOW.md`
- `docs/eval/CONTRACT.md`
- `docs/data/CONTRACT.md`
- `docs/data/PREPARATION.md`
- `docs/ARTIFACTS.md`

Relevant code / scripts:

- `scripts/run_infer.py`
- `scripts/evaluate_detection.py`
- `configs/infer/pipeline.yaml`
- `configs/infer/ablation/coco80_desc_first.yaml`
- `configs/bench/pure_ce_2b_1344_coco_val_1024_limit200.yaml`
- `public_data/run.sh`
- `public_data/datasets/coco.sh`
- `public_data/scripts/download_coco2017.py`
- `public_data/scripts/convert_coco2017_instances.py`

Known current state:

- The repo already has COCO `train2017` / `val2017` data artifacts under `public_data/coco/...`.
- The repo does **not** obviously have a prebuilt `test2017` CoordJSONL yet.
- `public_data/datasets/coco.sh` currently converts **train + val**; test support likely needs verification or a minimal extension.
- Local COCO evaluation requires GT, but **official COCO test-dev has no public labels**, so local `mAP` on test cannot be computed in the normal way.
- `docs/eval/CONTRACT.md` says COCO evaluation emits `coco_preds.json`; verify whether the existing evaluator/export path can be reused for official submission, or whether a small dedicated exporter is needed for test-only inference.

Repo guardrails:

- Do **not** add ad hoc CLI flags if a config-driven path is possible.
- Keep Qwen3-VL chat-template compatibility.
- Do **not** edit upstream HF model files like `modeling_qwen3_vl.py`.
- For any `*.py` exploration/editing, **Serena MCP is mandatory**.
- Use `conda run -n ms python ...` for Python commands.

---

## Main Question To Answer

What is the official COCO test-dev performance of the selected CoordExp model, and how does it compare to the local COCO-val score under the closest matching infer recipe?

---

## Required Outputs

Produce all of the following, or explain exactly which blocker prevented them:

1. A chosen model path and the rationale for selecting it.
2. A reproducible config for COCO test inference.
3. A COCO test submission JSON in official detection format.
4. A local COCO val sanity run using the same or closest possible infer recipe.
5. Official submission result(s): AP / AP50 / AP75 / APs / APm / APl.
6. A short progress note under `progress/benchmarks/` or `progress/diagnostics/` with:
   - model path
   - config path
   - dataset artifact path(s)
   - submission artifact path
   - official result
   - caveats / blockers / deltas vs local val

---

## Recommended Plan

### Phase 1: Verify the current benchmark path

Read these first:

- `docs/eval/WORKFLOW.md`
- `docs/eval/CONTRACT.md`
- `docs/data/PREPARATION.md`
- `configs/infer/ablation/coco80_desc_first.yaml`
- `public_data/datasets/coco.sh`

Answer these concrete questions before changing anything:

- What model checkpoint should be benchmarked?
- Does the current infer pipeline require inline GT objects, or can it run with test-only image metadata?
- Can the existing pipeline already produce a valid `coco_preds.json` for submission, or only during GT-backed evaluation?
- Does the current COCO public-data plugin support `test2017`, or only `train` / `val`?

Verification:

- Point to exact config/script handles and note any missing link.

### Phase 2: Build the smallest viable COCO-test path

Preferred option:

- Reuse the existing infer pipeline with a **test-only JSONL** that follows `docs/data/CONTRACT.md`.

If no existing test JSONL exists:

- generate one in a repo-consistent way from official COCO `test2017` image metadata,
- keep it minimal and contract-compliant,
- prefer extending the COCO public-data prep path rather than writing a one-off irreproducible script.

Important:

- official COCO test images have no public object labels,
- so the JSONL likely needs `images`, `width`, `height`, and an empty `objects` list or another evaluator-compatible minimal structure,
- but do not assume this blindly; verify against the infer pipeline contract before locking it in.

Verification:

- run a tiny smoke subset of the new test JSONL,
- verify `scripts/run_infer.py` can consume it,
- verify the output artifact parses cleanly.

### Phase 3: Create a val-matched sanity benchmark

Before full test submission:

- run a local COCO `val2017` benchmark using the same model and as-close-as-possible infer settings,
- use or clone a config near:
  - `configs/infer/ablation/coco80_desc_first.yaml`
  - or `configs/bench/pure_ce_2b_1344_coco_val_1024_limit200.yaml`

This is the sanity anchor for:

- prompt variant,
- object field order,
- resolution,
- generation settings,
- backend choice,
- score/export conversion path.

Verification:

- produce `summary.json`
- produce `metrics.json`
- if applicable, produce / inspect `coco_preds.json`
- confirm the run directory contains the expected artifacts from `docs/ARTIFACTS.md`

### Phase 4: Generate official submission artifact

Goal:

- produce a COCO detection submission JSON for **test-dev** in the exact format the official server expects.

Recommended route:

- if the current evaluator/export path already emits a correct `coco_preds.json`, reuse it;
- otherwise add the smallest config-first export path necessary.

Do not ship an unverified exporter.

Verification:

- compare the test export schema to a known-good val export, if available;
- confirm `image_id`, `category_id`, `bbox`, and `score` fields are correct;
- confirm bbox format matches COCO expectations.

### Phase 5: Submit and record the result

Because the official submission path can change over time:

- verify the current official COCO test-dev submission workflow on the web before submitting,
- confirm account / token / format expectations,
- submit the JSON,
- record the submission URL / ID / timestamp / returned metrics.

If submission is blocked by credentials:

- stop with the fully prepared submission artifact,
- document exactly what remains to be done manually.

Verification:

- preserve the official response artifact or copy the metrics into the progress note.

---

## Decision Points

### Decision 1: How to create COCO test input

Option A: Reuse an existing repo-native COCO prep path and extend it minimally for `test2017`.

- Pros:
  - reproducible
  - consistent with repo data contract
  - easiest to maintain
- Cons:
  - may require small code changes

Option B: Create a one-off test JSONL generator under `temp/`.

- Pros:
  - faster if the pipeline is very close already
- Cons:
  - less reproducible
  - easier to drift from contract

Recommendation:

- Prefer **Option A** unless the required code change is disproportionately large.

Verification:

- whichever option is chosen, validate the produced JSONL against `docs/data/CONTRACT.md` and a small infer smoke.

### Decision 2: How to produce submission JSON

Option A: Reuse the current evaluator/export path that emits `coco_preds.json`.

- Pros:
  - lower risk
  - uses repo-native artifact path
- Cons:
  - may require GT-backed evaluation assumptions to be bypassed safely

Option B: Add a tiny dedicated exporter from inference artifact to official COCO JSON.

- Pros:
  - explicit
  - easier to use on label-free test data
- Cons:
  - must be validated carefully

Recommendation:

- Prefer **Option A** if it works cleanly on val and test; otherwise implement **Option B** with a strict val-side equivalence check.

Verification:

- export on val with both routes and compare outputs structurally and numerically where possible.

### Decision 3: What to do if official submission cannot be completed

Option A: Stop with a ready-to-submit artifact and explicit instructions.

- Pros:
  - still highly useful
  - no guessing
- Cons:
  - no official score yet

Option B: Keep pushing through unofficial workarounds.

- Pros:
  - might get a result if the blockage is minor
- Cons:
  - higher risk of wasted time or irreproducible hacks

Recommendation:

- Prefer **Option A** once the technical pipeline is complete and the remaining blocker is external.

Verification:

- ensure the artifact is complete and the manual submission steps are exact.

---

## Constraints

- Stay config-first.
- Keep changes minimal and benchmark-scoped.
- Do not disturb existing train / val workflows.
- If a new config is needed, place it under `configs/bench/` or another appropriate infer/eval location.
- If a new progress note is added, index it in `progress/index.yaml`.
- If code changes are needed:
  - use Serena MCP for all `*.py` exploration/editing,
  - add the smallest possible verification path,
  - avoid generic refactors.

---

## Suggested Concrete Starting Point

Start from:

- `configs/infer/ablation/coco80_desc_first.yaml`

because it already matches the COCO-80 prompt contract:

- `prompt_variant: coco_80`
- `object_field_order: desc_first`
- `metrics: both`
- `use_segm: false`

Then decide whether to:

- clone this into a new `configs/bench/*coco_test*.yaml`, or
- create a test-specific variant under `configs/infer/`.

For local sanity validation, use a COCO val config close to:

- `configs/infer/ablation/coco80_desc_first.yaml`
- or `configs/bench/pure_ce_2b_1344_coco_val_1024_limit200.yaml`

depending on the selected model family and resolution.

---

## Success Criteria

This task is successful if:

- the selected model is run on official COCO test images,
- a valid official submission artifact is produced,
- the artifact is submitted successfully or is ready-to-submit with only external access remaining,
- the official result is recorded,
- and the exact infer/export path is reproducible from repo artifacts.

This task is **not** complete if it ends only with:

- a local val score,
- an unsubmitted prediction file with unknown schema,
- or a non-reproducible ad hoc export path.

---

## Final Deliverable Checklist

- [ ] chosen checkpoint / merged model path
- [ ] COCO test dataset artifact path
- [ ] infer config path
- [ ] run directory
- [ ] submission JSON path
- [ ] local val sanity metrics
- [ ] official test-dev result or explicit external blocker
- [ ] progress note with reproducibility details

