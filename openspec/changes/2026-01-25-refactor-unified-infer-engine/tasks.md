# Tasks: Refactor Unified Inference Engine

## 1. Proposal Validation Prep
- [ ] 1.1 Confirm current behavior/entrypoints to preserve: `scripts/run_infer.py`, `scripts/evaluate_detection.py`, `vis_tools/vis_coordexp.py`
- [ ] 1.2 Confirm evaluation contract: evaluation consumes inference dump JSONL that embeds GT; separate GT/pred evaluation is NOT exposed

## 2. Unified Pipeline (Config + Entry)
- [ ] 2.1 Add YAML config template under `configs/infer/` for a unified run (infer+eval+vis toggles) with explicit `run.name`
- [ ] 2.2 Define deterministic artifact layout:
  - Canonical run directory: `output/infer/<run.name>/`
  - Canonical artifacts:
    - `gt_vs_pred.jsonl`
    - `summary.json`
    - `eval/` (metrics + reports)
    - `vis/` (qualitative overlays)
- [ ] 2.3 Add a single pipeline entrypoint module (minimal CLI) that reads YAML and runs requested stages
- [ ] 2.4 Define and document the canonical prediction JSONL schema (single source of truth; embeds GT)

## 3. Stage Implementations (Reuse Existing Code)
- [ ] 3.1 Inference stage: wrap existing HF inference dump logic as the default backend
- [ ] 3.2 Eval stage: call detection evaluator on the prediction JSONL, producing `metrics.json` and optional overlays
- [ ] 3.3 Vis stage: render overlays from the prediction JSONL without requiring inference

## 4. vLLM Backend (Config Switch)
- [ ] 4.1 Define backend interface and add vLLM backend integration behind `infer.backend.type: vllm`
- [ ] 4.2 Add minimal validation checklist (schema consistency, determinism note recorded, metric sanity on a small subset)

## 5. Transition / Deprecation
- [ ] 5.1 Evolve `scripts/run_infer.py` into the unified YAML-first runner (`--config`), keeping legacy flags during the transition period
- [ ] 5.2 Convert legacy scripts (`scripts/evaluate_detection.py`, `scripts/run_vis.sh`) into thin wrappers OR document deprecation paths
- [ ] 5.3 Remove or fix drifted wrappers (e.g., `scripts/run_vis_coord.sh`) with clear guidance

## 6. Validation
- [ ] 6.1 Add lightweight smoke tests: parse+standardize determinism; pipeline "vis-only" run on a fixture `gt_vs_pred.jsonl`
- [ ] 6.2 Add a comparison recipe to verify outputs vs current scripts on a small sample (limit=10)
