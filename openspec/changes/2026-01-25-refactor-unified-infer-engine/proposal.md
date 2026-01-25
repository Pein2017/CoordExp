# Change: Refactor Unified Inference Engine (Dump + Eval + Vis) with HF + vLLM Backends

## Why
CoordExp inference/evaluation/visualization is currently split across multiple scripts (`scripts/run_infer.py`, `scripts/evaluate_detection.py`, `vis_tools/vis_coordexp.py`, plus shell wrappers). This fragmentation causes drift in JSONL contracts and makes it harder to run reproducible, paper-ready inference pipelines.

## What Changes
- Introduce a single *unified infer pipeline entrypoint* that can run stages: `infer` (generation + prediction dump), `eval` (metrics), `vis` (qualitative overlays) using one YAML configuration.
- Standardize on a single stable artifact contract: a combined JSONL (`gt_vs_pred.jsonl`) embeds GT (`gt`) and predictions (`pred`) per sample and is the only input to evaluation and visualization.
- Keep existing tools working during a transition period by turning legacy scripts into thin wrappers.
- Support a vLLM-powered generation backend behind a config switch, while keeping the HF/Transformers backend as the default.

## Non-Goals
- No changes to training (`src/sft.py`) behavior.
- No changes to upstream model code (e.g., `modeling_qwen3_vl.py` remains off-limits).
- No new dataset formats; use existing `docs/DATA_JSONL_CONTRACT.md` and the existing inference JSONL dump schema.

## Scope / Affected Areas
- Specs impacted:
  - `openspec/specs/inference-engine/spec.md`
  - `openspec/specs/inference-pipeline/spec.md`
  - `openspec/specs/detection-evaluator/spec.md`
- Code to be consolidated (behavioral reference only; implementation happens after approval):
  - `src/infer/engine.py` (current HF inference + dump)
  - `src/eval/detection.py` + `scripts/evaluate_detection.py` (metrics)
  - `vis_tools/vis_coordexp.py` + wrappers (visualization)
  - `scripts/run_infer_eval.sh` (end-to-end orchestration wrapper)
  - `scripts/report_rollout_stability.py` (summary/metrics reporting)
  - `src/callbacks/detection_eval.py` (training-time eval hook consuming prediction artifacts)
  - `scripts/run_vis.sh`, `scripts/vis.sh`, `scripts/run_vis_coord.sh` (visualization wrappers; some are drifted today)

## Impact
- User-facing UX: moves from fragmented CLIs/shell wrappers to one YAML-driven pipeline interface.
- Artifact contract: establishes `output/infer/<run_name>/gt_vs_pred.jsonl` as the stable interface between infer/eval/vis stages.
- Tooling drift reduction: wrappers become thin presets; core orchestration lives in one place.

Entrypoint choice:
- The unified pipeline runner will be implemented by evolving `scripts/run_infer.py` to accept `--config` (YAML-first) and optional stage toggles.

## Compatibility / Backend Behavior
- Contract drift: evaluation currently supports multiple schemas; the proposal makes the inference dump schema the single stable artifact consumed by eval/vis.
- Backend behavior differences: HF and vLLM backends may produce different raw text outputs; exact cross-backend reproducibility is not required. Runs remain reproducible by recording the resolved backend + generation config in the run artifacts.

## Compatibility / Deprecation
- Prediction artifact naming:
  - Canonical file name is `gt_vs_pred.jsonl` under `output/infer/<run_name>/`.
  - During transition, tools MAY continue to accept `pred.jsonl` as an alias, but new pipeline runs SHOULD emit `gt_vs_pred.jsonl`.
- Summary naming:
  - Canonical summary is `summary.json` under the run directory.
  - During transition, tools MAY auto-detect legacy names (e.g., sibling `.summary.json`) for backward compatibility.
- Legacy wrappers:
  - Existing scripts SHOULD continue working by calling the unified pipeline with an equivalent resolved config.
  - Drifted wrappers (e.g., `scripts/run_vis_coord.sh`) SHOULD be fixed or removed with a clear replacement command.

## Validation
- Parity recipe (small subset):
  - Run the legacy workflow (`scripts/run_infer_eval.sh`) with `LIMIT=10` and record `gt_vs_pred.jsonl`/metrics outputs.
  - Run the unified pipeline on the same checkpoint + dataset + generation config and compare:
    - JSONL schema (canonical keys, pixel-space points),
    - evaluation metrics JSON,
    - visualization outputs (count + sanity).
- Smoke tests (lightweight):
  - "vis-only from existing gt_vs_pred.jsonl" (no model load) produces overlays.
  - "eval-only from existing gt_vs_pred.jsonl" (no model load) produces `eval/metrics.json`.

## Rollout Plan (High Level)
1) Add unified pipeline config + entrypoint while preserving old commands.
2) Convert old scripts to wrappers (or deprecate clearly) after parity is validated.
3) Add/enable vLLM backend via config and validate it against the same artifact contract.
