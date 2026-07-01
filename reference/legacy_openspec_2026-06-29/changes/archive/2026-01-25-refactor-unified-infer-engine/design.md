# Design: Unified Inference Pipeline (Dump + Eval + Vis)

## Context
Today, CoordExp has:
- A centralized inference dumper (`src/infer/engine.py`) invoked by `scripts/run_infer.py`.
- A detection evaluator (`src/eval/detection.py`) invoked by `scripts/evaluate_detection.py`.
- A visualization renderer (`vis_tools/vis_coordexp.py`) invoked by shell wrappers.

The desired user experience is a single, minimal interface that runs any combination of:
- prediction dumps
- visualization
- metric computation

…with both an HF/Transformers backend and a vLLM-powered backend.

## Goals
- One orchestration layer drives dump/eval/vis from a single YAML config.
- One stable artifact schema (`gt_vs_pred.jsonl`) is the interface between stages.
- Reproducible outputs: deterministic seeding, stable artifact paths, stable ordering.
- Keep compatibility with Qwen3-VL chat template behavior.

## Key Design Decisions

### 1) Artifact-First Pipeline
The pipeline centers around a single per-sample JSONL dump produced by the inference stage. Downstream stages (eval/vis) MUST consume this artifact without re-running the model.

Rationale:
- Enables checkpoint ablations: run inference once, then run eval/vis multiple times.
- Prevents drift: a single schema becomes the contract.

### 2) Shared Parsing + Standardization
- Parsing model text into structured objects uses the shared parser.
- Coordinate standardization (norm1000/pixel → pixel-space, validation, clamping) uses the shared coordinate processing module.

Rationale:
- Avoid duplicated coord decode logic and subtle inconsistencies.

### 3) Generation Backend Interface
Define a small backend interface:
- Input: (image, messages/prompt, generation params)
- Output: `pred_text` (raw model output string)

Implementations:
- HF backend (default): Transformers generation.
- vLLM backend: enabled via config.

Rationale:
- Keeps the rest of the pipeline backend-agnostic.

### 4) Configuration (YAML-first)
The unified pipeline is driven by YAML (new config under `configs/`), aligning with CoordExp conventions.

### 5) Compatibility / Transition
- Keep existing scripts working by converting them to wrappers around the unified pipeline.
- Preserve existing prediction JSONL schema (or tighten it) so current visualizer/evaluator remain usable.

## Clarifications (Resolved)

1) **Evaluation input contract**: evaluation is based on the inference dump JSONL that *embeds GT* (`gt` per sample). Separate `gt_jsonl` + `pred_jsonl` evaluation is out of scope for the unified pipeline and should not be exposed.
2) **Default artifact layout**:
   - Canonical run directory: `output/infer/<run_name>/`
   - Canonical artifacts (paths relative to `run_dir` unless user overrides):
     - `gt_vs_pred.jsonl`
     - `summary.json`
     - `eval/metrics.json`, `eval/per_class.csv`, `eval/per_image.json`, etc.
     - `vis/vis_0000.png` ...
3) **vLLM backend**: `vllm` is a supported generation backend (not a speculative/optional compatibility experiment). It must produce outputs that conform to the same downstream artifact schema and stage interface as the HF backend.

## Normative YAML skeleton
The unified pipeline SHALL be configured via YAML. The example below is illustrative; exact key names MAY differ if documented consistently.

```yaml
# configs/infer/pipeline.yaml
run:
  name: lvis_val_ckpt1632
  output_dir: output/infer

stages:
  infer: true
  eval: true
  vis: true

infer:
  gt_jsonl: public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/val.poly_prefer_semantic_cap20.max60.coord.jsonl
  model_checkpoint: output/.../ckpt
  backend:
    type: hf   # hf|vllm
  mode: coord  # coord|text|auto
  pred_coord_mode: auto  # auto|norm1000|pixel
  generation:
    temperature: 0.01
    top_p: 0.95
    max_new_tokens: 1024
    repetition_penalty: 1.05
    seed: 42
  device: cuda:0
  limit: 0

# If omitted, these are derived deterministically from run.output_dir + run.name.
artifacts:
  run_dir: output/infer/lvis_val_ckpt1632
  gt_vs_pred_jsonl: output/infer/lvis_val_ckpt1632/gt_vs_pred.jsonl
  summary_json: output/infer/lvis_val_ckpt1632/summary.json

eval:
  output_dir: output/infer/lvis_val_ckpt1632/eval
  metrics: both
  unknown_policy: semantic
  overlay: false
  overlay_k: 12

vis:
  output_dir: output/infer/lvis_val_ckpt1632/vis
  limit: 20
```

## Notes
- The single run directory keeps artifacts co-located for paper-ready runs while still separating concerns via subdirectories (`eval/`, `vis/`).
- The pipeline YAML is single-file only (no `extends`/`inherit`) and does not rely on variable interpolation.
- The pipeline should default `ROOT_IMAGE_DIR` to the JSONL parent directory for consistent relative image resolution.
- The unified pipeline runner is `scripts/run_infer.py` (YAML-first via `--config`), with legacy CLI flags retained temporarily for compatibility.
