#!/usr/bin/env bash
set -euo pipefail
cd /data/CoordExp/.worktrees/research-probes
STAGE1="/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-successful-row-mechanism/stage1"
# GPU commands are prepared only; lead authorization is required before execution.
# CPU seam check (no model load):
python -m probes.training_set_completion.repetition_history_runtime --cpu-check --output-root "$STAGE1/qualification/runtime"
python -m probes.training_set_completion.repetition_history_runtime --panel "$STAGE1/panels/309264-SS.json" --case 309264 --mode prefix --output-root "$STAGE1/runtime/SS"
python -m probes.training_set_completion.repetition_history_runtime --panel "$STAGE1/panels/309264-FF.json" --case 309264 --mode prefix --output-root "$STAGE1/runtime/FF"
python -m probes.training_set_completion.repetition_history_runtime --panel "$STAGE1/panels/309264-SX_FY.json" --case 309264 --mode prefix --output-root "$STAGE1/runtime/SX_FY"
python -m probes.training_set_completion.repetition_history_runtime --panel "$STAGE1/panels/309264-FX_SY.json" --case 309264 --mode prefix --output-root "$STAGE1/runtime/FX_SY"
python -m probes.training_set_completion.repetition_history_scores --panel "$STAGE1/score-panel.json" --image 309264 --out "$STAGE1/score-runtime"
python -m probes.training_set_completion.repetition_history_score_check "$STAGE1/score-runtime"
