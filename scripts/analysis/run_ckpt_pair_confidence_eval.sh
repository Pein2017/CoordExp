#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[ERROR] scripts/analysis/run_ckpt_pair_confidence_eval.sh accepts environment variables only (no positional args)." >&2
  echo "[ERROR] Example: CKPT_A=output/.../ckpt-300 CKPT_B=output/.../ckpt-2400 GT_JSONL=public_data/... LIMIT=100 BATCH_SIZE=64 bash scripts/analysis/run_ckpt_pair_confidence_eval.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/backbone.sh"

# Required (with analysis-oriented defaults)
CKPT_A="${ckpt_a:-${CKPT_A:-output/stage2_ab/coco_bbox_max60/analysis/ckpt-300}}"
CKPT_B="${ckpt_b:-${CKPT_B:-output/stage2_ab/coco_bbox_max60/analysis/ckpt-2400}}"
GT_JSONL="${gt_jsonl:-${GT_JSONL:-}}"

# Run layout / devices
OUTPUT_BASE_DIR="${output_base_dir:-${OUTPUT_BASE_DIR:-output/stage2_ab/coco_bbox_max60/analysis/pair_eval}}"
RUN_A_NAME="${run_a_name:-${RUN_A_NAME:-ckpt_300}}"
RUN_B_NAME="${run_b_name:-${RUN_B_NAME:-ckpt_2400}}"
DEVICE_A="${device_a:-${DEVICE_A:-cuda:0}}"
DEVICE_B="${device_b:-${DEVICE_B:-cuda:1}}"

# Inference controls
MODE="${mode:-${MODE:-auto}}"                                # coord | text | auto
PRED_COORD_MODE="${pred_coord_mode:-${PRED_COORD_MODE:-auto}}"  # auto | norm1000 | pixel
PROMPT_VARIANT="${prompt_variant:-${PROMPT_VARIANT:-default}}"  # default | coco_80
OBJECT_FIELD_ORDER="${object_field_order:-${OBJECT_FIELD_ORDER:-geometry_first}}"  # desc_first | geometry_first
LIMIT="${limit:-${LIMIT:-100}}"
BATCH_SIZE="${batch_size:-${BATCH_SIZE:-64}}"
TEMP="${temp:-${TEMP:-0.01}}"
TOPP="${topp:-${TOPP:-0.95}}"
MAXTOK="${maxtok:-${MAXTOK:-1024}}"
REPPEN="${reppen:-${REPPEN:-1.05}}"
SEED="${seed:-${SEED:-}}"

# Evaluation controls
SEMANTIC_MODEL="${semantic_model:-${SEMANTIC_MODEL:-sentence-transformers/all-MiniLM-L6-v2}}"
SEMANTIC_THRESHOLD="${semantic_threshold:-${SEMANTIC_THRESHOLD:-0.6}}"
SEMANTIC_DEVICE="${semantic_device:-${SEMANTIC_DEVICE:-auto}}"
SEMANTIC_BATCH_SIZE="${semantic_batch_size:-${SEMANTIC_BATCH_SIZE:-64}}"
NUM_WORKERS="${num_workers:-${NUM_WORKERS:-0}}"
F1ISH_IOU_THR_0="${f1ish_iou_0:-${F1ISH_IOU_0:-0.30}}"
F1ISH_IOU_THR_1="${f1ish_iou_1:-${F1ISH_IOU_1:-0.50}}"

ensure_required "CKPT_A" "$CKPT_A"
ensure_required "CKPT_B" "$CKPT_B"
ensure_required "GT_JSONL" "$GT_JSONL"

mkdir -p "$OUTPUT_BASE_DIR"
cd "$REPO_ROOT"

echo "Checkpoint pair confidence-eval workflow"
echo "  CKPT_A:            $CKPT_A"
echo "  CKPT_B:            $CKPT_B"
echo "  GT_JSONL:          $GT_JSONL"
echo "  OUTPUT_BASE_DIR:   $OUTPUT_BASE_DIR"
echo "  LIMIT:             $LIMIT"
echo "  BATCH_SIZE:        $BATCH_SIZE"
echo "  DEVICE_A / B:      $DEVICE_A / $DEVICE_B"

write_infer_cfg() {
  local cfg_path="$1"
  local run_name="$2"
  local ckpt="$3"
  local device="$4"
  local seed_yaml="null"
  if [[ -n "$SEED" ]]; then
    seed_yaml="$SEED"
  fi

  cat > "$cfg_path" <<EOF
run:
  name: $run_name
  output_dir: $OUTPUT_BASE_DIR

stages:
  infer: true
  eval: false
  vis: false

infer:
  gt_jsonl: $GT_JSONL
  model_checkpoint: $ckpt
  mode: $MODE
  prompt_variant: $PROMPT_VARIANT
  object_field_order: $OBJECT_FIELD_ORDER
  pred_coord_mode: $PRED_COORD_MODE
  backend:
    type: hf
  generation:
    temperature: $TEMP
    top_p: $TOPP
    max_new_tokens: $MAXTOK
    repetition_penalty: $REPPEN
    batch_size: $BATCH_SIZE
    seed: $seed_yaml
  device: $device
  limit: $LIMIT
EOF
}

write_conf_cfg() {
  local cfg_path="$1"
  local run_dir="$2"
  cat > "$cfg_path" <<EOF
artifacts:
  run_dir: $run_dir
EOF
}

write_eval_cfg() {
  local cfg_path="$1"
  local pred_jsonl="$2"
  local out_dir="$3"
  local metrics="$4"
  local use_segm="$5"

  cat > "$cfg_path" <<EOF
pred_jsonl: $pred_jsonl
out_dir: $out_dir
metrics: $metrics
use_segm: $use_segm
overlay: false
overlay_k: 0
num_workers: $NUM_WORKERS
semantic_model: $SEMANTIC_MODEL
semantic_threshold: $SEMANTIC_THRESHOLD
semantic_device: $SEMANTIC_DEVICE
semantic_batch_size: $SEMANTIC_BATCH_SIZE
f1ish_iou_thrs: [$F1ISH_IOU_THR_0, $F1ISH_IOU_THR_1]
f1ish_pred_scope: annotated
EOF
}

run_one_checkpoint() {
  local label="$1"
  local run_name="$2"
  local ckpt="$3"
  local device="$4"

  local run_dir="$OUTPUT_BASE_DIR/$run_name"
  local infer_cfg="$run_dir/pipeline_infer.yaml"
  local conf_cfg="$run_dir/confidence_postop.yaml"
  local eval_f1_cfg="$run_dir/eval_f1ish.yaml"
  local eval_coco_cfg="$run_dir/eval_coco.yaml"

  mkdir -p "$run_dir"
  write_infer_cfg "$infer_cfg" "$run_name" "$ckpt" "$device"
  write_conf_cfg "$conf_cfg" "$run_dir"
  write_eval_cfg "$eval_f1_cfg" "$run_dir/gt_vs_pred.jsonl" "$run_dir/eval_f1ish" "f1ish" "false"
  # confidence-postop v1 is bbox-only, so COCO here is bbox-only (no segm).
  write_eval_cfg "$eval_coco_cfg" "$run_dir/gt_vs_pred_scored.jsonl" "$run_dir/eval_coco" "coco" "false"

  echo "[$label] infer start: run_dir=$run_dir device=$device"
  PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/run_infer.py" --config "$infer_cfg"

  echo "[$label] confidence post-op start"
  PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/postop_confidence.py" --config "$conf_cfg"

  echo "[$label] f1ish eval start"
  PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/evaluate_detection.py" --config "$eval_f1_cfg"

  echo "[$label] coco eval start"
  PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" "$REPO_ROOT/scripts/evaluate_detection.py" --config "$eval_coco_cfg"

  PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" - "$run_dir" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
f1_metrics = json.loads((run_dir / "eval_f1ish" / "metrics.json").read_text(encoding="utf-8")).get("metrics", {})
coco_metrics = json.loads((run_dir / "eval_coco" / "metrics.json").read_text(encoding="utf-8")).get("metrics", {})
conf_summary = json.loads((run_dir / "confidence_postop_summary.json").read_text(encoding="utf-8"))

f1_key = "f1ish@0.50_f1_full_micro"
if f1_key not in f1_metrics:
    # Fallback to the first available *_f1_full_micro key if thresholds changed.
    for k in sorted(f1_metrics.keys()):
        if k.endswith("_f1_full_micro"):
            f1_key = k
            break

trace_path = run_dir / "pred_token_trace.jsonl"
trace_rows = []
if trace_path.exists():
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        trace_rows.append(json.loads(raw))

token_lengths = []
token_total = 0
coord_total = 0
im_end_tail = 0
coord_re = re.compile(r"^<\|coord_\d+\|>$")
for row in trace_rows:
    toks = row.get("generated_token_text")
    if not isinstance(toks, list):
        continue
    clean = [str(t) for t in toks]
    token_lengths.append(len(clean))
    token_total += len(clean)
    coord_total += sum(1 for t in clean if coord_re.match(t))
    if clean and clean[-1] == "<|im_end|>":
        im_end_tail += 1

def _quantile(vals, q):
    if not vals:
        return 0
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * q))
    return vals[max(0, min(idx, len(vals) - 1))]

trace_diag = {
    "trace_rows": len(trace_rows),
    "token_len_mean": float(statistics.mean(token_lengths)) if token_lengths else 0.0,
    "token_len_p50": int(_quantile(token_lengths, 0.50)),
    "token_len_p90": int(_quantile(token_lengths, 0.90)),
    "token_len_max": int(max(token_lengths)) if token_lengths else 0,
    "im_end_tail_rate": (float(im_end_tail) / float(len(trace_rows))) if trace_rows else 0.0,
    "coord_token_fraction": (float(coord_total) / float(token_total)) if token_total > 0 else 0.0,
}

report = {
    "run_dir": str(run_dir),
    "f1_key": f1_key,
    "f1_value": f1_metrics.get(f1_key),
    "map_key": "bbox_AP",
    "map_value": coco_metrics.get("bbox_AP"),
    "ap50_value": coco_metrics.get("bbox_AP50"),
    "confidence_kept_fraction": conf_summary.get("kept_fraction"),
    "confidence_dropped_by_reason": conf_summary.get("dropped_by_reason"),
    "trace_diagnostics": trace_diag,
}
(run_dir / "pair_eval_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
}

run_one_checkpoint "A" "$RUN_A_NAME" "$CKPT_A" "$DEVICE_A" &
PID_A=$!
run_one_checkpoint "B" "$RUN_B_NAME" "$CKPT_B" "$DEVICE_B" &
PID_B=$!

set +e
wait "$PID_A"
STATUS_A=$?
wait "$PID_B"
STATUS_B=$?
set -e

if [[ "$STATUS_A" -ne 0 || "$STATUS_B" -ne 0 ]]; then
  echo "[ERROR] One or both checkpoint runs failed (status_a=$STATUS_A status_b=$STATUS_B)." >&2
  exit 1
fi

PYTHONPATH="$REPO_ROOT" "${COORDEXP_PYTHON[@]}" - "$OUTPUT_BASE_DIR" "$RUN_A_NAME" "$RUN_B_NAME" <<'PY'
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
run_names = [sys.argv[2], sys.argv[3]]
rows = []
for run_name in run_names:
    report_path = base_dir / run_name / "pair_eval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run": run_name,
            "f1_key": report.get("f1_key"),
            "f1": report.get("f1_value"),
            "mAP": report.get("map_value"),
            "AP50": report.get("ap50_value"),
            "kept_fraction": report.get("confidence_kept_fraction"),
            "im_end_tail_rate": report.get("trace_diagnostics", {}).get("im_end_tail_rate"),
            "coord_token_fraction": report.get("trace_diagnostics", {}).get("coord_token_fraction"),
        }
    )

summary = {
    "output_base_dir": str(base_dir),
    "runs": rows,
}
summary_path = base_dir / "pair_eval_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Wrote summary: {summary_path}")
PY
