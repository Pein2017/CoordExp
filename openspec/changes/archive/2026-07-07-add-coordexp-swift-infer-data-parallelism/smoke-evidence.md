# Smoke Evidence

## Scope

This note records implementation evidence for the
`add-coordexp-swift-infer-data-parallelism` change. It is a tiny two-row,
two-GPU HF smoke and is not production benchmark evidence.

## Run

- Date: 2026-07-05
- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`
- Smoke command exit status: 0
- Command shape:

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m src.infer --config configs/coordexp_swift/infer/dp-smoke-reviewfix-GAOS.yaml
```

The temporary launch config was removed after the run. Its SHA-256 and fully
resolved values are preserved in the stable resolved config evidence listed
below.

- Artifact root:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix`
- Stable resolved config evidence:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix/configs/resolved.yaml`
  and
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix/configs/resolved.json`
- Backend: `hf`
- Model:
  `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Input JSONL:
  `/data/CoordExp/.worktrees/CoordExp-swift/tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl`
- Rows: 2
- Visible CUDA tokens: `0,2`
- `generation.batch_size`: 1 per device
- Active ranks: 2

## Verified Evidence

- Required top-level merged artifacts exist:
  `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`,
  `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`,
  `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, and
  `run_manifest.json`.
- Required shard plan exists at `shards/data_parallel_plan.json`.
- Shard directories exist at `shards/rank-000/` and `shards/rank-001/`.
- `run_manifest.json` records `execution_mode: controller_worker`,
  `active_ranks: 2`, `per_device_batch_size: 1`, and
  `merge_status: completed`.
- Rank 0 recorded `worker_cuda_visible_devices: "0"`,
  `cuda_device_count: 1`, `cuda_current_device: 0`,
  `worker_logical_device: cuda:0`, and `model_first_parameter_device: cuda:0`.
- Rank 1 recorded `worker_cuda_visible_devices: "2"`,
  `cuda_device_count: 1`, `cuda_current_device: 0`,
  `worker_logical_device: cuda:0`, and `model_first_parameter_device: cuda:0`.
- Merged raw and scored row ids match the input order:
  `coco2017_train_000000000030__smoke2obj`,
  `coco2017_train_000000000036__smoke2obj`.
- Merged provenance records `merge_status: completed`, row binding for two
  rows, and generation policy with `batch_size: 1`, `max_new_tokens: 24`,
  `temperature: 0.0`, `top_p: 1.0`, and `repetition_penalty: 1.0`.
- Merged sidecar counts were 48 token-trace rows, 2 parse diagnostic rows, and
  2 image-plan rows.
- `src.eval.detection_consumer.evaluate_scored_detection_artifacts` consumed
  the merged scored artifact and wrote:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix/eval_detection/metrics.json`.
- The evaluator also wrote:
  `/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix/eval_detection/evaluation_receipt.json`.
- The evaluation receipt binds `gt_vs_pred.jsonl`,
  `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`,
  `summary.json`, and `run_manifest.json` by SHA-256.

## Metrics Scope

The evaluator reported `metric_family:
coordexp_swift_detection_coco_bbox_v1`, `row_count: 2`, `mAP: 0.0`, and
`mRecall: 0.0`. It also reported `benchmark_metric: false` and
`benchmark_eligible: false`. These values are smoke-only and carry no
benchmark meaning.

## Verification Commands

```bash
pytest tests/inference/test_pipeline.py tests/inference/test_worker_runtime.py tests/inference/test_data_parallel_runtime.py tests/inference/test_data_parallel_merge.py tests/inference/test_artifacts.py tests/inference/test_config_runtime.py tests/inference/test_backend_trace.py tests/eval/test_detection_consumer.py -q
openspec validate add-coordexp-swift-infer-data-parallelism --strict
git diff --check
```

Results:

- `Pytest: 154 passed`
- `Change 'add-coordexp-swift-infer-data-parallelism' is valid`
- `git diff --check` passed with no output

Post-run artifact probe:

```bash
python - <<'PY'
from pathlib import Path
import json

run = Path("/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/data_parallel_smokes/coordexp-swift-infer-dp-2gpu-smoke-reviewfix")
required = [
    "gt_vs_pred.jsonl",
    "gt_vs_pred_scored.jsonl",
    "gt_vs_pred_scored.jsonl.provenance.json",
    "pred_token_trace.jsonl",
    "parse_diagnostics.jsonl",
    "image_plan.jsonl",
    "summary.json",
    "run_manifest.json",
    "shards/data_parallel_plan.json",
]
missing = [p for p in required if not (run / p).is_file()]
assert not missing, missing

manifest = json.loads((run / "run_manifest.json").read_text())
provenance = json.loads((run / "gt_vs_pred_scored.jsonl.provenance.json").read_text())
raw_rows = [json.loads(line) for line in (run / "gt_vs_pred.jsonl").read_text().splitlines() if line]
scored_rows = [json.loads(line) for line in (run / "gt_vs_pred_scored.jsonl").read_text().splitlines() if line]

assert manifest["parallelism"]["execution_mode"] == "controller_worker"
assert manifest["parallelism"]["active_ranks"] == 2
assert manifest["parallelism"]["per_device_batch_size"] == 1
assert manifest["parallelism"]["merge_status"] == "completed"
assert provenance["parallelism"]["merge_status"] == "completed"
assert [row["row_id"] for row in raw_rows] == [row["row_id"] for row in scored_rows]
assert len(raw_rows) == len(scored_rows) == 2
assert sorted(path.name for path in (run / "shards").glob("rank-*")) == ["rank-000", "rank-001"]

for rank, token in [(0, "0"), (1, "2")]:
    shard = run / "shards" / f"rank-{rank:03d}"
    worker = json.loads((shard / "run_manifest.json").read_text())["parallelism"]["worker"]
    assert worker["rank"] == rank
    assert worker["world_size"] == 2
    assert worker["worker_cuda_visible_devices"] == token
    assert worker["cuda_device_count"] == 1
    assert worker["cuda_current_device"] == 0
    assert worker["worker_logical_device"] == "cuda:0"
    assert worker["model_first_parameter_device"] == "cuda:0"

metrics = json.loads((run / "eval_detection" / "metrics.json").read_text())
receipt = json.loads((run / "eval_detection" / "evaluation_receipt.json").read_text())
assert metrics["metric_family"] == "coordexp_swift_detection_coco_bbox_v1"
assert metrics["row_count"] == 2
assert metrics["benchmark_metric"] is False
assert metrics["benchmark_eligible"] is False
assert metrics["evaluation_receipt_json"] == "evaluation_receipt.json"
assert metrics["evaluation_receipt"] == receipt
assert receipt["run_manifest"]["terminal_status"] == "completed"
assert receipt["run_manifest"]["benchmark_eligible"] is False
assert receipt["benchmark_metric"] is False
for artifact_name in (
    "gt_vs_pred.jsonl",
    "gt_vs_pred_scored.jsonl",
    "gt_vs_pred_scored.jsonl.provenance.json",
    "summary.json",
    "run_manifest.json",
):
    assert receipt["artifacts"][artifact_name]["sha256"]
print("artifact_probe_passed")
PY
```

Result: `artifact_probe_passed`.
