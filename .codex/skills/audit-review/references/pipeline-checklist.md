## Pipeline / Processing Checklist (Audit-Focused)

Use this to find correctness, reproducibility, and evaluation-validity risks in end-to-end flows:
`data -> transforms/packing -> training/infer -> artifacts -> eval/vis`.

### 1. Inputs and Contracts
- Identify the contract boundary (JSONL schema, dataset adapter boundary, config schema).
- Confirm required keys and invariants (e.g., geometry is exactly one of `bbox_2d` xor `poly`).
- Check that “optional” fields cannot silently change semantics (defaults are explicit).

### 2. Geometry and Ordering (High Risk)
- Verify geometry is preserved (no dropped/reordered coords except explicitly defined canonicalization).
- Verify object ordering is deterministic (and tested), especially if training/eval depends on it.
- Verify coordinate-space transitions are explicit (pixel -> norm1000 ints -> tokens).

### 3. Transform Staging and Side Effects
- Stage order is deterministic and documented (planner contracts match implementation).
- In-place writes are avoided or protected (temp file + replace).
- Path rewriting (absolute vs relative) is consistent across stages and artifacts.

### 4. Determinism / Seeds / Multiprocessing
- Seeds are derived deterministically and scoped correctly (per-epoch / per-worker).
- Multiprocessing doesn’t introduce nondeterministic ordering (file writes, record ordering).
- Any randomness is either removed, seeded, or explicitly “best-effort” and isolated.

### 5. Failure Policy (Fail-Fast vs Best-Effort)
- Core paths do not swallow exceptions (dataset encoding, trainer steps, inference, eval).
- Best-effort behavior is narrow and justified (typically optional I/O sinks only).
- Error messages are actionable (include config key, file path, split, sample id).

### 6. Artifacts and Reproducibility Breadcrumbs
- Outputs include enough metadata to reproduce: dataset id, preset/run name, seed, max_objects, etc.
- Artifact naming is self-describing (avoid ambiguous “train.jsonl” meaning multiple things).
- Manifest/summaries are written deterministically and consistently.
- Training outputs include the expected manifest family: `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`.
- Infer/eval outputs include the expected artifact family: `summary.json`, `resolved_config.json`, `resolved_config.path`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`, and guarded companions when enabled.
- Downstream eval/vis jobs can recover authoritative root-image and config provenance without guessing from the current working directory.

### 6.1 Evaluation Validity
- Official COCO/LVIS style metrics consume scored artifacts, not raw unscored predictions.
- Raw-text `xyxy` norm1000 paths use numeric-text confidence alignment, not coord-token span alignment.
- `cxcy_logw_logh` / `cxcywh` results are only treated as evidence for checkpoints trained on that serialization.
- Proxy views are labeled explicitly (`coco_real`, `coco_real_strict`, `coco_real_strict_plausible`) and not compared as if they were the same benchmark.
- Benchmark notes state scope: `val200`, `limit=200`, first-200, full-val, checkpoint ids, raw-text vs coord-token, and GPU launch shape for timing.

### 7. Backward Compatibility / Deprecations
- Deprecated knobs are rejected (fail-fast) or removed; no warning-only no-ops.
- Stable external contracts remain stable (runner CLI, plugin interfaces, config keys).
- If compatibility shims exist, verify they can’t drift (duplication risk: bash vs python).

### 8. Tests and Verification Coverage
- There is at least one “parity” or regression test for contract-critical behavior.
- Add or suggest the smallest test that proves the invariant (avoid broad refactors).
- Prefer tests that do not require network; use fixtures or synthetic slices.
- Prefer targeted tests from `docs/IMPLEMENTATION_MAP.md` before broad suites.
