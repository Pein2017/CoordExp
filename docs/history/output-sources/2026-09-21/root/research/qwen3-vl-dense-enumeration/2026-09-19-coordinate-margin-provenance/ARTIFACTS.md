# Reconstruction map
- `coordination/selected-states.json`: frozen8 source states, hashes, offsets and tokens.
- `execution-plan.json`, `sources/`: prelaunch producer/reducer snapshots, actual installed forward source, unexecuted child draft.
- `runtime/<state>/capture.pt`: actual residual/component/norm/head/logit tensors, no-hook comparison, effective rows and source states.
- `runtime/<state>/receipt.json` and `native/`: exact model/config/media/input/group provenance and per-state counters.
- `reduction.json`, `reduction-recheck.json`: JSON-exact FP64 accounting, full rankings and signed per-layer contributions including rounding residuals.
- `verification.json`: independent source/no-hook/all-coordinate reconstruction checks.
- `contributions.csv`, `history-differences.json`: full signed tables and conditional-history differences.
- `launch/`: commands, logs, PIDs and observed exits; `cost.json`, `job-closure.json`.
- `result.json`, `integrated-terminal.json`: candidate scientific/technical result; root acceptance is separate.
- `direct-delivery.json`: one direct App Server completion delivery; no wake monitor.

Run from /data/CoordExp/.worktrees/research-probes:
```sh
python probes/training_set_completion/coordinate_margin/reduce.py --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-margin-provenance --output /tmp/coordinate-margin-recomputed.json
python probes/training_set_completion/coordinate_margin/verify.py --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-margin-provenance --output /tmp/coordinate-margin-verification.json
```
No full KV/attention maps or extra physical labels are captured. Positions are bound by native source/replay identity, not an additional all-position tensor archive.
