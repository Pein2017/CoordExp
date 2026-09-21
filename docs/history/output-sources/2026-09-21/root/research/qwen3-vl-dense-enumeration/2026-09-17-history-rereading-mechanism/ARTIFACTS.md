# History rereading artifact map

All model execution complete; final result/terminal govern candidate closeout. Paths below are relative to this directory.

- `bindings.json`, `source-snapshots/`, `panels/{S,F}.json`: exact accepted predecessor, original native inputs/model/config and new producer/protocol. Model inputs remain old processed source regardless of evaluation labels.
- `cpu-check.json`: original source-token boundary falsification, rejects shifted position and wrong source row.
- `runtime/{extract-S,extract-F}/`: 68 native forwards each, all four exact prefixes, source row cache and current states. `verify-extract-S-extract-F.json` verifies original fork logits/head/layer13/cache and MRoPE position identity.
- `runtime/{native-F,self-F,residual-only,cache-only,joint}/`: full raw tokens/text/stops, receipt and `capture.pt`. Conditions are target-only on the original heterogeneous batch; all companions remain native.
- `capture.pt`: nested `tensors.history` contains paired post-RoPE K and native V for actions54..62 and separate unpatched wrapper63..65; `restore` contains applied and restored region tensors. `residual` has layer13/14/17/20 outputs before/after; `head`, `logits`, `positions` capture selected offsets63/67/68/76/103; `cache` retains current-position all-layer K/V at67/68. Full batch tensors permit companion checks. `trace` binds consumed input IDs and physical cache positions54..68. Prefix, exact input identity and physical storage range are inside the payload. No full history-KV or attention-probability archive.
- `verify.py`: independent saved token/tensor/position/restore/companion checks. `compare_donors.py` / `donor-comparison.json`: measured source differences and equal positions; norms are descriptive, not causal attribution.
- `launch-*.sh`, `logs/`: exact commands, producer PID, start/end/exit and unfiltered logs. `cost.py` summarizes all counted receipts including failures; no hidden retries.
- `human-evaluation/`: separately owned read-only versioned annotation support. It cannot alter model inputs, old annotations or frozen scores. Admission/HOLD is explicit.

Reconstruction starts in `/data/CoordExp/.worktrees/research-probes`. Use `python <this-root>/verify.py <condition ...>` for saved evidence; new model calls are unnecessary for receipt verification. Final reduction and closeout paths will be appended once execution finishes.

## Final measurements and checks

- `pre-outcome-cache-gate-clarification.json`: records before main outputs that a later token change can also require the retained-state rebuild comparison. Original unit bytes are preserved.
- `conditional-decision.json`: cache-only selected; one rebuild is also the native control because their first68 emitted tokens are identical. F2 specificity not executed on the null/mixed repair result.
- `runtime/rebuild-cache-only/`, `rebuild-verification.json`: all native sequences and sampled current cache restored exactly, late patch drift removed.
- `verify-*.json`, `factorial-summary.json`, `donor-comparison.json`: raw margins/ranks, exact trajectory forks, current K/V changes and descriptive state distances. EOS remains separately recorded.
- `reduction-manifest.json`, `reduction.json`, `reduction-summary.json`: original frozen bank full/prefix/supplied/free owner sets and all output debt. Same action72 primary boundary for every condition. `independent-output-check.json` is a fresh JSON-exact saved-output replay with raw-regex checks and the inherited exclusion falsification binding. Run `PYTHONPATH=. python <this-root>/independent_output_check.py` from the worktree.
- `annotation-image-runtime-check.json`: bird export decoded image equals native media identity exactly. Annotation snapshots/evaluation preserve absent ignore semantics; no negative inference from missing or deleted annotations.
- `tensor-inventory.json`: eight actual captures with every tensor key/shape/dtype/byte count and file hash. `source-snapshots/` retains executed source and frozen protocol.
- `cost.json`: all eight model receipts/exits,18,640 forwards, bounded storage/runtime and no live owned model process. `team-observations.md` records the one annotation-admission correction. No cleanup.

## Human evaluation and final status

`human-evaluation/annotation-snapshot-v1/` preserves the original blanket-HOLD discovery receipt and immutable export. `human-evaluation/evaluation-v1/admission.json` narrows admission to positive-owner coverage only, leaving negative/ignore claims HOLD. Its `produce.py` imports the unchanged accepted parser/matcher; `result.json`, `summary.json` and `independent-check.json` bind ten old/new saved trajectories, both10/14 denominators, all four token views, per-owner assignment, current exclusion and exact historical-ledger replay. Reconstruct with `PYTHONPATH=. python <this-root>/human-evaluation/evaluation-v1/produce.py`. No annotation-bank or model-input rewrite.

`result.json` is the authoritative iteration receipt; `terminal.json` seals it and the local artifact inventory. `candidate-records/` freezes research unit/results/state for later lead integration. `knowledge-check.json` preserves actual checker outcome; lead owns the global frontier. No model or worker work remains.
