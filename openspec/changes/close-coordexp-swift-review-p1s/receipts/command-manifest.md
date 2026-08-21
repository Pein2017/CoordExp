# Command manifest — close-coordexp-swift-review-p1s

All Python via `conda run -n ms`, PYTHONDONTWRITEBYTECODE=1, no `env`-prefix
(rtk hook trap). Tree state for every entry: HEAD 04bbc1631 + this change's
uncommitted fixes (committed together as the Wave-2 close commit).

| # | Purpose | Command | Result |
|---|---|---|---|
| 1 | Builder A focused (P1-3) | `pytest tests/losses/ -x -q` | 206 passed |
| 2 | Builder B focused (P1-2) | `pytest tests/runtime/ -x -q` | 193 passed, 1 warning |
| 3 | Builder C focused (P1-1) | `pytest tests/training/ -q` | 1515 passed, 126 skipped |
| 4 | Lead P2-1 (RED then green) | `pytest tests/training/test_reporting.py tests/runtime/test_fp16_scaler_contract.py -q` | RED: TypeError at test_reporting.py:1228; then 62 passed |
| 5 | Lead P2-4 (RED then green) | `pytest tests/training/test_training_module_boundaries.py -q` | RED: 2 failed; then 20 passed |
| 6 | Lead P2-2 (rejected variant + landed) | `pytest tests/config/ tests/training/test_input_attestation.py -q` | proxy variant: 27 errors (mappingproxy not deep-copyable) — REJECTED; landed variant: 263 passed |
| 7 | Probe 2.2 zero-eligible gloo | `python scripts/probes/coordexp_swift/review_p1s_zero_eligible_gloo_probe.py` (+ `--simulate-pre-fix`) | PASS 16/16; pre-fix sim: real deadlock, SIGTERM at 30s bound → receipts/probe-zero-eligible-gloo.md |
| 8 | Probe 2.3 fp16 launch refusal | `python scripts/probes/coordexp_swift/review_p1s_fp16_launch_refusal_probe.py` (+ `--simulate-pre-fix`) | PASS 30/30; sim: 6 assertions fail, prepare reached → receipts/probe-fp16-launch-refusal.md |
| 9 | Wave 2.4 full suite (baseline argv replicated from archived obs change command-manifest.json:114) | `pytest tests/config tests/losses tests/runtime tests/training tests/artifacts tests/eval -q` (single invocation, detached; see receipts/run-wave-2-full-suite.sh) | 2600 passed, 126 skipped, EXIT=0, 969.22s; failure set EMPTY vs empty baseline; +35 = declared new tests → receipts/wave-2-full-suite.md |
| 10 | Scenario-diff pre-check | in-session python diff of delta vs stable `coordexp-swift-training-artifacts` | 0 problems; MODIFIED Wide-Step Logging Stream 20→21 scenarios (+1 ADDED) |
| 11 | Validation | `openspec validate close-coordexp-swift-review-p1s --strict` ; `openspec validate --all` | valid; 20/20 |

Known execution incidents (kept honest): the Wave-2.4 detached run was twice
misdiagnosed as dead by the lead (name-grep liveness check); Runner F refuted
with pid CPU-time deltas + child spawns and the run completed normally. Lesson
recorded: process liveness is proven by pid CPU-time delta, never by name-grep
absence. `conda run` (no `--no-capture-output`) buffers all child stdout until
exit — a frozen log is not death evidence.
