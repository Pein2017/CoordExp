# First-arrivals: admitted single-case qualification candidate

Status: **candidate, lead acceptance pending**. Only `mature:313465:0`, native arrival row 1, N owner `715278`, was admitted to Stage 1. The 87-cell pilot remains HOLD. This record does not accept the mechanism or release another cell.

## Exact case and technical result

Lead admission is `lead-stage0-admission-v1.json` (SHA-256 `146605205625078300ef00016c8e1f715e9f78b4afa7edf4282d61a32a44b555`). The executed source is the unchanged `supporting/selection-attempt-004.json` (SHA-256 `325dfcd67c3ef21c717e80686da92e9020a4fc151b98d23ef1217a15363e6eeb`), original untied source group `fresh-18`, batch index 3, in its full four-request batch. The admitted cells were native plus separately qualified A sham, complete A/N row scores, and first-coordinate N branch (`x1` `151675`→`151831`). The fresh effective model matched saved adapter, embedding payload and independent input/output row/delta hashes. Its maintained loader bytes matched the historical captured loader; only the loader **path** differed. The full original prompt/media/companion identity matched the source receipt.

Attempt-004 [case JSON](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/attempt-004/case.json) (SHA-256 `1585979fa9f889c6216f608f55fd2c714d56dc9c2c2255e5e25ec24843f24b57`) is the machine evidence. Native continuation exactly reproduced the 30 saved target token IDs through the current row plus two further complete rows; separate supplied-A sham matched that continuation exactly. All 30 native generated-token logits and all 10 A row-score token logits/top-two/log-normalizers matched saved source within `2e-4` (maximum errors `3.815e-5` and `7.057e-5`). A score included its opener and terminator. The actual score caller rejected dropped terminator and shifted-slot mutations. The actual model's recorded full-batch input contained supplied N x1 `151831`; deleting or replacing that token in the captured consumer input was rejected by the same input verification. No branch was credited for that supplied coordinate.

## Scientific observations and limit

At this native history, the complete-row scores are `log P(A|H) = -10.005468`, `log P(N|H) = -21.239322`, so **D = N−A = −11.233855 nats**, outside the 0.01-nat numerical deadband. The native x1 token log probability was `−1.931633`, versus `−9.672842` for N x1. This finite pair favors A; it is **not** a complete-row preference witness for N and does not prove an internal old-owner commitment.

Supplying only N's x1 made the free decoder emit first-row box `[161,0,415,241]` for bowl. The other three coordinates were freely generated. The box has IoU `0.813278` with the distinct upper bowl owner `715278` (and `0.029176` with A owner `716308`). [Branch overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/attempt-004/branch-first-row.png) (SHA-256 `eb3b94fa4e1dbe33a149f5b669ab9db454a63585325452958e029544e50cc616`) visually supports that unique owner assignment. Native/A-sham rows are A bowl → upper bowl → scissors; the N branch rows are upper bowl → scissors → carrot. All three paths stopped after the current plus two complete rows, before the 256-free-token cap. This is conditional N realization under a supplied coordinate, consistent with available grounding; it does not establish that the native decision preferred N or that the effect is recurrence-specific. No normal-control cell was run.

## Finalization, attempts, and bounded cost

Attempt-004 [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/attempt-004/receipt.json) (SHA-256 `3cb83e7fd4b5814b22bdec676133d98e28dcba655bf9ea57083fb6e650918785`) reports 80 model forwards, 5 vision forwards, 26.6645 allocated GPU-seconds, peak RSS 11,398,212 KiB, peak GPU allocated 11,193,154,048 bytes and reserved 12,022,972,416 bytes. Attempt-004 artifacts occupy 2,712,483 bytes after [cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/attempt-004/cold-readback.json) (SHA-256 `484b2a35c1df08cd2cefe90bc26c77f660c6ef7b3e2b5e90b85e87aa07300850`) passed in a separate process.

Attempts 001–003 remain terminal technical-invalid evidence: CUDA allocator initialization failed before model load (0 forwards); exact identity comparison detected a path-only historical-loader relocation before a forward (0 forwards); and CPU owner reduction used a nonexistent parser field after valid generation (80 forwards). The latter prompted durable `raw-evidence.json` write before owner reduction in attempt 004. [Cost and terminal ledger](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/cost-and-terminal.json) (SHA-256 `bd5b182a4569338c7ab295052c46aab3ead9b91df0d7f14663c6f3d5f7e7f8a1`) charges all failed attempts and auxiliary CUDA checks/diagnostic conservatively: **94.750 allocated GPU-seconds = 0.02632 GPU-hours** against the one-case 1-hour ceiling and package 8-hour ceiling. Total across attempts was 160 model forwards and 10 vision forwards. All four producer jobs are terminal; no owned GPU job remains.

The corrected CPU proposal `selection.json` (SHA-256 `435c333520f5f31078a44d07a31c986d07198f9c8140cd8f3ddf47395`) and dependent `supporting/cpu-qualification-attempt-008.json` (SHA-256 `9ba8403009d37658260dea81911815f37adcb85556f7eef13adab69e21f49e95`) retain all 13 family rulings and revised 35 possible diagnostic cells; this is accounting, **not GPU admission**. [Correction note](supporting/stage0-registry-correction-v1.md) records the row-1 part-only anchor and HOLD denominators.

Recheck without a model call:

```bash
python -B - <<'PY'
from pathlib import Path
from probes.training_set_completion.recurrence_first_arrivals.stage1_case import readback
p = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-first-arrivals/stage1-qualification/attempt-004')
print(readback(p))
PY
```

Source compilation, `git diff --check`, and the read-only two-root output-layout check also passed. No commit, push, or additional GPU cell was performed.
