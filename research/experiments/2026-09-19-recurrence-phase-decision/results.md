# Recurrence phase decision: worker candidate

Status: **candidate, pending root acceptance**. The frozen intervention matrix is complete. The stable machine entry is `analysis/summary-final.json` under the [artifact root](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-phase-decision); the full CPU result is `attempts/reduction-02/reduction.json`. No successor or training is released here.

## Question and frozen scope

The contrast is a one-bin x1 or y2 edit in the completed history at the seed, next, or third row of the **earliest chronological pairwise-near triple**, followed either by immediate original-greedy free release or by the **same edited history plus two complete native rows** before release. Native controls are paired within each release boundary. The 16-row free trajectory is secondary to the first-free choice and margin and the fixed-native-suffix scores. The two bridge rows receive no free-outcome credit.

The accepted panel contains 21 failure and 24 proxy states. All 45 were profiled; 16 failure sources (8 per model) and 4 proxy sources (2 per model) entered the intervention matrix. The crosswalk moved the chosen seed earlier than the predecessor source row in 11/21 failure sources, including 10/16 selected sources. Only **4/16 selected failure sources** have r+1 as their first-ever numerical repeat. No result below is a general first-ever onset, physical-owner, or training-origin claim.

## Admission and coverage

| Item | Observed |
| --- | ---: |
| Native profiles / available profile rows | 45 / 150 |
| Theoretical free cells | 520 |
| Out-of-vocabulary one-bin edit directions | 22 (44 mode cells absent) |
| Planned cells after those absent directions | 476 |
| Ready saved cells: immediate / bridge | 238 / 213 |
| Bridge HOLD cells lacking two complete native rows | 25 (10 failure, 15 proxy) |
| Edited versus same-boundary native comparisons | 352 (186 immediate, 166 bridge) |
| Candidate / technical binding HOLD / missing saved cells | 451 / 0 / 0 |

The 14-forward real-entry gate passed on both model packages. Source-trace top-two scores passed for the six tested model/source families; **full-vocabulary exact-versus-prefix parity at that gate was tested only for the numerical-feedback family**. The 45-source capture produced 3,751 registered causal slots and 2,023 native trace top-two checks, with maximum absolute top-two discrepancy 1.24e-4 below the 2e-4 threshold. For the **451 selected paired capture/free cells**, an independent saved-vector comparison found zero winner mismatches and maximum full-vocabulary difference 4.84e-5. That latter comparison is between two current execution paths at the same edited history; it does not extend full-vector parity to every original source trace.

Independent readback covered every capture and free-cell artifact, including plan/source hashes, edited site, prefix, consumed position, FP32 score and head vectors, first generated token, and stop bound. The final CPU reducer reports 451 candidates and 25 planned HOLDs. A separate parent replay and actual saved-output corruption checks are recorded under `attempts/reduction-02/`.

## Decision-bearing observations

Only **4/352** edited cells changed the first unforced argmax. All four belong to `untied-train-196924-failure`, whose r+1 is **not** its first-ever numerical repeat. Three are first-repeat **immediate** cells (x1−1, x1+1, y2+1): the native opener–EOS margin is 0.0153, and the edits choose EOS with margins 0.1255, 0.1701, and 0.0185. Their corresponding bridge cells are local HOLDs, so disappearance under bridge cannot be assessed. The fourth is a seed **bridge** x1+1 cell: native chooses EOS by 0.0165 and edit chooses opener by 0.0057. These margins exceed replay discrepancy, but the four switches are one source-specific low-margin boundary rather than cross-source onset evidence. No tied-model or proxy cell changes its first unforced winner.

The fixed-native-suffix comparison changes 60 registered winners across 55/352 edited pairs. Failure-source x1 edits account for 41 switches across 146 pairs; y2 edits account for 17 across 188 pairs. The proxy arm has two switches in one x1 pair. This is a descriptive role difference: x1 and y2 are at different token distances, 22 edit directions are unavailable, and fixed-native suffixes test their own conditional contexts. The per-model, phase, role, direction, and mode margins and centered score changes remain in the CPU result.

| Phase stratum, both modes | Edited pairs | First-free switches | Fixed-slot winner switches | Any free-trajectory divergence |
| --- | ---: | ---: | ---: | ---: |
| Seed | 114 | 1 | 23 | 49 |
| First repeat in qualifying triple | 110 | 3 | 17 | 34 |
| Third row | 110 | 0 | 18 | 36 |
| Proxy | 18 | 0 | 2 | 1 |

Bridge coverage differs by phase and source; these phase rows describe heterogeneity and are not a controlled repetition-count effect. Across all ready edits, 120/352 free token trajectories diverge from their same-boundary native control (72/186 immediate; 48/166 bridge). For the 166 edits with **both** modes available, trajectory divergence is absent in both for 94, immediate-only for 24, bridge-only for 3, and present in both for 45. Persistence after two supplied rows shows that the old edit can still affect later free output under a common token bridge. Direct rereading, hidden-state relay, decay, and later small-margin instability remain unresolved alternatives.

## Secondary numerical recurrence and stops

At common exposed free-row horizons among failure edits, longest near-run changes are infrequent and go in both directions:

| Release | Common 8-row pairs: lower / same / higher | Common 16-row pairs: lower / same / higher |
| --- | ---: | ---: |
| Immediate | 8 / 129 / 4 (141) | 11 / 102 / 6 (119) |
| Bridge | 1 / 122 / 3 (126) | 8 / 98 / 5 (111) |

Cells without both horizons are unobserved at that horizon, never counted as zero recurrence. Failure cells stop by natural EOS in 65 immediate and 65 bridge releases; 37 and 46 of these respectively end before eight complete free rows. No release hit the 256-token cap. Saved free outputs contain 85 complete invalid boxes across 78 cells and zero malformed openers; invalid boxes remain in numerical accounting. Annotation matches are relative only to the already bound source objects and do not establish physical recovery.

The evidence supports **localized score and occasional output sensitivity across the phase strata**, concentrated at one first-free EOS/opener boundary, with little consistent 8/16-row numerical escape. It does not support a general first-ever onset or a durable owner-recovery mechanism. The bridge contrast cannot distinguish direct rereading from a hidden-state relay, and decay remains possible.

## Reproduction, cost, and closure

From `/data/CoordExp/.worktrees/research-probes`, reproduce the final CPU result with the frozen `reduce.py`, `selection/plan.json`, `selection/windows.json`, and `reduction-inputs/{capture,release}` paths, using a **new** output path (for example `--output /tmp/recurrence-phase-replay.json`). The exact argv and source hashes are in `attempts/reduction-02/snapshot/` and `attempts/reduction-02/parent-replay-launch.json`. Run `PYTHONPATH=. python probes/training_set_completion/recurrence_phase_decision/reduce.py --selfcheck` for invalid-row and wrong-role sensitivity. The saved-output parent check under `attempts/reduction-01/corruption-check.json` rejects a wrong saved mutation offset and a dropped actual invalid-row claim without altering either original file.

All 20 GPU commands (2 gate, 2 pilot, 8 release, 8 capture) exited successfully; no unit model process remains. Runtime receipts total **54,206 model forwards and 2.476 allocated GPU-hours**. The closure measurement of the artifact root is 3,182,160,137 bytes (2.96 GiB), below the 32 GiB bound. No GPU attempt was technical-invalid. `reduction-01` and its replay are preserved as a superseded CPU candidate; `reduction-02` is the final worker reduction. Root alone may mark lead acceptance or authorize any next question.
