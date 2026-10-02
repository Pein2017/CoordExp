# CPU candidate: prefix exposure ranking

Lifecycle remains planned. Scientific outcomes are unmeasured; native release and lead acceptance remain false. Worker live App inspection confirmed `gpt-6.1-sol/high`, thread `01a0fe93-c919-72b0-b99b-f252464dbcbf`, at the canonical research checkout. Direct reports go to lead thread `01a0fdd8-26b6-7240-ab56-f021c05f3445` using the state-recorded `worker_turn.py --to lead` route.

## Frozen contexts

Accepted raw tokens and the maintained strict parser rederived the first eligible earlier row and coordinate without GT. Prefixes exclude the target x1; the last HF logit predicts that target. Each nonzero offset changes exactly one literal saved token. Allowed support is coordinate bins0..998; the complement contains every other vocabulary token, including bin999 and noncoordinates.

| Site | Offset | Generated position | Coordinate | Bin | Prefix tokens | Processed tokens |
|---|---:|---:|---|---:|---:|---:|
| 7511-626 | +0 | 620 | y2 | 638 | 626 | 1946 |
| 7511-626 | -1 | 620 | y2 | 637 | 626 | 1946 |
| 7511-626 | +1 | 620 | y2 | 639 | 626 | 1946 |
| 7511-626 | -2 | 620 | y2 | 636 | 626 | 1946 |
| 7511-626 | +2 | 620 | y2 | 640 | 626 | 1946 |
| 351017-1507 | +0 | 1499 | y1 | 915 | 1507 | 2869 |
| 351017-1507 | -1 | 1499 | y1 | 914 | 1507 | 2869 |
| 351017-1507 | +1 | 1499 | y1 | 916 | 1507 | 2869 |
| 351017-1507 | -2 | 1499 | y1 | 913 | 1507 | 2869 |
| 351017-1507 | +2 | 1499 | y1 | 917 | 1507 | 2869 |

These are structurally valid neighboring histories, without certified owner invariance. Training uses offsets0,-1,+1; held-out offsets are-2,+2. Native order is0,-1,+1,-2,+2 per site,7511 before351017.

## Inputs and training bindings

All18 image bytes, processed prompts, media hashes and grids were freshly checked against the accepted input ledger. Installed native placeholder expansion matches the original processed prompts and all ten extended histories. Input order and prompt lengths:

| Image | Processed prompt tokens |
|---|---:|
| 1584 | 1372 |
| 2299 | 1222 |
| 2685 | 1244 |
| 4134 | 1362 |
| 5001 | 1320 |
| 6040 | 1314 |
| 7116 | 1362 |
| 7511 | 1320 |
| 10707 | 1320 |
| 13348 | 1362 |
| 13923 | 1362 |
| 14038 | 1362 |
| 14439 | 1260 |
| 16228 | 1336 |
| 309264 | 1362 |
| 351017 | 1362 |
| 417044 | 1320 |
| 477415 | 1362 |

Maximum prompt1372 on1584 plus natural cap3084 gives4456. Labels18/570 remain evaluator-only. The manifest retains exact input/prompt/media identities, accepted historical receipts and raw token locators, checkpoint files, and the distinct current execution source envelope.

Checkpoint headers confirm588 FP32 language DoRA tensors, rank16/alpha32/dropout0, and two FP32[1004,2048] embedding deltas. Maintained composition uses BF16 FA2; the unit checks frozen base/vision/projector, zero attention dropout and activation checkpointing OFF at the first released real composition. A material mismatch stops the package.

Each arm reloads the accepted anchor independently, creates fresh AdamW (DoRA1e-5, deltas5e-6, betas.9/.999,epsilon1e-8,weight decay0), seed92711 and constant LR. Six singleton standalone Gmax terms each contribute1/6 before one global clip1 and one step. Both arms take16 updates; R-single repeats each original three times, R-multiple uses0,-1,+1. No label, CE, semantic, predecessor mixed loss, target refresh or checkpoint selection enters training.

## CPU evidence and release boundary

Preparation logs and focused CPU checks are retained under `outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/`. The immutable `cpu-candidate-01.json` records actual commands, exits, log hashes, scoped source qualification, manifest SHA and proposed phase commands after the source commit. The maintained research consumer is run after these record changes. Earlier failed logs are retained; the natural evaluator arm-field mismatch was corrected solely in the new probe.

The focused command `python -m pytest -q tests/probes/test_prefix_exposure_ranking.py` exited0:5passed in71.90s (`tests-04.log`). The research consumer exited0 with334 entries/10 current/167 claim references (`research-consumer-01.log`). Focused checks exercise the real exact-history builder, six-view update caller, independent arm reloads, all five native/training callers and final readback with CPU model/engine doubles. They check full vocabulary support and literal tie legality, evaluator-only context acquisition, repair versus retention denominators, frozen budgets/counters/order, and rejection of source drift, re-signed false natural/conditional credit, false training terms, swapped arm/checkpoint/weight receipts and false terminal summaries. CPU checks establish caller/consumer behavior; they do not establish real model training or native/HF parity.

The proposed released package has five serial phases: native anchor; train R-single; native R-single; train R-multiple; native R-multiple; then readback. All30 HF diagnostics occur inside the two training phases (10 anchor plus10 single endpoint;10 multiple endpoint). Each native phase performs ten scored emissions before18 empty-history outputs. Scientific non-repair does not truncate either endpoint readout. Technical source/input/checkpoint/budget/nonfinite failure preserves partial evidence and stops without replacement.

Resource ceiling: one GPU/rank/native sequence,2GiB native KV, context4456,32 optimizer steps,192 training replays,30 HF diagnostics,30 scores,54 natural outputs,166566 new native tokens. Each phase permits1800s active plus30s cleanup; active total9000s. Phase logs, wall/RSS, HF CUDA peaks and artifact bytes are measured by the runner. Native child CUDA allocation is explicitly unmeasured. No real model or GPU resources were acquired during preparation.

The lead clarification at44c1324ba preserves maintained native defaults (`enforce_eager=False`, `FULL_DECODE_ONLY`). Explicit training/diagnostic forward and native observation counters do not count every physical forward internal to startup/capture. Internal startup/capture forward count remains unmeasured; its costs are included in phase wall/resource totals. No extra caller-generated warmup/qualification requests or concurrent engines are added. Real BF16/FA2 trainable binding checks, native/HF disagreement and production resource bounds remain unmeasured until the exact release.

No execution checkout was created, edited or advanced. The existing execution checkout remained at5ce39ccca during inspection. The CPU source and qualified manifest are candidates, not native authority or self-acceptance. Stop at lead release; no next unit is scheduled.

## Candidate02: phase deadline and owned-group cleanup

The lead accepted the frozen scientific caller review and found an execution blocker in source52b074ff6: initial wait1830 exceeded1800 active, and parent exit after TERM did not prove owned descendants stopped. The immutable lead counterexample and candidate01 remain retained.

The correction changes only the package boundary. Initial wait derives1800 from frozen bounds. One absolute cleanup deadline derives30 from those bounds before group inspection. TERM gets half the remaining cleanup interval; KILL and live-group drain verification share the rest. A phase's new session/process group is inspected through Linux `/proc`; zombies/dead states are excluded from live members. A timeout, nonzero parent exit or successful parent with live descendants cannot advance to another phase. An undrained group produces a partial receipt and stops. Signals target only the package-spawned group; VllmDoraRollout and all scientific callers remain unchanged.

RED: the actual package caller failed all three bounded CPU subprocess cases (timeout/parent0, parent7, and parent0 without timeout), including live descendants surviving settled leaders and false continuation to readback. `package-red-01.log`, exit1:3failed. Each test owns a real descendant ignoring TERM, an unrelated live group as a protection control, and guaranteed finally cleanup.

GREEN: `python -m pytest -q tests/probes/test_prefix_exposure_ranking.py -k package_`, exit0:5passed/5deselected in7.49s (`package-green-02.log`). The same real-descendant cases now drain without advancing. Controlled time verifies1800 active/30 cleanup without waiting those durations; an undrainable double stops at30 with a partial receipt. A separate process-state check distinguishes live from zombie and other-session/group members, and completed groups permit the five normal phases.

Scientific-caller source/evidence is unchanged and reused from candidate01 and the lead review. Candidate02 and qualified manifest02 bind the new scoped clean commit; native authority remains false, lifecycle planned, and execution checkout5ce remains untouched. No real model/GPU work or observation, recipe, research-meaning or scheduling change occurred.
