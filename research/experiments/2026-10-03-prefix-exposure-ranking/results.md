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

## Released package execution

Exact release0fe3deac authorizes one serial five-phase package at clean execution sourcefef032092. Launched2026-10-02 UTC; invocation/PID/session and parent log are recorded at the execution-local `package-invocation-01.json`. Package deadlines govern phases; no outer supervisor-only timeout, extra model requests or CPU suites were added. Native outcomes are still unreviewed and require terminal evidence and separate lead acceptance.

## Released terminal candidate (worker report; lead review pending)


The single released package and separate final same-source CPU readback both exited0. All five phase exits are0, all five owned-group receipts report drained, and final Linux process checks find the supervisor, three native parents and their three children absent. No matching package/phase command remains. Training phase PIDs were not separately captured; their group completion is evidenced by the exact package receipts. Execution source remains clean at `fef0320926a5cde311caf203a5fbfa6a3e042ad8`. No CPU suites, model retries, extra warmup requests or replacements were run.

Immutable terminal candidate: `/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/native-terminal-candidate-01.json` SHA256 `78583948601bdb86b94e3d08d6086b82d452d0c8f3fd834cac235af375304080`. Raw terminal: `/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01/complete.json`; separate readback: `/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01/readback.json`. The candidate binds159 artifact/ledger/receipt/log/checkpoint hashes, original raw locators, release/lead-release identity, actual commands, sessions45343/48250 and cleanup evidence. Package file bytes533634088; phase-published artifact bytes529801948. Explicit native tokens26480 include30 scored tokens; natural outputs account for26450 tokens.

Observed counters:32 optimizer steps,192 training replays,30 HF diagnostics,30 native scores and54 natural continuations (84 requests). Sum of recorded phase active windows573.326s; observed supervisor wall662.649s. Each phase uses its frozen1800 active+30 cleanup deadline, within9000 active ceiling. Native internal startup/capture physical forward count and CUDA peaks remain unmeasured. Maintained capture defaults are unchanged; costs are included in native resource windows, while supervisor wall also includes validation and frontend work. One GPU/rank/sequence, context4456 and2GiB KV are unchanged.


| Phase | Active seconds | Resource seconds | Parent peak RSS KiB | Reaped-child peak RSS KiB | HF allocated/reserved bytes |
|---|---:|---:|---:|---:|---|

| native-R-multiple | 142.395 | 147.507 | 1254932 | 6832740 | None/None |

| native-R-single | 134.671 | 139.964 | 1235608 | 6801268 | None/None |

| native-anchor | 155.034 | 159.931 | 1236436 | 6802636 | None/None |

| train-R-multiple | 69.287 | 69.287 | 13504520 | 13504520 | 20999968256/22011707392 |

| train-R-single | 71.939 | 71.939 | 13472208 | 13472208 | 20996906496/22106079232 |


Both independent arms reloaded exact anchor07ea98e9 and passed runtime588 FP32 language DoRA+2FP32[1004,2048] delta, frozen BF16 FA2 base/vision/projector, zero dropout and checkpointingOFF checks. Fresh AdamW/constant LRs1e-5+5e-6/betas.9,.999/epsilon1e-8/weight decay0/clip1 remained fixed. Six equal1/6 losses precede each single clip/step. All losses, gradients and parameters passed finite guards.

R-single: endpoint `/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01/train-R-single/checkpoint-16`, weight identity `ff2cee614a1d30c2933a76df5933b731ee5ca6b4b3ced72e7636969ea5794a21`. First update mean loss1.35970587, preclip norm93.1567307; update16 mean loss4.46154718e-06, preclip norm0.00207804935. Both take exactly16 updates.

R-multiple: endpoint `/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01/train-R-multiple/checkpoint-16`, weight identity `7ab33db4db3d60b6d6e5037884cb632be428dc133882b28897e615b09877408f`. First update mean loss1.39115045, preclip norm91.9184189; update16 mean loss5.1085827e-06, preclip norm0.0026078925. Both take exactly16 updates.


Native literal legality: each endpoint repaired6/6 baseline-illegal and retained4/4 baseline-legal contexts, with0 legality losses. Original contexts:1/1 repaired and1/1 retained; training-neighbors:3/3 repaired and1/1 retained; held-out neighbors:2/2 repaired and2/2 retained. The351017 original is baseline-legal, so its repair denominator is0. The frozen original-error contrast is therefore observed at7511 only. Rounded zero native margins are ties; emitted tokens decide legality. HF anchor disagreements remain visible and do not replace native outcomes.


| Context | Anchor bin/legal/native margin | Single bin/legal/native margin | Multiple bin/legal/native margin | HF margins anchor/single/multiple |
|---|---|---|---|---|

| 7511-626/+0 | 999/no/-0.125000 | 981/yes/13.375000 | 981/yes/13.562500 | -0.125000/13.375000/13.437500 |

| 7511-626/-1 | 999/no/-0.125000 | 981/yes/13.500000 | 981/yes/13.625000 | -0.125000/13.937500/13.625000 |

| 7511-626/+1 | 982/yes/0.000000 | 981/yes/13.250000 | 981/yes/13.500000 | -0.250000/13.437500/13.375000 |

| 7511-626/-2 | 982/yes/0.000000 | 981/yes/12.875000 | 981/yes/13.625000 | -0.250000/13.250000/13.437500 |

| 7511-626/+2 | 999/no/-0.125000 | 981/yes/13.375000 | 981/yes/13.562500 | 0.000000/13.875000/13.250000 |

| 351017-1507/+0 | 966/yes/0.000000 | 966/yes/12.875000 | 966/yes/13.500000 | 0.000000/12.750000/13.125000 |

| 351017-1507/-1 | 999/no/-0.250000 | 966/yes/12.875000 | 966/yes/13.375000 | 0.000000/12.750000/13.375000 |

| 351017-1507/+1 | 999/no/-0.125000 | 966/yes/13.125000 | 966/yes/13.250000 | -0.125000/13.000000/13.250000 |

| 351017-1507/-2 | 999/no/-0.125000 | 966/yes/13.125000 | 966/yes/13.250000 | -0.125000/13.000000/13.375000 |

| 351017-1507/+2 | 966/yes/0.000000 | 966/yes/13.125000 | 966/yes/13.500000 | 0.000000/12.875000/13.375000 |


Full-vocabulary legal mass, token IDs, exact unrounded margins, HF logit identities and prefix identities remain in the terminal candidate and raw score files. Endpoint native margins span12.875–13.500(single) and13.250–13.625(multiple); these are separate from the HF diagnostics. These observations do not establish ranking-over-mass or certified owner-invariant histories.


Natural outputs, each arm versus the fresh anchor: category-aware known-owner counts single gained31/lost29/retained245, multiple gained24/lost30/retained244. Geometry-only owner counts single34/30/245, multiple26/30/245. Owner identities and matching details remain at the raw terminal locator. Labels18/570 are evaluator-only; unmatched remains neutral.


| Burden | Anchor | Single | Multiple | Single delta | Multiple delta |
|---|---:|---:|---:|---:|---:|

| caps | 2 | 1 | 1 | -1 | -1 |

| category_disagreements | 1 | 3 | 3 | 2 | 2 |

| eos | 16 | 17 | 17 | 1 | 1 |

| generated_tokens | 9922 | 7907 | 8621 | -2015 | -1301 |

| geometry_invalid | 430 | 151 | 159 | -279 | -271 |

| literal_complete_repeats | 502 | 202 | 227 | -300 | -275 |

| literal_valid_repeats | 89 | 88 | 112 | -1 | 23 |

| malformed | 2 | 1 | 1 | -1 | -1 |

| near_repeat_occurrence_pairs | 92 | 126 | 191 | 34 | 99 |

| unmatched | 377 | 426 | 505 | 49 | 128 |

| valid_rows | 652 | 705 | 776 | 53 | 124 |


Per-image category and geometry-only owner transitions (G/L/R), with every burden delta relative to anchor. Complete absolute burdens and stop reasons are in candidate `per_image`; full owner IDs remain in `package-01/complete.json`.


### R-single vs anchor


| Image | Category G/L/R | Geometry G/L/R | Tokens | Invalid | Complete repeats | Valid repeats | Near pairs | Unmatched | Valid rows | Category disagree | Malformed | Cap/EOS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

| 1584 | 1/1/9 | 1/1/9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

| 2299 | 2/3/31 | 2/3/31 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0/0 |

| 2685 | 0/1/11 | 0/1/11 | 19 | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0/0 |

| 4134 | 0/2/18 | 0/2/18 | -27 | 0 | 0 | 0 | -1 | -1 | -3 | 0 | 0 | 0/0 |

| 5001 | 3/1/15 | 3/1/15 | 2 | 0 | 0 | 0 | -1 | -2 | 0 | 0 | 0 | 0/0 |

| 6040 | 0/0/10 | 0/0/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

| 7116 | 0/0/4 | 0/0/4 | 9 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0/0 |

| 7511 | 3/1/8 | 4/1/8 | 0 | -104 | -77 | 49 | 88 | 101 | 104 | 1 | 0 | 0/0 |

| 10707 | 0/0/14 | 1/1/14 | 27 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 0/0 |

| 13348 | 1/0/3 | 1/0/3 | 144 | 0 | 4 | 4 | 18 | 15 | 16 | 0 | 0 | 0/0 |

| 13923 | 0/0/11 | 0/0/11 | -9 | 0 | 0 | 0 | 0 | -1 | -1 | 0 | 0 | 0/0 |

| 14038 | 2/1/6 | 2/1/6 | 18 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0/0 |

| 14439 | 1/1/21 | 1/1/21 | -1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

| 16228 | 2/5/23 | 2/5/23 | 9 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | 0/0 |

| 309264 | 3/1/0 | 3/1/0 | -18 | -1 | 0 | 0 | 0 | -3 | -1 | 0 | 0 | 0/0 |

| 351017 | 3/6/9 | 3/6/9 | -2173 | -174 | -227 | -54 | -70 | -67 | -70 | 0 | -1 | -1/1 |

| 417044 | 5/5/27 | 6/5/27 | -60 | 0 | 0 | 0 | 0 | -7 | -6 | 1 | 0 | 0/0 |

| 477415 | 5/1/25 | 5/1/25 | 45 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0/0 |


### R-multiple vs anchor


| Image | Category G/L/R | Geometry G/L/R | Tokens | Invalid | Complete repeats | Valid repeats | Near pairs | Unmatched | Valid rows | Category disagree | Malformed | Cap/EOS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

| 1584 | 2/0/10 | 2/0/10 | 0 | 0 | 0 | 0 | 0 | -2 | 0 | 0 | 0 | 0/0 |

| 2299 | 0/2/32 | 0/2/32 | -9 | 0 | 0 | 0 | 0 | 1 | -1 | 0 | 0 | 0/0 |

| 2685 | 0/0/12 | 0/0/12 | 28 | 0 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 0/0 |

| 4134 | 0/3/17 | 0/3/17 | -54 | 0 | 0 | 0 | -1 | -3 | -6 | 0 | 0 | 0/0 |

| 5001 | 2/2/14 | 2/2/14 | 20 | 0 | 0 | 0 | -1 | 2 | 2 | 0 | 0 | 0/0 |

| 6040 | 0/0/10 | 0/0/10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

| 7116 | 0/0/4 | 0/0/4 | 9 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0/0 |

| 7511 | 3/1/8 | 4/1/8 | 0 | -104 | -76 | 51 | 128 | 101 | 104 | 1 | 0 | 0/0 |

| 10707 | 1/0/14 | 1/0/15 | 10 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0/0 |

| 13348 | 1/0/3 | 1/0/3 | 144 | 0 | 4 | 4 | 15 | 15 | 16 | 0 | 0 | 0/0 |

| 13923 | 0/0/11 | 0/0/11 | -9 | 0 | 0 | 0 | 0 | -1 | -1 | 0 | 0 | 0/0 |

| 14038 | 2/1/6 | 2/1/6 | 18 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0/0 |

| 14439 | 2/2/20 | 2/2/20 | -1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |

| 16228 | 1/6/22 | 1/6/22 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0/0 |

| 309264 | 2/1/0 | 2/1/0 | 279 | -1 | 4 | 4 | 1 | 31 | 32 | 0 | 0 | 0/0 |

| 351017 | 2/5/10 | 2/5/10 | -1713 | -166 | -207 | -36 | -43 | -24 | -27 | 0 | -1 | -1/1 |

| 417044 | 4/6/26 | 5/6/26 | -50 | 0 | 0 | 0 | 0 | -4 | -5 | 1 | 0 | 0/0 |

| 477415 | 2/1/25 | 2/1/25 | 27 | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 0 | 0/0 |


Natural validity and complete-repeat burdens decreased in both arms, while owner losses, category disagreements, unmatched rows and near-repeat pairs remain visible. This is a measured selected conditional/natural contrast awaiting lead interpretation and acceptance. It makes no physical, population, global determinism or ranking-over-mass claim. No next unit is scheduled.


Execution commands (cwd `/data/CoordExp/.worktrees/greedy-prefix-native-01`), actual exits0:

```bash
python -m probes.prefix_exposure_ranking package --contract /data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/released-contract-01.json --contract-sha256 0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b --output /data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01
python -m probes.prefix_exposure_ranking readback --contract /data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/released-contract-01.json --contract-sha256 0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b --output /data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/package-01
```

Release SHA256 `0fe3deac8f3eff7ec542686bdda8c7673aeee91b993bd3ca1628ac2233bf441b`; separate lead-release SHA256 `169e146d67d602db14f571bfd014c1d1f407b682e4f335e918db4152116d8d66`. Accepted CPU candidate02 SHA256 `9c051db4439038bb1126776fe5c2116cb71c608cde0974c7365dc1602417e11b`. Package session45343/PID1687760; explicit readback session48250. Logs `package-01.log`, `readback-01.log` and five execution-local phase logs are hashed in the terminal candidate. All phase exits0: native-anchor, train-R-single, native-R-single, train-R-multiple, native-R-multiple.

Canonical record consumer `python -m scripts.check_research_knowledge check` exited0:334 entries/10 current/167 claim references; `outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/research-consumer-terminal-01.log` SHA256 `f9e734338f18ee0e7fc441240e1359066d1360bf1cde3a7420176ae8c0792ddd`. Consumer scope is record integrity, separate from scientific acceptance.
