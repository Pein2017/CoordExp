# Two-record routing: accepted shared prediction failure

Lead acceptance, 2026-09-24. The original five-arm production is technically accepted through its separately versioned CPU recovery. The frozen joint scientific prediction fails. [Acceptance](lead-acceptance-v1.json), [independent lead verification](supporting/lead-verification-v1.json), and [worker candidate](candidate-results.md) bind the exact source and all saved vectors.

At original train351017/refined-03 target2 row2, image, two-record length, positions, class tokens, native attention and four-request source batch were fixed. A denotes the broad left foreground person box [0,13,536,999]; F denotes the invalid tiny person proposal [0,0,33,86], whose physical owner remains UNKNOWN. Only historical coordinate IDs changed; complete current rows were freely generated.

| Written history | Free person box | Frozen component |
|---|---|---|
| Native AF and independent sham | [0,0,29,86] | Original row2 replay |
| FF | [1,13,536,999] | Broad-A PASS |
| AA | [205,105,407,499] | Neither region; fragment-F NONPASS |
| FA | [0,0,47,86] | Fragment-F PASS |

Thus FF∧AA∧FA is **NONPASS**, with all five arms technically valid. No exact-token secondary was declared. On the original image, AA identifies the distinct dark-haired man leaning forward at center-left; FF identifies the left foreground person. The AA rectangle is inside the broad A rectangle, so this is not evidence for geometric nonoverlap or a hard exclusion of previously boxed pixels. F/F2/FA owner attribution stays HOLD.

The strongest retained contrast is AF versus FF: identical latest F tokens/current position can yield fragment versus broad person when the older record changes. This rejects latest-written-token sufficiency locally. It does not tell whether current queries directly use the older row or use information already incorporated into the later F row. AF versus FA retains the fragment region, but their full distributions differ; this is not order invariance. The AA result rejects the proposed presence-of-A-implies-fragment account. None of these counterfactuals establishes a natural oscillation, owner ledger, general recurrence mechanism or system-level repair.

The original CPU reader failed before vector loading because JSON lists were compared with Python tuples. The admitted versioned reader accepts exactly the ordered JSON lists and rejects28 donor/order/container mutations. Original producer, raw files and failed receipt remain intact; the CPU sidecar provides the successful readback without another GPU execution. Lead independently rechecked45 vectors, exact source/history/own-prefix inputs, all28 consumed native masks, source traces, sham/companion states, binding hashes and terminal counts. Maximum source trace error was5.340576171875e-5 against2e-4; sham/companion differences were0.

Production:45 model/45 vision forwards and45 emitted tokens, no reuse; parent outer129.19039377570152s, sequence total0.537174243789956GPUh. CPU recovery and lead verification used no model/CUDA calls. The finite GPU unit is closed. A [CPU-only path-discriminator feasibility brief](supporting/context-relay-feasibility-brief.md) considers AF/FF cache recombination, with no GPU launch authority.
