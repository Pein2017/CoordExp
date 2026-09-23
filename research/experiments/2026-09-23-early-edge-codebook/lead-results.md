# Accepted early-edge pilot; not promoted

2026-09-23. Independent lead acceptance of the fixed candidate under [unit](unit.md) and [eight-GPU sequential ruling](lead-ruling-02-eight-gpu-sequential.md). Evidence is accepted; prospective eligibility fails. No successor, extension, commit or publication is authorized.

| Metric | Source | Late three-loss16 | Early-edge16 |
|---|---:|---:|---:|
| Train IoU50 /9519 |5580|6512|6538|
| Train IoU80 /9519 |3515|4635|4660|
| Train clean /1024 |320|457|455|
| Validation IoU50 /2033 |1224|1242|1229|
| Validation IoU80 /2033 |800|750|735|
| Validation clean /256 |104|93|94|

Early injection provides no clear practical advantage over late injection at this dose. Dense training clean falls22 to19/520; retained32 clean falls9 to8. This is not a statistical equivalence claim. Training improvement over source does not identify address-component benefit: there is no matched injection-off training control. Timing, encoding and projection change together; eight versus four training ranks also differ despite matched global packs/objective, so this is not a timing-only causal contrast.

Source-negative to early-positive bad/cap/owner-recurrent/severe counts are55/2/54/7 on training (limits51/10/51/10), and14/2/14/1 on validation (12/2/12/2). Bad and owner recurrence fail both panels. Validation CE/coverage is descriptive, not the veto. Compared with late, early repairs48/15 bad images and introduces42/11 on train/validation; owner recurrence repairs68/14 and introduces50/11. Aggregate parser drops nevertheless rise356 to707 and43 to446, with invalid geometry325 to511 and33 to369. Incidence and severity must stay separate. Exact-row recurrence and annotation-owner recurrence remain distinct; UNKNOWN remains annotation-unmatched, not physically false.

The32-image teacher-forced address diagnostic has exact identity replay. Cyclic tuple shift raises coordinate CE by0.014835/0.065744/0.004724/0.017066 for x1/y1/x2/y2 (679 targets each); description CE rises0.006160 (925 targets). Gain0.050896 and nonzero residual confirm an active route. This establishes sensitivity to this perturbation, not exact numerical address retention, native localization, owner binding or beneficial use. There is no matched generic-corruption control. The hypothesis that merely moving address injection earlier would solve the current bottleneck is weakened; whether coordinate semantics are learned remains unresolved.

## Independent verification and closure

Root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-early-edge-codebook`.
Candidate SHA256: `75ea743f02db9468397e2de6f1489e208c5bbf48b7fd4d3c3762fc2af53fc60f`.
Lead receipt: `lead-acceptance-v1.json` in that root.

All33 candidate bindings and24 current changed paths rehashed. Fresh saved-only reduction of4032 cells (1376new,2656reused) is byte-identical: `31356094d98fb2dd5c1041e1047187b32f370f12dfcd828df794062c2f71e52d`. Fresh1376-cell payload readback is byte-identical: `e8ae7e6d4690f7cdc4c069b1347b95943b063e73ef051ba121e0cfe2c743c4cd`. Lead separately recomputed panel aggregates and diagnostic token-weighted effects. Earlier technical qualification and eight-rank entry acceptance remain applicable.

The failed four-rank fit/grid repair is preserved. The disclosed postproduction test-only fixture correction does not alter executed runtime captures or warrant model replay. Final training completed984 calls,7872 global packs and16384 image presentations. Job intervals independently confirm eight-rank training ended before eight-worker epoch16 inference, which ended before epoch8 inference. All24 jobs terminal; recorded PIDs absent; no owned GPU allocation overlap. Recomputed cost31886.814773GPU-seconds (8.857449GPUh),5705.477577s model wall. No new model calls were made for acceptance.

See [candidate report](candidate-results.md) for per-image severity, reuse provenance, qualification debt and delegation lessons. This closes the package without promotion. Discuss a discriminating next experiment before further training; no automatic architectural or dose search.
