# Lane B candidate: visual position × history position

Lifecycle is `candidate`; root acceptance, policy promotion, and successor launch remain outside this worker boundary. The frozen question is whether numerical recurrence follows moved visual content, moved coordinate history, or fixed canvas preferences. The primary comparison is common `00` against coherent `11-`/`11+`, after inverse mapping output boxes to source bins. `10` and `01` are mismatch diagnostics.

## Frozen source and geometry

- Shared panel: `005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`; 45 states (21 failure, 24 proxy). Two states (`tied-417044-failure`, `untied-417044-failure`) reuse the accepted 14-cell pilot; the other 43 states have current native runtime receipts.
- Each cell copies the unchanged scene into a fixed black common canvas with no resize/interpolation, 32 px grid, round-half-even coordinate remapping, and ±128 px horizontal displacement relative to the centered 00 origin (00 uses visual/history offset 128 px; 10−/10+ use visual 0/256 with history 128; 01−/01+ use visual 128 with history 0/256; 11−/11+ use 0/0 and 256/256). History boxes receive the same affine map as the visual content. Cells are `00`, `10-`, `10+`, `01-`, `01+`, `11-`, `11+`; `00` is shared between signs.
- Source rows, including invalid source rows, remain in the bound histories. In the 45-state panel there are 503 source history rows: 3 invalid source rows across 3 states; 5 rounding-validity-change entries in `tied-7511-healthy`; mapped invalid rows and order checks are retained in the JSON result.

## Admission and primary comparison

| panel kind | states | admitted 00 + all 7 cells | 00 recurrent | 11− recurrent | 11+ recurrent | both 11 signs | either 11 sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| failure | 21 | 14 | 14 | 10 | 12 | 9 | 13 |
| healthy | 24 | 23 | 5 | 9 | 5 | 5 | 9 |

The declared denominator is shown in the first state column; the admitted column excludes common-00 admission holds. Failure states all had recurrent 00 among the 14 admitted states. Coherent 11 recurrence persists for both signs in 9/14 admitted failure states and 5/23 admitted proxy states. Predicate agreement with 00 is 10/14 for `11-` and 12/14 for `11+` in failure states; proxies are 19/23 and 21/23. The mapped repeat-anchor shift summaries among cells with anchors are `11−`: median `[1.5, 26.5]` source bins (n=15), `11+`: median `[1.0, -60.5]` (n=16). These are numerical mapped shifts, not physical owner identity.

## Mismatch diagnostics

| cell | present | recurrent | exact | near | alternative recurrence versus 00 |
|---|---:|---:|---:|---:|---:|
| 10- | 37 | 11 | 10 | 11 | 1 |
| 10+ | 37 | 13 | 8 | 13 | 0 |
| 01- | 37 | 14 | 10 | 14 | 0 |
| 01+ | 37 | 6 | 6 | 6 | 0 |

These cells deliberately break visual/history coherence and are not detection-quality controls. Anchor shifts are summarized in the result JSON by cell; missing anchors are retained as null rather than imputed.

## Per-cell recurrence and geometry totals

| cell | present | recurrent | exact | near | valid rows | invalid rows | malformed rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | 45 | 19 | 13 | 19 | 713 | 80 | 14 |
| 10- | 37 | 11 | 10 | 11 | 346 | 83 | 7 |
| 10+ | 37 | 13 | 8 | 13 | 611 | 24 | 9 |
| 01- | 37 | 14 | 10 | 14 | 697 | 59 | 17 |
| 01+ | 37 | 6 | 6 | 6 | 453 | 12 | 6 |
| 11- | 37 | 19 | 18 | 19 | 646 | 44 | 14 |
| 11+ | 37 | 17 | 14 | 17 | 617 | 50 | 9 |

All 267 free continuations emitted the EOS token; 191 ended below the 32-row bound (`natural_eos`) and 76 reached the producer row bound (`row_bound_eos`). None reached the 512-token cap. The raw and reduced receipts retain longest runs, descriptions, source-mapped boxes, pair-edge counts, output token counts, border rows, and out-of-bounds rows for each cell. Exact and near predicates are kept separate; long runs therefore have quadratic pair-edge counts.

## Opener/EOS and forced-description diagnostics

| cell | n | median opener log p | median EOS log p | median opener−EOS | median coordinate-family mass | median forced moved−old x1 log p |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 45 | -0.000216 | -8.454578 | 8.454363 | 6.076e-14 | 0.000000 |
| 10- | 37 | -0.001236 | -6.698269 | 6.697033 | 6.168e-13 | 0.000000 |
| 10+ | 37 | -0.000112 | -9.100419 | 9.100307 | 2.187e-14 | 0.000000 |
| 01- | 37 | -0.000118 | -9.048973 | 9.048855 | 4.140e-14 | 1.235125 |
| 01+ | 37 | -0.001857 | -6.290899 | 6.289041 | 4.307e-13 | 3.468387 |
| 11- | 37 | -0.000317 | -8.062466 | 8.062149 | 4.926e-14 | 6.879394 |
| 11+ | 37 | -0.000750 | -7.198788 | 7.198038 | 2.253e-13 | 8.532781 |

The opener and coordinate-family quantities are full-vocabulary values. The forced-description x1 old/moved window measurement is conditional and is not free continuation quality; no window renormalization is used. Full top competitors, coordinate distributions, and old/moved logits/log probabilities remain in each runtime receipt.

## Source and runtime strata

| source group | states | failure | proxy | tied | untied |
|---|---:|---:|---:|---:|---:|
| val-extra | 12 | 6 | 6 | 6 | 6 |
| refined-01 | 3 | 1 | 2 | 1 | 2 |
| refined-02 | 4 | 2 | 2 | 2 | 2 |
| refined-03 | 4 | 2 | 2 | 2 | 2 |
| fresh-18 | 2 | 0 | 2 | 1 | 1 |
| new-08 | 2 | 2 | 0 | 1 | 1 |
| new-18 | 2 | 2 | 0 | 1 | 1 |
| new-17 | 2 | 2 | 0 | 1 | 1 |
| new-14 | 2 | 2 | 0 | 1 | 1 |
| new-10 | 1 | 1 | 0 | 0 | 1 |
| new-27 | 1 | 1 | 0 | 0 | 1 |
| new-09 | 2 | 0 | 2 | 2 | 0 |
| new-06 | 1 | 0 | 1 | 1 | 0 |
| new-25 | 1 | 0 | 1 | 1 | 0 |
| new-19 | 1 | 0 | 1 | 1 | 0 |
| new-13 | 1 | 0 | 1 | 1 | 0 |
| new-28 | 1 | 0 | 1 | 1 | 0 |
| new-02 | 1 | 0 | 1 | 1 | 0 |
| new-07 | 1 | 0 | 1 | 1 | 0 |
| new-20 | 1 | 0 | 1 | 1 | 0 |

Model/policy/runtime/source identity is retained per state. Tied versus untied is a package comparison; it is not an untie-only estimate. Existing mature records, feedback selections, and new-cohort saved `image_plan` receipts are bound by path and SHA in the state summaries.

## Excluded or no-longer-recurrent states

The frozen admission rule generated no transformed cells for these eight states; this is an explicit denominator outcome, not a replacement selection:

| state | kind | model | source group | image | valid / complete rows | source invalid |
|---|---|---|---|---:|---:|---:|
| untied-885-failure | failure | untied | val-extra | 885 | 10 / 10 | 0 |
| tied-885-failure | failure | tied | val-extra | 885 | 10 / 10 | 0 |
| untied-5586-failure | failure | untied | val-extra | 5586 | 23 / 23 | 0 |
| tied-5586-failure | failure | tied | val-extra | 5586 | 23 / 26 | 3 |
| untied-7511-failure | failure | untied | refined-01 | 7511 | 18 / 21 | 3 |
| tied-7511-healthy | healthy | tied | refined-01 | 7511 | 0 / 0 | 0 |
| untied-train-196924-failure | failure | untied | new-10 | 196924 | 0 / 0 | 0 |
| untied-train-505967-failure | failure | untied | new-27 | 505967 | 16 / 16 | 0 |

The 7 failure holds no longer retained the frozen recurrence predicate in common 00. `tied-7511-healthy` is a proxy hold because common 00 emitted no valid row; it is retained as a healthy-control admission outcome.

## Cost and acceptance

- 267 free continuations: 45 common-00 plus 222 transformed cells; 42,270 model forwards and 801 vision forwards; summed producer wall time 4,408.467 s. GPU seconds were not instrumented and remain null; wall time is not relabeled as GPU time.
- Runtime JSON bytes: 56440507; reduced JSON bytes: 10881425. The 40 pre-fix new-cohort attempts (20 first batch, 20 v2 rerun) failed before native forward on missing `image_plan`; they are mechanical qualification attempts and are excluded from the scientific continuation denominator. The corrected 20-state rerun used saved Lane-A planning fields and verified input identity.
- No producer or launch processes remain. No model parameters were mutated.

### Artifacts

- Candidate result: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/result.json`
- Cost receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/cost-receipt.json`
- Job closure: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/job-closure.json`
- Manifest reconciliation: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifest-reconciliation.json`
- CPU reducer: `probes/training_set_completion/recurrence_spatial/reduce.py`
- Saved-output summary reducer: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/summarize_candidate.py`
- Frozen source snapshot: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/source-snapshot.json`

Acceptance commands:

```bash
python3 -m py_compile probes/training_set_completion/recurrence_spatial/producer.py probes/training_set_completion/recurrence_spatial/reduce.py probes/training_set_completion/recurrence_spatial/state_entry.py
python3 -m probes.training_set_completion.recurrence_spatial.reduce --model tied --manifest-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/manifests/tied-train-169872-failure.json --runtime-path /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/runtime/tied-train-169872-failure.json --reduced-path /tmp/recur-spatial-recheck.json
python3 /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-spatial-source/final/summarize_candidate.py --check-only
```

This is a lane candidate. The recurrence and mapped-location results do not identify a unique physical object or causal circuit, and optional grounding-bank absence limits physical interpretation.
