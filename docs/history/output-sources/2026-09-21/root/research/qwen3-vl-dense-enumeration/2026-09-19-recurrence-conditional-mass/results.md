# Lane C result: conditional numerical-repeat probability

Status: **candidate; root acceptance pending**.

The fixed original-policy native run completed all 45 final-panel states and 11,520 short ancestral draws. The primary event is a complete legal bbox plus terminator whose four coordinate bins fall within the frozen same-description historical `<=8`-bin union. This is a conditional numerical-repeat probability; `1-q` is not new-owner probability and no physical-identity claim is made.

## Bound inputs and denominators

- Panel: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json`; SHA256 `005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`; status `final_frozen`.
- Sources: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-sources.json`; SHA256 `8f7b43d2d9f4014bb42024324543eea27bad222b6c8432d3e44c93b45743fe9e`.
- Admitted/analyzable states: **45/45**; runtime failures during the scientific run: **0**.
- Draws: **11520 = 45 x 256**; unique image identities: **23**.
- Panel strata: **21 failures**, **24 healthy/non-recurrent proxies**; models: **19 untied**, **26 tied**.
- Greedy saved next-row audit at exact `source_row.end`: `29` legal repeats, `16` legal non-repeats, `0` unknown.
- The saved greedy audit is descriptive evidence at the conditioned prefix; it is separate from sampled q.

## Runtime and sampler entry

- Scientific run manifest: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/run-manifest.json`.
- Scientific model forwards: **7290**; vision forwards: **1530**; native generation calls: **1440**.
- Summed per-state elapsed time: **7708.110 s (2.14 h)** on logical GPU6 (`CUDA_VISIBLE_DEVICES=6`). GPU-kernel seconds were not separately instrumented; this is the recorded model-run wall-time bound.
- Qualification receipts account for **13** model and **13** vision forwards across the preserved pre-boundary, pre-batch-4, pre-stream and final attempts; the final successful qualification is **5/5** and all qualification attempts produced 0 scientific draws. Failed launch guards have measured **0** forwards; one source-binding attempt loaded the model and failed before a sampler entry, so no forward count is claimed for it.
- First counted sampler chunk: state `untied-885-failure`, batch **8**, seed `2979425262827546311`, offsets `[0, 1, 2, 3, 4, 5, 6, 7]`.
- Entry settings: T=1.0, top_p=1.0, top_k=0, RP=1.0, model defaults disabled, grammar constraint `None`, retry `False`.
- First chunk retained raw tokens and finite token logprobs: **40** values; grammar escapes observed: **0** (absence in this chunk does not imply zero probability).

## Overall sampled outcomes

| Outcome | Count | Rate |
|---|---:|---:|
| `legal_repeat` | 441 | 3.83% |
| `legal_nonrepeat` | 10119 | 87.84% |
| `invalid_geometry_near_repeat` | 4 | 0.03% |
| `invalid_extent` | 939 | 8.15% |
| `grammar_escape` | 17 | 0.15% |
| `early_eos` | 0 | 0.00% |
| `short_or_length` | 0 | 0.00% |
| **q legal repeat** | **441** | **3.83% descriptive captured-draw fraction** |

The pooled fraction above is a descriptive summary of these captured draws;
draws from heterogeneous states are not treated as iid observations from one
common q. No pooled or cross-image confidence interval is reported. Uncertainty
for the heterogeneous panel is represented by the per-state Wilson intervals
below, while the image table is a one-image-one-unit descriptive summary.

- Exact-bin invalid recurrence subset: **0** draws; invalid-geometry near-repeat (`<=8`) count: **4**.
- All non-events remain format, invalid-geometry, or legal non-repeat outcomes. They are not relabeled as owner changes.

## Prospective proxy conditioning audit

The prospective proxy cohort has ten tied states. Seven states use a
next-row description different from the source-row description:
`185502`, `119636`, `477785`, `328462`, `259312`, `523815` and `354063`.
For each of these seven states, the conditioned description's historical
same-description union is empty (`repeat_union_size=0`), so q is structurally
zero for that state; this is a conditioning/denominator property and must not
be interpreted as model suppression. The remaining prospective proxies
`59540`, `171564` and `131490` use matching descriptions and each has a
non-empty union of size one.

The raw draw records leave `source_policy` null, including these new states.
The canonical crosswalk is preserved in `state-bindings.json` and the frozen
panel/source manifests: these states are tied-original, split-aware
`source_example_id` bindings. The null raw field is a metadata omission, not a
new policy stratum or an inferred identity.

## Model, source and panel strata

| Panel stratum | Model | Cohort | States | Images | Draws | Legal repeats | q (descriptive draw fraction) | Greedy repeat states |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| failure | tied | existing | 5 | 5 | 1280 | 41 | 0.0320 | 5 |
| failure | tied | new_train | 4 | 4 | 1024 | 46 | 0.0449 | 4 |
| failure | untied | existing | 6 | 6 | 1536 | 98 | 0.0638 | 6 |
| failure | untied | new_train | 6 | 6 | 1536 | 110 | 0.0716 | 6 |
| proxy | tied | existing | 7 | 7 | 1792 | 48 | 0.0268 | 3 |
| proxy | tied | new_train | 10 | 10 | 2560 | 0 | 0.0000 | 1 |
| proxy | untied | existing | 7 | 7 | 1792 | 98 | 0.0547 | 4 |

The panel is intentionally imbalanced: all ten prospective proxies are tied; failure/proxy counts are not a matched causal comparison. Results retain model, source condition and cohort strata.

## Image-level summaries

| Source image identity | Numeric ID | States | Models | Panel kinds | Draws | Repeats | Mean state q | Greedy repeat states |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `coco2017_train_000000059540` | 59540 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000119636` | 119636 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000131490` | 131490 | 1 | tied | proxy | 256 | 0 | 0.0000 | 1 |
| `coco2017_train_000000169872` | 169872 | 2 | tied,untied | failure | 512 | 18 | 0.0352 | 2 |
| `coco2017_train_000000171564` | 171564 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000185502` | 185502 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000196924` | 196924 | 1 | untied | failure | 256 | 25 | 0.0977 | 1 |
| `coco2017_train_000000259312` | 259312 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000269858` | 269858 | 2 | tied,untied | failure | 512 | 19 | 0.0371 | 2 |
| `coco2017_train_000000301827` | 301827 | 2 | tied,untied | failure | 512 | 27 | 0.0527 | 2 |
| `coco2017_train_000000309264` | 309264 | 2 | tied,untied | proxy | 512 | 0 | 0.0000 | 0 |
| `coco2017_train_000000322768` | 322768 | 2 | tied,untied | failure | 512 | 49 | 0.0957 | 2 |
| `coco2017_train_000000328462` | 328462 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000354063` | 354063 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000417044` | 417044 | 4 | tied,untied | failure,proxy | 1024 | 15 | 0.0146 | 2 |
| `coco2017_train_000000477785` | 477785 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_train_000000505967` | 505967 | 1 | untied | failure | 256 | 18 | 0.0703 | 1 |
| `coco2017_train_000000523815` | 523815 | 1 | tied | proxy | 256 | 0 | 0.0000 | 0 |
| `coco2017_val_000000000632` | 632 | 4 | tied,untied | failure,proxy | 1024 | 147 | 0.1436 | 4 |
| `coco2017_val_000000000885` | 885 | 4 | tied,untied | failure,proxy | 1024 | 36 | 0.0352 | 4 |
| `coco2017_val_000000005586` | 5586 | 4 | tied,untied | failure,proxy | 1024 | 24 | 0.0234 | 3 |
| `coco2017_val_000000007511` | 7511 | 3 | tied,untied | failure,proxy | 768 | 25 | 0.0326 | 2 |
| `coco2017_val_000000014038` | 14038 | 4 | tied,untied | failure,proxy | 1024 | 38 | 0.0371 | 3 |

Each image summary gives one image identity one unit; image identities are
split-aware and never keyed only by numeric COCO ID. These are descriptive
image-level aggregates; no cross-image interval or population-rate claim is
made.

- opener logprob diagnostic across 45 states: mean `-0.093791`, median `-0.000280`, range `[-0.666208, -0.000018]`; these are not included in q.
- EOS logprob diagnostic across 45 states: mean `-7.086797`, median `-8.183380`, range `[-10.942634, -0.721067]`; these are not included in q.
- description-entry logprob diagnostic across 45 states: mean `-0.102979`, median `-0.068270`, range `[-0.412029, -0.000958]`; these are not included in q.

## Per-state q and greedy audit

| State | Source identity | Model | Panel stratum | Cohort | Greedy saved event | q | Wilson 95% | Repeat | Legal nonrepeat | Invalid near | Invalid extent | Grammar |
|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `untied-885-failure` | `coco2017_val_000000000885` | untied | failure | existing | yes | 0.0273 | [0.0133, 0.0554] | 7 | 230 | 0 | 18 | 1 |
| `untied-885-healthy` | `coco2017_val_000000000885` | untied | proxy | existing | yes | 0.0352 | [0.0186, 0.0655] | 9 | 225 | 0 | 21 | 1 |
| `tied-885-failure` | `coco2017_val_000000000885` | tied | failure | existing | yes | 0.0156 | [0.0061, 0.0395] | 4 | 234 | 0 | 18 | 0 |
| `tied-885-healthy` | `coco2017_val_000000000885` | tied | proxy | existing | yes | 0.0625 | [0.0388, 0.0991] | 16 | 227 | 0 | 11 | 2 |
| `untied-5586-failure` | `coco2017_val_000000005586` | untied | failure | existing | yes | 0.0000 | [0.0000, 0.0148] | 0 | 247 | 0 | 9 | 0 |
| `untied-5586-healthy` | `coco2017_val_000000005586` | untied | proxy | existing | no | 0.0156 | [0.0061, 0.0395] | 4 | 245 | 0 | 7 | 0 |
| `tied-5586-failure` | `coco2017_val_000000005586` | tied | failure | existing | yes | 0.0273 | [0.0133, 0.0554] | 7 | 235 | 0 | 13 | 1 |
| `tied-5586-healthy` | `coco2017_val_000000005586` | tied | proxy | existing | yes | 0.0508 | [0.0299, 0.0849] | 13 | 233 | 0 | 8 | 2 |
| `untied-7511-failure` | `coco2017_val_000000007511` | untied | failure | existing | yes | 0.0664 | [0.0419, 0.1038] | 17 | 195 | 0 | 44 | 0 |
| `untied-7511-healthy` | `coco2017_val_000000007511` | untied | proxy | existing | yes | 0.0234 | [0.0108, 0.0502] | 6 | 191 | 0 | 59 | 0 |
| `tied-7511-healthy` | `coco2017_val_000000007511` | tied | proxy | existing | no | 0.0078 | [0.0021, 0.0280] | 2 | 237 | 0 | 17 | 0 |
| `untied-14038-failure` | `coco2017_val_000000014038` | untied | failure | existing | yes | 0.0586 | [0.0358, 0.0944] | 15 | 191 | 0 | 50 | 0 |
| `untied-14038-healthy` | `coco2017_val_000000014038` | untied | proxy | existing | yes | 0.0117 | [0.0040, 0.0339] | 3 | 211 | 0 | 42 | 0 |
| `tied-14038-failure` | `coco2017_val_000000014038` | tied | failure | existing | yes | 0.0430 | [0.0242, 0.0753] | 11 | 193 | 0 | 52 | 0 |
| `tied-14038-healthy` | `coco2017_val_000000014038` | tied | proxy | existing | no | 0.0352 | [0.0186, 0.0655] | 9 | 223 | 0 | 24 | 0 |
| `untied-632-failure` | `coco2017_val_000000000632` | untied | failure | existing | yes | 0.2266 | [0.1795, 0.2817] | 58 | 180 | 0 | 18 | 0 |
| `untied-632-healthy` | `coco2017_val_000000000632` | untied | proxy | existing | yes | 0.2695 | [0.2189, 0.3270] | 69 | 170 | 0 | 17 | 0 |
| `tied-632-failure` | `coco2017_val_000000000632` | tied | failure | existing | yes | 0.0586 | [0.0358, 0.0944] | 15 | 232 | 0 | 9 | 0 |
| `tied-632-healthy` | `coco2017_val_000000000632` | tied | proxy | existing | yes | 0.0195 | [0.0084, 0.0449] | 5 | 231 | 0 | 18 | 2 |
| `untied-417044-failure` | `coco2017_train_000000417044` | untied | failure | existing | yes | 0.0039 | [0.0007, 0.0218] | 1 | 239 | 0 | 14 | 2 |
| `untied-417044-healthy` | `coco2017_train_000000417044` | untied | proxy | existing | no | 0.0273 | [0.0133, 0.0554] | 7 | 239 | 0 | 8 | 2 |
| `tied-417044-failure` | `coco2017_train_000000417044` | tied | failure | existing | yes | 0.0156 | [0.0061, 0.0395] | 4 | 242 | 0 | 10 | 0 |
| `tied-417044-healthy` | `coco2017_train_000000417044` | tied | proxy | existing | no | 0.0117 | [0.0040, 0.0339] | 3 | 248 | 0 | 5 | 0 |
| `untied-309264-healthy` | `coco2017_train_000000309264` | untied | proxy | existing | no | 0.0000 | [0.0000, 0.0148] | 0 | 230 | 0 | 25 | 1 |
| `tied-309264-healthy` | `coco2017_train_000000309264` | tied | proxy | existing | no | 0.0000 | [0.0000, 0.0148] | 0 | 221 | 0 | 35 | 0 |
| `tied-train-169872-failure` | `coco2017_train_000000169872` | tied | failure | new_train | yes | 0.0273 | [0.0133, 0.0554] | 7 | 220 | 0 | 29 | 0 |
| `tied-train-322768-failure` | `coco2017_train_000000322768` | tied | failure | new_train | yes | 0.1094 | [0.0768, 0.1535] | 28 | 218 | 0 | 10 | 0 |
| `tied-train-301827-failure` | `coco2017_train_000000301827` | tied | failure | new_train | yes | 0.0117 | [0.0040, 0.0339] | 3 | 199 | 0 | 54 | 0 |
| `tied-train-269858-failure` | `coco2017_train_000000269858` | tied | failure | new_train | yes | 0.0312 | [0.0159, 0.0604] | 8 | 223 | 0 | 25 | 0 |
| `untied-train-169872-failure` | `coco2017_train_000000169872` | untied | failure | new_train | yes | 0.0430 | [0.0242, 0.0753] | 11 | 197 | 1 | 47 | 0 |
| `untied-train-322768-failure` | `coco2017_train_000000322768` | untied | failure | new_train | yes | 0.0820 | [0.0543, 0.1221] | 21 | 210 | 0 | 24 | 1 |
| `untied-train-301827-failure` | `coco2017_train_000000301827` | untied | failure | new_train | yes | 0.0938 | [0.0638, 0.1357] | 24 | 153 | 3 | 76 | 0 |
| `untied-train-196924-failure` | `coco2017_train_000000196924` | untied | failure | new_train | yes | 0.0977 | [0.0670, 0.1402] | 25 | 199 | 0 | 32 | 0 |
| `untied-train-505967-failure` | `coco2017_train_000000505967` | untied | failure | new_train | yes | 0.0703 | [0.0449, 0.1084] | 18 | 215 | 0 | 23 | 0 |
| `untied-train-269858-failure` | `coco2017_train_000000269858` | untied | failure | new_train | yes | 0.0430 | [0.0242, 0.0753] | 11 | 204 | 0 | 41 | 0 |
| `tied-train-185502-healthy` | `coco2017_train_000000185502` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 256 | 0 | 0 | 0 |
| `tied-train-119636-healthy` | `coco2017_train_000000119636` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 252 | 0 | 4 | 0 |
| `tied-train-477785-healthy` | `coco2017_train_000000477785` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 256 | 0 | 0 | 0 |
| `tied-train-328462-healthy` | `coco2017_train_000000328462` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 256 | 0 | 0 | 0 |
| `tied-train-259312-healthy` | `coco2017_train_000000259312` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 248 | 0 | 8 | 0 |
| `tied-train-523815-healthy` | `coco2017_train_000000523815` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 254 | 0 | 2 | 0 |
| `tied-train-59540-healthy` | `coco2017_train_000000059540` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 255 | 0 | 1 | 0 |
| `tied-train-171564-healthy` | `coco2017_train_000000171564` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 249 | 0 | 7 | 0 |
| `tied-train-131490-healthy` | `coco2017_train_000000131490` | tied | proxy | new_train | yes | 0.0000 | [0.0000, 0.0148] | 0 | 250 | 0 | 4 | 2 |
| `tied-train-354063-healthy` | `coco2017_train_000000354063` | tied | proxy | new_train | no | 0.0000 | [0.0000, 0.0148] | 0 | 256 | 0 | 0 | 0 |

## Artifacts and CPU acceptance

- Raw draws: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/draws.jsonl`; SHA256 `bd42ccad21cdfed5f1e34cc4a9f7e05c45662afa9191df2c1140237250c644a1`.
- CPU reduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/reduction.json`; SHA256 `1f182ed94662323f9aaded6affd6ff8438f7a4ec57fb3851e5dedfb7647e0e01`.
- State bindings: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/state-bindings.json`.
- First sampler receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/sampler-entry.json`.
- Preserved pre-entry failures: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/launch-failures.json`; empty pre-entry draw file `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/draws.failed-preentry.empty.jsonl`.
- CPU commands:
  - `python -m probes.training_set_completion.recurrence_mass.reduce --self-test`
  - `python -m probes.training_set_completion.recurrence_mass.reduce --draws /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/draws.jsonl --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/reduction.json`

## Interpretation limits

- q conditions on the serialized description and row opener/box-start prefix. Description-entry likelihood is reported separately; free-description probability is not in q.
- The event is numerical membership in a frozen same-description `<=8`-bin union. It is not total physical duplicate probability.
- The saved greedy next-row event is a native recurrence audit, not a guarantee that an ancestral draw continues the same row.
- `1-q` is not new-owner probability. Unknown, grammar, EOS, invalid geometry and legal non-repeat outcomes retain their original meanings.
- Wilson intervals are per-state draw intervals only. Pooled and stratum q values are descriptive captured-draw fractions across heterogeneous states, not image-level uncertainty estimates. Image summaries are descriptive one-image-one-unit aggregates; no causal pooling across tied/untied or failure/proxy strata is performed.
- Root must independently verify the raw reducer and decide acceptance; this lane makes no policy promotion or successor launch.
