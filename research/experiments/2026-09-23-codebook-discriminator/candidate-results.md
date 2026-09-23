# Injection-off discriminator candidate

2026-09-23. Worker candidate, pending lead acceptance. The fresh injection-off fit completed the fixed seed-1729, 16-epoch, 984-call three-loss schedule from the mature live source. All 1,280 new off cells and 3,840 frozen reused source/late/early cells are present and unmutated. This is a matched ordinary-adaptation control, not a capacity-matched address-component ablation: off has 902 common trainable tensors, late has 903 and early has 904; historical late training used four ranks rather than the new eight-rank topology.

| Panel and condition | Clean | Class-consistent IoU50 | Class-consistent IoU80 | Teacher token CE | Teacher coordinate MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training source | 320/1024 | 5580/9519 | 3515/9519 | 1.66641 | 0.01643 |
| Training off16 | 466/1024 | 6575/9519 | 4730/9519 | 1.16517 | 0.00605 |
| Training late16 | 457/1024 | 6512/9519 | 4635/9519 | 1.17829 | 0.00618 |
| Training early16 | 455/1024 | 6538/9519 | 4660/9519 | 1.19156 | 0.00623 |
| Validation source | 104/256 | 1224/2033 | 800/2033 | 1.65458 | 0.01574 |
| Validation off16 | 90/256 | 1223/2033 | 735/2033 | 1.84727 | 0.01739 |
| Validation late16 | 93/256 | 1242/2033 | 750/2033 | 1.84337 | 0.01750 |
| Validation early16 | 94/256 | 1229/2033 | 735/2033 | 1.83600 | 0.01663 |

Off exceeds early by 11 training clean images and 70 IoU80 matches, and late by 9 clean and 95 IoU80 matches. The paired train IoU80 net improves/regresses/equal images are 189/229/606 for early versus off and 166/233/625 for late versus off. On the dense 520 training images, off has 27 clean and 3319/7914 IoU80 matches versus late 22/3222 and early 19/3244. On the retained 32, off has 8 clean and 200/679 IoU80, late 9/190 and early 8/191. The remaining 992 additions give off 458 clean and 4530/8840 IoU80, late 448/4445 and early 447/4469. These counts use the frozen known-positive annotation proxy; UNKNOWN predictions remain unmatched, not verified false objects.

Neither injected recipe meets the predeclared practical-advantage rule over off: both have lower training clean/IoU80, more paired IoU80 image regressions than improvements, and worse dense clean/IoU80. The strict rule to *deprioritize both* also does not fire because off does not dominate every failure measure. Training off has 105 owner-recurrent images versus early 94; off has 439 parser drops and 422 invalid geometries versus late 356/325, and four severe owner-run images versus late two. Validation coverage and CE are descriptive, not vetoes. This is a bounded inconclusive tradeoff, with no demonstrated injected-recipe advantage at this dose. It does not establish that coordinate codebooks are universally useless or identify address semantics causally.

The prospective off source-negative to off-positive bad/cap/owner-recurrent/severe counts are **53/1/68/4** on training against **51/10/51/10**, and **16/1/9/1** on validation against **12/2/12/2**. Training bad and owner recurrence, plus validation bad, fail their limits. Absolute train bad/cap/owner-recurrent/severe counts are source 76/34/112/8, off 74/1/105/4, late 80/1/112/2 and early 74/2/94/7; validation is source 21/11/26/3, off 22/1/18/1, late 26/0/27/0 and early 22/2/24/1. These source-relative incidences must not be replaced by aggregate error reductions. Off parser drops and invalid geometry fall from source 7946/7850 to 439/422 on train and 2635/2624 to 37/26 on validation, but new failures affect previously source-negative images. Exact-row and annotation-owner recurrence are distinct proxies.

Paired train transitions versus off make the severity tradeoff concrete. Early repairs/introduces 46/46 bad and 56/45 owner-recurrent images; it repairs one cap and introduces two, and introduces three severe images without repairing one. Late repairs/introduces 39/45 bad and 49/56 owner-recurrent images; it repairs/introduces one cap each, and repairs/introduces three/one severe images. In off image 380192, a natural source and injected completions contrast with an off 3084-token cap that repeats invalid `<|coord_0|>` geometry, with 303 parser drops and 302 invalid boxes. In off validation image 59262, a 3084-token cap contains a 304-row same-annotation-owner run; off train image 222037 ends naturally but has a 32-row owner run. These are saved token/parser traces at `production/off_epoch16/cells/off_epoch16-{image_id}.json`, not physical identity adjudications.

The frozen semantic assay ran exactly 48 teacher-forced forwards on four first-description cases under early16 and late16. All eight full-vocabulary correct-versus-identity maxima are zero (limit 2e-4); each axis override was invoked. For x1/y1 conditional mean coordinate, the signed plus-minus responses are:

| Model | Image | x-axis on x1 | wrong y-axis on x1 | y-axis on y1 | wrong x-axis on y1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early | 134886 | +0.000528 | +0.000149 | +0.000107 | +0.000353 |
| Early | 162581 | -0.000353 | -0.000261 | +0.000240 | -0.000026 |
| Early | 366711 | -0.000049 | -0.000340 | +0.000984 | -0.002344 |
| Early | 421834 | -0.000120 | +0.000429 | +0.000176 | +0.000065 |
| Late | 134886 | +0.000189 | +0.000044 | -0.000184 | -0.000060 |
| Late | 162581 | +0.000119 | +0.000075 | +0.000202 | +0.000009 |
| Late | 366711 | +0.000089 | +0.000150 | -0.000020 | +0.000448 |
| Late | 421834 | +0.000242 | +0.000258 | -0.000034 | +0.000049 |

The bounded cases show nonzero fixed-prefix sensitivity, but signs reverse across cases and wrong-axis responses are sometimes comparable or larger. They do not show coherent axis-selective numerical transport. The full artifact retains all 1,000 legal-coordinate logits/CDFs, full-vocabulary family mass, mass-weighted first moments and distribution displacements for every case/condition; family mass is near one at these selected coordinate slots. The y1 query fixes ground-truth x1, so it is not a whole-box translation test. The assay cannot select training or prove native owner binding.

Technical evidence: source-OFF full-vocabulary parity maxima are 0/0/0. Lead independently accepted the first two genuine eight-rank updates: 902 exact common optimizer tensors and frozen hashes, global segments 18/19, DDP scale 8, correct three-loss weighted gradients, valid LR-zero first call then positive-LR movement. The fit completed 984 finite/applied calls, 7872 packs and 16,384 image presentations from the original ordered schedule, then all eight training ranks joined before evaluation. Fresh inference loaded the saved step-984 payload; a 14-cell early readback and all-1280-cell final readback match its 902 trainable tensors exactly. All 1280 four-condition prompts, media, targets and greedy policies match. A saved-only replay is byte-identical to the final reduction; no missing, mutated, or technical-invalid cell remains. The fixed first distributed launch failed before a recorded update with unknown rank-0 cause, and seven evaluator launches failed on parent-supplied local CUDA ordinals; both are preserved with costs and repaired without a science change.

Final cost from 19 terminal job intervals is 27,620.318584 allocated GPU-seconds (7.672311 GPU-hours) and 5658.404 model wall-seconds (1.571779 hours) from the original start `1790146890.760539`; all recorded producer/supervisor PIDs are absent. Output occupies about 2.253 GB, below 8 GiB. Focused tests: 21 passed. Research-knowledge, both-root output-layout and `git diff --check` pass. This does not claim a whole-repository test-suite pass. The exact bindings, commands, per-image evidence, replay and jobs are in the stable `candidate-v1/manifest.json` under the output root. The delegation record identifies the actual Luna briefs, prompt gaps, parent corrections and the parent-owned launch error without ranking models.
