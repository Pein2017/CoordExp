# Local entrance CE learns and recovers targets, but causes collateral burden

Scientific disposition: **fixed-state learning and natural target realization,
without joint quality improvement**. Technical disposition: **lead-accepted**.
The checkpoint is **not promoted**. The registered training and18-output
evaluation stop is complete; no additional training, QP, objective variant,
sampling or confirmation read follows.

## Exact intervention and checkpoint choice

Start from original Source step2444, not round1. The two inputs are the exact
Source natural prefixes of the[verified branch bridge](../2026-09-10-verified-branch-update-bridge/results.md):
image368/person2022537 targets x1=455 at action index97; image7116/boat181378
targets x1=291 at index22. Their prior annotation-backed moderate-confidence
visual/occlusion qualifications remain unchanged.

Only these two x1 labels receive CE, with equal image weights. All preceding
rows, other coordinates, unmatched predictions and EOS are context, not new
supervised targets. The same588 language DoRA A/B/m tensors (18006016 scalars)
are trained; all base/vision/projector/selected-embedding/output-head parameters
remain frozen and are byte-hash verified unchanged.

Fresh AdamW uses lr1e-5, betas(0.9,0.999), eps1e-8, weight decay0 and gradient
norm clipping1.0. Two target losses accumulate before one update. The stop is
the first update with both actual full-vocabulary margins at least0.1, or32
updates. No natural/dev outcome participates in checkpoint selection.

| Fixed-prefix target | Initial margin | Selected checkpoint margin | Final target probability |
|---|---:|---:|---:|
|368, x1=455 |-3.215862 |+0.360796 |0.202027 |
|7116, x1=291 |-11.088956 |+0.120861 |0.264315 |

Both targets first become actual top1 at **update21**. The stricter registered
0.1-margin stop first fires at **update23**, which is the only evaluated saved
checkpoint. Update22 still has boat margin0.079802. Thus23 is not the first
literal argmax crossing; it is the predeclared stopping point. No earlier
checkpoint was selected from outcome evidence.

There are46 supervised target-token occurrences,46 differentiable forwards
and48 score forwards. Adapter movement L2 is0.46283. The actual initial logits
exactly match the retained Source bridge; independent cold candidate logits
exactly match the final trainer vectors. This is genuine parameter learning,
not an inference-time forced token or an unsaved in-memory calibration.

## Frozen natural evaluation

The evaluator prepared its manifest independently before seeing candidate
outcomes: the two training images plus16 historically used Source dev128 images,
selected by the frozen salted SHA256 ranking with no outcome filtering or
backfill. The training images and guard are disjoint. The guard is not an
untouched population test and was not used by this two-example optimizer.

All18 candidate outputs start from the original full image/prompt, with empty
forced prefixes, T0/top-p1/RP1 and the same3084-token cap. The Source control is
its retained bound native output. Matching is category-consistent global
one-to-one actual-pixel IoU50/60/80; FP remains annotation-relative.

| Outcome | Train2 Source | Train2 candidate | Guard16 Source | Guard16 candidate |
|---|---:|---:|---:|---:|
| GT owners |19 |19 |89 |89 |
| TP50 |16 |19 |56 |55 |
| TP60 |16 |16 |54 |52 |
| TP80 |9 |11 |43 |39 |
| FP50 |2 |34 |146 |155 |
| FN50 |3 |0 |33 |34 |
| F1@50 |0.864865 |0.527778 |0.384880 |0.367893 |
| Valid predictions |18 |53 |202 |210 |
| Strict later pixel-IoU>.95 repeats |0 |21 |44 |44 |
| Parser drops |0 |1 |7 |12 |
| Complete action tokens |167 |491 |2000 |2122 |
| Caps |0 |0 |0 |0 |

All18 outputs end naturally. At IoU50 the training pair gains3 owners and loses
none; the guard gains4 and loses5. At60 the training pair exchanges1/1 and the
guard4/6; at80 the training pair gains2/loss0 and the guard4/8. Do not pool the
training and guard populations into a single apparent net improvement.

## Target realization is real, but not along the exact trained histories

Both intended targets appear naturally at IoU50:

- **Image368/person:** TP12→13, FP2→4, F1.888889→.866667, tokens130→157,
  no strict repeats. The target's predicted bins are[455,181,492,229],
  IoU0.576003: it passes50, not60 or80.
- **Image7116/boat:** TP4→6, FP0→30, F1.800000→.285714,
  tokens37→334,21 strict repeats and one parser drop. The target boat's bins
  are[298,460,371,556], IoU0.947917, passing all three thresholds. Its actual
  natural x1 is298, not the specifically trained291. The other gained GT owner
  is person1759775.

Neither natural output reaches the original exact training prefix. First token
divergence is action16 for368 and5 for7116. This does not mean learning was
unusable: both owner targets are recovered via changed natural histories.
Nor does it establish a particular state mediator. The saved same-token-horizon
prefix comparisons can cut a different row and are explicitly descriptive,
not an equal-semantic-state experiment.

## Repetition is not simply repeated emission of the trained boat

A CPU recount of the36 valid image7116 predictions finds four strict geometric
repeat clusters, all described as person:

| Cluster boxes | Later repeats | Same-category direct GT incidence at IoU50 |
|---:|---:|---|
|3 |2 |None |
|5 |4 |None |
|2 |1 |None |
|15 |14 |Person1759775 |

The21 repeats do not directly match trained A181378 or original B176844 at50.
The largest cluster repeatedly overlaps another newly matched GT person;
the other clusters have low best IoU against person1739499. They are not
automatically classified as hallucinations or new physical entities.

Literal x1=291 appears in27 valid candidate rows versus zero Source rows.
Coordinate reuse is not owner identity. In particular, the accurate recovered
boat uses x1=298 while the repeated person rows heavily reuse291. This is
evidence of altered behavior outside the two supervised decision states, not
proof that the trained boat was emitted and then forgotten by an owner ledger.
Attributing it specifically to the291 training example, rather than the joint
two-example update or a parameter subfamily, would require a new contrast.

As a bound, merely deleting21 strict-repeat predictions while preserving all19
training GT matches would leave32 predictions and F1 at most38/51=0.745098,
still below the original0.864865. No such postprocessor was executed here, and
the bound remains annotation-relative; it does not adjudicate the residual
unmatched rows' physical truth.

## Interpretation and the registered stop

The frozen-prefix argmax barrier can be crossed by a short standard parameter
update, and the new owners can appear without inference-time location hints.
Thus this is not a failure to move logits or an inability to affect natural
owner output.

The failure is **selectivity and joint outcome preservation** for this recipe:
the desired movement coexists with large training-image prediction/repetition
burden and a small negative guard result. Clipping1.0 did not provide an
object-level preservation guarantee. The result does not establish that all
scalar reward/CE methods fail, nor that an explicit ledger or QP is required.

We did not replay the trained model at the old forced P+A continuation states.
Therefore the natural repetition cannot be uniquely assigned to a broken
post-emission transition at those states: the natural histories changed too.
The earlier Source/round1 forced-branch stability result remains true at its
own checkpoints and is not silently extended to this new CE adapter.

No joint owner/F1/burden benefit is demonstrated. Keep the candidate and all
negative evidence, but do not promote or extend it. A next intervention would
need a separately chosen preservation/credit/state-distribution question,
rather than more unconstrained steps on this recipe.

## Technical acceptance and resources

Training: one GPU0 load,23 updates,94 model forwards,123.225840 seconds,
peak CUDA allocation26.49GB, peak RSS11.58GB. The first actual update was the
vertical gradient/mask/frozen-parameter smoke, not an extra training arm.

Evaluation: one independent GPU1 candidate load, two cold score forwards,
18 natural outputs,2613 new tokens and2615 total model forwards,207.075663
seconds; peak CUDA allocation9.41GB and peak RSS9.72GB. Combined model time
is330.301503 seconds,0.091750 GPU-hours, not total preparation/coding/review
wall time. Both model processes exited and GPUs were released.

Root freshly passed six training tests and five evaluation tests, replayed both
verification CLIs, independently checked the first stopping point, initial and
cold full-vocabulary vector equality, checkpoint/source bytes, pooled metric
arithmetic, owner gain/loss arithmetic, repeat counts/GT incidence and actual
target-row coordinates. No additional model call was used for acceptance.

- [Training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/training/receipt.json).
- [Frozen evaluation selection](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/evaluation/selection.json).
- [Natural-output reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/evaluation/execution/reduction.json).
- [Raw natural trajectories](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/evaluation/execution/rows.jsonl).
- [Repeat diagnostics](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/evaluation/repeat-diagnostics.json).
- [Independent lead readout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/lead-readout.json).
- [Frozen protocol](unit.md).

CPU-only reproduction from the research-probes worktree:

```bash
python -m pytest -q probes/dora_owner_learning/tests/test_entrance_ce.py probes/dora_owner_learning/tests/test_entrance_ce_eval.py
python -m probes.dora_owner_learning.entrance_ce verify
python -m probes.dora_owner_learning.entrance_ce_eval verify
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-native-entrance-ce-feasibility/lead_verify.py
```
