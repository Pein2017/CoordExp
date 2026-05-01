---
doc_id: progress.diagnostics.et_rmp_rp_continuation_bias_hypothesis
layer: progress
doc_type: diagnostic-study
status: active-hypothesis
domain: stage1-et-rmp-ce
summary: Representative sample bank and hypothesis frame for repetition-penalty, continuation-bias, and perceived-but-not-emitted behavior on the pre-support-mass-enhancement ET-RMP checkpoint.
tags: [stage1, et-rmp-ce, repetition-penalty, continuation-bias, dense-detection, sample-bank]
updated: 2026-04-29
---

# ET-RMP RP Continuation-Bias Hypothesis

This note freezes the current representative visual sample bank and records the
working interpretation for the `rp=1.10/1.15/1.18` sweep on the old ET-RMP
checkpoint, before support-mass enhancement.

Do not treat these observations as evidence about the later support-mass
enhanced checkpoint unless the same bank is rerun on that checkpoint.

## Scope

- Checkpoint:
  `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/setcont-coco1024-sota1332-et-rmp-ce-v1/v0-20260429-022918/checkpoint-300`
- Eval slice: `val200`
- Decode: greedy, `max_new_tokens=3084`
- Compared knobs: `repetition_penalty in {1.10, 1.15, 1.18}`
- Training state: pre-support-mass-enhancement ET-RMP checkpoint
- Artifact bank:
  [artifacts/et_rmp_rp_sample_bank_2026-04-29/](artifacts/et_rmp_rp_sample_bank_2026-04-29/)
- Narrowed first-pass research subset:
  [artifacts/et_rmp_rp_sample_bank_2026-04-29/research_subset.json](artifacts/et_rmp_rp_sample_bank_2026-04-29/research_subset.json)

## Frozen Samples

The bank contains the rendered comparison PNG and per-`rp` canonical JSONL for
each case.

For manual relabeling and base-checkpoint ablations, use the `core_6` subset in
`research_subset.json` as the default. Do not start from all 200 validation
images. Add the two optional probes only if the first pass needs another
positive-recovery or negative-reranking example.

| Tag | Image idx | Source image | Primary reason |
|---|---:|---|---|
| `benefit_121` | 121 | `images/val2017/000000012639.jpg` | `rp=1.18` gains TP and reduces FN. |
| `benefit_010` | 10 | `images/val2017/000000001268.jpg` | `rp=1.18` recovers extra objects with only mild FP. |
| `benefit_158` | 158 | `images/val2017/000000016010.jpg` | `rp=1.18` opens up a previously under-emitting sample. |
| `hurt_025` | 25 | `images/val2017/000000002157.jpg` | `rp=1.18` sharply under-predicts versus `rp=1.10/1.15`. |
| `hurt_061` | 61 | `images/val2017/000000006471.jpg` | `rp=1.18` loses high-quality matches. |
| `hurt_178` | 178 | `images/val2017/000000017959.jpg` | dense kite/crowd case: `rp=1.18` shifts toward many tiny people while missing kite coverage. |

The attached kite/crowd review belongs to `hurt_178`. Its core ambiguity is
important: visually, many `rp=1.18` person boxes may correspond to real dense
crowd regions even when COCO annotations cannot fully adjudicate them.

## Aggregate Sweep Context

On this checkpoint and slice:

| rp | AP | AR100 | FN@0.50 | FP@0.50 | Pred total | Recall@0.50 | Duplicate guard suppressed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.10 | 0.4205 | 0.4690 | 661 | 176 | 992 | 0.5422 | 151 |
| 1.15 | 0.4190 | 0.4744 | 652 | 177 | 1037 | 0.5485 | 150 |
| 1.18 | 0.4155 | 0.4742 | 655 | 182 | 1077 | 0.5464 | 143 |

The sweep does not show monotonic conservatism with higher `rp`. In the
`1.10 -> 1.18` region, higher `rp` emits more objects and suppresses slightly
fewer duplicate-guard candidates, while AP softens after `1.15`.

## Rigorous Interpretation

The observed behavior is consistent with a decode-time redistribution effect,
not a simple global stop knob.

Autoregressive detection assigns a probability to a serialized object list:

```text
P(y | x) = product_t P(y_t | x, y_<t)
```

where each emitted object consumes many tokens:

```text
{"desc": d, "bbox_2d": [x1, y1, x2, y2]}
```

Stopping is just another continuation decision. At the boundary after object
`k`, the model compares probability mass for:

```text
continue_with_next_object  vs  close_array/EOS
```

A model can therefore visually represent more possible objects than it emits if
the boundary distribution, repeated-token dynamics, or self-generated prefix
state pushes `close` above `continue` before all visually plausible instances
are serialized.

The `rp` sweep suggests the penalty can change more than output length. It can
alter the local ranking among desc tokens and repeated coordinate tokens. This
can prevent staying inside one repeated class/box basin, and in some cases it
can make later or more diverse objects reachable. In the sample bank, `rp=1.18`
sometimes improves coverage and class diversity without catastrophic duplicate
burst, but it can also suppress or rerank useful repeated desc emissions such
as kites in `idx=178`.

The current evidence supports a softer claim:

```text
The old ET-RMP checkpoint appears to carry latent visual alternatives that
decode knobs can expose or hide, especially in dense scenes.
```

It does not yet prove the stronger claim:

```text
The model reliably perceives all missed objects and only fails to emit them
because of a continuation budget.
```

## Probabilistic Explanations

1. Boundary competition:
   At each object boundary, `P(close | prefix)` competes with
   `sum_o P(entry(o) | prefix)`. If `P(close)` rises with emitted count or
   sequence length, the model can stop despite still assigning nontrivial mass
   to plausible unseen objects.

2. Chain probability decay:
   Later objects require long token products. Even if the first token of an
   object has reasonable probability, the full entry probability can be fragile:

   ```text
   log P(entry) = sum_i log P(token_i | prefix, token_<i)
   ```

   Small per-token penalties compound across desc and bbox serialization.

3. Local basin allocation:
   Repeated desc and nearby bbox tokens can create an attractor where probability
   stays concentrated on a local same-class pattern. A repetition penalty can
   flatten that basin and redistribute rank mass to other descs or coordinates.

4. Count prior:
   If training examples frequently close after a certain object count, the
   hidden state may encode a learned count/length prior. This prior can override
   weak visual evidence for tiny or crowded objects.

5. Annotation incompleteness:
   In dense regions, COCO labels may underspecify visible instances. Some
   predicted FP under `rp=1.18` may be plausible true objects outside the GT
   contract, so raw FP alone can overstate quality loss.

## Neural-Dynamics Arguments

Structured detection generation mixes visual evidence with a strong language
model prior over JSON, object order, count, class frequency, and termination.
Teacher-forced training strengthens valid serialization but does not fully train
self-generated long-prefix recovery. As prefixes get longer, the hidden state
contains:

- accumulated object tokens and repeated desc tokens;
- a rough emitted-count signal;
- residual visual context from the image;
- local continuation history from the sampled order.

If visual evidence for small objects is weak, the language/count prior can win
at the boundary. If repeated-class evidence is strong, the model may keep
emitting one class unless decode-time penalties perturb the ranking. This makes
`rp` act partly like a local energy reshaper: it changes which continuations
survive the next-token ranking, not merely how long the sequence is.

## Alternative Explanations

These can falsify or weaken the perceived-but-not-emitted hypothesis.

1. Matching artifact:
   Extra `rp=1.18` boxes may be scored as FP because their boxes are shifted,
   too broad, or duplicated, not because GT is incomplete.

2. Prompt/order artifact:
   Different `rp` values may alter object order. Later matches can disappear
   due to ordering drift rather than perception/continuation budget.

3. Calibration artifact:
   Higher `rp` may produce more low-quality boxes that visually look plausible
   at thumbnail scale but fail localization under close inspection.

4. Class reranking without perception:
   A new desc may appear because repeated desc tokens are penalized, not because
   the model has stronger visual evidence for that class.

5. Old-checkpoint specificity:
   The effect may be specific to the pre-support-mass-enhancement checkpoint.
   If support-mass enhancement changes boundary mass or branch allocation, this
   bank must be rerun before generalizing.

## Experiments

The strongest test is to compare visual alternatives under fixed image and
prefix while measuring continuation probabilities, not just final rollouts.

1. Boundary continuation probe:
   For each sample-bank image, teacher-force the first `k` emitted objects from
   a reference rollout and measure:

   ```text
   log P(close | prefix_k)
   logsumexp_o log P(entry(o) | prefix_k)
   ```

   where `o` ranges over remaining GT objects and manually marked plausible
   unlabeled candidates.

2. Oracle-prefix continuation:
   Give the model a clean prefix containing all matched objects and ask whether
   it can emit the missed GT objects next. If missed objects become high
   probability after an oracle prefix, the issue is prefix/state dynamics rather
   than pure perception.

3. Forced-desc and forced-box probes:
   For a suspected missed object, force the desc token sequence and measure bbox
   token likelihood. Then force the bbox prefix and measure desc likelihood.
   This separates visual localization knowledge from class-token willingness.

4. Multi-sample coverage union:
   Decode each image many times using controlled stochastic settings and compute
   union coverage over GT and human-plausible unlabeled instances. If union
   recall is much higher than single greedy recall without large duplicate
   burst, the model has latent coverage not exposed by one rollout.

5. Count-prefix counterfactual:
   Compare identical image evidence under prefixes with different emitted counts
   but no overlap with the target object. If target continuation probability
   falls mainly with count, the length/count prior is measurable.

6. Repetition-penalty path analysis:
   At each object boundary, record top desc tokens and close/EOS logits before
   and after applying `rp`. This tests whether `rp` transfers mass across
   classes, across instances of the same class, or mainly away from close.

7. Support-mass-enhancement rerun:
   Re-run the fixed sample bank on the support-mass-enhanced checkpoint with
   identical decode settings and compare boundary mass, emitted count,
   duplicate clusters, and subjective dense-region plausibility.

## Metrics

Use raw metrics plus diagnostics that preserve the model-output view.

- emitted object count per image and per GT-count bucket;
- unique desc count and desc entropy per rollout;
- repeated-desc run length and same-desc duplicate cluster size;
- boundary close rank and close probability after each object;
- valid next-object mass at boundary;
- target missed-object rank under oracle prefix;
- union recall over repeated decodes;
- duplicate burst severity: max same-desc near-duplicate cluster size;
- dense-region FP review bucket: plausible unlabeled, box too broad, duplicate,
  wrong class, background;
- small/tiny-object recall by area bucket;
- class-transfer matrix across `rp`: classes gained/lost as `rp` changes;
- continuation survival curve: probability the rollout emits at least `k`
  objects.

## Visualization Protocol

Each bank case should keep these views:

- `rp=1.10/1.15/1.18` GT-vs-Pred panels with stable matching colors;
- per-rp object order list with desc and bbox;
- gained/lost object overlays between adjacent `rp` values;
- dense-region zoom crops for cases like `idx=178`;
- duplicate-cluster overlay with cluster IDs and sizes;
- manual review labels for FP: plausible unlabeled, duplicate, localization
  miss, class error, or background.

For future checkpoint comparisons, use the same image indices, same eval slice,
same renderer, same matching threshold, and same decode knobs. The point of the
bank is to make behavior changes visible rather than to discover fresh examples
each time.

## Current Read

The old ET-RMP checkpoint likely has a real continuation/ranking problem rather
than a pure visual-perception problem. `rp=1.18` can reveal additional objects
in several samples, and the aggregate sweep emits more objects as `rp` rises
from `1.10` to `1.18`. But `idx=178`, `idx=25`, and `idx=61` show that higher
`rp` is not uniformly better: it can rerank away from useful classes or lose
good matches.

The next decisive evidence is not another aggregate AP table. It is a
prefix-conditioned probability study over this fixed bank, followed by the same
bank on the support-mass-enhanced checkpoint.
