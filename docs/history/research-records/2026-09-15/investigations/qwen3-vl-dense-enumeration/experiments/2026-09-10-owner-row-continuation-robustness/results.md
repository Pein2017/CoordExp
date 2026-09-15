# Two verified conditional insertions connect to stable Source continuations

Scientific disposition: **positive local conditional realization and successor
compatibility on two provisionally admitted cases**. Technical disposition:
**lead-accepted**. This is not autonomous owner discovery, a population result,
or proof that the round1 update preserves these intervened histories.

## Denominator and visual eligibility

The fixed input was65 distinct image/GT-owner gains from the43 strong sampled
witnesses, not an arbitrary search for successful interventions. Same-category,
same-description/token-layout, ordering and native-prefix criteria leave5
CPU-eligible owner pairs. Numeric image/GT-ID ranking yields4 selected images,
one far-any-IoU<.1 and three other. Numeric-versus-string ID ordering was
clarified before root viewed cases or any model outcome; the first CPU version
is preserved as superseded.

Root reviewed all four context cards and enlarged the two small-object cases
before any continuation. Two cases were admitted, one per stratum:

- Image368, target person2022537: an occluded dark-clothed background person,
  distinct from the foreground child and the next red-capped adult B.
- Image7116, target boat181378: the rear blue vessel, distinct from the
  foreground pontoon and the next white motorboat B.

Admission is **annotation-backed, moderate-confidence single-instance IoU50
compatibility**, not precise amodal-extent or latent consumed-state certification.
The person is heavily occluded. The prior pontoon rectangle overlaps the rear
vessel in projection; root treated its principal physical referent as the
foreground pontoon, recording the overlap rather than claiming zero overlap.

Image72583 was omitted for overlapping painted-apple instance/extent and prior
binding ambiguity. Image575303 was omitted because prior cyan P4 already
substantially addresses the central cow targeted by A. These are qualification
limits, not claims that GT is false. No replacements or stratum backfill occurred.

The[admission receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-owner-row-continuation-robustness/admission-v1.json)
binds all reviews to numeric manifest-v2 before model outcomes. The result is
conditional on this narrow2/4 visual admission, not a success rate over65 owners.

## Fixed branches and actual behavior

At the exact Source natural row prefix P, the original next row is B. We ran:

1. original B, then greedy suffix (diagonal control);
2. A only through its first differing coordinate, then free row completion and
   suffix;
3. complete sampled A, then free suffix;
4. complete A' with x2 increased by one bin, then free suffix;
5. complete A at its original sampled history, then greedy suffix (descriptive
   history control).

In both partial-A cases the differing coordinate is x1. The preceding format
and description tokens are identical to Source's B row; the remaining three
coordinates are freely generated. This is one changed decision, but **not a
small coordinate perturbation**: x1 changes562→455 for the person and571→291
for the boat. It supplies a strong location cue and is not autonomous discovery.
The separate A/A' contrast is the one-bin x2 robustness test.

| Case / branch | TP50 /60 /80 | FP50 | FN50 | F1@50 | Action tokens |
|---|---|---:|---:|---:|---:|
|368 Source B |12 /12 /8 |2 |1 |0.888889 |130 |
|368 partial A |13 /13 /8 |2 |0 |0.928571 |139 |
|368 full A |13 /13 /8 |2 |0 |0.928571 |139 |
|368 A' |13 /13 /8 |2 |0 |0.928571 |139 |
|7116 Source B |4 /4 /1 |0 |2 |0.800000 |37 |
|7116 partial A |5 /5 /3 |0 |1 |0.909091 |46 |
|7116 full A |5 /5 /2 |0 |1 |0.909091 |46 |
|7116 A' |5 /5 /2 |0 |1 |0.909091 |46 |

Every branch has natural EOS, zero strict repeats, zero parser drops and no
caps. Both original B branches exactly reproduce retained Source tokens and
parser outputs. All three primary A variants add the target at IoU50/60,
recover B freely in the suffix, and retain all previously matched IoU50 owners.
One additional valid row accounts for the nine-token increase; longer output
alone is not penalized.

**A and A' produce exactly identical free-suffix token IDs in both cases.**
Thus the one-bin x2 change produces neither a persistent owner difference nor
even a token-level suffix difference at these two histories. This does not
establish general coordinate smoothness across axes, radii or contexts.

## Current row, suffix and geometry are different outcomes

The full forced A is not counted as autonomous discovery. Root independently
confirmed that A is absent from the prefix-covered set and present by completion
of the partial-A current row, rather than being recovered only later. The
remaining three coordinates after forced x1 are native model output.

For image368, the three old Source owners after B are retained. For image7116,
its one old Source owner after B is retained. B itself is recovered by a new
free suffix row in all primary A arms, not protected by a forced copy.

The image7116 IoU80 gain under full A/A' is an existing owner's localization
threshold crossing, not the new A passing IoU80. Partial A additionally realizes
the target itself at IoU80. The person target is added at IoU50/60, not80.
These distinctions prevent reporting uniform high-precision recovery.

## Sample-history control and correction

For image368, the two A-release histories leave the same four GT obligations,
all recovered by their free suffixes. The sample-history complete output has
TP80=7 versus8 for the Source-prefix A arms; equal IoU50 coverage is not exact
geometry invariance across those different histories.

For image7116, the sampled history leaves extra obligation1739499. The common
remainder has three owners, but the complete workloads differ. This control is
**non-discriminating for a controlled successor comparison**; common-subset
progress is descriptive only. Its total TP50/60/80 is5/5/1.

The first saved reducer used an insufficient control flag. Before acceptance,
the worker corrected it, added a regression case and published
`execution/reduction-v2.json`; original reduction and all ten raw trajectories
remain unchanged. Root's fresh native consumer/reducer reproduces v2 exactly.

## Implications and stop

These two Source histories admit a useful local connection: a supplied x1
choice can produce the missing row while retaining B and the previous suffix.
They do not require a harmful state transition after every correct addition.
They also provide no evidence for a one-bin x2 fragility at the tested releases.

This does not show that all strong witnesses can be attached locally, that
arbitrary GT insertions are safe, that Source autonomously finds the targets,
or that a new trained checkpoint preserves these continuations. The
[parallel likelihood read](../2026-09-10-fixed-witness-route-access/results.md)
is an actual-update diagnostic on fixed complete witnesses, not a paired
Source/round1 test of these new intervened histories.

The selected panel is exhausted. No expanded eligibility, extra perturbation,
longer forced prefix, post checkpoint, training or new sampling follows.

## Resources, evidence and CPU reproduction

One GPU1 Source load,10 continuations,279 new tokens/model forwards and10 image
forwards,32.12 model seconds including loading, peak allocated CUDA9.39GB,
peak RSS11734244KiB. Final execution artifacts are902396 bytes. GPU1 returned
to zero memory use and no owned model process remains.

The two packages together passed22 fresh root tests; this package contributes7.
Root inspected branch/consumer/reducer code and independently replayed the
tokenizer-backed raw-output consumer, both B diagonals, current-row A realization,
B suffix recovery, owner-set inclusion and exact A/A' suffix equality.

- [Final reduction v2](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-owner-row-continuation-robustness/execution/reduction-v2.json).
- [Native consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-owner-row-continuation-robustness/execution/consumer.json).
- [Raw token trajectories](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-owner-row-continuation-robustness/execution/rows.jsonl).
- [Lead checks](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-owner-row-continuation-robustness/lead-checks.json).

From the research-probes worktree, without a model forward:

```bash
python -m pytest -q probes/source_rweak_row_cross/tests/test_owner_row_robustness.py
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/verify_parallel_probes.py
```
