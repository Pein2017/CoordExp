# Better complete outputs exist; learning realization remains unresolved

Continuation after this CPU closeout: the separately authorized
[round1 greedy evaluation](../2026-09-09-round1-greedy-realization/results.md)
now supplies the immediate post-update outcome. Historical missing-evidence
statements below describe this retained-data phase, not the current route.

Scientific disposition: **observed, finite, train-side sampling opportunity**.
The claim that Source produced no better complete outputs is false for part
of this panel. This does not establish that learning failed, that the outputs
are learnable at the registered dose, or that a deployable selector exists.

Technical disposition: **lead-accepted**. The full256-image /1024-sample CPU
consumer reproduces the historical Source counts and round1 reward owner sets;
an independent full replay produces byte-identical summary and case artifacts.
This phase used zero GPU forwards, optimizer steps, new samples or protected
confirmation reads. The retained-data stop is reached.

## Complete-output opportunity, not a sample union

All256 images /1955 GT objects and exactly four natural T1 samples per image
remain in the analysis. The baseline is original Source step2444 greedy T0;
checkpoint, embedding, native prompt/media and RP1/cap3084 are matched.
Every imperfect, empty, dropped or capped output is retained.

| Nested descriptive witness at IoU50 | Images with any witness /256 | Sample outputs /1024 |
|---|---:|---:|
| More matched GT objects than greedy, allowing exchanges | 54 (21.09%) | 89 |
| Strictly more, preserving every greedy-matched GT object | 49 (19.14%) | 77 |
| Also no increase in annotation-relative FP, repeats, parser drops or cap indicator | 32 (12.50%) | 43 |

These are descriptive levels, not new training admission rules. Of the32
strong-witness images,29 have an uncapped greedy baseline: the opportunity
is not solely avoidance of the four greedy cap failures. On23 of these32
images at least one strong IoU50 witness also has no reduction in TP80.
Neither stronger observation changes the primary IoU50 definition.

Greedy already covers every annotated object on96 images. The remaining160
have at least one FN:54 have an observed net-improving sample, and106 do not
among these four samples. Thus the202-image no-net-improvement group includes
96 images with no room to improve this count. K4 absence on the other106 is
not proof of zero probability or of an output-space impossibility.

## Selection tradeoffs

Best-of4 is selected using GT TP50, then FP, repeats, drops, length and seed.
The optimistic oracle can also retain greedy. **Both use labels; neither is
a deployed policy or an actual learned checkpoint.** F1 is pooled,
annotation-relative F1 under the same global one-to-one matching.

| Complete-output panel | TP50 /60 /80 | Recall50 | F1@50 | FP50 | Strict repeats | Drops | Caps |
|---|---|---:|---:|---:|---:|---:|---:|
| Source greedy | 1259 /1190 /908 | 64.40% | 0.5886 | 1064 | 467 | 794 | 4 |
| Sampled best-of4 | 1275 /1129 /740 | 65.22% | 0.6675 | 590 | 0 | 35 | 0 |
| Oracle over greedy plus four samples | 1358 /1246 /882 | 69.46% | 0.7040 | 545 | 1 | 19 | 0 |

Sampled-best gains151 but loses135 original matched object IDs, net+16.
Its image-level TP50 is better/equal/worse on54/163/39 images. The oracle
gains124 and loses25, net+99; it is count-preserving per image, not necessarily
object-ID-preserving. Even this oracle is below greedy at IoU80. Finding a
good IoU50 output must not be relabeled universal localization improvement.

All1024 samples pooled have4424 TP50 across four copies of the GT population:
the per-seed total averages1106, below greedy1259. Therefore the result does
not recommend replacing greedy with an arbitrary T1 sample. The shorter,
less repetitive sampled outputs and localization tradeoff both matter.

The sample-only matched-owner union contains1425 objects:217 gained and51
lost versus greedy. On40 images its union gains an object but no individual
sample increases TP50. This union is an unattained combination bound, not
evidence of one complete good output or a detection score.

## Concrete witness and geometry limits

On `coco2017_train_000000007116`, seed2026090604 changes TP50 from4 to5 of6,
F1 from0.8000 to0.9091, and TP80 from1 to3. FP50, repeats and drops stay zero;
both outputs end naturally. Complete length is37 versus46 tokens, so extra
correct predictions are not disallowed by an output-length gate.

The gained boat annotation is owner `181378`. Greedy's best same-category
geometric IoU is0.13918; the sampled matched box has IoU0.74280. This is a
continuous geometric diagnostic, not a pixel-verified claim of a newly seen
physical entity or a causal classification. The case artifact retains1392
changed-owner diagnostics across all candidates, with boxes, any/same-category
IoU, selected matches and competing eligible predictions.

## What was supplied to learning, and what remains unknown?

A supplementary CPU join to the already executed round1 RLOO plan gives:

| Candidate class | Positive advantage | Zero advantage | Negative advantage |
|---|---:|---:|---:|
| Net improvement over greedy (89) | 76 | 5 | 8 |
| Owner-preserving improvement (77) | 66 | 5 | 6 |
| Strong joint witness (43) | 37 | 2 | 4 |

All1024 stored advantages reproduce `reward - mean(other three rewards)`.
RLOO compares each sample to the other samples, **not to greedy**, and its
reward is image-normalized TP50, not the joint witness definition. A complete
output that is better than greedy can therefore receive zero or negative
advantage when its peers score as well or better.

The persisted round1 receipt binds this exact plan to one mechanically valid
update with1024 forward/backward calls and72750 replayed action tokens. This
establishes signed objective supply, not the resulting likelihood change of
each candidate: shared gradients, clipping and optimizer movement may interact.
The37 positive-advantage strong witnesses also rule out a blanket explanation
that no good candidate received positive objective weight.

There is **no accepted immediate post-round1 train256 greedy read**. The
[original course](../2026-09-06-source256-ce-rloo/results.md) scheduled train
evaluation only at round4. Its terminal train TP50 is Source1259, CE1264,
RLOO1278. RLOO used four fresh banks; CE used canonical GT. Those course-end
results cannot isolate whether this first bank's good outputs were learned
or whether their likelihood increased without changing greedy decoding.

Bounded next proposal, **not launched**: evaluate the already retained round1
adapter on the same train256 natural-greedy panel, without training or new
candidate sampling. This directly closes the missing immediate outcome read.
If attribution between objective response and greedy realization is needed,
score fixed retained witness/baseline sequences before and after that update;
likelihood change remains a diagnostic, not the final owner/F1 outcome. Freeze
the GPU budget and stop before either operation. Do not change objective or
architecture merely because the K4 oracle is better than greedy.

## Evidence and reproduction

- Primary [summary](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-natural-candidate-opportunity/full-v2/summary.json), SHA256 `4db5e3bb5ee363b18def0876b2cc15f04eec664aa33cb257f50d821960b3e3e3`.
- All [case scorecards](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-natural-candidate-opportunity/full-v2/cases.json), SHA256 `62a072ab9d8d56c638936666bfc76ec094b89047465e24e9f8b5338ba684a1fd`.
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-natural-candidate-opportunity/lead-verify-WkshDe/lead-acceptance.json), fresh13-test pass, full replay, logs, effective code snapshot and supplementary sign counts.
- [Consumer](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/candidate_opportunity.py) and [tests](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/tests/test_candidate_opportunity.py).

The summary records source and analysis dependency hashes. Native greedy
traces contain20580 trailing batch-pad tokens, explicitly validated and
excluded from action lengths. Samples store bodies without their observed EOS;
only observed `im_end` adds that terminal token. One invalid projected sample
prediction is retained in annotation-relative FP. This reduction does not
load model weights or claim a fresh model execution.

Run from `/data/CoordExp/.worktrees/research-probes`, using an absent result
directory; publication is exclusive:

```bash
python -m pytest -q probes/dora_owner_learning/tests/test_candidate_opportunity.py
python -m probes.dora_owner_learning.candidate_opportunity \
  --plan /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json \
  --greedy-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1 \
  --output-dir <absent-output-directory>
```

Reproduce the supplementary signed-supply read directly from those immutable
inputs (zero uses absolute tolerance1e-12):

```python
import json, math
from pathlib import Path
root = Path('/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration')
plan = json.loads((root / '2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json').read_text())
cases = json.loads((root / '2026-09-09-natural-candidate-opportunity/full-v2/cases.json').read_text())
groups = {g['example_id']: g for g in plan['population']['groups']}
for g in groups.values():
    for a in g['actions']:
        expected = a['reward'] - sum(b['reward'] for b in g['actions'] if b['seed'] != a['seed']) / 3
        assert math.isclose(expected, a['advantage'], rel_tol=1e-12, abs_tol=1e-12)
for level in ('net_owner_improvement', 'owner_preserving_improvement', 'strong_joint_witness'):
    values = []
    for c in cases:
        actions = {a['seed']: a for a in groups[c['example_id']]['actions']}
        values.extend(actions[s['seed']]['advantage'] for s in c['samples'] if s['comparison'][level])
    print(level, sum(v > 1e-12 for v in values), sum(abs(v) <= 1e-12 for v in values), sum(v < -1e-12 for v in values))
```
