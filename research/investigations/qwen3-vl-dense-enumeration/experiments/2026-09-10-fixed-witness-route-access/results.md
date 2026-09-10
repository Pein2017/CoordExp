# Fixed witnesses gained some probability, but no first-fork crossings

Scientific disposition: **bounded actual-update movement, without fixed-route
greedy access or owner realization**. Technical disposition: **lead-accepted**.
The registered read and stop are complete. No new training, stochastic samples,
gradient diagnosis, confirmation evaluation or architecture is authorized.

## Population, conditioning and actual execution

The original Source step2444 and its retained first RLOO update score exactly
43 frozen strong complete-output samples from32 train images containing260 GT
annotations. Their advantage
strata are37 positive,2 zero and4 negative. The before/after greedy references
on those same images are also scored; deduplication gives85 distinct routes per
model and170 forwards in total. All original prompt/media/continuation IDs and
EOS/pad semantics are retained. These are teacher-forced score reads joined to
already observed natural outputs, not new natural generation.

Source and post each reproduce every target winner on their own32 retained
greedy references, with zero mismatched positions. The same FP32/SDPA,
patch-linearized, unmerged-DoRA execution uses the original selected embeddings.
The historical authored config path belongs to a retired worktree. The current
maintained profile has different authored/resolved hashes, explicitly recorded;
model/data identities, actual prompts/media/grids and winner parity bind the
executed semantics. Identical historical config bytes are not claimed.

## Actual likelihood and margin movement

| Fixed-sample readout | Result |
|---|---:|
| Complete log-likelihood increased |32/43 |
| Increased among positive-advantage witnesses |29/37 |
| Mean sample change, nats per complete sequence |+0.062262 |
| Median complete-sequence change, nats |+0.045008 |
| Image-balanced mean sample change, nats |+0.076125 |
| Gained-target-row log-likelihood increased |28/43 |
| Source-first-fork target-vs-best-other margin improved |20/43 |
| Target becomes argmax at that fork after update |0/43 |
| First non-argmax position unchanged |43/43 |

The median first-fork margin is-0.782579 before and-0.786036 after. Mean
margin movement is+0.002952 but median movement is-0.000565; movement is not
uniformly favorable. No actual first-fork margin reaches zero, and the closest
post margin is-0.002804. These are measured fixed-state margins, not a linear
learning-rate extrapolation or a prediction of how many further steps suffice.

CPU group accounting gives positive direct target-logit ascent coefficients
at37 first forks, negative at5 and zero at1. Only20 functional margins improve.
This distinguishes local objective credit from the shared model's actual
functional update; it does not isolate gradient interference, Adam, clipping
or dose as the cause.

## Two important interpretation controls

**Absolute reinforcement is not relative preference.** The saved scores also
allow a paired, fixed-sequence log-odds read:

`Delta log[p(candidate)/p(original Source greedy)]`.

It improves for only19/43 candidates, or18/37 positive-advantage candidates.
The sample median is-0.016128 nats and mean-0.187524; lengths and repeated
images affect these aggregates, so they are descriptive, not a utility score.
The image-balanced mean is-0.084586. The fixed Source greedy route itself gains
log-likelihood on22/32 images. Thus32 candidate likelihood increases do not
establish selective preference for the better complete outputs. This read is
not an estimate of expected sampled reward or a deployable selector.

**A fixed sampled fork is not a necessary owner route.** Of the43 first forks,
35 occur in other parsed object rows and8 in gained-target rows. Earlier rows
may matter causally, but their fork cannot automatically be called the required
entrance to the later gained owner. The result describes these exact witnesses,
not every valid route to their owner sets.

On these32 images, the retained natural-greedy comparison has zero owner gains
and zero losses at IoU50: all124 previously covered owners remain. None of65
strong-candidate gained owners appears, including none of61 from the positive
subset. Across the full256 panel the separately accepted result remains9 gains,
6 losses; those changes must not be attributed to this32-image subset.

## What this closes and what remains open

- A blanket claim of zero candidate likelihood movement is false: many fixed
  witnesses increase, including29 positive-advantage ones.
- The stronger claim that this update acquired greedy access to those exact
  routes is false: all43 first obstacles remain at the same positions.
- The data do not show that everything useful was learned and only a decode
  temperature adjustment is missing; relative preference and margin movement
  are mixed, and most first forks are not in gained-target rows.
- Expected T1 policy quality, necessary owner-specific entrances, and the
  decomposition among optimizer/dose/interference remain unmeasured.
- The[parallel row probe](../2026-09-10-owner-row-continuation-robustness/results.md)
  independently tests whether a correct row can connect to a stable Source
  suffix; it is not gated on this diagnostic and does not test update-induced
  damage at the same intervened state.

## Technical correction, resources and acceptance

An immutable-config assignment failed after all85 Source scores, before loading
post. The original terminal failure and Source scores remain intact. A narrowly
approved config-copy correction loaded post once and scored its85 routes; no
Source rerun or scientific retry occurred. Separate terminal receipts record
72.64 seconds for the first invocation and146.77 cumulative seconds after the
continuation. Do not add those values together.

Total: two loads,170 score forwards,44554 scored token positions,146.77 model
seconds, peak allocated CUDA14.76GB, peak RSS across invocations12.11GB, and
about178MB package artifacts. No persistent GPU process remains. Model time
does not include all CPU preparation, coding, review or orchestration time.

Root freshly ran the combined22-test suite (15 here and7 for the row probe),
inspected native replay/scoring, reproduced both score and credit reductions,
and independently computed the supplementary fixed-sequence contrasts. No
additional model read was needed for acceptance.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access`.

- [Primary reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/reduction.json).
- [Per-token credit context](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/credit.json).
- [Lead supplementary readout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/lead-supplement.json).
- [Lead checks](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/lead-checks.json).
- [Package receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/package_receipt.json).

CPU reproduction from the research-probes worktree:

```bash
python -m pytest -q probes/dora_owner_learning/tests/test_route_access.py
python -m probes.dora_owner_learning.route_access reduce
python -m probes.dora_owner_learning.route_access credit
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/verify_parallel_probes.py
```
