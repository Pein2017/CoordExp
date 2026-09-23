# Fixed first-row template for distributed repeated-history cache

2026-09-22. Final distinct discriminator in the current local mechanism loop,
authorized by the user's ongoing autonomous-loop grant. The preceding one-row
position-correct substitution is independently accepted in both cases. That
result removes the need for freshly computed LAST-row cache under a particular
nearby donor; it leaves distributed contextual evolution across repeats open.

## Question, cases and frozen prediction

Can a stationary FIRST-row contextual template pool, at native per-row phases
with freely recomputed current S, reproduce each native exit endpoint?
Use mature untied+axis step2444 and the same exact native batches and prefixes.
Root verified every9token row and both adjacent boundaries in selection.json:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/selection.json.

| Case | Identical run | First-row source | ALL destination | LAST destination | Current S | Native winner |
|---|---|---|---|---|---|---:|
| val7511 | rows27–88,62 rows |1563:1572|1572:2121,61 rows|2112:2121|2121:2127|999|
| train269858 | rows16–19,4 rows |1506:1515|1515:1542,3 rows|1533:1542|1542:1546|348|

Three cells per case:
- NN: native historical cache, current S recomputed.
- LAST: replace last repeated row by the FIRST row's pre-K rotated to that
  last row's native phase, and the FIRST row's V.
- ALL: same fixed FIRST-row template at every later repeated row, each with its
  OWN native phase, retaining the original first row. This is61/3 destination
  rows, respectively. No freshly computed cache remains in those destinations.

The FIRST-row template comes from each layer's actual normalized K before RoPE
and actual source V during native prefill. Source context is retained. All
nonrepeat history, first row, other batches, masks, current positions and S
remain native; S adapts freely. There are no interposed nonrepeat historical
positions in either run. This is an explicitly hybrid cache sufficiency assay,
not a sequentially generated counterfactual history.

Advance prediction: ALL retains native global999/348 with top-two gap>0.001
in BOTH cases. LAST is the fixed donor-age compatibility control. Interpret:
1. LAST and ALL retain: the fixed contextual template pool is sufficient for
   that endpoint under this assay; position-specific contextual row variation
   after the first row is not required for that categorical result.
2. LAST retains, ALL decisively differs: reject fixed-template sufficiency;
   distributed replacement changes the endpoint. Do not claim general necessity
   of an accumulator or a specific stored latent variable.
3. LAST differs: donor-age compatibility already fails; ALL cannot isolate
   replacement extent. Report its result but no distributed-extent attribution.
Near ties<=0.001 are inconclusive. Any third winner is preserved; no favorable
case substitution, donor rescue, fraction sweep or component partition.

Even full success cannot explain timing: this same surrogate might cause an
exit at earlier positions. Do not claim that it predicts the native transition
row, removes all context, proves count-only dynamics or restores physical owners.
The preceding mass-only rule remains rejected.

## Execution, invariants and cost

Root owns interpretation, selection and acceptance. /root/trace_dynamics owns
only recurrence_fixed_template.py and attempt001. Reuse maintained native case
preparation, identity and cache helpers; leave accepted producers unchanged.
Bind original raw/batch/media/model and accepted positioned-duplicate receipts
before forwards. Source/readback metadata must name FIRST donor distinctly from
the preceding one-row test's penultimate donor.

One native historical prefill per case captures actual first-row pre-K/V and
native phases for the entire repeated block. Require exact token-row boundaries,
unchanged native identities and pre-K→native source post-K replay, with root
FP64 complex-pair oracle<=2e-4. Save template pre-K/V, phase coefficients and
actual source/destination hashes; full historical cache storage is unnecessary.
For every28 actual self-attention consumer, check all destination row K/V at
its assigned phase and exact template V, the first-row and outside-history
invariance, mask/slots/current phases and all companions. LAST must leave all
other repeated rows exact. Restore native cache by layer reference after S
append and verify its digest. Do not clamp current Q/K/V or change native SDPA.

NN must reproduce accepted full logits<=2e-4 with identical global winner.
Save each full-vocabulary vector, top-five/gap, FP64 probabilities/logprobs for
all declared and observed competitors, val z38-z999, train z350-z348 and
z591-z348, and distances from NN. CPU check must reject wrong destination,
row phase/template alignment, V assignment and append/restoration leakage.
Preserve failures before stopping; root independently verifies consumed tensors,
phase construction, bindings and full vectors.

Budget: two cases x (prefill+NN+LAST+ALL) =8 model/2 vision forwards; cap10 model
calls and15minutes, one GPU4 initially. Artifact cap128MiB/case. No generation,
new trajectory or automatic retry. Close this local branch after the verdict,
including mixed/negative outcomes; synthesize the tested mechanism and remaining
natural-timing question without further donor/layer/head scans.

Producer: probes/training_set_completion/recurrence_fixed_template.py.
Artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/attempt-001/.
