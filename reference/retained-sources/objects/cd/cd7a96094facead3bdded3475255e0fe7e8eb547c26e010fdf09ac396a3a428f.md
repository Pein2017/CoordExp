# First-coordinate mediation

Candidate completed 2026-09-19T03:05:19.498392+00:00; root acceptance pending.

First x1=1 is sufficient for a low-recurrence conditional original-readout route in untied donut, but not tied donut. It is not necessary for the numerical effect of sign20/22/23/24 in either model: all remain near-run1 when first x1 is forced0. Continuing readout can act through other choices; the post-hoc first-token association is not a complete mediator.

## Fixed evidence

Both mature donut417044 original failure histories were crossed with original/sign19–26 readout and forced first x1=0/1. All36 cells completed. All18 concordant controls exactly reproduce accepted source sequences. Independent CPU reduction is JSON-exact; saved head/logit/carrier traces verify11,248 steps, exactly one offset5 override percell, and forced-token consumption at offset6. CPU corruption checks reject wrong forced position and wrong token.

The following outcomes use **only fully free rows after the first completed row**. Each entry is longest near-run / invalid rows / free rows / stop; R means32 whole rows (31 primary rows), E means naturalEOS. First-row geometry/validity and whole-sequence exact/near/native/alternate metrics are retained in `reduction.json` and `outcomes.csv`.

| Model | Readout | First x1=0 | First x1=1 |
|---|---|---|---|
| tied | original | 17/1/31/R | 19/1/31/R |
| tied | sign19 | 9/1/31/R | 9/1/31/R |
| tied | sign20 | 1/0/31/R | 1/0/31/R |
| tied | sign21 | 11/0/31/R | 11/0/31/R |
| tied | sign22 | 1/0/31/R | 1/0/31/R |
| tied | sign23 | 1/0/31/R | 1/0/31/R |
| tied | sign24 | 1/0/31/R | 1/0/31/R |
| tied | sign25 | 8/0/31/R | 9/0/31/R |
| tied | sign26 | 11/1/31/R | 11/1/31/R |
| untied | original | 13/0/31/R | 1/0/29/E |
| untied | sign19 | 9/1/31/R | 9/1/31/R |
| untied | sign20 | 1/0/30/E | 1/0/27/E |
| untied | sign21 | 17/0/31/R | 16/0/31/R |
| untied | sign22 | 1/0/29/E | 1/0/26/E |
| untied | sign23 | 1/0/31/R | 1/0/27/E |
| untied | sign24 | 1/0/27/E | 1/0/25/E |
| untied | sign25 | 17/0/31/R | 17/0/31/R |
| untied | sign26 | 9/2/31/R | 9/2/31/R |

All first rows are complete and geometrically valid; all primary/whole malformed-opener counts are zero and no512-token cap occurs. Validity does not establish physical identity. No controlled row receives recovery credit.

## Interpretation and remaining alternatives

- **Untied original:** changing only emitted first x1 from0 to1 changes primary longest near-run13→1, with29 free rows thenEOS instead of31 free rows at the whole32-row limit. This establishes conditional sufficiency for that numerical route, with termination and physical coverage limits.
- **Tied original:** the same change gives17→19, both at31 free rows and one invalid row. The choice is not sufficient across these two model packages.
- **All four association-positive fields, both models:** forcing first x1=0 still yields primary near-run1. Initial x1=1 is therefore not necessary for their low-recurrence outcome here. Other first-row slots or later readout decisions provide alternate routes.
- **Concrete later route:** untied sign23/sign24 forced0 preserve the entire first-row token sequence of original0. Their first subsequent difference is offset15, the next row x1, where they emit1 instead of0. Other positive fields already differ at first-row x2/y2. This locates observed alternate decisions, not a unique causal circuit or proof a later single token is sufficient.
- **Other fields retained:** sign19/21/25/26 remain recurrent under both first choices, with geometry debt in some cases. The first choice does not erase continuing-readout effects.

## Technical closure

Eight launch wrappers pre-created output directories and failed before model loading/forward. Evidence preserved; wrapper corrected once,32 remainingcells executed once. Pre-launch role guard corrected before anyGPU calls; CPU gate catches wrongoffset/token. No producer failure or scientific cell omission.

Measured successful runtime: 1760.588 GPU-seconds. Conservatively adding 61.889 seconds of eight pre-model failed launch intervals gives 1822.477 budget-charged seconds (0.506 GPU-hours). Total 11,248 forwards and 706,621,509 tensor bytes; all8 GPUs used. Pilot and8 successful scaleout wrappers have observed0 exits; failed pre-model wrappers have observed1 exits. All owned jobs and Luna-max child ended.211 unique bindings verify with no unresolved gap.

An idle-child dispatch delay occurred before qualification; it consumed no model calls and duplicated no work. Root-owned catalog result registration is the only expected knowledge-check gap. Predecessors remain immutable. Exact artifacts/reproduction: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-first-coordinate-mediation/ARTIFACTS.md`.

## Claim boundaries

- Primary metrics exclude the entire first completed partiallycontrolledrow. Whole metrics remain separately labeled.
- Outcome-selected two-model single-image follow-up, not prevalence or held-out efficacy.
- Near-row recurrence is a numerical proxy; EOS/fewer rows/another pattern is not physical recovery.
- Continuing readout changes many coordinate logits. Several bins share signpatterns; no bin0 coefficient uniqueness or universal circuit claim.
- T/U compares mature model packages; no untie-only or training-origin conclusion.
- Target outputs only; companions outside estimand.
