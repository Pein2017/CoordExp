# Covered history with a common last row

**Candidate complete; awaiting root acceptance.** Full-row E is negative in all16 supplied/candidate extent combinations under both original and equal-norm readout. This supports conditional **relative** owner/location-sensitive suppression with fixed literal last row C. It does not establish absolute suppression: both A and B candidate probabilities are higher after earlier A than after earlier B, with B increasing more. The shared-x1 first-fork effect has the opposite sign—relative facilitation toward A. These surfaces must remain separate.

[Authoritative result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail/result.json) · [complete reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail/reduction.json) · [artifact map](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail/ARTIFACTS.md) · [frozen protocol](unit.md)

## Admission and fixed conditioning

One image417044, original R16 plus paired embeddings. h is genuine original row1, person[0,406,500,999],9tokens, with no A/B/C owner coverage. Earlier ambiguous native rows3/4 are not used or assumed distinct. C was fixed from original row2 before scores: donut[0,310,57,373], owner -5529525760667096, the distinct bottom-left visible donut.

A1=[0,256,61,315] and A2=[0,256,66,315] designate owner -1693019979812657. B1=[52,274,106,327] is a saved norm-generated extent; B2=[57,274,105,327] is its refined secondary extent, owner -8380415314849442. All four and C have the same10-token donut serialization. Two h/C checks plus four reused displayed extents; no new owner search,6 displayed realizations within12-row ceiling, no current admission HOLD.

[Full-image overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail/review/triple-full.png) and [local context](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-covered-history-common-tail/review/triple-local.png) support distinct dominant owners despite legitimate box overlap. A→C is mostly downward; B→C is down-left with larger horizontal displacement and different older scan ordering. This is a material surviving route confound, not matched displacement.

All four h→A/B→C histories have29actions and identical literal C at19..28, identical positions, images and companions. Their C contextual states may differ. These are supplied histories,20tokens earlier than the prior confirmed recurrence boundary49; only h is claimed genuinely native.

## Entire coherent-row matrix

Each cell is **original / equal-norm log probability**, including all10 row tokens and excluding the separately captured next boundary. Full probabilities, token contributions, D(H), EOS/opener/ranks and logits remain in reduction.json and matrix.tsv. Later candidate states differ after the first fork; these are coherent conditional rows, not products of independent coordinate marginals.

| Supplied history | Candidate A1 | Candidate A2 | Candidate B1 | Candidate B2 |
|---|---:|---:|---:|---:|
| h A1 C | -15.760358 / -16.304015 | -14.448673 / -15.167226 | -11.430849 / -11.148288 | -12.584698 / -12.727811 |
| h A2 C | -18.047835 / -18.553576 | -16.485628 / -17.145040 | -11.276870 / -10.988566 | -12.471150 / -12.610846 |
| h B1 C | -18.191774 / -18.576372 | -17.170838 / -17.732068 | -19.695943 / -19.613872 | -18.430915 / -18.605961 |
| h B2 C | -18.622560 / -18.994637 | -17.789243 / -18.326237 | -19.604020 / -19.523702 | -18.579587 / -18.745936 |

## Entire E matrix

D(H)=logP(A_candidate|H)−logP(B_candidate|H); E=D(h A_supplied C)−D(h B_supplied C). Rows vary supplied extents; columns vary candidate extents. Each cell is **original / equal-norm E**. No averaging, owner-mass summation or independence claim.

| Supplied A vs B | Candidate A1−B1 | A1−B2 | A2−B1 | A2−B2 |
|---|---:|---:|---:|---:|
| A1 vs B1 | -5.833678 / -6.193226 | -3.414801 / -3.605793 | -5.542930 / -5.900741 | -3.124053 / -3.313308 |
| A1 vs B2 | -5.310969 / -5.684792 | -3.132687 / -3.327503 | -4.832601 / -5.216402 | -2.654319 / -2.859113 |
| A2 vs B1 | -8.275133 / -8.602509 | -5.815825 / -5.972319 | -7.733863 / -8.038277 | -5.274555 / -5.408088 |
| A2 vs B2 | -7.752424 / -8.094075 | -5.533711 / -5.694029 | -7.023534 / -7.353939 | -4.804821 / -4.953893 |

All16 raw E values are negative (−8.275133 to−2.654319), as are all16 equal-norm E values (−8.602509 to−2.859113). Extent changes alter magnitude substantially but do not reverse this differential sign. D itself is not uniformly a clean novelty preference: after B2, A1 versus B2 remains slightly negative (raw−0.042973; normalized−0.248701).

For primary A1/B1 supplied and candidate rows, D changes−4.329509 versus+1.504169, giving E=−5.833678. However, logP(A1) is **2.431416 higher** after A1 than after B1, while logP(B1) is8.265094 higher. Across all variants both absolute changes remain positive. Thus “relative suppression” means B gains more relative preference, not a verified inhibitory operation on A. Replacing B with A also removes B history; facilitation and removed inhibition are not separately identified.

## First fork disagrees with full-row direction

Every A/B pair first diverges at x1, action34. A variants share x1=0; B1/B2 use52/57. The A0-minus-B52 margin remains positive under every history and both readouts:

| Supplied history | Raw A0−B52 | Equal norm A0−B52 | Raw global x1 → equal norm |
|---|---:|---:|---|
| h A1 C | 2.019356 | 1.092789 | 0→21 |
| h A2 C | 1.885767 | 0.956924 | 0→21 |
| h B1 C | 1.109342 | 0.367339 | 19→23 |
| h B2 C | 1.131448 | 0.404339 | 23→23 |

First-fork E is **positive in all16 comparisons**: raw+0.664676..+0.910014; normalized+0.526111..+0.725450. Earlier A therefore increases the local A-versus-B x1 margin while decreasing its relative complete-row likelihood. Under h A1 C and h A2 C, raw x1 selects0 even though both coherent B rows have higher complete likelihood than both A rows. This is finite local-versus-row preference disagreement, not evidence that a free continuation would realize either complete candidate.

Primary raw E decomposes algebraically into x1+0.910014, y1−4.356628, x2−3.068882, y2+0.681825 and box-end−0.000009; earlier common-token contributions cancel. After x1, A/B candidates have different internally consistent prefixes. These conditional terms explain the score sum, not unique causal responsibility of a slot or a circuit. No free continuation was run.

## Numerical/runtime acceptance and evidence preservation

All16 scientific calls retain full-vocabulary logits, actual final head inputs and full positions. All four native-incremental qualifications consumed39 supplied actions under the original3084-cap generate path, then stopped after capturing action39 logits **before selecting/emitting a free token**. Original companions/order/padding and MRoPE matched;44 selected full/native argmax and position checks passed. No old3084-token baseline was regenerated.

Maximum cache/full logit discrepancy is0.000184417; maximum2×error/argmax-margin is0.006012156. Effective head reconstruction maximum error is0.000005502. Same-history candidate-fork logit differences are zero. All16 full position tensors are identical. Original effective base+shared-delta rows, tied input rows, absent bias and fixed norm factors are checked; parameters unchanged.

Equal-norm row probabilities use the full vocabulary after coordinate scaling, with FP64 multiplication rounded to FP32; no non-coordinate/EOS logit changes. No input or output parameter was edited. The old onset correction is respected: shared-x1 and whole-row orderings are separate outputs.

Independent verification reconstructs all16 row scores and32 E values with selected logits minus logsumexp, checks all44 native positions, exact common tail/histories, source hashes and current annotation bytes, and rejects a corrupted C token. Maximum independent row-logp error is1.63e-13; FP32-versus-FP64 softmax difference is0.000002908.

Accepted snapshot/current exports and image bytes are unchanged:63 positive rows, no explicit ignore semantics. UNKNOWN is not physical negative. No annotation bank was rewritten; no new physical coverage score is claimed for supplied histories.

Actual cost:176 batch forwards (16scientific+160qualification),20vision calls,95.634 allocated GPU-seconds (0.02656 GPU-hours), peak reserved11,981,029,376 bytes, tensors202,395,272 bytes. GPUs0..3 handled four conditions after the first parity gate. No failed/HOLD/unexecuted scientific cells, no retry or cleanup deletion. All four producer PIDs and the owned launch session ended. Knowledge checker leaves only root-owned frontier integration.

## Interpretation and stop

The fixed literal last row and positions are insufficient to determine the conditional scores: earlier history changes them. Complete-row relative suppression survives both extent variants and norm equalization, while first-fork relative facilitation survives too. This rules out a literal-last-row-only description on this panel, not a contextual last-row state carrying older history.

Strongest remaining alternatives are older spatial/order information and broad contextual compatibility under the synthetic insertion. Distinct visual identities are fixed but do not make coveredness separable from their different locations. The result does not identify an abstract owner ledger, unique suppression mechanism, training origin, missing strength, or native-policy recovery. No Q/K/V/layer source is localized.

The package stops here with a stable unreviewed candidate. Root decides acceptance and whether any new contrast has decision value. No escape rollout,0→30 intervention, norm pulse, checkpoint comparison, training or successor is launched.
