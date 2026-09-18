# Owner-grounded recurrence onset: donut417044

**Lead-accepted; bounded onset diagnosis, no successor launched.** Under the original readout, at the earliest physically confirmed recurrence in this bounded review, the emitted repeated-owner row wins both its first competing token and complete-row likelihood against both supported uncovered-owner extents. This finite original-policy contrast does **not** show distinct-owner local greedy selection failure. Equal output norms remove coord0's global x1 win but do not reverse the shared-x1 A-versus-B ordering. They **do reverse complete-row likelihood for primary B**, while secondary B remains below A. These are distinct decision surfaces; neither predicts a free counterfactual continuation. [Root receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/lead-acceptance.json) owns acceptance and this interpretation correction; sealed candidate bytes remain unchanged.

[Authoritative result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/result.json), [full reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/reduction.json), [artifact map](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/ARTIFACTS.md), [frozen protocol](unit.md).

## Physical admission and onset boundary

Nine target rows were inspected: original rows1–8 and saved norm row4, with full-image context and selected local overlays. No other image or visual census was opened.

- Original rows3→4 are the **earliest possible** recurrence. Their left-edge boxes overlap a partially visible neighbor and the dominant adjacent donut; exact intended identity/extent remains **HOLD**. They are not silently treated as distinct owners.
- Original row5 `[0,256,61,315]` seeds admitted owner A; row6 `[0,256,66,315]` clearly revisits the same dominant donut. Thus row6 is the **earliest confirmed** recurrence in this review, not necessarily the actual first recurrence. Its action boundary is49; x1 is action54. Both rows remain geometrically valid.
- A is current positive `-1693019979812657`. Distinct neighboring B, `-8380415314849442`, is unvisited through original row5. Its two supported realizations are saved norm row4 `[52,274,106,327]` and refined extent `[57,274,105,327]`. The saved generated extent is primary; GT geometry is not the sole realization.
- Rows7/8 revisit the ambiguous upper region/A region, respectively. First exact literal repeat is only row15, so literal-repeat onset is a later proxy.

[Full contrast image](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/review/contrast-full.png), [local contrast](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/review/contrast-local.png), [physical sidecar](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-owner-recurrence-onset/review/admission.json). Overlap alone does not establish identity. Resolving rows3/4 could move onset earlier and change the answer at that earlier boundary; it does not erase the measured row6 contrast.

## Finite coherent candidates

All four candidates use the same `donut` spelling/wrapper and10-token row serialization. Complete-row scores include opener, description, delimiters and four coordinates; the following boundary is captured separately, not included in row likelihood. Later token states differ after each candidate's first fork.

| History / next row | A seed extent logp | A recurrence extent logp | B saved extent logp | B refined extent logp |
|---|---:|---:|---:|---:|
| Original seed, row5 | -10.636638 | -10.629973 | -15.103026 | -15.516281 |
| Original recurrence, row6 | -13.140623 | **-12.577035** | -13.124262 | -13.608137 |
| Saved norm normal transition, row4 | -22.336354 | -22.279493 | **-8.963479** | -10.590100 |

At recurrence, the actually emitted A row exceeds primary B by0.547227 log units and secondary B by1.031102. This is a finite candidate preference mismatch with the goal of selecting a distinct unvisited owner. It does not establish the best likelihood over every owner/extent or an owner-level probability. B slightly exceeds the alternative A seed extent by0.016362, illustrating extent sensitivity; that alternative A row is not the actual recurrence output. At the seed window, the alternate A extent slightly exceeds the greedy row while denoting the same owner, a separate within-owner serialization effect.

The normal-transition comparison strongly prefers B under original readout on the **saved equal-norm target history**. Original batch companions are retained. It is a separate, deliberately conditioned history, not an original-policy natural trace or an isolated A-count intervention: earlier geometry, route and history length differ. It shows available coherent B preference in another same-image context without identifying which earlier difference supplies it.

Earlier possible-onset rows3/4 retain descriptive observed/previous-row scores and full tensors. Their physical identity uncertainty prevents assigning a covered/uncovered interpretation to those likelihood differences.

## Same-state readout decomposition

Actual bias-free effective output rows are base+shared embedding delta; output/input coordinate rows are equal and tied in this runtime. No parameter changed. Original fixed FP64 norm factors are applied offline to all1000 coordinate logits and rounded back to native FP32, leaving non-coordinate logits unchanged.

| History | A(x1=0) minus B(x1=52), raw | Equal norm | Raw global x1 winner → equal norm |
|---|---:|---:|---|
| Seed | +4.909229 | +3.908205 | 0→14 |
| Recurrence | +2.856792 | +1.863773 | 0→30 |
| Normal transition | -2.684990 | -3.509225 | 46→52 |

At recurrence, coord0 falls from rank1 to15; primary B remains rank50. Equal norms expose another interior competitor, not the selected uncovered-owner row. The0-versus52 angular preference survives: cosine0=0.271494 versus cosine52=0.247409, with norms1.357017 versus1.293269 and hidden norm58.958023. On the separate normal history, their cosines reverse to0.250800 versus0.306121. Hidden norm alone cannot change bias-free coordinate ordering. Input-embedding influence remains upstream and untested.

### Root correction: full-row probability changes differently from the x1 fork

At the same recurrence row boundary, applying the exact output-only norm policy at every position of each coherent forced candidate yields the following full-vocabulary softmax log likelihoods:

| Candidate | Original row logp | Equal-norm row logp |
|---|---:|---:|
| Actual A recurrence | -12.577035 | -13.362865 |
| Primary B, saved generated extent | -13.124262 | **-12.872099** |
| Secondary B, refined extent | -13.608137 | -13.786149 |

Primary B therefore moves from0.547227 below A to0.490765 above A; secondary B remains0.423284 below A after scaling. Output-only scaling leaves the model states unchanged along each specified forced history, so these are valid conditional row probabilities under the normalized softmax readout. They exclude the next boundary. They are neither owner-level probability masses nor evidence that greedy emits B: the shared-x1 pair still favors A0 over B52, and the full-vocabulary winner is a third choice, coord30. The candidate introduction's unqualified no-ordering-reversal statement is narrowed to the shared-x1 fork.

On the recurrence A candidate's later internally conditioned states, equal norms also change x2 winner66→69 and y2 winner315→317. These are **separate fixed-state sensitivities**, not one generated counterfactual row: choosing a different x1 would change the later states. Neither global x1 reversal nor changed later coordinates establish useful free continuation.

At the original recurrence row opener, opener-minus-EOS is9.058554; after the complete A row it is8.907362. Geometry-invalid conditional coordinate mass is0.00006522 at x2 and0.00202022 at y2. Thus this onset is a valid-box recurrence with strong continuation preference, not the later invalid-box or premature-EOS mode. Full-vocabulary ranks, all candidate slot masses, endpoint/interior norm/cosine terms and next-boundary states remain in the reduction/tensors; no coordinate marginals are multiplied into owner probabilities.

## Annotation version and saved-output accounting

Current working/source export and last-export receipt match the accepted snapshot; original image bytes match.63 current positives, no explicit ignore semantics. No image preprocessing or historical labels were replaced. Independent saved-output rescoring gives:

| Saved policy | Old17 matches | Current63 matches / FN | Current UNKNOWN rows | Complete invalid / malformed | Strict valid repeats | Stop |
|---|---:|---:|---:|---:|---:|---|
| Original | 2 | 4 /59 | 304 | 0 /1 | 266 | cap3084 |
| Equal norm | 14 | 30 /33 | 2 | 0 /0 | 0 | EOS320 |

These are geometric one-to-one positive matches, not exhaustive physical TP/FP/FN labels. UNKNOWN includes duplicate/extent and potentially real unlabeled predictions. This package did not regenerate the norm trajectory, relabel unmatched predictions, or convert the earlier14/17 claim to14/63. Exact old/current owner identities and parse ledgers are retained.

## Technical validity and cost

One instrumented original native heterogeneous bs4 replay reproduces **all four full token sequences and stops exactly**, with original media/prompt/padding/MRoPE, R16 adapter/paired embeddings, FP32/SDPA/RP1 and cap3084.16 full-prefix candidate calls recompute multimodal context.98 matching-history native/full comparisons preserve argmax; maximum logit discrepancy0.000138283 and maximum2×error/argmax-margin0.0138433. These are shared-position checks, not98 independent scientific observations. Effective-row/head-input reconstruction maximum error is0.000006287. Candidate-fork numerical differences are below their decision margins. Positions were checked exactly.

Independent CPU verification reconstructs likelihoods by selected logits minus logsumexp, fork margins, source hashes, current-bank scores and all four native sequences. A one-token corruption fails the exact-sequence gate. No model gradients or parameter version changes occurred.

An inherited whole-file native.py binding first failed **before model loading/forwards**. Git recovery found the exact old bytes; the three used preparation/replay functions are AST-identical, with only unused helpers added. The failed receipt, old panel, old/current source and repair record remain. No scientific contrast changed and no successful model cell was rerun.

Actual cost:3100 batch forwards,17 vision forwards,415.599 allocated GPU-seconds (0.11544 GPU-hours), peak reserved17,666,408,448 bytes,173,916,954 tensor bytes. One target image, one exact native replay and16 candidate forward calls. One zero-forward mechanical failure. All owned producer PIDs ended, exits0; no active worker/job or cleanup deletion. Controls351017/7116 and all Q/K/V/attention/layer captures are explicitly unexecuted. Knowledge validation leaves only root-owned frontier integration.

## Remaining alternatives and stop

The strongest account supported here is **history-conditioned coherent preference plus local output-scale sensitivity**, with no identified upstream origin. Spatial/extent realization, positional/order context and generic route dependence remain alternatives; an owner ledger or specialized repetition circuit is not established. R16 includes ranking, so this is not pure-SFT or training-origin evidence. The earliest possible physical recurrence remains HOLD.

No specific query/key/value source is localized by these head-level measurements. If root wants one next causal decision, the smallest new test is the measured recurrence x1 choice0→30 followed by original greedy, compared with the saved original continuation and judged by distinct-owner coverage, incumbent retention and debt. This would test sufficiency after withdrawal, not prove why the hidden state formed. It is a recommendation only; no successor or repair trial is launched. Stop for root acceptance.

## Independent lead acceptance

Root verified all107 bound files, reran the sealed reduction JSON-exact, replayed the independent verifier without overwriting candidate artifacts, and separately reconstructed all16 normalized row likelihoods. The98 shared-history position/argmax comparisons pass; no new model forwards were used. Root viewed full/local bbox evidence and retains rows3/4 as HOLD while admitting rows5/6 as the same dominant visible donut and B as its distinct right neighbor. One bounded alignment review found no runtime blocker and identified the full-row normalization distinction corrected above. All owned producer PIDs are absent with successful recorded exits; the designated worker is idle. The initial root replay lacked the launcher's PYTHONPATH and performed no model work; the corrected CPU verifier inserts the worktree import path and passes.

This package closes at finite conditional preference/readout evidence. No new token intervention, layer/cache sweep, training, or checkpoint-stage probe is launched. The user's explicit checkpoint-probe deferral remains in force; root will discuss the accepted result before choosing any successor.
