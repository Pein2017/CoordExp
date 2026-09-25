# Reciprocal y1 selection under continued historical reading

Lead-frozen 2026-09-24. One original image and one A history, two continuing read modes, six arms. The user's autonomous-continuation instruction authorizes this bounded investigation; the exact execution grant is `lead-admission-v1.json`. Worker evidence is candidate until independent lead acceptance.

## Decision and prior boundary

At the matched one-record A-history fork, does exchanging selected y1=0/13 between native reading and coordinate-query history blocking exchange the complete fragment/broad regions, while each read mode remains active for the free remainder?

The [accepted read-stage partition](../2026-09-24-recurrence-history-read-stage/results.md) failed both complete-row signatures. Under A history, native reading gives fragment F, coordinate-read blocking gives broad A, and header-read blocking gives invalid geometry. Saved identical header+x1 inputs localize a y1 choice difference; a later matched input also shows a header-dependent x2 difference. Those retrospective observations motivate this prospective intervention, not a relabeling of the closed nonpass. The [CPU feasibility acceptance](../2026-09-24-recurrence-history-read-stage/supporting/lead-y1-bridge-cpu-acceptance-v1.json) independently binds the fork and route.

Leading hypothesis: the selected y1 contributes enough token feedback to redirect subsequent x2/y2 into the opposite region. Strongest comparator: later coordinates still follow their continuing historical-read mode, despite matching the other mode's y1. One-way or other outcomes reject the shared reciprocal claim and remain in the denominator. The [closed x1 intervention](../2026-09-24-recurrence-x1-row-routing/results.md) failed in a different AF/FF two-record state; this is not a repeat of that state or a search across coordinate tokens. No internal mediation fraction or generic recurrence cure is at issue.

## Bound source and fork

Original refined-03 four requests, target index 2, train351017; original untied step2444 FP32 SDPA model, images, prompts, media, companion tokens, positions and source identities. A is the sole history, raw[:9]:

`[151646,8987,151647,151648,151670,151683,152206,152669,151649]`, box **A=[0,13,536,999]**. The numerical fragment reference is **F=[0,0,33,86]**; its physical identity remains HOLD. No F-history or AF/FF-history arm is admitted.

The original padded prompt width is1362; A keys occupy physical[1362,1371). Each arm starts before the next opener, uses its own full-vocabulary greedy tokens, and recomputes the full four-request prefix with vision at every step. All four original rows remain active through raw24. Companions always use their original source tokens. No cache split, vocabulary mask, rephasing, hidden-state patch, supplied header or supplied x1.

Both accepted modes freely emit current `[151646,8987,151647,151648,151670]` before the y1 decision. At current step5 the full input width is1376; x1 at physical1375 predicts y1. Native raw argmax is151670 (0), coordinate-block raw argmax is151683 (13). The policy selects y1 **after saving that unmodified vector**. It first becomes model input at step6, raw14/physical1376, three-axis rotary position[401,401,401]. Other input differences later are only the arm's own generated continuation.

Native mode retains native attention throughout. Coordinate mode blocks only target2 queries physical[1375,1371+t) from prior keys[1362,1371), all28 text layers; the interval is empty when t≤4. Selected cells are9×max(t−4,0):9 at t5,18 at t6,99 at t15. Header queries, current-to-current edges, all complements, companions and rotary positions remain unchanged. **The y1 write never releases or changes the mask.** At t0 every arm executes the native mask.

## Fixed order and selection

| Order | Arm | Continuing mode | Selection at current step5 | Maximum emissions |
|---|---|---|---|---:|
| 1 | native_A | native | full-vocabulary argmax | 9 |
| 2 | coordinate_A | coordinate block | full-vocabulary argmax | 9 |
| 3 | native_sham | native | explicit identity151670 | 9 |
| 4 | coordinate_sham | coordinate block | explicit identity151683 | 9 |
| 5 | native_flip | native |151683, replacing raw argmax151670 |16 |
| 6 | coordinate_flip | coordinate block |151670, replacing raw argmax151683 |16 |

Qualify all four controls before either flip. Every arm independently executes every forward including t0; no reuse. Control nine-token rows must reproduce their accepted same-mode vectors/tokens. Flips must reproduce their own same-mode input, vector and raw argmax through t5 before applying the sole write. All other selections are unrestricted greedy argmax. Record raw argmax, selected ID, policy and actual next-forward consumption separately; identity writes use the same policy/consumer path.

Stop each flip at its first complete row, EOS, early terminator, malformed serialization or16-token cap. Canonical serialization and strict valid geometry are distinct. Coordinate IDs are[151670,152670), with the upper endpoint excluded. Other class, invalid geometry, incomplete, neither-region, malformed, EOS and cap remain scientific outcomes after technical qualification. No second row, automatic retry, extra scoring forward or added arm.

## Frozen complete-row endpoint

Use complete canonical `person` rows (description IDs[8987]) with strict x1<x2 and y1<y2. Coordinate-bin IoU defines broad_A by IoU(A)≥0.5 and IoU(F)≤0.1; fragment_F reverses these roles. A threshold distance≤1e-6 is numerical HOLD. Report both raw IoUs and invalid/other outcomes; these regions are not physical-owner probabilities.

**Shared primary:** native_flip is broad_A **and** coordinate_flip is fragment_F.

**Symmetric persistence comparator:** native_flip is fragment_F **and** coordinate_flip is broad_A.

Both require all technical gates and no numerical HOLD. If neither conjunction holds, category `mixed_or_other`; a qualified failed clause makes its conjunction NONPASS. Technical failure leaves the affected question unanswered, never a scientific negative. Finish both fixed directions regardless of the first scientific outcome. Retain each clause separately. There is **no exact-token secondary** and no fitted threshold. The selected y1 itself gets no prediction credit; save freely emitted x2/y2/terminator and full boxes to show what changed beyond that write.

A reciprocal pass establishes conditional one-token sufficiency between these two continuing read modes. Persistence weakens that shared bridge and supports a role for context beyond the selected y1. Neither is a unique attention/copying mechanism, natural mediation percentage, physical owner resolution, new-owner progress or explanation of a whole recurrence burst. Broad A is an already reported person localization. No semantic-role specificity follows merely from selecting coordinate-position queries.

## Technical qualification

1. CPU preflight reconstructs the original full source, model/embedding identity, all18 accepted same-mode vector/input references and source trace applicability. Freeze exact commands, the new maintained producer and every directly used dependency before model load. Bind the accepted maintained-loader crosswalk; never execute a captured historical source. Reforecast actual shape and storage.
2. Reuse maintained source/mask/parser operations and the qualified CPU/CUDA comparison semantics. Adapt only the local own-prefix/selection/cold-reader route for the six arms, step5 write and possible t9–t15 continuation; do not edit accepted producers. Test the actual caller and serialized checker at prewrite, next-consumption and maximum steps. Reject wrong arm, source/history, policy step/token, dropped/replaced y1, extra write, supplied prefix, companion/position drift, wrong coordinate rectangle, current-to-current/complement mutation, wrong raw-versus-selected ID and incomplete/reordered receipts. Keep parser stop/boundary cases. Use the nearest actual caller tests, not a new test framework or a target test count.
3. Preserve `.detach().cpu()` comparison of serialized CPU inputs/masks with reconstructed tensors. A single bounded non-model CUDA boundary check of the actual new checker may qualify that seam before the model run; record device placement, positive and intended-mutation outcomes and charge its outer process. It must not substitute for actual model consumer qualification; failure stops and returns, with no automatic fixture repair/retry.
4. Each native mode's nine all-four vocabulary vectors and actual source inputs match its accepted references at max-absolute logit error≤2e-4, with exact tokens. Both explicit identity shams match fresh same-mode vectors and states. The native target source trace applies throughout native controls and through t5 before a native flip; coordinate target is counterfactual. Companions retain original trace qualification at every executed step. Never impose original target trace parity on a changed history or counterfactual mask.
5. All28 actual attention consumers must match declared masks, selected/complement counts, positions and original historical state. All current-to-current edges remain native. Prior target states and companion states/full vectors remain equal at available matched steps; later current target states may respond. Verify actual t6 input at physical1376 and every subsequent own-prefix token. At t6 onward no flip has a matched target full-vector reference, including the opposite arm because its mask differs. Beyond the original nine steps use source input/companion traces and consumer invariants, never invented same-step reference vectors.
6. Persist full four-row vectors, exact inputs, actual consumer evidence and finalized cell records before reduction. A separate CPU process reconstructs original source, policy selections, next-token consumption, own prefixes, masks/positions, trace/reference gates, parser, region arithmetic, call order/count and outer charge. Cold reader mutations must test the actual selected-token boundary. Save one stable candidate, terminal receipts and original-image overlay; invalid boxes remain in the ledger even if undrawn.

## Finite resources and ownership

Nominal54, hard maximum68 **each** model forwards, vision forwards and logical emissions; zero reuse. Four explicit policy writes:two identity and two flips. Separate greedy versus policy selections. No extra model diagnostic. CPU feasibility forecasts297.580/375.631 outer seconds and1.537/1.918GB artifacts at nominal/maximum; these are forecasts, not runtime ceilings. Maximum width1386, four requests,24,502,272 pixel elements. A2GiB artifact planning envelope must cover the fresh preflight; a capacity conflict returns before model work. Peak-memory basis and full shape are bound in admission. Charge full outer startup, fixtures and failures. Prior accepted sequence0.8574518933237164GPUh; no elapsed-time or cumulative-hour ceiling is reinstated.

Worker923 `01a0ce4b-9b55-7392-8a25-6a76f9e12c3a`, gpt-6-sol/xhigh, owns only the new `probes/training_set_completion/recurrence_y1_row_routing/`, this unit's supporting/candidate files and its named raw root. Lead `01a0c831-b332-7e12-b931-6ebe2359c99f`, gpt-6-astra/xhigh, owns protocol, admission, state, acceptance and synthesis. No Git/index, peer, shared-runtime or predecessor writes. Technical failure stops the fixed queue and returns evidence promptly. No retry, self-acceptance or successor. Lead independently accepts or rejects; local synthesis is updated in place.
