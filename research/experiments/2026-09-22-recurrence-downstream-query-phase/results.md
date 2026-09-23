# Matched-state phase transfer chooses a third coordinate

2026-09-22. Scientific status: BOTH advance categorical predictions fail. Root
predicted retained38; the portable-state-plus-phase alternative predicted999.
Actual global winner is579. Technical status: root independently accepted the
corrected three-cell attempt. The first vision-routing failure remains separate.

## Result

Original mature untied+axis step2444, val7511 row42 x2, query input1703 predicts
action384. Both held conditions receive actual row89 layer0 whole-head output.
Their actual layer1 incoming state matches the native row89 state EXACTLY.
The new treatment also advances live normalized Q and its current self-key in
layers1..27 to saved row89 rotary phase, keeping visible historical K/V native.

| Condition | Global bin | z38 - z999 | P38 | P999 |
|---|---:|---:|---:|---:|
| native | 38 | +1.901578903 | 0.054691631 | 0.008167246 |
| held_native_phase | 38 | +1.780111313 | 0.052605006 | 0.008870223 |
| held_late_phase | 579 | -3.503167152 | 0.000035712 | 0.001186372 |

The new phase treatment changes d by-5.283278465, but999 probability
falls as well. Its stronger odds against38 therefore do not mean absolute
reinforcement of999. The winning bin579 has probability0.022592350
and a top-two gap0.212467194. Its top five are579,575,592,605,582;
38 ranks420 and999 ranks124. No binary margin can substitute for
the global categorical result. Treated future-query logits were captured for
technical completeness and are explicitly outside the scientific endpoint.

## What is established, and what changed in the explanation

Current downstream rotary phase can strongly redirect the multi-candidate
coordinate competition even with identical incoming layer1 state. The late
incoming state plus late query/self-key phase is insufficient to reproduce the
late native999 decision on earlier visible memory. Root's stronger claim that
the earlier context would retain38 is falsified, not rescued by the failure of999.

The treatment preserves current Q/K norms and current self-attention scores.
It leaves actual strictly historical K/V at physical positions[0,1703) identical
in all27 treated layers. The common earlier history is part of the same native
causal trajectory. After incoming state/current phase matching, the late native
query additionally reads positions[1703,2126):423 cached tokens. This identifies
a remaining history-package difference; it does not identify which entries,
their semantic content versus multiplicity, or an accumulated scalar counter.

The strongest limitation is local row coherence. Current row42 begins at1698;
the five prefix keys[1698,1703) retain early phase, while the query and its
self-key at1703 move to late phase. Those five keys contain the row opener,
person label, delimiters and x1. Native row89 instead has its corresponding
five prefix keys at[2121,2126), close to its query. Co-rotating the self-key
preserves self-score but does NOT preserve query-to-prefix relative phase.
The423-token difference includes these late prefix-role cues, not only extra
complete repeated rows. A missing local structural cue could therefore explain
the third winner without a whole-history accumulation mechanism.

An explicit, untested conjecture is that phase separation disrupts current-row
role/coordinate retrieval; several new leading values are near the repeated
y coordinates571/575. That proximity alone does not prove axis confusion.
The next decision-changing control would restore only those five prefix-key
phases for the current query, with live key content/V and older history fixed.
Its implementation must act only on the target query's attention computation:
globally rephasing historical keys would change earlier prefix queries and
their later-layer states, spoiling this isolation. A selected-query SDPA
recomputation with a matched identity control is a bounded route. No layer/head
scan, natural generation, training or new successor has been launched.

This unit is closed at its failed categorical prediction. It does not predict
natural onset/exit timing or establish physical recovery. Coordinate38 was a
user-adjudicated water-person hallucination; neither999 nor this forced579 has
been accepted as useful owner recovery. Peer codebook work remains untouched.

## Independent acceptance and execution repair

Root verified51 source/artifact bindings, model and original native inputs,
both real LM query indices and all28 masks/cache slots. The actual F.sdpa
consumer sees16 heads after native GQA expansion. Root independently rebuilt
Q/K rotation with FP64 complex multiplication and recomputed all vocabulary
outcomes from raw tensors. All off-target Q/K and V/masks remain unchanged at
each consumer; all companion batch vectors are exactly unchanged. Layer0's
actual consumer and composed output are checked separately.
Native full-vector parity error0; held-state/native-phase parity
against the accepted reverse transfer1.907348633e-05. Layer1 incoming
state error is0 in both held cells. In the late-phase cell, maximum errors are
self-score1.161464044e-06, Q norm
2.019582258e-06, K norm
1.271692625e-05, all below2e-4.
The actual Q treatment is nonzero (maximum element change34.020042).

Attempt001 failed when the global registry intercepted vision attention; it
never reached a text layer and produced no cell logits. Root took over after
the worker correctly stopped. The [repair ruling](repair-ruling-01.md) also fixes
a would-be rejection of the intentional layer0 output substitution and replaces
a self-referential CPU check with actual-consumer positive/mutation tests.
The wrong-query and wrong-sine mutations now reach the installed SDPA consumer
and fail. Vision passthrough is freshly proven by attempt002. Failed producer
bytes and receipt are preserved; no failed run is labeled a scientific null.

Attempt002 uses3 model/3 vision calls,33.841s package time,
peak reserved13,214,154,752 bytes on GPU4, tensor payload
19,215,367 bytes. Including attempt001 (7.162s), total cost is
4 model/4 vision invocations. Both processes are terminal. Research-knowledge
and output-layout checks pass; these certify record plumbing, not the science.

## Prompting observation

The separate short mapping turn worked: Luna-max identified live pre-Q versus
post-RoPE layout and the SDPA entry before implementation. The coupled global
registry/vision boundary and composed layer0 intervention still failed in its
producer, despite being named in the packet. More prose alone did not close the
gap. Root completed a bounded repair and strengthened the test to exercise the
actual dispatch/consumer path, using the observed failure as the counterexample.
For future novel instrumentation, freeze that small real caller and mutation
test as a separate execution milestone before combining it with the full probe.
Luna remains useful for bounded implementation; root retains coupled boundary
decisions, scientific hypotheses and independent acceptance. No capability ceiling,
autonomous-lead equivalence or measured cost advantage is established. No global
skill or memory was edited.

- [Frozen protocol](unit.md).
- [Independent readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-downstream-query-phase/lead-checks/phase-readback.json).
- [Acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-downstream-query-phase/lead-acceptance.json).
