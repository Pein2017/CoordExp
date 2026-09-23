# Recurrence exit crossing: accepted local history-following result

Date: 2026-09-22. Lead-accepted on the frozen [unit](unit.md).
[Machine acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/lead-acceptance.json) and
[raw execution result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/attempt-002/result.json) own exact bindings.

## Finding

The predeclared bidirectional positional transport prediction failed in both
crossed cells. Both choices instead followed their history package. Current-row
position still had a large causal effect on the fixed-pair margin, opposing the
added-history effect in this local crossing. This is one exposed numerical exit,
not a population mechanism or a physical owner-exclusion result.

Same mature untied+axis step-2444 composition, val:7511, original native batch4,
FP32/SDPA. E/L histories precede native rows88/89 (zero-based). The identical six
current-prefix tokens are placed at native positions1176–1181 or1185–1190 on all
three MRoPE axes. Crosses create intentional positional gaps/overlaps while
preserving actual causal order and history tokens within each H.

| H / current-prefix positions | z38 minus z999 | Actual winning bin | Vocabulary winner gap |
|---|---:|---:|---:|
| E_E | +0.206699 | 38 | 0.206699 |
| E_L | +1.133533 | 38 | 0.344786 |
| L_E | -0.643066 | 999 | 0.643066 |
| L_L | -0.006081 | 999 | 0.006081 |

The full-vocabulary winner remains38 in E_L although999 is no longer runner-up;
the reported fixed-pair margin must not be confused with the vocabulary gap.
Both crossed winner gaps exceed the predeclared0.0004 numerical guard.

At fixed history, advancing S by nine positions changes the margin by
+0.926834 (E history) and+0.636986 (L history),
favoring38. At fixed S positions, adding the last repeated row and its contextual
history changes it by-0.849766 (E positions) and
-1.139614 (L positions), favoring999. Factorial interaction is
-0.289848. These deterministic contrasts are not independent
samples and carry no estimated population uncertainty.

## Interpretation and limits

This rejects the specific claim that this native exit is transported by swapping
current-prefix positions alone. It does not reject positional involvement: the
position effects are substantial and oppose the observed exit direction. The
natural diagonal change combines both factors. The history factor transports a
whole additional contextual row, including length and its keys/values; it does
not yet isolate literal content, attention multiplicity, or an abstract state.

Moving the whole current prefix also changes its internal contextual states.
The result is about this explicit prefix-position intervention, not isolated
last-query rotation. No claim about a position-only clock or hidden accumulator
follows. Nor does repeated38 select a real person: the user identified this
water-region prediction as hallucination and the immediate999 rectangle as a
bad multi-owner box. Later valid kite/person predictions remain a separate event.

The next useful distinction is whether the added-history effect depends on what
row was written at matched length/positions. A new bounded content contrast, if
chosen by the lead under current authorization, is a separate experiment; no
layer/position sweep follows from this unit.

## Technical acceptance and cost

Root independently reconstructed native token offsets798/807 and their exact
six-token common prefix from literal openers, verified all source hashes, loaded
all five saved vocabulary vectors and recomputed margins, argmax, gaps, effects
and no-op equality. Native top-two error is at most7.6294e-06 against the saved
source; the full-vocabulary no-op difference is exactly0.0. The maintained
producer selfcheck passed, including wrong-slot rejection.

The executed rotary hook asserts complete equality between consumed and
manifest-bound requested positions before incrementing its counter. Every call
retained one successful hook. This equality chain, the saved S vectors and input
hashes establish consumption without an additional run just to serialize equal
values again. First/last attention mask and physical cache-position hashes agree
within each history. Root also inspected the installed text forward's shared
mask construction and rotary path. The complete
[consumption proof](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/lead-checks/consumption-proof.json) preserves that basis.

Attempt001 stopped before forward because an exact identity comparison included
a relocated loader-source path. Root verified identical bytes and size at the
current maintained path and authorized only that path normalization; every other
identity field still matched exactly. Attempt002 supplies the accepted cells.
The loader capture supplement was written after forwards, explicitly labeled;
its hash equals the pre-forward identity and the old source. No historical source
was executed. Attempt003 was proposed only for redundant serialization and was
not launched; the maintained producer was restored to the executed hash.

Total:5 model forwards,5 vision forwards, one physical GPU4,
14.552 seconds in the forward phase; loading and the
zero-forward failed attempt are additional process time in their receipts.
Peak reserved memory was11,230,248,960 bytes. Both executed processes are terminal.
No training, free generation, other images or checkpoint changes.
