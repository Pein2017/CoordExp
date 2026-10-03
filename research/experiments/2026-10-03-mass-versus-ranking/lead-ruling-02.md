# Accepted equal conditional repair with mixed natural outcomes

The lead accepts the released package and final readback as
`completed_as_released`. Scientific status is
`both_objectives_repair_with_mixed_preservation_and_recurrence`;
user scientific acceptance remains false.

Both fixed one-update endpoints repair6/6 fresh baseline-illegal contexts and
retain4/4 legal contexts. The original351017 emission is already legal at a
literal zero-margin tie; its original repair denominator remains0. Thus Gmax
shows no additional conditional-legality benefit over Gmass at this fixed update
rule. High aggregate legal mass does not guarantee greedy legality, but the
aggregate-mass update was sufficient to repair these selected errors.

| Natural readout | Anchor | Gmax1 | Gmass1 |
|---|---:|---:|---:|
| Category-correct known owners |274|267|272|
| Geometry-only known owners |275|269|274|
| Category owners gained / lost from anchor |—|15 /22|10 /12|
| Invalid rows |430|193|555|
| Literal valid repeats |89|100|215|
| Near-repeat occurrence pairs |92|987|1026|
| Generated tokens |9922|7924|12729|
| EOS / cap |16 /2|17 /1|15 /3|

Gmass preserves ten more original category owners than Gmax, but gains five
fewer; neither improves aggregate known coverage over the anchor. Gmass has
more invalidity, recurrence and length. These outcomes do not establish objective
dominance. Direct Gmax-to-Gmass exchange is18 gained /13 lost /254 retained
category owners, not a sequential training trajectory. Full geometry transitions,
per-image IDs and burden vectors remain in [results](results.md) and the terminal.

The excess Gmass burden is localized. Relative to Gmax, images13348/351017/7511
contribute respectively +173/+162/+28 invalid rows, +91/+51/−27 valid-repeat
excess and +577/+220/−755 near pairs. The other15 images together contribute
−1 invalid row and−3 near pairs. All555 Gmass invalid rows and all215 valid
repeat excess occur in those three scenes. Images13348 and351017 reach the3084
cap under Gmass versus344/943 tokens under Gmax;7511 caps in both arms.

For a concrete saved-output witness, Gmass13348 contains42 occurrences of
`person [966,915,999,999]` and13 of`[966,913,999,999]`: their IoU.976744 yields
546 of580 near pairs. Raw rowp138 at generated span[1243,1252) witnesses the
first pattern. It also contains72 invalid`person [999,999,999,999]` rows, including
p224 at[2017,2026). This scene has3 known owners versus4 under Gmax. In351017,
the longer fork burst coexists with15 versus12 known owners. These are saved
row occurrences and annotation-relative coverage, not physical entity counts
or proof of a changed stopping mechanism.

Gmass's preclip gradient norm5.214 is much smaller than Gmax's historical93.156,
but their combined serialized update L2 norms are similar:.042588 and.042434.
The output-delta component differs more:.003599 versus.0004524, about8×.
First-step AdamW and the distinct gradient supports make loss or gradient size
an insufficient description of the update. This descriptive difference motivates
a separate parameter-block intervention; it does not explain the natural result.

The lead verified the terminal candidate and24 nested small record bindings,
all three per-image owner transitions, burden aggregates/deltas and fresh
conditional denominators. Existing final-consumer evidence for89 raw artifacts
and the previously verified real training/publication seam were reused. No model,
test suite or successful readback was repeated. All phase exits are0; groups
drained and all11 observed PIDs are freshly absent. The training phase PID was
not separately captured; its tested package cleanup receipt owns that boundary.

Source remains clean/pinned`2876f43db0d81569937a4ff1142129039204d038`.
Counters are1 update,6 training replays,20 HF diagnostics,30 native scores and54
natural requests:30605 new tokens. Active time500.767s,package wall575.239s.
Native child CUDA peaks/internal startup forwards remain unmeasured.
The immutable acceptance at the execution-local round07 output root is
`lead-acceptance-01.json`, SHA256
`ff045f47565bdce6c92e1c8e08eca207876d0b591a20c2db660491acb1825d3f`.

Fresh anchor and Gmax natural measurement records exactly match round06; this
does not establish global determinism. Gmax is a saved realized training outcome,
not a fresh training replicate. Evidence remains the fitted18-image/570-label
cohort with one fresh native observation per input. Unknown/unmatched is neutral.
No physical recovery, causal owner-loss explanation or GT training feedback follows.
Round07 ends here; any following unit has a separate lead-owned protocol.
