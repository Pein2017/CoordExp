# Coordinate-pair readout support

Lead-accepted 2026-09-19T04:53:31.887023+00:00. Independent CPU reduction and saved-score verification reproduce the candidate exactly; the bounded review found no blocker. [Acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/lead-acceptance.json).

Sparse continuing changes on logits0/1 are sufficient for sign23-like low numerical recurrence on both selected donut models, while removing that pair from sign23 loses the near-run1/zero-all-pairs outcome. Untied responds to only0; tied requires the pair among these tested supports. The pattern is case-specific, not a universal two-logit cause or physical recovery.

## Complete fixed panel

All66 cells run from the original11 failure prefixes (six images), under original/sign23/only0/only1/pair01/except01. All22 controls reproduce accepted direction-study outputs exactly. Every generated row is free. CPU reduction is JSON-exact; all17,909 savedsteps satisfy score-support formulas, untouched-coordinate bit identity and actual greedy emission. A wrong-support CPU corruption is detected.

Each entry below is **longest near-run / all-pairs near count / invalid rows / stop / complete rows**. R=32-row limit; E=EOS. Exact/all-pairs exact counts, native/alternate returns and row-level tokens remain in `reduction.json`/`outcomes.csv`; invalid literal rows are included in all-pairs comparisons.

| Boundary | Original | sign23 | only0 | only1 | pair01 | except01 |
|---|---|---|---|---|---|---|
| untied-885-failure | 21/253/8/R/32 | 26/406/0/R/32 | 18/254/0/R/32 | 21/253/8/R/32 | 15/156/0/R/32 | 12/163/0/R/32 |
| tied-885-failure | 12/171/0/R/32 | 21/244/0/R/32 | 12/159/0/R/32 | 12/171/0/R/32 | 12/161/0/R/32 | 23/289/0/R/32 |
| untied-5586-failure | 30/436/0/R/32 | 10/86/0/R/32 | 3/3/0/E/31 | 30/436/0/R/32 | 20/245/0/R/32 | 12/157/0/R/32 |
| tied-5586-failure | 25/352/0/R/32 | 11/125/0/R/32 | 25/352/0/R/32 | 25/352/0/R/32 | 14/173/0/R/32 | 16/193/0/R/32 |
| untied-7511-failure | 11/125/0/R/32 | 17/185/0/R/32 | 6/40/1/E/25 | 11/125/0/R/32 | 10/52/1/E/25 | 17/241/0/R/32 |
| untied-14038-failure | 5/18/0/E/16 | 4/8/0/E/15 | 5/18/0/E/16 | 5/18/0/E/16 | 5/18/0/E/16 | 4/8/0/E/15 |
| tied-14038-failure | 8/41/11/E/24 | 9/61/1/E/27 | 8/41/11/E/24 | 8/41/11/E/24 | 8/41/11/E/24 | 9/61/1/E/27 |
| untied-632-failure | 18/250/0/R/32 | 25/321/0/R/32 | 18/250/0/R/32 | 18/250/0/R/32 | 18/250/0/R/32 | 25/321/0/R/32 |
| tied-632-failure | 3/16/1/R/32 | 6/37/1/R/32 | 3/16/1/R/32 | 3/16/1/R/32 | 3/16/1/R/32 | 6/37/1/R/32 |
| untied-417044-failure | 14/139/0/R/32 | 1/0/0/E/28 | 1/0/0/E/26 | 14/139/0/R/32 | 1/0/0/E/30 | 10/117/0/R/32 |
| tied-417044-failure | 17/146/1/R/32 | 1/0/0/R/32 | 17/145/1/R/32 | 17/146/1/R/32 | 1/0/0/R/32 | 15/169/0/R/32 |

All cells have zero malformed openers and zero512-token caps. All-pairs counts are supplementary numerical accounting, not owner counts or independent trials.

## What the contrast establishes

- **Tied donut:** pair01 gives near-run1 and zero near pairs, zero invalid rows, at32 rows; sign23 has the same numerical endpoints. Both singletons retain near-run17. except01 gives near-run15/169 near pairs at32 rows. This supports a joint sparse score contribution in this selected case, without an early-EOS explanation.
- **Untied donut:** pair01 and only0 each give near-run1/zero near pairs/zero invalidity, ending after30 and26 rows; sign23 ends after28. except01 retains near-run10/117 near pairs at32 rows. Termination remains a limit; no physical coverage claim. only0 first emits x1=13, whereas pair01 first emits1: the numerical result is not tied to emitting1 specifically.
- **only1:** all11 entire continuations are identical to original. This is a null for this frozen singleton operator on these routes, not general irrelevance of coordinate1.
- **Beyond donut:** untied5586 only0 produces near-run3/3 near pairs at31 rows thenEOS, versus original30/436; pair01 still has20/245 and except01 has12/157. Tied5586 pair01 and except01 both partly reduce runs. These support multiple/partial routes rather than one universal pair explanation.
- **Counterexamples retained:** both14038 and632 model pairs show no benefit from pair01, while sign23/except01 preserve their own beneficial or adverse behavior. Untied885 full sign23 worsens recurrence, while pair01 and except01 each reduce its numerical burden. Global-support effects are path-dependent and not a simple sum of per-support rollout benefits.

## Technical closure

All8 GPUs used. 17,909 model forwards, 66 vision passes, 2505.334 allocated GPU-seconds (0.696 GPU-hours), 795,095,099 tensor bytes. No failed GPU attempt, retry, new arm or predecessor rewrite. All pilot/scaleout wrappers observed exit0; all owned jobs and the Luna-max child ended.271 source/artifact bindings verified with no unresolved gap.

The producer qualification fields named `...fp64` compare final-cast scores toFP64 formula values; they include cast residuals. Independent `verification.json` separately reconstructs pre-cast support identities and final-cast scores. The initial parent file-discovery pass found zero cells because the pilot was nested; discovery was corrected before gate release and all6 pilot cells checked, without GPU repetition.

Root registered the result at acceptance. Exact commands and artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-pair-readout/ARTIFACTS.md`.

## Limits

- Support magnitude intentionally differs; this is not a magnitude-matched test. Coefficients/sign23 unchanged.
- Outcome-informed support/field selection, evaluated on all11 original failure boundaries; not a fresh cohort or population rate.
- All generatedrows free; no first-row exclusion or forced token.
- Near/all-pairs numerical boxes are not physicalowners; EOS/another pattern/less repetition is not recovery.
- Changing0/1 output scores does not identify input embeddings, unique circuitry or training origin; legal0/1 values must not be treated as banned.
- T/U is a model-package comparison; targetoutputs only, companions not interpreted.

## Root forward-path observations

Comparing raw argmax with actual emission on each saved trajectory, tied donut pair01 changes the winner only at offsets5 and15, the first two x1 slots:0→1, then0→30. Untied pair01 changes only offset5,0→1; untied only0 changes only offset5,0→13. Later native raw argmax already agrees with these low-recurrence paths. This shows sparse realized decisions under a continuing operator; it is not a fresh pulse replay or universal two-token repair. Untied5586 only0 instead changes seven decisions across x1/y1, retaining a counterexample to an all-case early-two-gate explanation. Exact records: `coordination/lead-intervention-positions.json`.

At tied pair01 offset15, the original0-versus30 logit margin is+0.332750. A symmetric raw-margin accounting separates a+0.921884 output-norm term and−0.589133 directional term. Equalizing the two row norms reverses this selected pair, but full coordinate normalization ranks23 above30 at that state. Untied only0's first0-versus13 margin similarly comprises+0.853137 radial and−0.328986 directional terms. Conversely, the first0-versus1 margins remain positive after norm equalization. These are exact descriptive decompositions of saved scores, not causal shares of a circuit or proof that a particular competitor is correct. `coordination/lead-gate-margin-accounting.json` retains the arithmetic.

The next [forward-only margin provenance unit](../2026-09-19-coordinate-margin-provenance/unit.md) traces a fixed eight-state panel through residual attention/MLP contributions and the final output readout. It does not add a repair or layer-patching sweep.
