# Accepted fixed-first-template counterexample

2026-09-22. The advance ALL-retains-native-winner prediction fails in both
fixed cases. Technical status: accepted. The matched LAST-only control also
fails in both, so the result does not isolate distributed replacement extent.

| Case | Native | Corrected penultimate donor, last row | Corrected FIRST donor, last row | FIRST template, all later repeats |
|---|---:|---:|---:|---:|
| val | 999 | 999 | 38 | 38 |
| train | 348 | 348 | 599 | 599 |

The penultimate-donor result is the separately accepted
[position-correct substitution](../2026-09-22-recurrence-positioned-duplicate/results.md).
Both donor tests use identical destination phase, row count, other native history
and freely recomputed current S. The FIRST donor and penultimate donor share all
nine literal token IDs but have different contextual K/V. Their different
outcomes establish source-state noninterchangeability at these two native exits.

| Case | Cell | Global top-two gap | Repeated-coordinate minus native-exit margin | Full-vector max difference from NN |
|---|---|---:|---:|---:|
| val | NN | 0.006084442 | -0.006084442 | 0.000000000 |
| val | LAST | 0.368934631 | 0.368934631 | 1.209962130 |
| val | ALL | 0.187389374 | 0.187389374 | 3.961851835 |
| train | NN | 0.028825760 | -0.028825760 | 0.000000000 |
| train | LAST | 0.117696762 | -0.094009399 | 0.780993104 |
| train | ALL | 0.201005936 | -0.132448196 | 1.145339370 |

The margins are val z38-z999 and train z350-z348. In train the global winner is
599, although348 still exceeds350; this again exposes the insufficiency of a
binary repeat-versus-exit margin. Full probabilities and top-five competitors
are retained in the independent readback.

## Mechanistic decision

The FIRST-row contextual template is not locally interchangeable with the
recent donor. Because LAST fails before changing replacement extent, ALL cannot
establish that distributed contextual accumulation is necessary. Its margins
can change further without resolving that missing premise.

An onset transient is a live alternative: the first repeated row follows a
different preceding history. Earlier-layer positional computations can also
already affect pre-K and V; correcting the final K rotation does not remove
those effects. Gradual accumulation is neither identified nor ruled out.
The intervention jointly replaces K and V and does not assign the failure to
one of them. Layer0 first-versus-last pre-K and V are exactly equal in both
cases, while subsequent contextual processing changes representations; these
saved norm differences are descriptive, not a layer attribution.

Together with the accepted phase crossing, mass/profile counterexample and
current-state clamp, the supported local mechanism is a multi-candidate
competition affected by BOTH relative key phase and contextual cache state.
The attention softmax changes total row weight and within-row weighting; the
resulting current state changes subsequent Q/K/V across layers. A nearby donor
can substitute at the proper phase, while a first-row donor cannot. Neither a
context-free repeated-evidence model nor a gradual scalar accumulator has been
established. The precise natural exit time and physical-owner recovery remain
outside these endpoints.

The next decision-changing prediction would distinguish a transient at entry
from sustained decision-relevant evolution within the established repeat
plateau. That requires a separately frozen temporal test; this branch closes
without a rescue donor, fraction sweep, head/layer scan or further model work.
The user's broad continuation grant remains valid.

## Independent acceptance and cost

Root checked65 bindings, all six full-vocabulary vectors, and every substituted
actual-consumer K/V hash reconstructed from the saved first pre-K/V and native
per-row phase: val LAST28 / ALL1708 layer-rows; train LAST28 / ALL84. All28 layers'
source, native mask/slots/current phase, unchanged outside history/companions,
and LAST's unchanged earlier repeats qualify. Both native vectors reproduce the
accepted references exactly. FP64 complex-pair oracle errors are2.940e-5 and
2.618e-5, below2e-4. Source pre-K→native post-K replay is exact0. Root reran the
cache selfcheck successfully and inspected the actual patch/consumer path.

Eight model/two vision calls,57.4751s package execution,64.7717s elapsed,
16,489,906,176 bytes peak reserved. Process terminal, no retries or rescue cells.

- [Frozen protocol](unit.md)
- [Root-verified exact-row selection](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/selection.json)
- [Original result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/attempt-001/result.json)
- [Independent readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/lead-checks/template-readback.json)
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-fixed-template/lead-acceptance.json)
