# Both anchor aging and repeated-pool growth move the first-layer readout

2026-09-22. Scientific status: root's advance anchor-greater-than-pool prediction
passes narrowly under the frozen symmetric decomposition. Both components are
substantial. Technical status: independently accepted. This is a local attention
computation result, not final-logit or numerical-exit causal identification.

The assay uses mature untied+axis step2444, val7511, original batch/history and
layer0. It crosses native score groups at row42 (15 complete repeats) and row89
(62 complete repeats). Anchor means ALL keys before the first exact repeated
row: image, prompt and pre-repeat output. It does not mean image tokens alone.
The six current-prefix tokens are held at the new native score/value state.

| Factorial cell | L2 distance from old/old | L2 distance from new/new |
|---|---:|---:|
| Old anchor / old pool | 0.000000 | 0.939008 |
| New anchor / old pool | 0.659178 | 0.571740 |
| Old anchor / new pool | 0.461709 | 0.580942 |
| New anchor / new pool | 0.939008 | 0.000000 |

The all16-head2048-dimensional displacement has norm0.939008, well
above its active threshold4.3211215e-05. The prespecified symmetric
contributions have the following signed projections on that displacement:

| Contribution | Projection coefficient | Vector norm | Perpendicular norm |
|---|---:|---:|---:|
| Anchor score aging | 0.565764 | 0.611790 | 0.303402 |
| Repeated-pool growth | 0.434236 | 0.508246 | 0.303402 |

These coefficients sum to1 by the declared allocation convention; they are not
probabilities or model-wide causal fractions. Anchor exceeds0.5, as predicted,
but56.6% versus43.4% is balanced enough that neither factor can be ignored in
this readout. The interaction norm is0.216493, or
23.06% of the net displacement norm.
The equal perpendicular contribution norms describe cancellation; simply
ranking raw component magnitudes would obscure it.

All16 heads were included before inspection. As a supplementary observation,
head13 contains80.15% of squared net head displacement. That is a
geometric concentration before o_proj; it neither proves final-logit importance
nor authorizes a selected-head intervention. All head data are retained.

## Concrete computation and evidence boundary

The accepted trajectory already showed exact identical layer0 pre-Q at the
coordinate role and per-role pre-K/V across repeated rows. Native capture here
confirms the same Q, phases and stationary V. Anchor keys/values are the same
stored tensors for the two queries, while the query phase changes relative to
them. The repeated group changes from15 to62 complete rows; its score profile
can be expressed in relative ages. Concatenating either anchor score group with
either repeated score/value group and applying the actual softmax computes all
four head outputs without changing the network or inventing a latent counter.
Softmax normalization couples the factors, hence the interaction.

This supplies a concrete account of first-layer attention-state movement during
an exact-token plateau. The strongest single-factor reading is weakened: both
anchor aging and repeated-pool growth contribute appreciably in the fixed
comparison. It does not establish that either contribution is necessary for
the final999 decision, that an image is forgotten, or that a count threshold
predicts natural exit. The original water-person and exit rectangle remain
user-adjudicated bad predictions.

The next decision-changing causal contrast would pass these already computed
whole-head counterfactual outputs through the unchanged downstream network at
the fixed exit query, with an exact native/sham control and original history.
That would test whether these local changes alter the full-vocabulary decision.
It is distinct from the present head-space decomposition and was not launched.
No new donor/head/layer scan is needed to formulate it.

## Qualification, cost and Luna trial

Root verified56 source/artifact bindings, the original input identities,
both actual selected LM rows, all28 original masks/cache slots and the actual
layer0 o_proj input. All captured Q/phases agree exactly with the prior native
trajectory. Full-vocabulary parity error is3.4332275e-05, with
native winners38/999. FP64 head reconstruction error against actual attention
output is8.7005308e-07. Replacing old C with fixed new C
changes the computed output by at most3.5017492e-06;
the old/new factorial corners match their corresponding actual native outputs
within3.4034301e-06/8.7005308e-07,
all below2e-4. Synthetic anchor-only and pool-only checks recover projection1
and0 respectively. Root reran the source/selector CPU qualification.

One model/one vision forward on GPU4, no retry;9.144s package
time (3.516s forward), peak reserved
12,907,970,560 bytes on the actual device. Native tensor payload
18,683,583 bytes, below64MiB. Process terminal.
No parallel codebook job or unrelated source was changed.

Luna-max executed the bounded capture successfully on its first model attempt,
reusing the qualified native hooks. Root supplied source-binding corrections,
the scientific contrast and independent analysis. The worker's prior suggested
layer1 deletion-versus-anchor clamp was held because layer0 already changes and
those unequal perturbations do not support its proposed comparative verdict.
This supports Luna as a supervised execution worker here; it does not establish
autonomous research-lead equivalence or a measured total-cost advantage.

- [Frozen protocol](unit.md).
- [Independent four-cell readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/lead-checks/groups-readback.json).
- [Figure PNG](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/lead-checks/first-layer-groups.png), [PDF](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/lead-checks/first-layer-groups.pdf).
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/lead-acceptance.json).
