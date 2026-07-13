# Model Diagnosis Probe Gates

Read only the branch needed for the current diagnosis.

## Tiny causal probe

Before expensive training, use tens to a few hundred examples plus a held-out
slice and enough steps for repeated exposure. Match model, tokenizer,
preprocessing, decode, and batch semantics to the closest baseline. Label the
scope exactly and retain resolved config, sample counts, tokenizer IDs,
trainable groups, loss weights, raw predictions, parse/drop counters, and
metrics.

Track valid versus invalid target mass, entropy/KL inside the valid set,
stop/continue margin, allowed token-type mass, malformed/duplicate/truncation
rates, effective target counts, mask density, support size, gradients, LR,
NaN/Inf, and update norms when relevant. The gate passes only when intended
terms move, schema health holds, free rollout agrees with the teacher-forced
trend, and no provenance, mask, or precision contradiction remains.

## Mechanism-conditioned negative result

Before calling a denoising, robustness, prefix, hidden-state,
coordinate-basin, binding, or duplicate-control mechanism irrelevant, verify:

- a known healthy baseline and exact artifact/config identity;
- that the perturbation can reach the documented state basin;
- slot-wise `x1`, `y1`, `x2`, `y2`, boundary/control, and stop/continue effects;
- mechanism-sensitive rows rather than aggregate averages alone;
- wrong-control or same-description competitor prefixes where relevant;
- the hidden-state, visual-region, or generated-bad-prefix handle identified by
  prior evidence;
- sparse-sampling traps such as one object per image or mostly insensitive rows.

Reconcile conflicts with prior mechanism notes. Use `probe handle mismatch` or
`inconclusive-needs-mechanism-panel` when the perturbation does not target the
claimed surface.

## Stage-1 coordinate locality

For SoftCE, Gaussian, or hard-CE coordinate objectives, compare `x1`, `y1`,
`x2`, and `y2` separately under both teacher-forced and self-prefix logits.
Pair distribution tables with plots, keep variants separate, and label rollout
scope with checkpoint and decode settings.

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_hard_ce_coord_logit_locality.py --help
python -m pytest tests/test_hard_ce_coord_logit_locality.py -q
```

Set `PYTHONPATH=/data/CoordExp` when a direct script cannot import `src`.
