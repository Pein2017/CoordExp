---
title: Image2299 checkpoint contrast reveals late, prefix-conditioned coordinate readouts
description: Both models resolve selected exact coordinates late; overfit strongly sharpens its own route without generally continuing the Source terminal prefix.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-08-image2299-logit-lens
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

## Outcome and evidence boundary

Mechanics are **lead-accepted**; the scientific result is a descriptive
same-prefix Logit Lens comparison on one training image. It nominates late
coordinate readout and prefix dependence, not a proven causal computation.
[Unit](unit.md) owns the protocol; [acceptance](lead-acceptance.json) binds the
immutable execution receipt and CPU-derived summary.

Two adapters: sorted-xy Source step-2444 and the prior 140-update Human13
magnitude-only cross-entropy overfit. Both use the same base, image, prompt,
tokenizer, embedding delta and unchanged output head. Full final-norm and
sampled head checks additionally agreed across loaded sessions; the head check
is explicitly a 4,096-element sample, not a full head hash.

Two native greedy RP1.0 generations yielded respectively 262 tokens/29 closed
rows and 415 tokens/46 closed rows, both with EOS, neither capped. These are
row counts, **not fresh unique-owner or IoU measurements**. Each trajectory was
replayed on both adapters, at the same 21 text positions across 28 decoder
blocks. Every statement below is conditional on that prefix-origin stratum.

## Observations

### Exact coordinates are late-readable in both models

All layer numbers below are **1-based**, whereas raw JSON uses 0-based indices.
At the 12 selected coordinate positions per trajectory, the actual next token
is not rank one in any of blocks 1–24, for either checkpoint and either prefix.

| Prefix origin / readout checkpoint | Block 25 | Block 26 | Block 27 | Block 28 |
| --- | ---: | ---: | ---: | ---: |
| Source / Source | 1/12 | 3/12 | 7/12 | 12/12 |
| Source / overfit | 0/12 | 0/12 | 1/12 | 1/12 |
| overfit / Source | 0/12 | 0/12 | 0/12 | 0/12 |
| overfit / overfit | 0/12 | 0/12 | 3/12 | 12/12 |

This does **not** support the simple story that overfit makes these exact
coordinates become the top readout earlier. It does not rule out useful
earlier representations in a different basis or distributed state.
Own-trajectory final top-one agreement is partly expected from greedy decoding;
the informative quantities are intermediate behavior and cross-checkpoint
differences at matched prefixes, not the own-prefix 12/12 itself.

### Overfit sharpens its route; it is not a prefix-independent coordinate upgrade

Mean probability of each trajectory's actual next coordinate token:

| Prefix origin | Source readout | overfit readout |
| --- | ---: | ---: |
| Source | 0.08308 | 0.04973 |
| overfit | 0.02656 | 0.71082 |

The diagonal values condition on different prefixes and target tokens; their
ratio is **not** a matched treatment effect. Same-row comparisons are matched.
Coordinate-family mass can already be near one while the exact coordinate
differs, so family selection alone does not explain these selected-site gaps.

Exact-token disagreement is also not automatically an instance switch. On
Source prefixes, overfit's top-coordinate absolute differences from Source's
actual token have median 7 bins, maximum 71. On overfit prefixes, Source has
median 6, maximum 534. The latter maximum is the selected middle-row y1
decision (reference 190, Source top token 724). This is a useful candidate
site, not proof of a wrong physical owner. Each coordinate is separately
conditioned on preceding reference coordinates; these top tokens must not be
assembled into an independently generated box. No IoU is inferred from them.

### More emitted rows does not mean a universally weaker stop preference

At selected first/middle/last completed-row boundaries, final raw
`opener minus EOS` logit margins are:

| Prefix origin / checkpoint | First | Middle | Last (terminal) |
| --- | ---: | ---: | ---: |
| Source / Source | 9.114 | 8.487 | -0.878 |
| Source / overfit | 9.380 | 11.963 | -5.128 |
| overfit / Source | 9.513 | 11.349 | -0.073 |
| overfit / overfit | 10.405 | 11.707 | -7.735 |

Both adapters still choose EOS at the Source trajectory's terminal prefix;
overfit strengthens rather than reverses that local preference. Consequently,
its longer own trajectory cannot here be explained simply as always suppressing
EOS on the same histories. This is evidence against that simple local account,
not an exhaustive claim about stopping or coverage.

### DeepStack pre/post instrumentation

At decoder blocks 1–3, all 874 visual positions receive nonzero additions.
Immediate text positions are exactly unchanged; later block boundaries have
zero injection delta. This agrees with the actual visual-mask-only in-place
implementation. Pre-state clones prevent aliasing from manufacturing equality.
This check validates placement; it does not show that vision is unused or
causally unimportant downstream.

## Interpretation and next discriminator

**Supported observation:** in this slice, overfit changes sharp, late-readable,
prefix-conditioned exact-coordinate preferences and terminal margins. There
is no evidence of a simple global shift toward earlier top-one coordinates
or a prefix-independent 'keep enumerating' policy.

**Hypothesis:** improved fitted enumeration may depend on maintaining its
learned geometric route rather than rescuing arbitrary Source histories.
**Strongest alternative:** ordinary representation-to-head alignment and
local coordinate calibration can generate these patterns without an explicit
covered-set mechanism. Raw Logit Lens cannot adjudicate that alternative.

**Proposed, not executed:** one bounded patching test at the large middle-row
y1 disagreement, with an unchanged prefix and donor-site controls, could ask
whether late-state replacement changes the final coordinate preference.
It would establish only local causal leverage, not remaining-owner coverage;
any natural continuation/owner claim would need its own frozen evaluation.
No Human13 expansion, Tuned Lens training, or automatic successor was launched.

## Artifacts and verification

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-image2299-logit-lens`.

- `run-v2/receipt.json`: authoritative execution identities, counts and checks.
- `run-v2/lens.jsonl`: 112 layer records; `selected-residuals.pt`: four compact tensors.
- `run-v2/trajectory-source.json`, `trajectory-overfit.json`: exact generated tokens.
- `analysis-v2/summary.json`: paired rank trajectories and coordinate-bin differences.
- [Paired ranks](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-image2299-logit-lens/analysis-v2/paired-ranks.png)
  and [boundary margins](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-image2299-logit-lens/analysis-v2/boundary-margins.png).

Four hook-on/off raw-logit comparisons and final norm/head reconstructions
have max absolute difference zero (fixed tolerance 2e-4). Lead replayed four
focused tests including an intentionally rejected text-change mutation,
checked artifact hashes/source-gate provenance, recomputed summaries and
inspected the rendered chart. Execution took 82.885 seconds, peak allocated
GPU memory 15.62 GB; raw payload is 21.63 MB. No training took place.

Attempt 1 failed closed before GPU allocation because the existing config
lacked live embedding source-gate evidence. Attempt 2 staged hash-verified
existing accepted evidence; no validator was bypassed. Failed receipt/log
remain in `run-v1`. `analysis-v1` is the retained initial rank-only reduction;
`analysis-v2` adds the coordinate-bin check and is the current derived analysis.

Reproduce analysis (choose a new output path, never overwrite):

```bash
conda run --no-capture-output -n ms python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-image2299-logit-lens/analyze.py /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-image2299-logit-lens/run-v2 --output /path/to/new-analysis
```

The single-image stop is reached. All work remains in its isolated direction
worktree, uncommitted; unrelated work and the feedback experiment are unchanged.
