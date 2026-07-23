# vLLM sampled-only breadth panel completed

Verified at `2026-07-23T03:57:46Z` in the `research-probes` worktree.

## Durable state

The complete 2,432-image source-model trajectory panel is available at:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-vllm/production-v2/`

It contains exactly sixteen low-temperature sampled trajectories per image:
38,912 routes total, all naturally closed, with zero length stops. Every image
has `sample_index` 0 through 15. The output image set exactly matches the
2,432-image candidate pool. All 152 persisted sampled-artifact SHA-256 hashes
match their manifests.

The eight workers used one vLLM engine each, 32-sequence scheduler capacity,
16 images per persisted batch, 4,096 maximum prompt-plus-generation tokens,
1,024 maximum generated tokens, temperature 0.4, nucleus probability 0.95, and
repetition penalty 1.0. Request seed is not a research variable.

## Important claim boundary

The panel estimates object support across sixteen sampled trajectories. It is
not a matched greedy-versus-sampling panel.

A targeted four-image smoke showed that all four greedy routes at repetition
penalty 1.0 entered exact or near-exact repeated-row loops and consumed the
1,024-token allowance. The matched 64 sampled routes all closed naturally.
Greedy was removed from production rather than increasing its loop length,
allowing truncation, or changing its repetition penalty.

Before the constant-dose training StateBanks are frozen, define a separate
finite Source-baseline policy or revise the original route-admission rule. A
`StateBank` is the persisted collection of model states, prefixes, target rows,
and token-level supervision consumed by the training pipeline. Do not treat
the truncated greedy loops, a different repetition penalty, or one sampled
route as the missing matched baseline without an explicit research decision.

## Infrastructure provenance

- upstream `CoordExp-swift`: `f263beaa0` adds configurable positive
  `backend.vllm.max_model_len` with default 2,048;
- research synchronization: `6fa61cf4d`;
- research configuration: `c8259aedf` selects 4,096;
- collector: `cfc24ea61`;
- sampled-only mode: `41e530c01`.

Full receipt:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-vllm-receipt.md`
