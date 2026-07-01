# Phase 4 Layer-17 Head-1 Mechanism Chain Synthesis

Date: 2026-06-11

## Purpose

This note consolidates the layer-17 head-1 branch after the route/content,
Q/K-origin, score-bias, and per-source score-pattern probes. It is intended to
prevent the next slice from chasing another attention-score-only variant when
the evidence now points to a joint route-and-value visual-state circuit.

## Evidence Chain

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

The primary comparison below uses `none_latest_ckpt32`, phase
`post_y1/pre_x2`.

### 1. Route/Content Causal Patch

Artifact:

```text
phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_summary.json
```

For duplicate-basin `masked_to_control`:

| effect | n | prob recovery | radius4 recovery | expected abs-error recovery |
| --- | ---: | ---: | ---: | ---: |
| `total_delta` | 62 | 0.003976 | 0.027956 | -3.342128 |
| `route_delta_masked_values` | 62 | 0.003703 | 0.026634 | -1.972511 |
| `value_delta_control_route` | 62 | 0.000343 | 0.002422 | -1.427814 |

Interpretation: the route term explains most of the layer-17 head-1 projected
movement at the aggregate phase level, but value content is still present and
becomes decisive in the high-signal primary-anchor value-source patch.

### 2. Q/K Route-Origin Readout

Artifact:

```text
phase4_qk_route_origin_layer17_head1_visual_triplet_top4_allshards_summary.json
```

For duplicate-basin:

| n | score mean delta | query delta with masked keys | key delta with control query |
| ---: | ---: | ---: | ---: |
| 62 | 1.860993 | -0.501067 | 2.362059 |

Interpretation: the duplicate-basin route advantage is mostly key-side in
score space. The coordinate-slot query alone does not become more compatible
with masked duplicate-basin keys; the useful route appears when the control
query meets control duplicate-basin visual keys.

### 3. Scalar Score-Bias Patch

Artifact:

```text
phase4_attention_score_bias_patch_layer17_head1_duplicate_top4_allshards_summary.json
```

For duplicate-basin `masked_to_control`:

| n | attention recovery | prob recovery | radius4 recovery | expected abs-error recovery |
| ---: | ---: | ---: | ---: | ---: |
| 26 | 0.303442 | 0.004768 | 0.032295 | 1.580239 |

Interpretation: scalar bucket score bias is causally sufficient to restore
duplicate-basin attention mass and local coordinate probability/radius mass,
but it is not sufficient to restore coordinate-basin quality. Expected
absolute error worsens in the primary average.

### 4. Per-Source Score-Pattern Patch

Artifact:

```text
phase4_attention_score_pattern_patch_layer17_head1_duplicate_top4_allshards_summary.json
```

For duplicate-basin `masked_to_control`:

| n | attention recovery | prob recovery | radius4 recovery | expected abs-error recovery |
| ---: | ---: | ---: | ---: | ---: |
| 26 | 0.304048 | 0.004902 | 0.032808 | 8.639782 |

Interpretation: even restoring the detailed within-bucket score pattern gives
nearly the same attention and local-probability movement as scalar score bias,
but still does not reproduce the good coordinate-basin behavior. Therefore the
missing piece is not another attention-score-only variant.

### 5. Direct Value-Source Patch

Artifact:

```text
phase4_value_source_patch_layer17_head1_top4_allshards_summary.json
```

Recorded in:

```text
progress/diagnostics/2026-06-11_phase4_layer17_head1_routing_value_patch_findings.md
```

For the primary anchor, `masked_to_control` duplicate-basin value-source patch:

| n | control prob | masked prob | patched prob | prob recovery | rank recovery |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.026926 | 0.013462 | 0.032568 | 0.019106 | 20.916667 |

Interpretation: direct value contribution replacement is much more effective
than score-only patches at repairing the primary coordinate target, even
overshooting the control probability mean.

## Current Mechanism Sketch

The best current picture is a joint visual key-value route/content circuit:

1. The duplicate-basin visual key state opens a strong layer-17 head-1 route
   at the coordinate slot.
2. That route alone is not enough. It mainly restores attention mass and some
   local probability/radius mass.
3. The duplicate-basin value contribution carries a coordinate-basin direction
   that is much more directly useful for target probability/rank repair.
4. The final coordinate-basin behavior depends on the route and value together:
   key-side score structure gates access to the basin token, while the value
   vector supplies target-relevant coordinate evidence.

This explains why attention maps and score patches are necessary but not
mechanistically complete. The failure is not simply "the model attends to the
wrong place"; it is "the model loses the right visual key-value state coupling
at the coordinate slot."

## Exclusions

The following explanations are now too shallow for the final diagnosis:

- Pure residual-layer explanation: layer-17 self-attention/value-source probes
  localize a concrete head/source pathway.
- Attention-mass-only explanation: scalar and per-source score patches restore
  mass but not the full coordinate-basin quality.
- Query-only explanation: Q/K decomposition shows the masked-key query-side
  hybrid is negative on average for the duplicate basin.
- Smoothness-only coordinate-token explanation: coordinate-basin movement is
  modulated by visual route/content state, not just the coordinate-token atlas.

## Next Causal Design

The next intervention should not be another score-only patch. It should bind
route and content in one controlled forward pass.

Preferred design:

1. Capture masked and control post-RoPE K states, V states, and attention
   probabilities for layer-17 head-1 duplicate-basin visual tokens.
2. For `masked_to_control`, patch the masked run with:
   - the per-source score pattern or equivalent K-side score shift;
   - the matching duplicate-basin value contribution computed under the patched
     route, not merely the unpatched masked-route contribution.
3. Record:
   - duplicate-basin attention mass recovery;
   - target probability/rank recovery;
   - coord radius mass and expected absolute error;
   - projected value-contribution delta against the coordinate-token basin.

Acceptable near-term proxy:

- Implement a joint score-pattern plus `o_proj`-slice value contribution patch,
  but label it as a proxy unless the old contribution subtracted from the
  `o_proj` input is recomputed under the patched route. A naive composition can
  double-count or under-subtract source contribution and should not be treated
  as a clean K/V-state intervention.

## Stop Rule For This Branch

This branch should stop chasing layer-17 head-1 attention-score variants unless
they also manipulate value content or visual K/V states. The score-only family
has served its purpose: it showed that routing is causal for attention mass
and insufficient for the final coordinate basin.

The next successful slice should either:

- produce a clean joint K/V or score/value causal patch, or
- switch to coordinate-slot basin projection of the already successful
  duplicate-basin value-source delta.
