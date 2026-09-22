# FN-Guidance Continuation Behavior

Date: 2026-06-27

Scope: greedy continuation check for the cross-sample FN-guidance prefix-history panel. This follows the hidden-state readout and route probes in `2026-06-27_cross_sample_fn_guidance_prefix_history.md`.

## Question

When a prefix-history ablation produces a target-near post-x1/pre-y1 hidden-state readout, does greedy generation actually emit the missing object's `[y1,x2,y2,box_end]` tail?

This test uses the same plan rows as the readout panel, starts exactly after:

```text
<|object_ref_start|>{missing desc}<|object_ref_end|><|box_start|><|coord_true_x1|>
```

and greedily emits six tokens from the same post-x1 state. It is a behavior check, not a patching experiment.

## Artifacts

Plan rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val_cases_632_5586_18380_plan/fn_guidance_prefix_history_rows.jsonl
```

Continuation output:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_prefix_history/v1_sorted_val_cases_632_5586_18380_continuation_orig_nohist_droprecent_scramble_gpu0
```

## Integrity

- selected rows: `12`
- cases: `632`, `5586`, `18380`
- variants: `original`, `no_history`, `drop_recent8`, `coord_scramble_all_to_target_band`
- generated steps: `6`
- errors: `0`
- first-token target hits: `0`
- complete coordinate-tail hits: `0`

## Results

| image | variant | expected `[y1,x2,y2]` | generated first tokens | generated bbox | IoU |
|---:|---|---|---|---|---:|
| `632` | `original` | `113,784,172` | `593,829,622,<box_end>` | `[776,593,829,622]` | `0.0` |
| `632` | `no_history` | `113,784,172` | `93,829,178,<box_end>` | `[776,93,829,178]` | `0.1048` |
| `632` | `drop_recent8` | `113,784,172` | `319,814,444,<box_end>` | `[776,319,814,444]` | `0.0` |
| `632` | `coord_scramble_all_to_target_band` | `113,784,172` | `999,829,624,<box_end>` | `[776,999,829,624]` | `0.0` |
| `5586` | `original` | `100,461,183` | `710,625,944,<box_end>` | `[425,710,625,944]` | `0.0` |
| `5586` | `no_history` | `100,461,183` | `281,520,278,<box_end>` | `[425,281,520,278]` | `0.0` |
| `5586` | `drop_recent8` | `100,461,183` | `2,480,182,<box_end>` | `[425,2,480,182]` | `0.2971` |
| `5586` | `coord_scramble_all_to_target_band` | `100,461,183` | `716,625,947,<box_end>` | `[425,716,625,947]` | `0.0` |
| `18380` | `original` | `287,557,426` | `946,526,999,<box_end>` | `[516,946,526,999]` | `0.0` |
| `18380` | `no_history` | `287,557,426` | `281,536,317,<box_end>` | `[516,281,536,317]` | `0.1031` |
| `18380` | `drop_recent8` | `287,557,426` | `704,539,681,<box_end>` | `[516,704,539,681]` | `0.0` |
| `18380` | `coord_scramble_all_to_target_band` | `287,557,426` | `545,544,679,<box_end>` | `[516,545,544,679]` | `0.0` |

The generated tokens after `<box_end>` often continue with another object start, so these are coherent object-tail continuations. The failure is not malformed output; it is wrong coordinate selection.

## Interpretation

The continuation check refines the previous readout/route conclusion.

For `632` and `18380`, `no_history` does produce target-near first y coordinates:

- `632`: original y1 `593` becomes `93`, target is `113`.
- `18380`: original y1 `946` becomes `281`, target is `287`.

But neither emits the exact target y1, and neither completes the correct box. The no-history continuations look like partial recovery into a nearby object/anchor basin:

- `632` no-history box `[776,93,829,178]` uses the true x1 and a target-near y band, but x2 is too wide compared with target `[776,113,784,172]`.
- `18380` no-history box `[516,281,536,317]` is close to the partial generated wine-glass anchor, not the full GT extent `[516,287,557,426]`.
- `5586` remains unstable: `drop_recent8` gives nonzero IoU (`0.2971`) because the lower-right extent overlaps, but y1 collapses to the top boundary `2`.

So the mechanism is not "remove history and the model solves the FN." A more accurate decomposition is:

1. **History removal can restore a target-near first-coordinate basin** for some FNs.
2. **Greedy autoregressive emission still snaps to a nearby local object/anchor basin**, often one already implied by previous generated predictions or scene geometry.
3. **Full object-span recovery requires x2/y2 sharpening**, not only y1 rescue.

This helps interpret the cross-sample readout honestly. The no-history readout for `18380` (`target rank 9`) was real, but it corresponded to a partial wine-glass span, not the complete missing GT object.

## Consequence

For false negatives, the strongest current statement is:

```text
Some missed objects are not pure visual non-perception: true desc+x1 with harmful history removed can move the model into a target-near object basin.
But the recovered basin is often a partial/nearby-object basin, so exact target-span completion remains a separate bottleneck.
```

This supports the broader two-stage model:

- prefix/history controls which object basin the current coordinate token enters;
- late coordinate decoding then chooses a local basin and must sharpen the full span.

## Next

The next higher-value surgery is full-span state intervention, not more y1-only readout. For `18380`, patch the no-history state into the original prefix across post-x1, post-y1, and post-x2 positions. If y1 becomes near-target but y2 remains partial, the missing object is an extent/boundary problem. If the whole span transfers, the main blocker is prefix-history state formation.
