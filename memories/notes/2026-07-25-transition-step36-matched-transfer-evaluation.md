# Transition Step-36 Matched Transfer Evaluation

## Decision boundary

After the long-training discussion, the user approved only a matched transfer
evaluation of Source versus first-divergence transition step 36 on the frozen
development and heldout splits. Full-pool evaluation, a successor direction,
and architecture promotion remain outside the authorized round.

## Frozen execution contract

- original `list all objects` prompt;
- one autoregressive completion in the canonical row schema;
- Hugging Face deterministic greedy decoding;
- `batch_size=4`, `max_new_tokens=3084`, `temperature=0`, `top_p=1`, and
  `repetition_penalty=1`;
- scoring enabled;
- eight controller-worker ranks for every run; and
- identical Source/treatment shard-plan fingerprint and rank-to-device mapping
  within each split.

Development contains 256 images and heldout contains 128. Their input JSONL
hashes are `e4600b1998ad1269e351373e1c03caa39ff0a97b0474eb6852268b3c1736d0e4`
and `8378af4429cc3cf2084da50a34fdf9b163e8d210291b3a787cdc05c1bdbd266e`;
their example-ID overlap is zero.

Source inherits the production step-4887 adapter. Treatment replaces it with
transition step 36. The Source and treatment special-token embedding payloads
are content-identical by hash; the model adapter is the intended payload
difference.

## Result

All four runs completed. All 768 rows decoded and scored, with zero parser,
score, or image-validation failures.

| Split | Source / treatment owners | Gained / lost | Net | Positive / negative / unchanged images | Prediction delta | Strict duplicate delta | Common-owner mean Intersection over Union delta | Source / treatment length stops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development-256 | 1,452 / 1,520 | 128 / 60 | +68 | 60 / 21 / 175 | +350 | -14 | +0.002068 | 1 / 3 |
| heldout-128 | 729 / 736 | 46 / 39 | +7 | 21 / 17 / 90 | -118 | -13 | +0.006447 | 0 / 1 |

Development coverage rises 2.60 percentage points. Held-out coverage rises
0.54 percentage points. Heldout is directionally favorable without verbosity:
it uses 118 fewer predictions, has 13 fewer strict duplicate candidates, adds
no invalid predictions, and slightly improves common-owner geometry. Its net is
nevertheless only seven owners across 1,303 annotations, it adds one length
stop, and a post-hoc paired image bootstrap interval crosses zero.

## Current belief and stop

Transition step 36 now has evidence of transfer beyond its 64 training-panel
images. The evidence is promising but not robust enough to call it a usable
model improvement or promote the mechanism. Stop for user discussion. Do not
launch full-pool evaluation, choose a successor direction, or change the final
architecture before that discussion.

## Evidence roots

- configs and receipt:
  `configs/coordexp_infras/infer/research/qwen3_vl_2b_transition_step36_transfer_max3084_matched_b4_v1/`;
- run artifacts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/clean-rollouts-transition-step36-transfer-max3084-matched-b4-v1/`;
- paired owner ledgers:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/owner-comparisons-transition-step36-transfer-max3084-matched-b4-v2/`.
