# Targeted Record-33 Value and Key+Value State Patch

Date: 2026-06-11

## Scope

This slice splits the remaining gap after the direct key-state patch for the
cross-phase top case:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`
- component: `duplicate_basin`

Prior targeted result:

- route/content route-only output patch recovery: `0.018168`
- score-bias per-source proxy recovery: `0.010221`
- direct key-state patch recovery: `0.009652`

This run asks whether source value-state replacement or paired key+value-state
replacement closes the remaining route/content gap.

## Implementation

Extended:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_qk_route_origin.py
src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py
```

Added behavior:

- `QKRouteStateCaptureHook` now also captures post-`v_proj` value states.
- `AttentionValueStatePatch` patches selected source values only for the target
  query row.
- `parse_score_bias_modes` now accepts:
  - `value_state_control`
  - `key_value_state_control`
- `build_attention_value_state_patch_row` emits:
  - `patch_kind=attention_value_state_patch`
  - `patch_kind=attention_key_value_state_patch`
  - value-delta stats: `value_delta_mean`, `value_delta_count`,
    `value_delta_min`, `value_delta_max`, `value_delta_l2`

The existing score-bias materialization path is reused for replay, source
bucketing, readout, and row writing.

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Value/key+value state patch output:

```text
value_key_value_state_patch_layer17_head1/attention_score_bias_patch_rows.jsonl
value_key_value_state_patch_layer17_head1/phase4_attention_score_bias_patch_summary.json
```

Synthesis output:

```text
targeted_state_patch_synthesis_layer17_head1/targeted_state_patch_synthesis_summary.json
targeted_state_patch_synthesis_layer17_head1/targeted_state_patch_synthesis_report.md
```

Run scale:

- target cases: `5`
- state patch rows: `164`
- reduced groups: `14`
- layer/head: `17/1`
- component: `duplicate_basin`
- direction: `masked_to_control`
- GPU used: `CUDA_VISIBLE_DEVICES=0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/value_key_value_state_patch_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin \
  --score-bias-modes value_state_control,key_value_state_control
```

## Result

Main case comparison:

| probe | mean probability recovery | attention mass recovery | rank recovery | abs-error recovery |
|---|---:|---:|---:|---:|
| route/content total output | 0.018371 | n/a | 19.4167 | -9.8000 |
| route/content route-only output | 0.018168 | n/a | 19.3333 | -9.8141 |
| route/content value-only output | 0.000121 | n/a | 0.5000 | 0.2754 |
| score-bias per-source proxy | 0.010221 | 0.292988 | 15.3333 | -6.2925 |
| direct key-state patch | 0.009652 | 0.237000 | 16.0000 | -5.9484 |
| direct value-state patch | -0.000068 | 0.000000 | -0.9167 | 0.1441 |
| direct key+value-state patch | 0.010027 | 0.237000 | 16.2500 | -6.1240 |

Top grouped rows:

| rank | checkpoint | record | phase | mode | rows | probability recovery | mass16 recovery |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `key_value_state_control` | 12 | 0.010027 | 0.180497 |
| 2 | `aligner_parent_ckpt1824` | 54 | `post_y1/pre_x2` | `key_value_state_control` | 11 | 0.001314 | 0.035280 |
| 3 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `key_value_state_control` | 12 | 0.000768 | 0.009141 |
| 10 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `value_state_control` | 12 | -0.000068 | -0.004061 |

## Interpretation

For the main record-33 failure, source value content is not the missing piece:

- Value-only replacement is effectively zero or slightly harmful.
- Key+value replacement is only slightly stronger than key-only:
  `0.010027` versus `0.009652`.
- Both stay close to the score-bias proxy and well below the route/content
  output patch (`0.018168`).

This sharpens the remaining mechanism picture. The route/content output effect
is not simply "control key plus control value for duplicate-basin source
tokens." The remaining gap likely comes from the downstream attention output
path after score/value assembly:

- `o_proj` and post-attention residual geometry;
- normalization/residual interaction at the coordinate slot;
- source competition or distributed route effects outside the selected
  duplicate-basin bucket;
- coordinate-slot basin amplification after the attention block.

The next deterministic probe should therefore patch later sites with the same
source-state controls:

1. compare `self_attn_output` key+value patch against direct
   post-attention-residual patch for the same rows;
2. or run component expansion beyond `duplicate_basin`:
   `visual_near_ring`, `visual_far_background`, `non_region_complement`, and
   `whole_head` under key+value mode.

## Verification

Red test first:

- direct importlib harness failed on missing `AttentionValueStatePatch`.

Deterministic green checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py \
  src/analysis/autoregressive_duplication_mechanism/phase4_qk_route_origin.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py')
spec = importlib.util.spec_from_file_location('test_phase4_attention_score_bias_patch', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    getattr(mod, name)()
    print(f'PASS {name}')
PY
```

GPU run completed with:

- `attention_score_bias_patch_row_count=164`
- `checkpoint_count=3`
- `replay_case_count=5`
- `score_bias_modes=["value_state_control","key_value_state_control"]`

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py -q`
still hits the local wrapper issue: `Pytest: No tests collected`, so the direct
importlib harness remains the meaningful deterministic test for this file.
