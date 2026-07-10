# Targeted Record-33 Key-State Patch

Date: 2026-06-11

## Scope

This slice follows the targeted Q/K and score-bias probe for the cross-phase
top case:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`
- component: `duplicate_basin`

The prior targeted evidence said:

- route/content `route_delta_masked_values` recovery: `0.018168`
- attention score-bias proxy recovery: `0.010221`
- Q/K origin: score delta is dominated by visual-key-side state
  (`key_delta_control_query_score_mean=5.61591`, while
  `query_delta_masked_keys_score_mean=-1.45842`)

This run tests a more direct key-state intervention: replace the selected
masked duplicate-basin source key vectors with the aligned control key vectors
for the target query row, then measure attention-mass and coordinate-basin
recovery.

## Implementation

New hook:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py
```

Added symbols:

- `AttentionKeyStatePatch`
- `build_attention_key_state_patch_row`
- `key_state_control` mode in `parse_score_bias_modes`

New named runner:

```text
scripts/analysis/run_autoregressive_duplication_phase4_key_state_patch_shard.py
```

The runner currently reuses the score-bias materialization path for capture,
region bucketing, replay, and readout, but emits rows with:

- `task=phase4_attention_key_state_patch`
- `patch_kind=attention_key_state_patch`
- `score_bias_mode=key_state_control`
- key-delta stats: `key_delta_mean`, `key_delta_count`, `key_delta_min`,
  `key_delta_max`, `key_delta_l2`

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Key-state patch output:

```text
key_state_patch_layer17_head1/attention_score_bias_patch_rows.jsonl
key_state_patch_layer17_head1/phase4_attention_key_state_patch_summary.json
```

Synthesis output:

```text
targeted_key_state_patch_synthesis_layer17_head1/targeted_key_state_patch_synthesis_summary.json
targeted_key_state_patch_synthesis_layer17_head1/targeted_key_state_patch_synthesis_report.md
```

Run scale:

- target cases: `5`
- key-state patch rows: `82`
- reduced groups: `7`
- layer/head: `17/1`
- component: `duplicate_basin`
- direction: `masked_to_control`
- GPU used: `CUDA_VISIBLE_DEVICES=0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_key_state_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/key_state_patch_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin
```

## Result

Grouped key-state patch result:

| rank | checkpoint | record | phase | rows | attention mass recovery | probability recovery | rank recovery | abs-error recovery | mass16 recovery |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | 12 | 0.237000 | 0.009652 | 16.0000 | -5.94837 | 0.176720 |
| 2 | `no_aligner_parent_ckpt3668` | 36 | `post_y1/pre_x2` | 11 | 0.047982 | 0.002354 | 3.3636 | -1.38745 | 0.043199 |
| 3 | `aligner_parent_ckpt1824` | 54 | `post_y1/pre_x2` | 11 | 0.084961 | 0.001679 | -2.2727 | -6.43412 | 0.035498 |
| 4 | `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | 12 | 0.119441 | 0.000414 | 78.2500 | -14.5676 | 0.009892 |
| 5 | `none_latest_ckpt32` | 33 | `box_start/pre_x1` | 12 | 0.104137 | 0.000180 | 60.2500 | -9.71593 | 0.005333 |
| 6 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | 12 | -0.015471 | -0.000073 | 0.5000 | 0.02840 | 0.002744 |
| 7 | `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | 12 | 0.379688 | -0.000898 | -5.8333 | -0.30692 | -0.012316 |

Main case comparison:

| probe | mean probability recovery | attention mass recovery | rank recovery | abs-error recovery |
|---|---:|---:|---:|---:|
| route/content `route_delta_masked_values` | 0.018168 | n/a | 19.3333 | -9.8141 |
| score-bias per-source proxy | 0.010221 | 0.292988 | 15.3333 | -6.29252 |
| direct key-state patch | 0.009652 | 0.237000 | 16.0000 | -5.94837 |

Top individual key-state anchors in the main case include rows `23-31`, with
probability recovery around `0.00946-0.01780` and radius-16 recovery around
`0.17386-0.32775`.

## Interpretation

The direct key-state intervention is causal and phase-specific, but it does not
fully explain the route-only output patch.

For the main record-33 failure:

- Replacing duplicate-basin key vectors recovers a large part of the
  attention-mass and coordinate-basin movement.
- The key-state patch is close to the score-bias proxy
  (`0.009652` vs `0.010221` probability recovery), which validates that the
  score-bias proxy was a reasonable causal score-space abstraction.
- Both are still well below the route/content output patch (`0.018168`
  probability recovery), so key-side score repair is not the whole mechanism.

This leaves three plausible remaining contributors:

- value-side or post-attention projection interactions that are preserved by
  the route/content output patch but not by score-only/key-only interventions;
- attention normalization or multi-source competition outside the selected
  duplicate-basin bucket;
- downstream coordinate-slot basin amplification after attention output.

The strongest next deterministic step is to split the remaining gap by running
paired key+value source patching, or by adding a direct value-state patch under
the repaired key route on the same target panel.

## Verification

- Red test first:
  direct importlib harness failed on missing `AttentionKeyStatePatch` import.
- Deterministic green checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_attention_score_bias_patch.py \
  scripts/analysis/run_autoregressive_duplication_phase4_key_state_patch_shard.py \
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

- Named GPU runner completed with:
  `attention_score_bias_patch_row_count=82`,
  `checkpoint_count=3`, `replay_case_count=5`,
  `score_bias_modes=["key_state_control"]`.

`python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_attention_score_bias_patch.py -q`
still hits the local wrapper issue: `Pytest: No tests collected`, so the direct
importlib harness remains the meaningful deterministic test for this file.
