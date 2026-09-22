# Phase 4 Cross-Row Route/Content Swap And Candidate-Head Pilot

Date: 2026-06-11

Scope: Phase 4 mechanism diagnosis, focused on layer 17 route/content effects.
This note records the dynamic deeper path added after the paired
constructive/destructive manifest: cross-row effect transfer and a small
candidate-head pilot. Evidence is artifact-backed but still partial; shard-04
pilot evidence must not be interpreted as full all-shard validation.

## Code Commits

- `dd82a607 feat: add cross-row route content swap probe`
- `fb01f538 feat: add donor baseline deltas to cross-row swap`

## Cross-Row Swap Artifacts

Primary full-manifest run with donor-target masked baselines:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_cross_row_route_content_swap_layer17_head1_full_manifest_donor_delta
```

Inputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_paired_case_manifest_layer17_head1_duplicate_constructive_destructive/paired_case_manifest_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase2_region_rows.jsonl
```

Run shape:

- `attention_layer=17`
- `attention_head=1`
- `patch_site=self_attn_output`
- `role_limits=constructive_core:8,destructive_core:6`
- `effect_kinds=total_delta,route_delta_masked_values,value_delta_control_route`
- `manifest_row_count=14`
- `cross_pair_count=96`
- `swap_row_count=288`

Aggregate JSON:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_cross_row_route_content_swap_layer17_head1_full_manifest_donor_delta/phase4_cross_row_route_content_swap_donor_delta_aggregate.json
```

Key full-manifest means, `n=48` per direction/effect:

| donor -> recipient | effect | recipient prob delta | recipient r4 delta | donor-target prob delta | donor-target r4 delta |
|---|---:|---:|---:|---:|---:|
| constructive -> destructive | route_delta_masked_values | -0.013280 | -0.072883 | +0.003134 | +0.025236 |
| constructive -> destructive | total_delta | -0.012239 | -0.066467 | +0.003241 | +0.024994 |
| constructive -> destructive | value_delta_control_route | -0.001243 | -0.005491 | +0.000291 | +0.002379 |
| destructive -> constructive | route_delta_masked_values | +0.002056 | +0.014115 | +0.000000 | +0.000001 |
| destructive -> constructive | total_delta | +0.000223 | +0.000722 | +0.000207 | +0.002260 |
| destructive -> constructive | value_delta_control_route | -0.000937 | -0.006964 | +0.000144 | +0.001514 |

Interpretation:

- Constructive donor route vectors are not portable generic repairs. Inserted
  into destructive recipients, they modestly pull probability toward the donor
  coordinate basin while substantially damaging the recipient target basin.
- Destructive donor route vectors inserted into constructive recipients barely
  attract the donor basin. Route-only transfer slightly improves the recipient
  target basin, but total/value effects do not form a strong portable
  destructive signature.
- This supports a route-as-specific-basin-vector picture rather than a simple
  role label such as "constructive route helps everywhere".

## Candidate-Head Pilot

Purpose: decide whether layer 17/head 1 is uniquely worth deeper cross-row
analysis, or whether high-recovery all-head attention-output candidates should
also receive paired manifest construction.

Pilot heads selected from existing layer-17 all-head attention-output recovery:

- `head=13`
- `head=3`
- `head=15`
- `head=14`

Shard-00 pilot:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head13_pilot_shard00
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_pilot_shard00
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head15_pilot_shard00
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head14_pilot_shard00
```

Shard-04 pilot:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head13_pilot_shard04
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_pilot_shard04
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head15_pilot_shard04
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head14_pilot_shard04
```

Shard-04 aggregate:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_candidate_heads_shard04_pilot_aggregate.json
```

Shard-04 target slice:

- `checkpoint_label=none_latest_ckpt32`
- `phase=post_y1/pre_x2`
- `patch_direction=control_to_masked`
- `patch_component=duplicate_basin`
- `n=21` rows per head/effect

Key shard-04 means:

| head | effect | patched prob delta | r4 recovery | r8 recovery | expected abs error recovery | projected delta L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | route_delta_masked_values | +0.000196 | +0.067000 | +0.121073 | -10.372085 | 0.596569 |
| 3 | route_delta_masked_values | +0.000159 | +0.066930 | +0.120944 | -10.233614 | 3.934373 |
| 15 | route_delta_masked_values | +0.000207 | +0.066768 | +0.121412 | -10.345957 | 0.005545 |
| 14 | route_delta_masked_values | -0.000009 | +0.067987 | +0.123416 | -10.375524 | 0.003037 |
| 1 | route_delta_masked_values | -0.007575 | +0.013119 | +0.023367 | -0.539188 | 52.293086 |
| 1 | value_delta_control_route | +0.000778 | +0.070612 | +0.125782 | -9.653003 | 16.268775 |

Interpretation:

- The high-recovery attention-output candidate heads do not reproduce the
  layer17/head1 route-only geometry. Their route-only projected deltas are much
  smaller, and their broad radius recovery resembles a low-amplitude value or
  residual basin smoothing effect.
- Head 1 remains the sharper route/content mechanism candidate: it has large
  route-projected vectors and the strongest role-specific cross-row behavior,
  even though route-only patching can reduce exact target probability on shard
  04.
- The next higher-leverage step is not to fan out cross-row swaps to all heads
  yet. First, run all-shard route/content patch rows for the best alternate
  candidates only if they show nontrivial projected deltas or role-specific
  pairing in richer shards. Current pilot suggests head 3 is the only alternate
  worth a second look because its projected L2 is non-negligible; heads 14 and
  15 look too small.

## Verification

Code verification:

```bash
python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_cross_row_route_content_swap.py scripts/analysis/run_autoregressive_duplication_phase4_cross_row_route_content_swap.py tests/analysis/autoregressive_duplication_mechanism/test_phase4_cross_row_route_content_swap.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_cross_row_route_content_swap.py')
spec = importlib.util.spec_from_file_location('test_phase4_cross_row_route_content_swap', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    getattr(mod, name)()
    print(f'PASS {name}')
PY
```

Direct tests passed. The repo pytest wrapper still reports `Pytest: No tests
collected` for this local test file, matching earlier behavior.

GPU execution:

- Cross-row 2x2 smoke: passed, `swap_row_count=24`.
- Cross-row full manifest with donor baselines: passed, `swap_row_count=288`.
- Candidate-head shard-00 pilots: passed, `1128` rows each.
- Candidate-head shard-04 pilots: passed, `1032` rows each.
