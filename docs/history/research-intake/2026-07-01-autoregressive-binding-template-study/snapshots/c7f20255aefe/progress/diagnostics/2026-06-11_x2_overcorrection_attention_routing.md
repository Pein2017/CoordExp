# X2 Overcorrection Attention Routing

Date: 2026-06-11

## Scope

Ran the route-head attention-routing probe for the validated x2 overcorrection
candidate panel and joined it with the candidate rows plus the hidden-shift
artifact. This is the first direct bridge from the x2 overcorrection candidate
surface to attention-route movement around the pinned route head.

GPU use was restricted to `CUDA_VISIBLE_DEVICES=0`, within the available device
set `0,1,2,3`.

## Inputs

- Candidate token windows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl`
- Candidate region rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl`
- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl`
- Hidden-shift rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_hidden_shift_whole_head_layer17_head1/hidden_shift_rows.jsonl`

## Outputs

Attention-routing output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1`

Joined summary root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1/candidate_attention_hidden_summary`

Key files:

- `attention_routing_rows.jsonl`
- `phase4_attention_routing_summary.json`
- `candidate_attention_hidden_summary/candidate_attention_routing_rows.jsonl`
- `candidate_attention_hidden_summary/candidate_attention_hidden_summary_rows.jsonl`
- `candidate_attention_hidden_summary/phase4_candidate_attention_hidden_summary.json`
- `candidate_attention_hidden_summary/phase4_candidate_attention_hidden_summary.md`

Counts:

- Replay cases: `2`
- Region rows: `14`
- Attention-routing rows: `49`
- Candidates in joined summary: `7`
- Hidden-shift rows in joined summary: `63`
- Attention layer/head: `17/1`

## Main Readout

For the six auxiliary-checkpoint x2 candidates, masking the duplicate basin
causes a large route redistribution at head `17/1`:

| component | n | control mass | masked mass | control-minus-masked |
|---|---:|---:|---:|---:|
| duplicate_basin | 6 | 0.461202 | 0.061242 | 0.399960 |
| visual_near_ring | 6 | 0.294915 | 0.636851 | -0.341937 |
| visual_non_basin | 6 | 0.494902 | 0.891674 | -0.396772 |
| non_region_complement | 6 | 0.538475 | 0.938354 | -0.399878 |
| text_prefix | 6 | 0.041357 | 0.044114 | -0.002757 |
| special_control | 6 | 0.002216 | 0.002565 | -0.000349 |

The single no-aligner parent control candidate has much smaller duplicate-basin
movement:

| component | n | control mass | masked mass | control-minus-masked |
|---|---:|---:|---:|---:|
| duplicate_basin | 1 | 0.005310 | 0.003830 | 0.001480 |
| visual_near_ring | 1 | 0.854836 | 0.872049 | -0.017213 |
| visual_non_basin | 1 | 0.972460 | 0.974624 | -0.002164 |

The strongest auxiliary candidates align attention-route movement with late
hidden-state movement:

| candidate | dup delta | near delta | hidden L16 | hidden L20 | hidden L24 | patched-target |
|---|---:|---:|---:|---:|---:|---:|
| row 21 exact | 0.602295 | -0.485881 | 44.2300 | 124.0570 | 304.1290 | -12 |
| row 20 exact | 0.534668 | -0.524053 | 46.2831 | 123.9550 | 294.5100 | -7 |
| row 28 near | 0.674316 | -0.545800 | 37.5543 | 111.9550 | 284.2580 | -5 |

## Interpretation

This supports a sharper mechanistic picture for the auxiliary x2 overcorrection
surface:

- In the unmasked/control image, head `17/1` places substantial attention mass
  on the duplicate basin for the problematic x2 slots.
- Masking the duplicate basin removes that route and shifts mass into broader
  visual alternatives, especially near-ring and non-basin visual buckets.
- The same rows show large later-layer hidden-state movement, especially after
  the route-head neighborhood.
- The no-aligner parent candidate does not show comparable duplicate-basin
  routing dependence at this head.

Guardrail: the source buckets are not all disjoint; for example
`visual_non_basin`, `non_region_complement`, and `whole_head` can overlap by
construction. Interpret component means as route diagnostics, not a partition
whose masses must sum to one.

This is still not causal sufficiency for the final token choice. The next high
value step is to connect this route redistribution to coordinate-basin logits:
for the same rows, compare whether duplicate-basin attention mass or hidden
delta predicts movement into the lower-control coordinate basin.

## Reproduction

Attention routing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_attention_routing_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --patch-components duplicate_basin,visual_near_ring,visual_non_basin,non_region_complement,text_prefix,special_control,whole_head
```

Joined summary:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_candidate_attention_summary.py \
  --attention-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1/attention_routing_rows.jsonl \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --hidden-shift-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_hidden_shift_whole_head_layer17_head1/hidden_shift_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_attention_routing_layer17_head1/candidate_attention_hidden_summary
```

Verification:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_candidate_attention_summary.py \
  scripts/analysis/run_autoregressive_duplication_phase4_candidate_attention_summary.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_attention_summary.py
```

The direct harness for
`tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_attention_summary.py`
passed.
