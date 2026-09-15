---
doc_id: docs.standards.porting
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Porting guidance and compatibility notes.
updated: 2026-06-07
---

# Porting notes between Qwen3-VL and CoordExp

## Qwen-VL Family Differences

Use [`UPSTREAM.md`](UPSTREAM.md) and
[`upstream/QWEN_VL.md`](upstream/QWEN_VL.md) for detailed upstream handles.
Do not assume Qwen2-VL, Qwen2.5-VL, and Qwen3-VL share processor, video, RoPE,
or attention behavior.

| Surface | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| Model type | `qwen2_vl` | `qwen2_5_vl` | `qwen3_vl` |
| Vision path | older Qwen-VL baseline | windowed/full-attention vision patterns | DeepStack visual feature injection |
| Video timing | older video flow | `second_per_grid_ts` path | text timestamp/frame-block path |
| Position IDs | 3D MRoPE | 3D MRoPE with Qwen2.5 variants | 3D MRoPE; optional 4-row tensor where row 0 is text positions |
| Packing risk | attention/position reset handling | attention/window/grid handling | row-0 text boundaries plus rows 1-3 MRoPE geometry |
| CoordExp rule | inspect exact installed source | inspect exact installed source | inspect exact installed source |

For CoordExp, the stable rule is to preserve image order, `image_grid_thw`,
video grids, token order, and coordinate geometry before crossing the upstream
model/template boundary. Any port that changes processor class, model family,
or `attn_implementation` must rerun placeholder-count, grid, position-id, and
loss/logit availability checks.

## Token-type metrics (desc/coord/format)
- Feature source: Qwen3-VL `data_collators/dataset_metrics.py`, `token_types.py`, `metrics/dataset_metrics.py`.
- Intentional delta in CoordExp: aggregate-only metrics (no per-dataset buckets) and packing support.
- Metric keys (no `agg_` prefix): `loss`, `token_acc`, and `{desc,coord,format}_token_acc`.
- Packing support added in CoordExp: token types computed per sample pre-pack and concatenated; on length mismatch metrics are skipped (IGNORE) instead of erroring.
- Defaults differ: CoordExp includes only `lvis` by default; Qwen3-VL defaults to `target,lvis` includes and excludes `coig_lang_chat`.
