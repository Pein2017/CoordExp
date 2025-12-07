# Porting notes between Qwen3-VL and CoordExp

## Token-type metrics (desc/coord/format)
- Feature source: Qwen3-VL `data_collators/dataset_metrics.py`, `token_types.py`, `metrics/dataset_metrics.py`.
- Intentional delta in CoordExp: aggregate-only metrics (no per-dataset buckets) and packing support.
- Metric keys (no `agg_` prefix): `loss`, `token_acc`, `token_count`, and `{desc,coord,format}_token_acc`, `{desc,coord,format}_entropy`, `{desc,coord,format}_token_count`.
- Packing support added in CoordExp: token types computed per sample pre-pack and concatenated; on length mismatch metrics are skipped (IGNORE) instead of erroring.
- Defaults differ: CoordExp includes only `lvis` by default; Qwen3-VL defaults to `target,lvis` includes and excludes `coig_lang_chat`.
