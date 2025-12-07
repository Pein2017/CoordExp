# Design: Packing-compatible token-type metrics

## Goals
- Provide desc/coord/format token-level metrics in CoordExp with aggregate logging only.
- Work for both padded and packed batching without misalignment; avoid NaNs when supervision is absent.

## Key decisions
- Compute token types per raw sample **before** packing; when packing, concatenate per-sample token_types in the same order the padding-free collator concatenates labels/input_ids.
- If concatenated length mismatches packed labels after packing, drop token-type metrics for that batch and log a debug warning (no NaNs).
- Keep metrics aggregate-only (no per-dataset buckets); include default targets `lvis` and allow configurable include/exclude.
- When eval packing is disabled (default today), metrics still work on padded eval batches; if eval packing is turned on, packed alignment logic applies.
- Maintain metric key parity with upstream Qwen3-VL for dashboards: keys without prefix `loss`, `token_acc`, `token_count`, and `{desc,coord,format}_token_acc|entropy|token_count`; document intentional delta (aggregate-only vs per-dataset in upstream).

## Edge cases
- Samples with zero supervised tokens: emit IGNORE types, skip metrics for that sample; aggregate denominators guard division-by-zero.
- Oversized single samples (> packing length) that are emitted alone: concatenate their token types directly; still counted in aggregate.
- Mixed image/text prompts share same format as upstream; only assistant payload needed to compute spans.

## Testing strategy
- Unit: compute_token_types span mapping; include/exclude filtering; IGNORE when no supervision.
- Collator: padded batch attaches token_types with correct shape vs labels/attention_mask.
- Packing: two samples packed into one—concatenated token_types matches concatenated labels; mixed include/exclude packed case; misalignment path skips metrics without crashing.
- Trainer smoke: mixin logs aggregate token_acc and desc/coord/format metrics; NaN-safe when masks empty.
