# VL packing wrapper – merged final plan (no external deps touched)

## Root causes (from `issues.md`)
- `PackedCaptionDataset` is an `IterableDataset` but exposes `__len__`; ms-swift sees `__len__`, injects `BatchSamplerShard`, and PyTorch rejects `batch_sampler` for iterables → immediate crash.
- With a lengthless dataloader, HF/accelerate requires a positive `max_steps`; leaving `max_steps = -1` aborts before training.
- Secondary: prior LVIS path slip (use `public_data/lvis/rescale_32_768_poly_20/*`).

## Chosen approach
- Keep the **streaming, rank-local packer**; do not modify ms-swift or HF.
- Make the wrapper **pure iterable**: drop/guard `__len__` (or add `packing_iterable` flag) so ms-swift skips `BatchSamplerShard`. Cache `length_hint` from the base dataset before wrapping for logging/schedulers only.
- **Require or auto-derive `max_steps`** when packing is on: if unset, set `ceil(base_len / (grad_accum_steps * world_size))` as a conservative default (using un-packed length and expected fill); log the derived value. Fail early if it can’t be resolved.
- Enforce `per_device_train_batch_size = 1`; retain throughput via `gradient_accumulation_steps`. Error if user overrides after packing is enabled.
- Maintain template flags (`packing=true`, `padding_free=true`), multimodal passthrough, buffer bin-packing with carry + `min_fill_ratio`, and allow-single-long handling.
- Apply packing **after dataset construction**; forward `set_epoch` so buffers reset each epoch. Eval remains un-packed unless `eval_packing=true`.

## Implementation steps
1) **Wrapper (`src/datasets/wrappers/packed_caption.py`)**
   - Remove/guard `__len__`; add optional `length_hint` (cached base length) for telemetry.
   - Keep streaming iterator; log packs, avg fill, single-long, skipped counts per epoch; warn on samples missing `length`.
2) **sft wiring (`src/sft.py`)**
   - Cache `base_len = len(dataset)` before wrapping.
   - When packing enabled: force batch size 1; if `max_steps<=0`, derive as above and warn; use `base_len` for curriculum/telemetry instead of `len(wrapper)`.
   - Ensure dataloader path treats the wrapped dataset as iterable (no `batch_sampler` injection).
   - Ensure `set_epoch` continues to propagate after wrapping so packing buffers reset each epoch.
3) **Validation & docs**
   - Fail fast on invalid `packing_length` or unresolved `max_steps`.
   - Document iterable semantics, `max_steps` requirement, rank-0 dispatcher behavior, and recommended knobs (`packing_buffer`, `min_fill_ratio`).
4) **Smoke tests**
   - Tiny LVIS with packing on and small `max_steps`: expect no `batch_sampler` error and >0 steps.
   - 2-GPU run to confirm unique packs per rank and fill stats logging.
   - Oversized sample path with `allow_single_long` on/off logs, not crashes.

## Risks & mitigations
- **Rank-0 dispatcher overhead:** DataLoaderDispatcher centralizes loading for iterables. Mitigate with modest `packing_buffer` (256–512) and a few workers; consider length-cache map-style variant later if throughput is lacking.
- **Step-count drift:** Derived `max_steps` uses un-packed length; log estimated vs observed pack counts and allow manual override.
- **Binpacking missing:** ImportError already surfaced; keep noted in runbook.

## Why this is optimal
- Minimal, targeted edits; preserves streaming memory efficiency.
- Resolves both blockers (batch_sampler conflict and max_steps requirement) without touching external dependencies.
- Fully aligned with OpenSpec: rank-local packing, multimodal preservation, config-first surface, aggregate metrics only.
