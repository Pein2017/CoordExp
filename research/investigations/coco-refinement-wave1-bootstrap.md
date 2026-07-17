# Standalone COCO Refinement Wave 1 Bootstrap Receipt

## Status

**Passed on 2026-07-17.** This receipt attests one fresh and one idempotent
restart bootstrap of the complete approved train and validation sources. It is
execution evidence for OpenSpec task 1.7, not a latency optimization claim.

## Evidence Boundary

- Checkout HEAD: `01a453e099828ab6e225b1f77b3a5630a3f7a7cd`.
- Isolated runtime root:
  `outputs/coco_refinement/probes/wave1-full-20260717-20260717-024651/`.
- The accepted preflight receipt recorded FastAPI `0.136.3`, Uvicorn `0.37.0`,
  Starlette `0.52.1`, one worker, reload disabled, and
  `WEB_CONCURRENCY=1` before bootstrap.
- Relevant implementation SHA-256 values:
  - `bootstrap.py`: `58a5192bd663975b098ce6c401a8f8aea7c719294dd783ee5334a75deae96f0e`
  - `repository.py`: `d034ae8d3655d5953692a23f61e2f3e5e5c0dc47a38f79209257fe244ea4c47b`
  - `canonical.py`: `6127eebbfbd570b450dbbc851f2508c7e2cf7e552027666efb3abab756435468`
  - `preflight.py`: `73044351b891aa96892f4763c3e8976ef384e7a9a35584e856c41fe4565b2caa`

The probe called `bootstrap_workspace()` for both production source contracts,
called it a second time against the same root, opened first/middle/last tasks
and images in each split, queried SQLite counts, measured non-symlink runtime
bytes, and recomputed protected source/sample image hashes.

## Full-Index Result

| Measurement | Result |
| --- | ---: |
| Fresh train + validation bootstrap | 104.423 s |
| Idempotent restart/bootstrap validation | 46.419 s |
| SQLite projects | 2 |
| SQLite tasks | 122,218 |
| SQLite train tasks | 117,266 |
| SQLite validation tasks | 4,952 |
| SQLite Draft rows | 0 |
| SQLite mutation rows | 0 |
| Runtime regular files | 19 |
| Runtime regular-file bytes | 226,583,203 |

Both split stores reported `created=true` on the first call and `created=false`
on the restart. Both remained at generation 0. The SQLite task rows equal the
exact source counts and contain no baseline object arrays.

Largest derived files were:

| File | Bytes |
| --- | ---: |
| `train/working.norm.jsonl` | 116,111,083 |
| `state.sqlite3` | 58,531,840 |
| `train/task_index.json` | 45,119,374 |
| `val/working.norm.jsonl` | 4,921,651 |
| `val/task_index.json` | 1,863,035 |

No images were copied. `train/images` and `val/images` are managed symlinks to
the same approved root:
`public_data/coco/rescale_32_1024_bbox/images/`.

## Protected Inputs

The complete source SHA-256 values matched before and after bootstrap:

- train: `d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a`
- validation: `a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795`

First/middle/last image samples in each split also matched byte-for-byte before
and after. The sampled image IDs were train `9`, `291780`, `581929` and
validation `139`, `289586`, `581781`.

## Representative Open Latency

The supported store baseline read intentionally attests the complete working
JSONL before returning a row. Measured first/middle/last latencies were:

- train: `1.859 s`, `1.776 s`, `1.622 s`;
- validation: `0.069 s`, `0.069 s`, `0.069 s`.

Indexed image resolution, byte hashing, and dimension validation measured
`0.0004-0.0134 s` across the same six tasks.

The train baseline latency is a recorded Gate A experience risk, not a failed
throughput gate. The approved change explicitly defers optimization; user
experience on the isolated service will determine whether a generation-bound
offset/cache is needed.
