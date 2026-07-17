# Standalone COCO Refinement Gate A Receipt

## Status

**Running for operator UAT on 2026-07-17.** The isolated lightweight service is
available at `http://127.0.0.1:19172/`. This receipt proves the full-index
human-loop launch and read-only browser opening boundary; it does not constitute
operator approval and does not authorize real ROI implementation.

## Runtime Identity

- Checkout commit: `801cda3f` (`Allow full COCO worker recovery at startup`).
- Runtime root: `outputs/coco_refinement/gate-a-20260717/`.
- Process: one Uvicorn worker, reload disabled, numeric loopback only.
- Full restart began at approximately `2026-07-17T14:03:15Z`; the runtime
  recorded `state=ready` at `2026-07-17T14:05:27.543112Z`.
- Both train and validation workers were `idle`, `healthy=true`, and alive;
  `accepting_writes=true` and the sole-writer lock was held.
- The legacy service remained unchanged on `127.0.0.1:8080` under PID
  `1240251` throughout launch and browser probing.

The first launch attempt exposed a production-only five-second worker recovery
timeout after the complete workspace was materialized. Commit `801cda3f`
separated the 300-second production startup budget from the 10-second launch
cleanup budget and added fail-fast timing validation. The launcher/runtime
regression suite passed 89 tests and the fixed-point contract audit reported no
P0/P1/P2 findings before the successful relaunch.

## Full Data and Image Boundary

| Split | API tasks | First task | Last task |
| --- | ---: | --- | --- |
| train | 117,266 | `train:9` | `train:581929` |
| val | 4,952 | `val:139` | `val:581781` |

- SQLite contained 122,218 task identities, zero Draft rows, and zero mutation
  rows after the read-only browser probe.
- The category API returned the official 80 categories from `person` (ID 1) to
  `toothbrush` (ID 90), registry fingerprint
  `8f77d563aaf6579a09d534641befbfdd12cb163b13d3e7d65cb576e49bf7c7a7`.
- Train/validation first and last task payloads opened successfully, with 8/2
  and 20/13 baseline objects respectively.
- All four sampled JPEG routes returned HTTP 200 with `image/jpeg` and
  `Cache-Control: no-store`.
- `train/images` and `val/images` remained symlinks to
  `public_data/coco/rescale_32_1024_bbox/images`; no images were copied.

## Browser Receipt

Cypress 14.5.0 with Electron 130 ran a read-only full-index smoke against the
live `19172` service. The browser opened the first train task, navigated to the
next train task, switched to validation, and observed rendered SVG bbox regions
in every view. A catch-all API interceptor observed no POST, PUT, PATCH, or
DELETE requests. Final result: one test passed in six seconds.

The browser-generated first-failure screenshot and one ignored probe script are
local harness residue only. AgentGuard prohibited their deletion; neither is in
the parent Git commit or the standalone runtime contract.

## Gate

OpenSpec task 3.7 is satisfied by the earlier focused CRUD/autosave/visibility/
Commit browser probes plus this real full-index opening receipt. Task 3.8 remains
open until the operator personally tests train and validation and gives explicit
human-loop approval. Real ROI work remains prohibited before that approval.
