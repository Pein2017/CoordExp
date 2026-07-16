# Label Studio runtime feasibility

Date: 2026-07-15 UTC

Scope: exact checkout `/data/CoordExp/label-studio` at
`acbe4550f6ed81d1a0342913bb309f63cbe1452e` (`1.24.0.dev0`) and the dedicated
ignored runtime under `outputs/label_studio_coco_refinement/`. No source data,
source images, project records, or tracked vendor files were changed by these
checks.

## Corrected environment finding

The first automated receipt checked only conda environment `ms` and therefore
reported that Label Studio and Django were unavailable. That conclusion does
not describe the intended runtime: the existing dedicated virtual environment
`outputs/label_studio_coco_refinement/.venv` is 729 MiB and contains Python
3.12.11, Django 5.1.15, the editable pinned Label Studio checkout, and the
`label-studio` executable.

With
`LABEL_STUDIO_BASE_DATA_DIR=outputs/label_studio_coco_refinement/label-studio/state`,
the dedicated interpreter executed `manage.py check` successfully:

```text
System check identified no issues (1 silenced).
```

The existing ignored SQLite state was empty at inspection time: zero users,
projects, tasks, annotations, and Drafts.

## Loopback execution probe

The dedicated executable started Django on loopback. Port 18080 was already in
use, so the development launcher selected 18081. Requests explicitly bypassed
environment HTTP proxies.

| Request | Result |
| --- | --- |
| `GET /health` | HTTP 200, 16 bytes |
| `GET /api/projects` without authentication | HTTP 401 JSON |
| `GET /` followed to login | HTTP 200 at `/user/login/`, 27,440 bytes |
| `GET /react-app/main.css` | HTTP 404 |

The process was then stopped cleanly with SIGINT. This proves the backend,
database, authentication boundary, and loopback launch are executable. It does
not prove a usable editor because the React application bundle is absent.

## Frontend runtime and build

Node 22.22.0 and Corepack 0.34.0 are present. Corepack resolves Yarn 1.22.22;
the executable is invoked as `corepack yarn` because no global `yarn` shim is
on `PATH`. The pinned `label-studio/web/yarn.lock` dependencies were installed
with `--frozen-lockfile` and the configured HTTP(S) proxy. The install was
completed with `CYPRESS_INSTALL_BINARY=0` after the optional Cypress 14.5.0
desktop binary download was truncated at 176,350,912 of 202,897,924 bytes and
failed its checksum. A separate resumable download then produced the expected
202,897,924-byte archive with the published SHA-512
`7aa4d87e...e573b3d9`; Cypress installed successfully from that verified local
archive. The test-only Xvfb, GTK, NSS, and related runtime libraries were then
installed, and `cypress verify` completed with `Verified Cypress!` for the
14.5.0 binary. Browser E2E is therefore executable against this pinned runtime.

The exact pinned frontend then executed:

```text
corepack yarn ls:build
NX Successfully ran target build for project labelstudio
webpack compiled successfully (f48dc2f42dfd8d50)
Done in 76.81s.
```

Full ignored logs are under
`outputs/label_studio_coco_refinement/frontend-runtime/`. The build emitted
only dependency/resolution, peer-dependency, and stale Browserslist-data
warnings; it did not modify tracked vendor sources.

After the build, the dedicated Django runtime served both
`/react-app/main.css` (HTTP 200, 233,110 bytes) and `/react-app/main.js`
(HTTP 200, 2,399,379 bytes), closing the earlier missing-bundle observation.
The `label-studio start --host 127.0.0.1` wrapper still announced a
`0.0.0.0` development bind, so it is not the accepted V1 launcher. Direct
`manage.py runserver 127.0.0.1:18083 --noreload` bound exactly to loopback and
served both `/health/` and the built CSS with HTTP 200. The operator launcher
must retain this direct loopback binding (or an equivalently attested fixed
launcher) rather than relying on the wrapper's `--host` flag.

## Gate status

- Backend bootstrap and Django endpoint tests are no longer blocked by Python
  dependencies; use the dedicated ignored virtual environment rather than
  conda `ms` for vendor execution.
- Frontend source, build, unit-test, and browser-E2E work can now proceed
  against the pinned checkout; the checksum-valid Cypress binary and its host
  runtime dependencies are verified.
- Task 1.4 remains open until the specified fake-backend Draft/status/direct-
  insertion spikes execute; the production build portion is now attested.

## Full-project production preview execution

On 2026-07-16, the explicit `serve_coordexp_refinement` command started the
full ignored runtime on `127.0.0.1:8080` with the deployed Label Studio revision
`eb40d7d000b8110d0a853b402bcd1c48e98a3c9c`.  Construction completed in about
710 seconds on the first successful run.  The dominant cost was repeated
full-train bootstrap/task attestation; model weights were not loaded during
server startup.

The executed runtime, not a synthetic adapter, attested:

| Split | Tasks | Authoritative annotations | Drafts | Predictions |
| --- | ---: | ---: | ---: | ---: |
| train | 117,266 | 117,266 | 0 | 0 |
| val | 4,952 | 4,952 | 0 | 0 |

Both split-local `images` entries are symlinks resolving to the same immutable
`public_data/coco/rescale_32_1024_bbox/images` root.  Their project-bound local
storage records remain restricted to `train2017` and `val2017`.  The source
hashes remained:

- train: `d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a`;
- val: `a34afb33c567690f56fa3704e213cfc00dc3c000f018d2bb89a7f60139efd795`.

Generation-zero working files are ordinary derived JSONL with 117,266 and
4,952 rows.  Their manifest-attested hashes are respectively
`5cbd91b6e7e45a2ba211e3f3a77c72bc6f847b8c6548a1d6b3feb0bf74c35ac5`
and `8b8a79d4e465b09418a3c6b4ae4963817491fa7bbbebb6d374b65608ee595cfb`.
Authenticated smoke checks returned HTTP 200 for both projects, task 1, its
authoritative annotation, the shared JPEG, managed project state, and the safe
`step917` profile projection.

Warm request samples on the full train project were approximately 104 ms for
the project page, 92 ms for task 1, and 12 ms for its JPEG.  The managed
`project-state` request took 1.7--2.3 seconds for train but about 124--128 ms for
val.  The size-proportional cause was isolated: the read-only Draft catalog
called `restore_drafts(())` even with zero Drafts, forcing a complete
hash-attested working-file scan, and the UI polls this endpoint every 1.5
seconds.  This is a runtime-performance defect, not a source-data or inference
failure.  The correction must preserve full attestation for explicit Commit
capture and invalidate any read-only baseline cache when the published
generation or working hash changes.

## Exact-current recovery and materializer execution

On 2026-07-16, a disposable three-row store executed seven real subprocess
publication cuts using `SIGSTOP` followed by `SIGKILL`.  The durable receipt is
`outputs/label_studio_coco_refinement/recovery-concurrency/20260716T1500Z/receipt.json`.
It binds store SHA-256
`10acc1744679bd75f5a9004418e9b53459b7f14ee530660246ba321e6e4935a2`
and records:

- pre-rename recovery retaining generation 0 with terminal failure;
- six post-rename/fsync/manifest/journal/queue cuts reconciling exactly once to
  generation 1 success;
- supported readers and status queries failing closed while publication was
  paused;
- second admission returning the active batch or blocking while the queue
  terminal lock was held;
- byte-identical repeated recovery, no orphan candidates, no partial rows, and
  unchanged synthetic source/image hashes.

The current full train materializer then ran against a shared-lock snapshot,
not the live split.  Receipt
`outputs/label_studio_coco_refinement/materializer-probes/full-train-current-20260716-a/probe.json`
binds the executed materializer/store/loader source hashes and reports:

| Measure | Result |
| --- | ---: |
| Working rows | 117,266 |
| Objects | 849,947 |
| Wall time | 325.83085 s |
| Peak RSS | 245,284,864 bytes |
| Coord output size | 156,908,539 bytes |
| Coord output SHA-256 | `24bda9a1f360253d8d7e26ad403622e1e1c1bdfeebc7675b6871cfb9195d08ab` |

`src.data.iter_raw_examples` accepted all 117,266 derived rows.  The selected
source remained
`d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a`
and live `working.norm.jsonl` remained
`5cbd91b6e7e45a2ba211e3f3a77c72bc6f847b8c6548a1d6b3feb0bf74c35ac5`
before and after.  Separate standards and intent audits found no remaining
P0/P1 for the materializer or recovery tasks.

The filesystem did not support reflinks, so each executed snapshot copied
about 1.6 GiB, mostly the canonical bootstrap task index.  AgentGuard denied
deleting these already-created disposable roots and explicitly prohibited a
retry or workaround:

- `outputs/label_studio_coco_refinement/full-materializer-probes/20260716T1438Z/repo`;
- `outputs/label_studio_coco_refinement/materializer-probes/full-train-current-20260716-a/repository`.

They are probe residue, not project/source authority, and require an operator
to remove them.  The tracked `materialize_full.py` probe is now schema v2: it
embeds the full compact receipt and cleans its copied repository by default;
`--retain-snapshot` is the explicit opt-in.  This cleanup-only harness change
does not alter the executed production materializer/store hashes above.

## Managed Draft lifecycle fixed point

The current source adds one authenticated, CSRF-protected per-task lifecycle
endpoint.  Task, annotation, current-user Draft, source-row, working generation,
and committed result authority are resolved server-side.  Exact terminal
rebase requires the persisted Draft ID, canonical UTC `Z` revision, server
semantic hash, and exact browser serialization to remain unchanged.  A newer
Draft receives only stable region key, committed ID, and committed-bbox
baseline metadata; newer geometry, class, membership, creation order, training
metadata, inference provenance, presentation, and undo history remain intact.

Responses that cross into an active Draft save or editor history freeze defer
without mutating metadata or consuming their retry key.  Explicit persisted
Draft discard/reset uses the same exact-token and store-generation checks.
The real DRF serializer's `Z` timestamp was probed directly and the catalog was
corrected from the non-identical `+00:00` representation.

Current fixed-point verification is 699 parent tests, 164 Django refinement
tests, and 114 managed frontend tests, with Ruff/Biome and diff checks clean.
Two independent audits closed both discovered P1s.  These source changes are
not deployed into the active UAT process: it deliberately remains on revision
`eb40d7d000b8110d0a853b402bcd1c48e98a3c9c` until an explicit restart window.

## ROI cancellation and orphan-receipt closure

On 2026-07-16, nested revision
`ff0d82ecd272d7fc4ba938015da4b68fdef7fd84` and parent revision
`a89ce02ee9441110abd42156ceba02b0c953c50d` closed the production cancellation and receipt-expiry gaps without
restarting the active UAT process.  The fixed point now:

- registers the authenticated project/split/principal request before target
  capture and atomically binds the server-frozen target;
- preserves bounded abandon-first intents for 60 seconds, never evicts an
  accepted intent, and fail-closes unknown inference during capacity
  saturation;
- propagates cancellation through the managed resident single-flight path,
  records backend-observed/CUDA-synchronized terminal evidence, reuses a safe
  backend, and poisons a backend after synchronization failure;
- accepts exact existing no-insertion terminals without annotation mutation,
  while enforcing same-reason idempotence for both early and produced-result
  abandonment receipts;
- removes the raw receipt-layer expiry mutator and routes overdue candidates
  through the finalizer's transition fence and locked Draft authority, including
  a disposition-winner recheck.

Verification at this fixed point is 197 parent resident/launch/runtime tests,
71 targeted Django tests, the 26-test service slice under the ordinary Label
Studio venv, and 98 managed frontend tests.  Independent execution additionally
attested the capacity/TTL fence, exact reasons, zero-linkage expiry, disposition
winner, real synchronization-failure poisoning, deterministic pre/post-backend
cancellation, and a Django setup that does not import Torch before the lazy ROI
runtime boundary.  Engineering and intent/contract audits both report no
remaining P0/P1 and approve OpenSpec tasks 4.3 and 4.6.

This is source evidence only.  The current localhost UAT process remains on
`eb40d7d000b8110d0a853b402bcd1c48e98a3c9c`; browser E2E and one accepted
real-profile smoke remain tasks 5.5 and 5.6.
