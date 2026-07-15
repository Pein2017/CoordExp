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
