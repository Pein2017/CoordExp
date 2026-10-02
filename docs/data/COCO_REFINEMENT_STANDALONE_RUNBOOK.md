---
doc_id: docs.data.coco-refinement-standalone-runbook
layer: docs
doc_type: runbook
status: draft
authority: active-change
change: coco-refinement
domain: data
summary: Operate the sole active human-only COCO refinement Gate A editor.
tags: [data, coco, annotation, standalone, gate-a, runbook]
updated: 2026-10-01
---

# Standalone COCO Refinement Gate A Runbook

Gate A is the sole active COCO refinement editor. The former Label Studio V1
service is retired; its two pending drafts are preserved separately for human
recovery and are not imported here. See
[`COCO_REFINEMENT_RUNBOOK.md`](COCO_REFINEMENT_RUNBOOK.md) for the retired
editor's recovery notes.

Gate A is a lightweight, build-free FastAPI service with a browser-native
HTML/CSS/ES-module SVG editor. SQLite stores sparse Drafts only for touched
tasks. The accepted full indexes contain 117266 train tasks and 4952 val tasks.

## Dependencies

Run from `/data/CoordExp` in the `ms` conda environment. Launch preflight
requires exactly:

- FastAPI 0.136.3;
- Uvicorn 0.37.0;
- Starlette 0.52.1;
- one Uvicorn worker, reload disabled, and `WEB_CONCURRENCY` unset or `1`.

There is no Node build, Label Studio frontend, Django, GPU, or ROI-model
dependency for Gate A. Check the HTTP stack before launch:

```bash
conda run -n ms python -m pip show fastapi uvicorn starlette
```

## Launch and shutdown

The isolated Gate A identity is:

```text
URL:          http://localhost:53662/
runtime root: /data/CoordExp/outputs/coco_refinement/gate-a-20260717
```

The UI opens on train and provides a train/val split selector. Numeric
loopback is mandatory; the retired Label Studio service no longer uses port
8080.

Gate A now has one fixed direct browser endpoint and does not require a
separate browser relay:

```text
browser URL: http://localhost:53662/
bind URL:    http://127.0.0.1:53662/
```

Launch one foreground process from any working directory:

```bash
bash /data/CoordExp/scripts/launch_coco_refinement_gate_a.sh
```

The wrapper fixes the numeric bind and accepted browser authority to port
`53662`, fixes the runtime root above, enters the repository root, and uses the
`ms` conda environment. `--print-config` reports those values without starting
the runtime. Port overrides are intentionally rejected so bookmarks remain
stable. Both `localhost:53662` and `127.0.0.1:53662` are accepted authorities;
the fixed launcher also accepts a canonical non-default local port chosen by
VS Code when Host remains `localhost` or `127.0.0.1`. Mutations still require
Origin to exactly match that browser-selected Host and a valid CSRF token.
Other hosts and forwarded authority headers remain forbidden.

Startup deliberately validates the complete train/val workspace before it
binds the browser port, which normally takes about one to two minutes on the
current Pod. The launcher prints this pre-bind phase immediately. VS Code can
discover and offer to forward `53662` only after the terminal reports
`Uvicorn running on http://127.0.0.1:53662`.

The launcher enforces a latest-instance-wins policy on the fixed port. If
`53662` already has a listener, it reports the owning process, sends `TERM`,
waits up to ten seconds for the listener and process to exit, then uses `KILL`
only if necessary. Startup proceeds only after the port is confirmed free.

Only one process may own this runtime root. A second launcher fails before
binding the port. Stop the owning foreground process with `Ctrl-C` and wait
for exit so both split workers and the writer lock shut down cleanly. Do not
use `kill -9` for normal shutdown.

## Runtime roots and status

All mutable standalone state currently remains below the existing `outputs/` root:

```text
outputs/coco_refinement/gate-a-20260717/
  runtime.json
  runtime-health.json
  state.sqlite3
  train/
    working.norm.jsonl
    project.json
    task_index.json
    queue.jsonl
    journal.jsonl
    images -> public_data/coco/rescale_32_1024_bbox/images
  val/
    ...
```

`runtime.json` records the accepted dependency/process receipt.
`runtime-health.json` records the current runtime state, source identities,
writer-lock state, and train/val worker health. Inspect them without editing:

```bash
conda run -n ms python -m json.tool \
  outputs/coco_refinement/gate-a-20260717/runtime.json
conda run -n ms python -m json.tool \
  outputs/coco_refinement/gate-a-20260717/runtime-health.json
curl --noproxy '*' -fsS http://127.0.0.1:53662/api/splits/train/state | \
  conda run -n ms python -m json.tool
curl --noproxy '*' -fsS http://127.0.0.1:53662/api/splits/val/state | \
  conda run -n ms python -m json.tool
```

The UI distinguishes task state (`Committed`, `Saving`, `Draft`, `Conflict`)
from batch state (`Queued`, `Running`, `Reconciling`, `Succeeded`, `Failed`).
Durable enqueue is `Queued`, not dataset success.

## Draft, navigation, and Commit

- A completed edit gesture autosaves the task's complete canonical object list
  as a sparse SQLite Draft.
- In-app navigation waits for the active Draft save, but never waits for a
  running Commit batch.
- `Commit Drafts` captures all pending Drafts for the selected split into one
  immutable batch and returns after durable enqueue.
- The split worker publishes that frozen batch atomically. Later edits remain
  newer Drafts for a later Commit.
- Only `Succeeded` advances that split's `working.norm.jsonl` generation.
  `Failed` preserves the Drafts and does not advance a partial generation.

Final-bbox deletion is currently blocked pending an explicit operator policy.
Do not use Gate A to commit a task with its final bbox removed. Visibility,
selection, zoom, pan, color, and Undo presentation state are not saved data.

## Recovery

Drafts, queued batches, journals, and completed SQLite transactions survive a
browser reload or process restart. After an interruption, restart the exact
launcher command against the same runtime root. Startup reconciles train and
val before accepting writes; wait for `state: ready`, healthy workers, and no
`health_error` in `runtime-health.json` before continuing.

If the browser lost a save or Commit response, reload the authoritative task
or poll the displayed batch identity. Do not edit SQLite, queue, journal, or
working JSONL files by hand, and do not launch a fresh runtime root as a
substitute for recovery.

The current public-data source hashes differ from the original bootstrap
contracts. Preserve this complete accepted runtime, including its per-split
project receipts, publication receipts, journals, and working JSONL, so startup
can verify the exact published generation. A fresh empty runtime would fail
the frozen source-hash check; do not relax that check or create a replacement
runtime to bypass it.

## Output and materialization status

The current Gate A output is the generation-bound per-split
`working.norm.jsonl` under the runtime root. It is derived output, not a source
file, and advances only after terminal Commit success.

Relocating these mutable annotation views out of the current runtime is
explicitly USER-DEFERRED to the next round. Keep the complete accepted runtime
at its existing `outputs/coco_refinement/gate-a-20260717` path until then.

Standalone coord materialization is not yet an approved Gate A operator step:
the active change still requires a representative terminal generation to be
materialized and loaded through the current coordexp-infras data path. Do not
treat `working.norm.jsonl` as training coord JSONL, copy it over source data,
or claim downstream materialization acceptance from this gate.

## Known exclusions

Gate A does not include ROI selection or inference. It also excludes remote/LAN
access, multiple users or browser writers, masks, polygons, rotated boxes,
arbitrary classes, automatic import of the two pending retired Label Studio
Drafts, synchronous publication, and automatic promotion into training.

## Protected inputs and fallback

These inputs are immutable:

```text
public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl
public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl
public_data/coco/rescale_32_1024_bbox/images/
```

The runtime validates their identities and fails closed on drift. It must not
rewrite or copy source JSONL or images; each split's `images` entry is a
managed symlink to the shared image root.

Gate A owns the active editor route. The retired Label Studio port 8080 is not
a fallback and must remain unused. Do not restart its Django service or ROI
model; consult [COCO_REFINEMENT_RUNBOOK.md](COCO_REFINEMENT_RUNBOOK.md) only
for pending-draft recovery context.
