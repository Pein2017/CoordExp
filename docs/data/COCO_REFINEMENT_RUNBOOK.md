---
doc_id: docs.data.coco-refinement-runbook
layer: docs
doc_type: runbook
status: canonical
domain: data
summary: Operate the loopback-only Label Studio workflow for COCO-80 bbox refinement.
tags: [data, coco, label-studio, annotation, runbook]
updated: 2026-07-16
---

# COCO Refinement Runbook

This is the operator route for the `label-studio-coco-refinement` V1. It runs
one Label Studio instance with separate managed `train` and `val` projects.
Only official English COCO-80 axis-aligned bboxes are editable.

The only accepted inputs are:

- `public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl`
- `public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl`

They use integer norm1000 `xyxy` on `0..999`. Source JSONL and source images
are immutable. Runtime state and derived JSONL live below
`outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000/`; each
split's `images` entry is a managed symlink to the shared source image root,
never an image copy.

## Runtime prerequisites

The accepted local runtime is:

- Python 3.12.11 in the pinned checkout's ignored `label-studio/.venv`
  Label Studio environment;
- the `ms` conda environment's installed CoordExp/Torch dependencies, exposed
  to that interpreter through `PYTHONPATH`;
- Node 22.22.0, Corepack 0.34.0, and Corepack Yarn 1.22.22 for frontend builds;
- one visible CUDA device when provisioning or loading a resident ROI profile;
- the pinned `label-studio/` checkout and its frozen `web/yarn.lock`.

Node is not a second production server. It is needed only to install/build the
checked-out frontend or run its tests. Do not use `label-studio start`: its
host wrapper did not preserve the required loopback bind in the accepted
probe. `serve_coordexp_refinement` is the production entrypoint and starts the
two split workers in-process.

From the repository root, establish one shell environment:

```bash
export REPO_ROOT=/data/CoordExp
export RUNTIME_ROOT="$REPO_ROOT/outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
export IMAGE_ROOT="$REPO_ROOT/public_data/coco/rescale_32_1024_bbox/images"
export LS_STATE="$RUNTIME_ROOT/label-studio/state"
export LS_PY="$REPO_ROOT/label-studio/.venv/bin/python"
export MS_PY="$(conda run -n ms python -c 'import sys; print(sys.executable)')"
export MS_SITE="$("$MS_PY" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$REPO_ROOT:$MS_SITE"

export DJANGO_DB=sqlite
export DJANGO_SETTINGS_MODULE=core.settings.label_studio
export BASE_DATA_DIR="$LS_STATE"
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT="$IMAGE_ROOT"
export FEATURE_FLAGS_OFFLINE=true
export LATEST_VERSION_CHECK=false
export SENTRY_DSN=''
export FRONTEND_SENTRY_DSN=''
```

Verify the accepted toolchain and an existing frontend build:

```bash
"$LS_PY" --version
node --version
corepack yarn --version
test -f "$REPO_ROOT/label-studio/web/dist/apps/labelstudio/main.js"
```

If the frozen frontend dependencies or build are absent, rebuild from the
pinned checkout; do not update the lockfile:

```bash
cd "$REPO_ROOT/label-studio/web"
CYPRESS_INSTALL_BINARY=0 corepack yarn install --frozen-lockfile
corepack yarn ls:build
```

## One-time Label Studio identity setup

The managed commands require one existing active Label Studio user and that
user's active organization. For a fresh ignored state, migrate the database,
start stock Django on loopback only, create/sign in to the operator account in
the browser, then stop it with `Ctrl-C`:

```bash
cd "$REPO_ROOT/label-studio"
"$LS_PY" label_studio/manage.py migrate --noinput
"$LS_PY" label_studio/manage.py runserver 127.0.0.1:8080 --noreload
```

Record the numeric IDs without changing database state:

```bash
export OPERATOR_EMAIL='<operator-email>'
"$LS_PY" label_studio/manage.py shell -c \
  'import os; from django.contrib.auth import get_user_model; u=get_user_model().objects.get(email=os.environ["OPERATOR_EMAIL"]); print(f"OPERATOR_USER_ID={u.pk}\nORGANIZATION_ID={u.active_organization_id}")'
export OPERATOR_USER_ID='<printed-user-id>'
export ORGANIZATION_ID='<printed-organization-id>'
```

The account must be active, authenticated, and an active member of that exact
organization. The bootstrap fails closed otherwise.

## Provision an ROI profile

The provisioning CLI loads the exact resolved CoordExp infer config, assembles
the real runtime on exactly one visible CUDA device, fingerprints all model,
adapter, embedding, tokenizer, processor, prompt/parser, and runtime inputs,
and atomically updates the saved profile and launch documents. Use values
approved for the selected model for the axis, total-pixel, acknowledgement,
and deadline fields; they are profile semantics, not universal defaults.

Production infer leaves must live below `configs/coordexp_swift/infer/`; an
authored YAML under `outputs/` is intentionally rejected. The command below is
the execution-verified local step-917 profile, not a global bound for other
models.

```bash
export ROI_PROFILE_STORE="$RUNTIME_ROOT/label-studio/roi-profiles.json"
export ROI_LAUNCH_CONFIG="$RUNTIME_ROOT/label-studio/roi-launch.json"
export ROI_RECEIPT_STORE="$RUNTIME_ROOT/label-studio/roi-receipts.jsonl"
export ROI_INFER_CONFIG="$REPO_ROOT/configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_label_studio_roi.yaml"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT" "$MS_PY" \
  scripts/provision_label_studio_roi_profile.py \
  --infer-config "$ROI_INFER_CONFIG" \
  --profile-store "$ROI_PROFILE_STORE" \
  --launch-config "$ROI_LAUNCH_CONFIG" \
  --receipt-store "$ROI_RECEIPT_STORE" \
  --profile-name step917-coco80 \
  --selector step917 \
  --bind-host 127.0.0.1 \
  --bind-port 8080 \
  --ack-timeout-seconds 30 \
  --min-axis-pixels 32 \
  --max-axis-pixels 2048 \
  --max-total-pixels 4194304 \
  --deadline-seconds 120
```

Omitting `--default-width` and `--default-height` keeps the V1 default at
`1024 x 1024`. Patch alignment is captured from the processor (32 for the
accepted Qwen3-VL profile); it is not a hand-authored CLI flag. Re-running the
command may add another unique selector/profile while preserving identical
global launch settings. A selector conflict, artifact drift, changed global
bind, or missing saved profile fails closed. The reviewer chooses the selector
in the AI Region panel; that selection becomes the active profile for that
project.

## Bootstrap train and validation together

`bootstrap_coordexp_refinement` always plans both managed projects. `attest`
does not mutate source data or live Label Studio rows; a cold attest may
populate ignored deterministic task-index planning sidecars. On a fresh state
it reports `create`, and after a correct bootstrap it reports `reuse`. `apply`
creates or reconciles without deleting rows and performs a post-apply `reuse`
attestation.

```bash
cd "$REPO_ROOT/label-studio"
"$LS_PY" label_studio/manage.py bootstrap_coordexp_refinement attest \
  --repo-root "$REPO_ROOT" \
  --operator-user-id "$OPERATOR_USER_ID" \
  --organization-id "$ORGANIZATION_ID"

"$LS_PY" label_studio/manage.py bootstrap_coordexp_refinement apply \
  --repo-root "$REPO_ROOT" \
  --operator-user-id "$OPERATOR_USER_ID" \
  --organization-id "$ORGANIZATION_ID"
```

This is the only full train+val bootstrap command. It creates one editable
annotation per task, no predictions, separate task/storage/queue/journal
namespaces, and two managed image symlinks. Re-running an exact bootstrap is
idempotent; source, vendor, category, label-config, image-link, task, or
manifest drift is an error, not an implicit migration.

## Launch and shutdown

Launch one process. The configured launch document supplies the loopback bind;
host/port command-line overrides are intentionally unsupported.

```bash
cd "$REPO_ROOT/label-studio"
CUDA_VISIBLE_DEVICES=0 "$LS_PY" label_studio/manage.py serve_coordexp_refinement \
  --repo-root "$REPO_ROOT" \
  --operator-user-id "$OPERATOR_USER_ID" \
  --organization-id "$ORGANIZATION_ID" \
  --roi-launch-config "$ROI_LAUNCH_CONFIG"
```

Open the printed `http://127.0.0.1:<port>` URL on the same machine. The command
applies/attests the two projects, reconciles each split, starts one worker per
split, loads the resident ROI service, registers the managed HTTP boundary,
and then starts Django without autoreload.

Stop with `Ctrl-C` or `SIGINT` and wait for exit. The lifecycle first closes
HTTP admission, then unregisters project bindings, closes ROI inference, and
stops both workers. Avoid `kill -9`; if the process is interrupted, restart the
same command before committing or consuming derived output so startup recovery
can reconcile durable state.

## Reviewer workflow and output lag

Opening a managed task enters the sole editable annotation immediately. Create,
move, resize, relabel, or delete bboxes using canonical COCO-80 English names.
Dense-scene show/dim/hide controls and nearby inference colors are presentation
only; color, badge, conflict, focus, visibility, and ROI metadata never enter
canonical JSONL.

On in-app navigation, the current task's in-memory edit is durably saved as a
Label Studio **Draft** before navigation continues. Navigation does not wait
for dataset publication. Accumulate Drafts across several images, then use the
project **Commit** action:

1. the active Draft is saved;
2. all eligible pending Drafts for the current user and current split are
   frozen into one immutable batch;
3. durable enqueue returns `Queued` without claiming dataset success;
4. the split worker validates and publishes the whole batch atomically;
5. only terminal `Succeeded` advances `working.norm.jsonl` and its generation.

At most one batch is active per split; train and val can progress independently.
Later edits remain newer Drafts and are never overwritten by the frozen batch.
An empty Draft is preserved, but V1 rejects committing any captured member with
an empty object list. Hard reload/tab close warns only for unsaved in-memory
edits; durable Drafts and queued work survive the browser.

For AI Region, draw one temporary ROI, choose independent width/height (default
`1024 x 1024`, both valid for the selected profile), and click Infer. Valid
mapped results are appended as ordinary editable bboxes in one undo action.
There is no NMS, replacement, merge, prediction-copy, or per-box acceptance
stage. Adjust or delete the results and include them in a later normal Commit.

## State, recovery, and diagnostics

Interpret state literally:

- `Queued` / `Running`: the prior complete `working.norm.jsonl` remains
  authoritative; keep editing and let the worker finish.
- `Succeeded`: the complete batch is terminal and the split generation has
  advanced. Enqueue alone is not success.
- `Failed`: no partial member was published; the previous generation and all
  live Drafts remain. Fix the reported Draft validation/staleness problem and
  enqueue a new batch.
- `Reconciling` or an unknown/lost response: do not enqueue another batch, edit
  queue/journal files, materialize coords, or claim success. Preserve the
  runtime tree and restart the same `serve_coordexp_refinement` command. It
  repairs queue/manifest projections from the journal and hash-attested working
  file before admitting work.

If startup remains fail-closed, retain `project.json`, `queue.jsonl`,
`journal.jsonl`, receipt files, and the server log for diagnosis. Never repair
these files manually and never retry a logical batch under the same `batch_id`
with different content.

The important split paths are:

```text
$RUNTIME_ROOT/{train,val}/project.json
$RUNTIME_ROOT/{train,val}/task_index.json
$RUNTIME_ROOT/{train,val}/working.norm.jsonl
$RUNTIME_ROOT/{train,val}/working.coord.jsonl   # explicit derived export only
$RUNTIME_ROOT/{train,val}/working.coord.receipt.json
$RUNTIME_ROOT/{train,val}/queue.jsonl
$RUNTIME_ROOT/{train,val}/journal.jsonl
$RUNTIME_ROOT/{train,val}/images                # symlink to $IMAGE_ROOT
$RUNTIME_ROOT/label-studio/state/               # SQLite and Label Studio state
```

`working.norm.jsonl` is always the last terminal complete editing generation,
so it intentionally lags saved and queued Drafts. `working.coord.jsonl` is not
automatic and may be absent or older; only the explicit command below, bound
to the exact reconciled split store and schema-v3 bootstrap receipt, makes it a
loader-ready derivative.

```bash
cd "$REPO_ROOT"
for SPLIT in train val; do
  PYTHONPATH="$REPO_ROOT" "$MS_PY" \
    scripts/materialize_label_studio_coco_refinement.py \
    --repo-root "$REPO_ROOT" \
    --split "$SPLIT"
  RECEIPT="$RUNTIME_ROOT/$SPLIT/working.coord.receipt.json"
  OUTPUT="$RUNTIME_ROOT/$SPLIT/working.coord.jsonl"
  MANIFEST="$RUNTIME_ROOT/$SPLIT/project.json"
  OUTPUT_SHA256="$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
  GENERATION="$(jq -r '.generation' "$MANIFEST")"
  jq -e \
    --arg output_sha256 "$OUTPUT_SHA256" \
    --argjson generation "$GENERATION" \
    '.code == "label_studio.working_coord_materialized"
      and .loader_attestation.status == "passed"
      and .materialization.destination_sha256 == $output_sha256
      and .materialization.generation == $generation
      and .store.generation == $generation' \
    "$RECEIPT"
done
```

## Validation

Run checks under the same environment and require exact `reuse`:

```bash
cd "$REPO_ROOT/label-studio"
"$LS_PY" label_studio/manage.py check
"$LS_PY" label_studio/manage.py bootstrap_coordexp_refinement attest \
  --repo-root "$REPO_ROOT" \
  --operator-user-id "$OPERATOR_USER_ID" \
  --organization-id "$ORGANIZATION_ID"

test "$(readlink -f "$RUNTIME_ROOT/train/images")" = "$IMAGE_ROOT"
test "$(readlink -f "$RUNTIME_ROOT/val/images")" = "$IMAGE_ROOT"
test "$(wc -l < "$REPO_ROOT/public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl")" -eq 117266
test "$(wc -l < "$RUNTIME_ROOT/train/working.norm.jsonl")" -eq 117266
test "$(wc -l < "$REPO_ROOT/public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl")" -eq 4952
test "$(wc -l < "$RUNTIME_ROOT/val/working.norm.jsonl")" -eq 4952
```

Run the bounded real-fixture materializer/training-path probe into a new ignored
receipt root:

```bash
cd "$REPO_ROOT"
export MATERIALIZER_PROBE_ROOT="outputs/label_studio_coco_refinement/materializer-probes/$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONPATH="$REPO_ROOT" "$MS_PY" \
  scripts/probes/label_studio_coco_refinement/materialize.py \
  --output-root "$MATERIALIZER_PROBE_ROOT"
jq -e \
  '.code == "label_studio.materializer_probe_passed" and .output.object_ids == ["589229", "-1", "-2"]' \
  "$REPO_ROOT/$MATERIALIZER_PROBE_ROOT/probe.json"
```

Before the first apply, record the two source JSONL hashes outside the source
tree; after any recovery, batch, export, or rollback, re-run `sha256sum -c`.
Project attestation, startup, batch validation, and the materializer all fail
closed on schema, identity, geometry, COCO mapping, image-link, generation, or
hash drift. A terminal working object contains canonical dataset fields only;
presentation metadata must not appear there.

## V1 exclusions

V1 deliberately excludes arbitrary JSONL, image-only/no-coordinate tasks,
empty committed samples, crowd regions, polygons, masks, rotated boxes,
free-form or translated classes, multi-user adjudication, LAN/remote-browser
deployment, multiple active batches per split, queued/batch ROI inference,
automatic NMS/merge/deletion, and automatic training-data promotion.

## Runtime-only rollback

Rollback is destructive only to ignored derived state. First stop the service
and optionally archive the runtime receipts/journals. Then verify the exact
root before removal:

```bash
EXPECTED="/data/CoordExp/outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000"
if test "$(realpath -e "$REPO_ROOT")" = /data/CoordExp \
  && test "$RUNTIME_ROOT" = "$EXPECTED" \
  && test -d "$RUNTIME_ROOT" \
  && test ! -L "$RUNTIME_ROOT" \
  && test "$(realpath -e "$RUNTIME_ROOT")" = "$EXPECTED"; then
  rm -rf -- "$RUNTIME_ROOT"
else
  echo "Refusing rollback: runtime root identity check failed" >&2
  false
fi
```

This removes the Label Studio database, Drafts, queues, profiles, receipts, and
derived JSONL. It removes the managed image links, not their targets. Never
delete or rewrite `public_data/coco/rescale_32_1024_bbox_len12000/` or
`public_data/coco/rescale_32_1024_bbox/images/`. A later clean bootstrap starts
again from the unchanged accepted sources.
