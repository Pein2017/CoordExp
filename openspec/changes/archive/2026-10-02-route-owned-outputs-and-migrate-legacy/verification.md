# Output ownership migration — final acceptance

Verified on 2026-10-02. All 11 tasks are **lead-accepted**; the earlier annotation
deferral is superseded by the user's completed-annotation adoption and draft
retirement ruling. Global migration is complete within this change's scope.
Root `outputs/` contains only four selected shared checkpoint packages with
required metadata: 26 files, 346,671,422 logical bytes, no symlinks. Inference
results/metrics are the other permitted root class; none are currently retained.

## Accepted annotation result

Five submitted human Label Studio completions (image IDs 7116, 309264, 351017,
417044, 477415) were missing from the prior edited train views. Both current
`train.norm.jsonl` and `train.coord.jsonl` now contain the full submitted lists:
83 to 178 objects, with 105 added, 10 deleted and seven box edits. The other
117,261 training rows retain their bytes and order. Both train views contain
117,266 rows; val remains 4,952 rows and its views were not modified.
The unsuffixed train/val JSONLs remain original pixel-coordinate views.

The exact sanitized completed source and truthful offline publication receipt
are under `public_data/coco/annotation_sources/label-studio-project3-20260918/`.
The lead checked every selected object against the raw completed annotations,
non-object fields, native IDs, norm/coord parity, and the 7116 deletion/two-edit
golden case. Installation guards rejected changed candidate inodes; actual
hash equality requalified those identical bytes before installation. No frozen
receipt was rewritten to bypass the guard. The installed files have the
validated candidate identities. No native Gate A generation was invented.

The current six-file manifest was stale for every materialized JSONL. It now
records the resulting train pair, previously verified Gate A val generation 7,
and measured unchanged pixel files. Its original processing command is explicitly
historical; recovery of edited views also requires preserved annotation sources.
The old manifest remains in Git history.

Gate A's eight committed generations were already published and verified.
Their unchanged receipts/journals now belong to
`public_data/coco/annotation_sources/gate-a-20260717/{train,val}/`.
The exact old service stopped gracefully; its five unfinished drafts and obsolete
state were discarded. After submitted-source publication, the qualified old
Label Studio family (414 files, 5,493,634,309 bytes, 78 aliases) and its two
unfinished drafts/export manifest were retired. Image targets and current views
were preserved. There is no new editor deployment.

The self-contained review HTML/JSON now belongs to Research Probes. Its static
server serves the same origin, filename, source bytes and browser storage key at
`http://localhost:8766/unmatched-prediction-review.html`; HEAD returned 200.
Its initially stale server directory produced 404 and was repaired by restarting
only that exact static server against the accepted directory. Review judgments
are research evidence, not newly adopted annotation ground truth.

## Retained earlier acceptance

- Main `46cc16c0c`, `818fba0b8`, `25e8d95d7`: canonical document routes, current
  base COCO builders, and retirement of obsolete constructors plus 395 copied
  Python, 277 copied YAML, one shell file and 23 exclusive callers/tests.
- Main `04aa7dbdf`, Research Probes `dd67e331e`, Web `426b7f044`: strict frozen
  model-card placement and future producer packaging. All 86 historical and
  34 newer card payloads/identities remain unchanged. Placement qualification
  does not establish tensor integrity or scientific validity.
- User-authorized v1 and weighted-v2/hard-v2 datasets are retired. Seven small
  original v2 provenance/consumer records remain at their existing research
  artifact owner. Two visual interpretations live at their research question;
  redundant output prose is retired. Prior exact receipts remain unchanged.
- Main `7c4b3545a`: editor defaults use application state under `.local/state/`;
  the native CLI rejects root output/dataset/image destinations and path escapes.
- Main `a2d46c325`: current annotation provenance, closed editor records, retired
  draft manifest and the two-class shared-root storage policy.

## Validation

| Check | Result |
|---|---|
| Earlier base builder / metadata packages | Reused unchanged accepted evidence: 27 / 32 CPU tests. |
| Editor default/path package | Reused unchanged accepted evidence: 77 tests; initial RED covered obsolete defaults and forbidden destinations. |
| Annotation capture / build / budget | All final commands exited 0. Five selected rows reached the current loader, renderer and encoder; max 1,950 / 12,000 tokens, no model loading or image-pixel materialization. |
| Actual publication consumer | Exit 0; actual installed payload identities and all six manifest sizes agree with accepted receipts. |
| Manifest schema/path/shape/policy | Four selected tests passed, exit 0. Full JSONL-tree validation is not claimed; unrelated extra smoke JSONLs remain outside this six-file identity. |
| Exact root allowlist | Exit 0; only four previously selected checkpoint packages, no editor/review/cache roots or symlinks. |
| Four-root output checker | Exit 0; 26 / 514 / 39,183 / 1 files at the scan snapshot, no symlinks or findings. |
| Research knowledge | Exit 0; 331 catalog entries, 167 claim references, no errors; five external links not revalidated. |
| OpenSpec strict validation | One change valid, zero errors, exit 0; no delta specs. |
| Git/directory boundaries | Explicit owned-path commits and diff checks; unrelated active research and worktree locks preserved. |

The final consumer commands are:

```sh
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/coordexp-infras/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs \
  --root /data/CoordExp/.worktrees/research-probes-web-codex/outputs
# From the Research Probes checkout:
python scripts/check_research_knowledge.py check
# Before archiving this change, from main:
openspec validate route-owned-outputs-and-migrate-legacy --strict --json
```

Current structured acceptance and raw checker outputs are at
`/data/CoordExp/.worktrees/coordexp-infras/outputs/maintenance/output-storage-closeout-20261002/final-global-closeout.json`.
`inventory.json` binds the unchanged original machine inventory and describes
current dispositions; Git history preserves older maintenance snapshots.

This closes output migration and the explicitly authorized five-row adoption.
It does not qualify GPU research, full raw-data preparation, model quality,
or the separate editor feature/UAT changes. The max60 proxy remains a consumed
input outside the selected length-budget retirement. No remote push occurred.
