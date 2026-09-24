# Documentation ownership and retained-source cleanup

Status: implemented and CPU/file-identity checked on 2026-09-22.
Scope: canonical `research-probes` at `/data/CoordExp/.worktrees/research-probes`.
No root-checkout or probe implementation mutation, model execution, Git commit,
publication, or service stop is part of this maintenance.

## Current placement

| Content | Owner |
|---|---|
| Global interfaces, engineering behavior and operating guidance | `docs/` |
| Useful research protocols, results, negative/invalid evidence and supporting observations, irrespective of age | `research/experiments/<unit-id>/` and the appropriate question |
| Reusable research checkpoints, datasets and fixed panels | `research/assets.md` |
| Maintained implementation | Existing `src/`, `probes/` and thin `scripts/` owners |
| Conclusions, useful process and necessary hyperparameters | Owning `research/experiments/<unit>/` record |
| Source capture for a current run | Local receipt inputs under `reference/retained-sources/runs/`; not a Git archive |
| Unresolved legacy reference value | Temporary `docs/history/`, with extraction or deletion conditions |

`run-sources` was a storage location for frozen provenance, not a functional
module. Its placement under documentation was incorrect. There are no source,
script, notebook, bytecode or executable configuration files under `docs/`
after this cleanup. `docs/catalog.yaml` remains documentation metadata.

## Implemented changes

- Rehomed 214 catalog records from documentation archives to their scientific
  units. Existing evidence/lifecycle labels were not promoted. Three retained
  design records were catalogued as designs, not executed experiments.
- Moved the authored research asset index to `research/assets.md`, preserving
  its substantive contents, and updated routing references.
- Reassigned 223 original June records using their existing `absorbed_into`
  mapping to eight existing synthesis units, with unit-owned `sources.md` and
  `supporting/` notes. The exact lineage table is now
  `manifests/duplication-source-lineage.tsv`.
- Integrated the old specialization-preservation and sampling-support decision
  boundaries into the existing research question owners. Removed their obsolete
  standalone decision authority and the already consumed intake/atlas routers.
- Removed byte-identical document copies and superseded global snapshots, while
  preserving original-byte recovery. Research plans and named retired-worktree
  records were grouped with their matching scientific units.
- Moved frozen Python/shell sources and YAML configuration copies out of docs.
  The migration-time object corpus was later retired from Git; its implementation
  snapshots are no longer required for interpreting the research records.
- Preserved the conflicting existing curriculum unit; the divergent old unit
  remains separately named rather than overwriting current work.
- Updated `research/CONVENTIONS.md`, storage policy and the history entry:
  completed work is not automatically history. Unique legacy content is not
  automatically valuable. Remaining salvage has explicit extraction/drop rules.

The original 5,289 documentation files were classified during migration. The
current `manifests/documentation-layout.json` keeps only path routes to current
owners; it no longer binds original bytes or supports source-code recovery.

## Verification

- Targeted source-location, source-capture, research layout and real exposure
  consumer tests: **55 passed**, no failures or skips.
- Original documentation/source recovery: **4,613 moved/removed originals**
  checked against original SHA-256 and size, through retained objects or pinned
  Git blobs.
- Original September 21 source manifest: **10,877 entries verified** by the
  research-checkout reader with its explicit local location overlay.
- Imported research Markdown: **803 documents** preserve body and numerical
  text exactly, excluding only Markdown URL location edits.
- The frozen exposure consumer still reads the same **106 JSON sources** and
  **135 image IDs**; its real-consumer and failure-path tests pass.
- Knowledge checker passes with **810 original source versions**. Its **36
  historical unresolved links** are pre-existing, not silently converted to
  successful live links. Root/branch HEAD is unchanged; `git diff --check` passes.

These counts describe the 2026-09-22 migration checks, not current source
retention. Source-code recovery was retired in the 2026-09-24 cleanup; current
research interpretation lives in its unit records.

## Explicit residual salvage

At cleanup closeout, `docs/history/` still contains 500 Markdown files with
unresolved branch-specific design/context or global contract reference value,
plus immutable provenance metadata. Their remaining uses and exit conditions
are documented in `docs/history/README.md`. This is not a claim that all 500
have been deeply scientifically synthesized. Routine results, yesterday's notes,
and code snapshots must not be added to this queue.

The 106 frozen exposure JSON files remain at their original docs/history paths
because `probes/parallel_owner_research/transfer.py` still reads them directly.
Changing that consumer was excluded by the user's concurrent-development scope.
Their eventual move requires owner-coordinated hash/image-ID parity. This is
explicit compatibility debt, not a permanent documentation-data policy.

## Preservation and concurrency

The local backup is `.local/maintenance/docs-ownership-2026-09-22/before.tar.gz`:
5,491 original files were re-read and hash-verified after packing. This is a
same-machine recovery backup, not off-machine disaster recovery.

Probe files present at baseline were neither modified nor deleted by this
maintenance. New concurrent probe files, research state/catalog changes and an
independent change to `src/artifacts/checkpoints.py` were preserved, not claimed
as this maintenance's work or included in any commit.

Detailed commands, baseline, JUnit, content checks and closeout receipt live in
`.local/maintenance/docs-ownership-2026-09-22/`. No blanket restore should be used
on this shared checkout; resolve exact source paths and hashes instead.
