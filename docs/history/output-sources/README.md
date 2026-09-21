# Historical output sources

These files preserve original source, script and document bytes from the
September 21 output migration. The nested paths identify their original
locations; they are evidence, never an import path or runnable dependency.

The authoritative mapping and migration report live in the root checkout:

- `/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json`
- `/data/CoordExp/docs/history/output-sources/2026-09-21/MIGRATION.md`

The mapping spans 1,998 files here, 112 files in the root archive and 8,767
pre-maintenance checkout files below
`/data/CoordExp/.local/maintenance/2026-09-21-source-artifact-boundary/before/`.
The last directory is an archive dependency, not disposable temporary storage.
Copying this worktree alone does not preserve the complete recovery set.

Use `python -B -m src.artifacts.source_archive --manifest <manifest> --verify`
to verify the whole mapping, or pass `--source <original> --sha256 <expected>`
to locate one file. Never rewrite sealed hashes, execute these copies, or
bulk-restore them over maintained code. Current placement rules are in
[Output storage policy](../../OUTPUT_STORAGE_POLICY.md).

See [independent acceptance](2026-09-21/ACCEPTANCE.md) for the bounded repairs,
verification and intentionally uncommitted research changes.
