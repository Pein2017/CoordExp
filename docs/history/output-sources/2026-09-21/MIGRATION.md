# Source/artifact boundary migration, 2026-09-21

## Outcome

The source/artifact boundary migration is complete in the canonical root and research-probes checkouts. No commit, push, merge, GPU experiment, training run or deployment was performed. Existing scientific records, model weights, data and raw outputs were not rewritten by this maintenance.

The root checkout remains main at `ce1188a40198c7acfc61ab655b62fccd7bb3422e`; research-probes remains at `cedea9c8908f9a6ccebce3466295628d5acf7abf`. Both retain uncommitted changes. The infras and research-probes-web-codex Projects were not modified.

## Storage changes

| Item | Verified action |
| --- | --- |
| Root outputs before | 133,657 ordinary files, including 22,371 Python, 1,163 Markdown and 209 shell files |
| Worktree-local outputs before | 29 files, including two model-card Markdown files |
| Source/prose/script relocation | 2,109 files: 1,381 Python, 542 Markdown and 186 shell; exact-byte copies verified before removing originals |
| Retired runtime/vendor roots | Two virtual environments and one existing Label Studio Git worktree moved outside outputs; 49,022 ordinary file contents verified, with one Git pointer updated by git worktree move |
| Bytecode | 206 remaining files separately backed up and removed; empty __pycache__ directories removed |
| Broken links | Two already-broken historical links recorded and removed; no substitute checkpoints installed |
| Other note | outputs-lane-spatial-proposal.md retired from the research checkout root, preserving its bytes and lookup mapping |
| Layout check | Both physical output roots contain zero .py/.md/.sh/.pyc/.pyo, no virtual environments and no broken links at validation |

The Label Studio worktree retains its detached HEAD `4fd750eafae199711de99ef9c0f04c66fefa8fb7` and all pre-existing dirty files. Retired virtual environments are recovery material, not relocated runnable installations. Unrelated legacy services were not stopped.

## Maintained owners

- `probes/training_set_completion/row_scoring.py`: readable frozen saved-row accounting. Eleven characterization cases cover eight real saved outputs plus empty/EOS/malformed edges; all returned fields are preserved. The golden JSON is losslessly compressed to 265,867 bytes from 3,681,405 bytes.
- `src/qwen/input_identity.py`: literal tensor/native-input identity. Twenty-four consumers no longer import generic helpers from the readout-norm experiment. `probes/training_set_completion/artifacts.py` distinguishes authored-path binding from resolved-path binding and preserves their different serialization contracts.
- `probes/dora_owner_learning/composition.py`: shared effective composition verification across three trainers, retaining frozen receipt semantics. Scientific training loops, denominators and stop rules were not generalized.
- `src/artifacts/source_provenance.py`: verified source capture outside outputs. Eleven DoRA preparation writers and the two current spatial/visual runtime capture paths were migrated.
- `src/artifacts/source_archive.py`: explicit expected-hash historical recovery. It reads and verifies; it never executes archived sources or makes a new run satisfy an old source gate.
- Previously executed output scripts now have maintained owners: `src/qwen/untied_embeddings.py`, `probes/training_set_completion/row_branch.py`, the positive-branch endpoint and escape-witness modules, and the retained CoDETR profile capsule. The untied payload is a distinct byte-preserved implementation, not an unvalidated replacement of the general embedding owner.
- Historical endpoint and annotation readers explicitly verify retained producers. Current execution validation remains strict. Original sealed receipts and expected historical hashes were not rewritten.

The old `scripts/research/` closure remains: it contains optional admission/evidence consumers and knowledge tooling. Large files or old experiment names alone were not treated as proof of dead code. This migration is not a claim that every historical recipe has been unified or is runnable with today's producer hashes.

## New artifact behavior

New checkpoint publication stores generated PEFT card bytes in `adapter/model_card.json`, with original name, SHA-256 and UTF-8 content. Adapter configuration and tensor payloads are unchanged; the DoRA payload inspector addresses those files rather than model-card metadata.

Visualization descriptions now live in `manifest.json` as `summary`; `VisualizationResult.summary` replaces `readme_path`. Renderers no longer emit README.md. Known maintained consumers and tests were updated; no obsolete forwarding property was added.

`docs/OUTPUT_STORAGE_POLICY.md` owns the placement boundary. The research checkout has a short AGENTS.md router, repairing the existing CLAUDE.md/GEMINI.md links. Research conventions, infrastructure documentation and the shared research-flow Skill route to the policy. The user's pre-existing dirty `.codex/AGENTS.md` was preserved byte-for-byte.

## Validation and limits

The final research command covered all `probes`, `tests/artifacts`, `tests/adapters`, the visualization tests and research-knowledge contracts: **921 passed, 5 failed, 15 skipped**. The five failures are exactly the names and messages recorded before refactoring, all `sources.producer source bytes changed` in older Source256 integration paths. No new failures remain. They were not skipped or repaired by replacing sealed hashes. Source hashes for all 668 Python files under src/probes/tests were unchanged throughout the final test run.

The root artifact/visualization tests passed **51/51**. The source archive verified **10,877** entries. Both output-layout checks passed. The local knowledge checker passed while continuing to disclose its pre-existing historical gaps and unverified external handles. Syntax parsing covered 774 first-party Python files. These are mechanical, CPU and saved-output checks, not full-model/GPU parity or new scientific evidence.

All 82,348 remaining original artifact files were checked for existence, size and nanosecond mtime and were unchanged at the preservation check. This is deliberately not described as a pre/post content hash of all model and data payloads.

An existing process-identity test had a pre-exec /proc race exposed by the broader suite. Only the test was changed to wait for its owned child's command line with a deadline; production process handling was not changed.

## Backups and recovery

These are same-machine recovery copies, not offsite disaster-recovery backups.

- Backup root: `/data/CoordExp/.local/maintenance/2026-09-21-source-artifact-boundary/`
- Original checkout copies: `before/coordexp` and `before/research-probes`, including pre-existing dirty/untracked source and Git binary patches.
- Compressed source/runtime backup: `output-sources-and-runtimes-before.tar.gz`, 1,075,044,488 bytes, SHA-256 `e8b30b221f227271db2770141ec83f0c99a273059945c3a9e2a31fb7886cd7ab`; 51,132 regular members verified individually.
- Exact mapping: `/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json`
- Retired environments/vendor worktree: `/data/CoordExp/.local/retired-output-runtimes/2026-09-21/`
- Detailed evidence: `verification-receipt-v2.json`, `source-change-review-v2.json`, `retained-artifact-validation.json`, runtime-move and removal receipts under the backup root.
- Research test logs, JUnit reports, source-stability snapshots and refactor journals: `/data/CoordExp/.worktrees/research-probes/.local/source-artifact-boundary/`.

Resolve one historical file with `python -B -m src.artifacts.source_archive --manifest <manifest> --source <original-path> --sha256 <expected>`. Read the returned file as evidence. Do not bulk-extract a backup over a checkout or restore old output executable paths to bypass source gates. Any restoration first checks new dirty work and demonstrated consumers.

## Concurrent work preserved

During maintenance, `research/index.md` and `research/experiments/catalog.jsonl` changed independently of the maintenance edits, adding `2026-09-21-spatial-progress-recovery`. Its state record says running; that is a recorded status, not an independently verified GPU process claim. This unit was not created, launched, stopped, archived or overwritten by this maintenance. Its routing changes are preserved and separately identified in the source-diff review. Do not stage or revert them as though they were maintenance-owned changes.

See `CODEX_HANDOFF.md` for the operating handoff.

## Human13 compatibility closure

A final dependency check found that the finite Human13 adapter materializer still required README.md. Its maintained source and tests were updated to the version 2 metadata boundary described in the storage policy. Missing, Markdown and JSON source-card cases pass; version 1 metadata is verified through an explicit expected-hash archive lookup while live payload hashes remain mandatory. Changed current version 2 metadata is rejected without archival fallback. The focused Human13/model-card checks passed 33 tests. Existing materializations and their receipts were not rewritten.
