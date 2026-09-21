# Source, evidence, and output storage

This document owns the storage boundary in this checkout. Research meaning and
knowledge layout remain owned by `research/CONVENTIONS.md`; checkpoint, decoder,
loss, and evaluation semantics remain with their existing owners.

## Maintained source and reusable operations

Keep maintained experiment code and tests in `probes/<direction>/`, reusable
cross-direction mechanics at the relevant `src/` owner, and thin operator entries
in `scripts/`. New code uses ordinary imports. Neither `outputs/` nor an archive
is a module search path or a source of executable Python/shell dependencies.

A first exploratory consumer may stay local. At the second real consumer, decide
whether the shared contract is a reusable operation or merely similar scientific
recipes. Share the former; keep cohorts, objective denominators, geometry,
interventions, masks, release/stop rules and evaluators explicit. Do not build a
generic trainer just to remove similar loops, or leave public helpers owned by
whichever experiment implemented them first.

Use short inline queries for disposable inspection. Repeated/nontrivial task-local
scripts belong in a named maintained package, or in `.local/scratch/<task>/` while
strictly disposable. Promote a script before citing its result as reproducible
evidence or reusing it in another unit. Ignored scratch files must never become
runtime dependencies of maintained code.

## Outputs contain artifacts, not a second codebase

`outputs/` contains data, model/checkpoint payloads, raw outputs, JSON/JSONL
receipts, resolved configuration, logs, metrics and rendered assets. Do not add
loose `.py`, `.md`, `.sh`, Python bytecode, virtual environments, vendor checkouts
or Git worktrees. Human-authored protocols, interpretation and current status
belong to their research/document owners, not inside a run directory.

Source captures are evidence, not implementation. Use
`src.artifacts.source_provenance.preserve_source` for exact source copies outside
outputs, below this checkout's `docs/history/run-sources/`. It returns a verified
path for the caller's existing receipt; it does not change scientific admission
or create a second receipt hierarchy. Include newly used shared dependencies.
Current runs bind current code. Existing sealed receipts remain immutable.

Generated PEFT cards are stored losslessly in `adapter/model_card.json` before
atomic checkpoint publication. This is generated checkpoint metadata, not a place
for research notes. Tensor payloads and adapter configuration are unchanged.
Exporting a model package elsewhere can reconstruct the original README bytes
from `content_utf8`, checking its SHA-256 first.

Visualization descriptions are generated metadata in `manifest.json` under
`summary`; `VisualizationResult.summary` returns that text. Renderers do not
create a separate Markdown report in the output directory.

## Historical recovery is not current execution validation

The September 21 migration map is
`/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json`.
`python -B -m src.artifacts.source_archive --manifest <map> --source <original>
--sha256 <expected>` verifies and locates original bytes without restoring or
executing them. `--verify` checks the complete archive.

Never replace an old receipt's hashes with today's hashes, resolve a current
source failure by silently substituting an old snapshot, or infer permission to
rerun from a preserved command. A reader explicitly consuming a checksum-pinned
historical packet may verify archived sources; a new run must pass its current
source checks and receive its own authorization. Recoverability and runnable
original-context replay are different claims.

Before moving/deleting a source or artifact, verify fresh Git, live consumers and
holders, source/destination hashes, a recoverable backup, and a source-to-target
mapping. Preserve original scientific records and unrelated dirty work. Retired
virtual environments are backups, not runnable relocatable environments; rebuild
them in an environment-owned location when reuse is separately authorized.

## Closeout check

Run the read-only checker with explicit physical roots:

```sh
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs
```

Add `--details` to list findings. Missing/unreadable roots fail, and directory
symlinks are reported but not traversed. This checks placement, not transitive
import correctness, scientific validity or model behavior. Consumer changes also
need focused tests and saved-output parity where their semantics are frozen.

## Human13 adapter metadata boundary

Human13 finite materialization now writes a version 2 receipt: `adapter_files`
contains only live adapter configuration and tensor payloads; optional generated
model-card JSON is separately recorded in `metadata_files`. A source adapter may
omit a model card. A source Markdown card, when present, is packaged losslessly
before publishing the new adapter directory.

Reading a preserved version 1 materialization still verifies the actual live
configuration/tensors and explicitly resolves its old README hash as historical
metadata through the source archive. It does not claim that a README remains in
the adapter directory, modify the old receipt, or permit an archive fallback for
current version 2 metadata. This metadata change does not alter learned tensors,
optimizer behavior, stage populations or natural evaluation gates.
