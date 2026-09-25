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

Current runs may use `src.artifacts.source_provenance.preserve_source` to keep
local source captures beside their receipts. These captures are not maintained
implementation and are not tracked as a historical source library. Include newly
used shared dependencies when the run requires them. Existing sealed receipts
remain immutable.

Generated PEFT cards are stored losslessly in `adapter/model_card.json` before
atomic checkpoint publication. This is generated checkpoint metadata, not a place
for research notes. Tensor payloads and adapter configuration are unchanged.
Exporting a model package elsewhere can reconstruct the original README bytes
from `content_utf8`, checking its SHA-256 first.

Visualization descriptions are generated metadata in `manifest.json` under
`summary`; `VisualizationResult.summary` returns that text. Renderers do not
create a separate Markdown report in the output directory.

## Historical source snapshots

Migration-time source-code recovery is retired. Conclusions, useful process
details and necessary hyperparameters remain with their research owners; old
implementation snapshots and their hashes are not required for interpreting
those records. Keep sealed data, checkpoint and result receipts unchanged, and
do not infer permission to rerun from a preserved command.

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

The maintained materialization reader accepts version 2 only. Preserved version
1 receipts remain historical evidence; this checkout does not supply the retired
source-archive README fallback. Do not rewrite those receipts or restore the old
archive merely to make them pass the current reader. This support boundary does
not change learned tensors, optimizer behavior, populations or evaluation gates.

## Documentation and retained source locations

No implementation files, source snapshots, scripts, notebooks or code caches belong anywhere under `docs/`, including `docs/history/`. Inline explanatory examples in documentation are not executable source ownership. Useful research documents, including old or completed results, belong under `research/` according to its convention. Global documentation stays in `docs/`. History is a temporary salvage queue, not a date-based warehouse.

The migration-time `reference/retained-sources/objects/` corpus is no longer
tracked. `manifests/documentation-layout.json` keeps path routes for migrated
documents, not source-byte identity. Current-run captures may remain local under
`reference/retained-sources/runs/`; they are separate from maintained code and
from the research conclusions, process notes and hyperparameters that remain in
their owning records.
