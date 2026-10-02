# Source, evidence, and output storage

This document owns the storage boundary in this checkout. Research meaning and
knowledge layout remain owned by `/data/CoordExp/.worktrees/research-probes/research/CONVENTIONS.md`; checkpoint, decoder,
loss, and evaluation semantics remain with their existing owners.

## Choose the physical owner before writing

`/data/CoordExp/outputs/` is an ignored **shared-asset store**, not the default
run directory. A branch-owned run uses its owner's physical worktree `outputs/`.
Infrastructure runs use `/data/CoordExp/.worktrees/coordexp-infras/outputs/`;
research runs use their retained research worktree. The four retained checkouts
are listed in [worktree policy](BRANCH_AND_WORKTREE_POLICY.md). The temporary
main-runs checkout is retired. No worktree is a default legacy destination;
retain only content with a concrete use and an explicit current owner.
Research Probes uses `/data/CoordExp/.worktrees/research-probes/outputs/` and keeps
its scientific summaries in its own `research/` tree. Other branches use their
own checkout. Resolve `pwd -P`, Git identity and explicit output arguments; a
relative `outputs/` launched from repository root is **not** branch-local.

Promote only a deliberately selected, long-lived asset that has an explicit
cross-worktree retention purpose. Another worktree reading a run once does not
promote that run. Copy the selected checkpoint/JSONL/gallery to a fresh shared
location, verify every file, and retain producer checkout/commit (or explicitly
unknown commit plus captured source identity), run, config, data and original
path in its provenance. A consumer's current HEAD is not the producer. Do not
copy a whole branch run or retain only a symlink. Never overwrite a destination.
Update current consumers to the selected destination, verify its bytes, then
remove the original. Do not keep an old-path symlink or a second historical
payload tree. Frozen receipts keep their original provenance fields; those
fields do not make the old filesystem path a supported current input.

Human-authored protocols, reports and interpretation go to existing Git-managed
document/research owners; maintained scripts/configs go to their source owners.
A disposable worker message uses task-owned `.local/scratch/`, while a JSON
transport/run receipt can use that branch's output directory. Do not choose a
message path by appending `.md` to an artifact root. Generated checkpoint cards
are package metadata, not research notes; preserve compatibility before changing
them. Resolved run configs are evidence, not maintained source configurations.

COCO images, annotation sources and published annotation views belong under
`public_data/coco/`, with their maintained provenance under
`manifests/public_data_provenance/`. Preserve official raw inputs and the existing
edited-view semantics. Editor databases, pending drafts, journals, locks,
sessions and inference caches use the application's explicitly assigned runtime
owner; the active cross-worktree Gate A service may use a selected shared root.
They are not substitutes for the published dataset. For this maintenance round,
the user explicitly defers annotation relocation: Gate A stays at
`/data/CoordExp/outputs/coco_refinement/gate-a-20260717/`, and existing annotation
payloads may remain under root outputs until the next public-data migration.
The retired Label Studio service must not be restarted; its two exported drafts
remain pending annotations, not published ground truth.

Existing root trees are not retrospectively declared shared. The bounded
[2026-09-30 migration](../openspec/changes/route-owned-outputs-and-migrate-legacy/design.md)
records the finite source-to-owner migration. Legacy root run paths are retired
inputs and forbidden destinations for new training/inference runs. Explicitly
selected shared checkpoints may be read as inputs; runs never write into the
root shared-asset store. Close live writers before moving their state, update
current entrypoints and consumers, verify the destination, and remove old paths.
Temporary execution blockers must be resolved rather than retained as permanent
layout exceptions. Moving bytes does not authorize a research rerun.

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

Current execution binds maintained Git source and records its commit, exact
paths/digests and any permitted dirty-source identity in the existing run
receipt. Do not retain copied executable sources, vendor trees or binary
payloads in historical directories. Distill useful code into its maintained
owner before removing obsolete copies. Historical Markdown records may retain
original locators and interpretation. Existing sealed receipts remain immutable.

Generated, manifest-bound checkpoint README files are package metadata and may
remain with their owning checkpoint; they are not loose research reports.

Generated PEFT cards are stored losslessly in `adapter/model_card.json` before
atomic checkpoint publication. This is generated checkpoint metadata, not a place
for research notes. Tensor payloads and adapter configuration are unchanged.
Exporting a model package elsewhere can reconstruct the original README bytes
from `content_utf8`, checking its SHA-256 first.

Visualization descriptions are generated metadata in `manifest.json` under
`summary`; `VisualizationResult.summary` returns that text. Renderers do not
create a separate Markdown report in the output directory.

## Retire old payloads after recording their identity

Decide whether content is useful before choosing a destination. Retain a payload
only for an identified current consumer or a specific necessary reproduction
of a maintained result. A historical path reference, unknown producer, available
disk space or successful move is not evidence of usefulness. Integrate useful
implementation into its existing source owner and keep its proportionate checks;
distill useful conclusions into the existing research/document owner. Remove
obsolete payloads and redundant copies after verifying those owners. Do not move
whole legacy families into another ignored directory to call the migration done.

Keep small maintained migration summaries and human interpretation in Git.
Large per-file machine receipts belong to the maintenance owner's worktree
outputs. The migration's `inventory.json` locates and hashes the complete raw
receipt; that receipt records retired paths rather than supporting old-path
execution. Do not preserve obsolete source or binary trees in `docs/history/`.

Never replace a sealed receipt's hashes with today's hashes or silently replace
its source with a different snapshot. A new run binds current maintained source,
explicit current inputs and its normal research authorization. Historical Git
records and Markdown explain past work; they are not default runtime fallbacks.

Before removing an original, verify fresh Git and live holders, the exact
source-to-target mapping, and destination bytes (or same-filesystem inode
preservation for opaque data). Preserve scientific records, annotation drafts
and unrelated dirty work. Do not retain obsolete environments as historical
backups; use an environment-owned installation when execution is authorized.

## Closeout check

Run the read-only checker with explicit physical roots:

```sh
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/coordexp-infras/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs \
  --root /data/CoordExp/.worktrees/research-probes-web-codex/outputs
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

Historical version 1 receipts retain their original metadata identities. A
current consumer uses the migrated checkpoint's actual configuration, tensors
and packaged metadata; it must not fall back to an old executable archive or a
retired path. This storage migration does not change learned tensors, optimizer
behavior, stage populations or natural evaluation gates.
