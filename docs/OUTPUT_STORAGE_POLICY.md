# Storage, recovery and retention

This is the single operator policy for repository content, external assets and
output transfer. It does not authorize dataset edits, uploads, deletion, process
stops or Git publication. [Worktree rules](BRANCH_AND_WORKTREE_POLICY.md) govern
source integration; [research conventions](../research/CONVENTIONS.md) govern
scientific interpretation and evidence.

## Resolve this branch's output owner

Research Probes writes branch-owned runs, JSON transport receipts and rendered
assets to `/data/CoordExp/.worktrees/research-probes/outputs/`. Do not derive a new
output root from an old input path or a historical run constant. Those constants
remain frozen consumer bindings, not defaults for a new direction.
Human reports/protocols belong to their existing `research/` question/unit;
maintained scripts/configs belong to their source owner. Disposable dispatch
messages belong to task-owned `.local/scratch/`, not the run directory.

The shared-root rule is owned by `/data/CoordExp/docs/OUTPUT_STORAGE_POLICY.md`:
root `outputs/` contains only selected durable cross-worktree assets, never an
entire branch run merely because another worktree reads it. Sharing requires a
real verified copy and original producer/run/config/data provenance, not a
symlink or the destination checkout's HEAD. Main-owned assets use main's linked
worktree; they do not become Research Probes assets.

This explicit maintenance task can migrate verified inactive assets; ordinary
research cleanup still does not authorize external moves. Existing root runs
with live writers, frozen paths, unknown recovery or generated package metadata
remain named HOLDs in `/data/CoordExp/openspec/changes/route-owned-outputs-and-migrate-legacy/`.
Do not rewrite old receipts or recreate retired archive/code-snapshot trees.

## Keep purpose and ownership separate

- Current reusable execution belongs to its existing `src/` owner; retained
  research operators belong to `probes/`; supported entries are documented at
  their package. Tests belong to the contract they protect.
- Current explanation belongs to `docs/`; stable behavioral requirements belong
  to `openspec/specs/`; bounded ongoing change work belongs to its named change.
- Accepted research knowledge belongs to question/story/catalog. Closed operating
  records and superseded implementation are recovered from their exact Git
  commit, not copied into a new history/archive tree. Do not recreate retired
  `docs/history/`, `progress/`, or `openspec/changes/archive/` directories.
- Datasets, checkpoints, raw outputs, logs and galleries remain at their declared
  external/run owner. A repository cleanup never follows data symlinks to delete
  them. Credentials, model caches, sessions and generated plugin state are not
  source merely because they are near source files.
- Task-owned `.local/` is ignored validation scratch, not a permanent source or
  evidence archive. Existing ignored files are not automatically disposable.

## A checksum is not a recovery path

A currently consumed dataset must have a reachable verified immutable copy or a
maintained, dependency-complete regeneration path. A local directory, a manifest
file or a historic command by itself does not satisfy this requirement. Distinguish
identity verification, sampled reading, complete regeneration and independent
backup. Never infer a remote backup from a path naming convention.

[Public-data contracts](../manifests/public_data_provenance/README.md) distinguish
current recovery from historical identity. Current COCO recovery uses checksum-bound
official raw ZIPs plus the minimum necessary observed annotation delta; it does
not use an old tokenizer/config/factory or assume a processed-data mirror.
Historic proxy/view identities have no current regeneration promise.

```bash
python -m public_data.recover_coco check
python -m public_data.recover_coco regenerate \
  --manifest manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json \
  --raw-archives /path/to/verified/coco-zip-inputs \
  --destination /path/to/absent-restored-workspace --dry-run
```

Remove `--dry-run` only for an explicitly requested full data restoration with
sufficient disk/time. Existing destinations are never overwritten. Failed
restorations retain an incomplete marker and are not accepted materializations.
A new observed annotation version must not silently replace a historical checksum
or rewrite old run receipts. Preserve a precise original Git locator and explain
the version boundary. Image serialization, geometry, object order and annotation
IDs are part of the recovery contract, not cosmetic implementation details.

## External outputs and cross-machine transfer

Model caches and raw public datasets normally come from their verified upstream
source or an explicitly verified mirror. Processed datasets are not mirrored by
default, so maintained regeneration is mandatory for currently supported inputs.
If a new independent snapshot becomes authoritative, qualify its actual location,
content and retrieval before changing that policy. No upload is automatic.

Experiment-specific outputs may use the established `/CoordExp/outputs/...` remote
namespace, but that is a convention, not evidence that a particular run is backed
up. Verify the exact source and destination with the available transfer capability
and keep credentials out of commands, logs and Git. Do not assume an old tool
installation, proxy port or repository skill path still exists.

Do not rename an active writer's output tree. Any explicitly authorized transfer
must preserve the source, check conflicts, avoid overwriting existing destination
bytes, and verify what was transferred. Staging a live directory is not an atomic
snapshot. This repository no longer supplies the old output-absorption utility;
there is no fallback command or standing instruction to perform that migration.

## Execution identity and publication

Decision-bearing runs bind clean Git commit/tree/source paths, independently of
model/data/config identity. Legacy receipts lacking a verifiable source binding
are historical and unsupported for continuation. Requalification creates a new
identity; it does not repair an old hash or justify executing archived code.

Preserve each owner's exclusive/idempotent publication rule, JSON byte convention,
path semantics and schema. Never discard meaningful checks to make old inputs
pass. Before retiring code/config/docs, verify both its running consumers and the
recovery closure of any retained asset that depended on it.
