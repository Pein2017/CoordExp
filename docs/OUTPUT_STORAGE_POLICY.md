# Research source, evidence and recovery

Research runs belong under
`/data/CoordExp/.worktrees/research-probes/outputs/`. Do not derive new destinations
from historical input paths. Shared cross-worktree assets follow the root
`/data/CoordExp/docs/OUTPUT_STORAGE_POLICY.md`; sharing retains actual producer
identity and verified bytes, not the consumer's HEAD or a symlink.

Maintained mechanics belong in `src/`, research operators in `probes/`, and thin
commands in `scripts/`. Protocols, interpretation and current scientific state
belong to their question/unit under `research/`. Datasets, checkpoints, raw
outputs, logs and galleries remain at their declared data/run owner. Source
cleanup never follows a data symlink to delete its target. Ignored scratch must
not become a hidden runtime or evidence dependency.

A checksum is not a recovery path. Currently consumed inputs require a verified
immutable copy or a dependency-complete maintained regeneration route. Distinguish
identity checks, sampled reads, full regeneration and independent backup. Use
[public-data provenance](../manifests/public_data_provenance/README.md) for the
actual recovery contract; historical tokenizer/factory commands and processed
path names are not a regeneration promise. Preserve label, geometry, ordering,
serialization and original/edited-version identities. No automatic data upload,
restoration, annotation change or destructive migration is authorized here.

Closed scientific records use the existing question/story/catalog and exact Git
recovery described by [research conventions](../research/CONVENTIONS.md). Do not
recreate `docs/history/`, `progress/` or an in-tree archive of old implementations.
The separate [docs policy](RETENTION.md) governs long-lived engineering assets.
Historical inspection never qualifies model continuation. Missing or changed
execution source fails closed; requalification creates a new identity rather
than resealing old evidence. Clean source alone does not prove numerical parity.

Before an authorized asset move, verify live holders, conflicts, destination
bytes and current consumers. Preserve the source until acceptance, never
rewrite sealed receipts or use an obsolete source tree as a fallback. Preserve
exclusive/idempotent publication, schema and package metadata semantics.
