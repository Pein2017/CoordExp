# Source, data and output ownership

This page owns the cross-checkout storage boundary. Scientific interpretation
belongs to canonical Research Probes; runtime semantics belong to the selected
checkout's source, tests and stable specs.

## Select an owner before writing

`/data/CoordExp/outputs/` is an ignored shared-asset store, not a run destination.
It holds deliberately selected durable checkpoint packages and inference
results/metrics with a concrete cross-worktree purpose. Another checkout reading
a run once does not promote the entire run to shared storage.

Infrastructure runs belong under
`/data/CoordExp/.worktrees/coordexp-infras/outputs/`; research runs use their
own retained physical worktree. Resolve the canonical checkout and explicit
output argument before launch. A relative `outputs/` in main is not branch-local.
No unknown legacy payload gets a default owner merely because space is available.

Maintained code/configs belong in their source package; human-authored protocols
and conclusions in their existing document/research owner. Task-owned ignored
scratch is disposable working space, never an untracked execution dependency.
Do not put loose source scripts, human reports, environments or vendor checkouts
inside outputs. Generated manifest-bound model cards are package metadata, not
research reports; preserve their bytes and package authentication on migration.

## Identity, recovery and promotion

Keep original and edited dataset versions separate at `public_data/` and record
provenance under `manifests/public_data_provenance/`. A hash is not a recovery
path: preserve a verified immutable copy or a maintained regeneration closure.
Do not infer remote backup from a naming convention. Annotation publication,
draft disposal and application-state retirement need their own explicit decision.
Do not derive a publication status from this evergreen policy.

A shared asset needs verified destination bytes and original producer checkout,
commit or explicit unknown source identity, run, config, data and original path.
The consumer's current HEAD is not the producer. Never overwrite an existing
destination or treat a symlink as a durable verified copy. Before retiring an
original, verify the destination, live holders, all current consumers and the
asset's recovery closure. Do not move an active writer's directory.

Preserve sealed receipts without updating their hashes or historical source
fields. Current consumers must use a qualified current location rather than an
old-path fallback. Model tensors, generated metadata, transport receipts and
human conclusions have different owners even when produced by one task.
Mutable databases, editor sessions, caches and credentials are application state,
not a permanent exception in the shared-asset store. Never publish credentials.

## Retention without accumulation

Keep a payload only for a named current consumer or a specific necessary
reproduction. Extract useful implementation and reasoning into their real owners;
do not transfer whole retired trees into a new ignored directory and call that
cleanup. Ordinary documentation retirement uses [Git recovery](RETENTION.md),
not a new source-snapshot or archive directory. The finite historical extras
sealed by that migration are evidence, not a new intake destination.

New execution binds maintained source and explicit inputs. Recovering a historical
record does not qualify continuation or authorize training, deletion or upload.
The existing placement checker is `python -B -m src.artifacts.output_layout`;
select its explicit physical roots through its current CLI. Placement validation
is not tensor integrity, transitive dependency validation or model qualification.
