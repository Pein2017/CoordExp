# Infrastructure invariants

Configuration, data preparation, training, checkpoint publication, inference,
evaluation and visualization are separate contracts. Validate at the owner of an
invariant and preserve useful failure evidence rather than quietly repairing an
invalid input. Current fields, commands, defaults and supported topologies are
in the local typed source, configs and [stable specs](../../openspec/README.md).

## State and identity

A checkpoint's inference payload is not its exact-resume training state.
Training continuation needs complete runtime state and matching identities at an
admitted optimizer boundary. Loading a model for inference cannot prove exact
replay or topology migration. Preserve original/derived model identity and any
required package authentication during composition and publication.

Packing caches are derived data whose semantic identity must cover the work they
represent. Stale, corrupt or mismatched caches must not silently change training.
Parallel preparation must preserve authored order; a bound on submitted tasks
is not proof of streaming memory usage. Data recovery is independent of cache
availability and is owned by [public-data provenance](../../manifests/public_data_provenance/README.md).

## Qualification and claims

A backend import, operational smoke, forced-replay alignment and cross-backend
numerical fidelity answer different questions. Qualification and source identity
must match the actual implementation; never reseal an old receipt to admit new
code. The applicable numeric tolerance belongs to the current owning contract,
not a duplicated number in this guide. Composed-model probabilities are not
implicitly the dynamic model's probabilities.

Bound execution and verify cleanup of owned processes/resources without broad
process-name kills. A topology exposed by config is not a measured execution
witness. Small smokes are implementation evidence, not full-training or
model-quality conclusions. Evaluation/visualization are downstream consumers;
they neither change the checkpoint nor prove training resume. Preserve raw/scored
artifact binding and explicit geometry conversion.
