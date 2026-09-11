## REMOVED Requirements

### Requirement: Checkpoint Handoff Manifest Is Canonical For Handoff Identity

**Reason**: Explicit inference configuration already names the base model,
adapter payload, and optional selected-token embedding payload required for
composition. A second `checkpoint_handoff.json` identity graph duplicates
loader-visible information and is not required to load adapter checkpoints.

**Migration**: New training checkpoints save only standard adapter and optional
embedding-delta payloads. Inference configs point to those payloads directly;
extra handoff manifests beside older checkpoints are ignored.

### Requirement: Inference Provenance Distinguishes Handoff From Manual Composition

**Reason**: Canonical-versus-manual composition status depends on discovery of
the removed neighboring handoff manifest rather than on the actual configured
payloads. The inference engine will validate and record the base, adapter, and
optional embedding-delta identities it actually loads.

**Migration**: Remove automatic neighboring-handoff discovery and composition
mode changes. Preserve ordinary inference provenance for explicitly configured
model payloads.

### Requirement: Readiness Validator Is Read-Only

**Reason**: The handoff/eval/placeholder-production gate framework validates a
metadata dossier that is no longer produced and adds a second readiness system
outside the actual inference loader and evaluation workflow.

**Migration**: Validate adapter and optional embedding-delta payloads through
their inference loaders. Establish evaluation acceptance through the existing
inference/evaluation artifact contracts and named smoke/eval evidence, not a
checkpoint readiness gate.
