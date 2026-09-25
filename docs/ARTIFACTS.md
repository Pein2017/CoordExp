# Artifact identities and continuation

Ordinary result publication uses `src.artifacts` with explicit collision behavior.
Data, model tensors, runtime and source identities are different facts; a source
commit does not identify a mutable dataset or prove numerical repeatability.

`src.artifacts.git_identity` records a clean full commit, tree and explicit regular
source path/blob/content identities. It rejects dirty tracked/untracked state,
missing files, symlinks, legacy envelopes and mismatched current commits/trees.
Historical Git retrieval is read-only explanation, not a weaker execution gate.

Training checkpoint state schema2 additionally publishes
`training_state.source.json`, binding the clean source envelope and exact `.pt`
bytes. Resume validates that JSON and payload hash before deserialization and
before accelerator/model/output initialization. The configuration, schedule,
data and optimizer compatibility checks still apply. A legacy checkpoint without
this gate is historical/unsupported for continuation. A new qualification is not
a silent migration of its receipt.

Output-QP and norm CLI results bind explicit numerical inputs and clean source.
Their certificate is bounded to those arrays; no natural trajectory or physical
owner claim is implied. Pure operator tests remain separate from qualification.

Inference payload import validates model metadata/tensors but is not training
continuation. Only trusted local checkpoint payloads belong on the deserialization
path; source hashes are integrity checks, not cryptographic producer signatures.
External saved receipts and captures are not modified by repository cleanup.
