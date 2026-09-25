# Current selected-embedding qualification contract

The approved mechanism is the custom Qwen wrapper pair as V1 recommendation,
with semantics: additive_delta. Only the explicitly selected wrapper/coordinate
rows are trainable; nonselected input/output rows remain unchanged. Tied and
untied payloads have distinct identity and composition checks.

The original qualification is recorded by
`outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json`.
The owner-commit wrapper variant requires its own corresponding receipt. Counts,
selected-token identity, payload metadata, finite gradients and reload behavior
are checked by `src.qwen.special_token_embeddings`; this page cannot substitute
for missing or failed probe evidence.

The historical detailed study is recoverable at Git
`108dede0154abfd90a54d18234d9e0bac780a3ba` in the original June 27 architecture
proposal. This is a current mechanism contract, not a new qualification run or
permission to continue a legacy experiment. External receipts remain untouched.
