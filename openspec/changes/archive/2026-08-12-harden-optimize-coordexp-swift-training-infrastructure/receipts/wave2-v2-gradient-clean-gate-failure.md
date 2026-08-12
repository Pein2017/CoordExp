# Wave 2 v2 gradient clean-gate failure

Status: terminally failed single replacement execution; replacement authority
consumed; Wave 2 is not accepted.

- Original receipt:
  `/tmp/coordexp-wave2-v2-authorized.vXMV2a/gpu-receipt.json`
- Durable byte-identical receipt:
  `wave2-v2-gradient-clean-gate-failure.json`
- Receipt SHA-256:
  `fbd0556f597aab3facae4af1ef6bc7ebb24c2f447cf172df1c24f4d127ac2820`
- Receipt schema: `coordexp-swift-wave2-packed-parity-receipt-v2`
- Plan internal SHA-256:
  `e5c2b1eb0c7ff99b7a66de8dc172f5af0331959d9b5f12a07e61096dd79b4197`
- Plan file SHA-256:
  `aadcfe046938315d90df05633b1da00d5a86e368eee3304b9d822183c1cdb681`
- Requested device: `cuda:0`
- Terminal status: `failed`
- Failure code: `qwen.parity.clean_failed`

The bounded terminal error records the clean-gate decisions computed before
failure:

- semantic atoms: passed;
- exact denominator projection: passed;
- supervised logits: passed;
- total loss: passed;
- all four mandatory BF16-derived per-term loss scalars: passed;
- complete trainable-gradient comparison: failed.

The harness's generic failure path then replaced the in-memory partial
observation with a minimal failure receipt. Consequently `arms`, `comparisons`,
`proof`, `negative_discriminator`, `timings`, `gpu_memory`, and `measurement`
are empty in the durable artifact. Exact failing parameter identities,
gradient deltas, all-layer proof details, negative-control result, and resource
high-water values are therefore unavailable and MUST NOT be inferred.

This artifact is both an unfavorable clean-gate outcome at the recorded
boolean boundary and an incomplete failure artifact for diagnosis. It does not
establish that packing is semantically wrong, that FA2 proof failed, or that a
particular parameter/tolerance class caused the gradient failure. It cannot
close Wave 2 tasks 3.4–3.7 or support promotion.

The single replacement command genuinely entered Conda, Python, model load,
CUDA execution, all three arms, comparison, and the `qwen.parity.clean_failed`
terminal path. The replacement authorization is therefore consumed. No retry
or tolerance amendment is authorized from this artifact; any further real-
model execution requires a new user-owned stop-rule decision after the failure
receipt contract is repaired and independently audited.
