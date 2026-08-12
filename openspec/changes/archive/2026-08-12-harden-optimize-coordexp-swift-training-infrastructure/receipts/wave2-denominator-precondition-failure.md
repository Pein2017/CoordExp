# Wave 2 denominator-precondition failure receipt

Status: immutable failed technical observation; not an accepted numerical
comparison and not a Wave 2 semantic result.

- Original artifact:
  `/tmp/coordexp-wave2-post-identity.b0DWrs/gpu-receipt.json`
- Receipt schema: `coordexp-swift-wave2-packed-parity-receipt-v1`
- Receipt SHA-256:
  `ff8eb69913a7722b82009626937e5b4413e5181c38274e316757a62ab2196ed5`
- Authenticated plan SHA-256:
  `875750a17489fb674c65073bb10654e2434680e9423ca0a6aab2fddd512fb4af`
- Terminal status: `failed`
- Failure code: `qwen.parity.denominator_mismatch`
- Executed numerical arms: none (`arms={}`)
- Produced comparisons: none (`comparisons={}`)
- Produced FA2 proof: none (`proof={}`)
- Produced negative-control result: none (`negative_discriminator={}`)
- Produced measurement result: none (`measurement={}`)

The failure was caused by pre-v2 whole-record denominator equality treating the
declared structural `context_count` values (`1` packed, `2` shared-separate) as
a semantic mismatch. All equality-bearing denominator fields visible in the
bounded failure receipt matched. The v2 contract replaces that comparator and
rejects this v1 plan/receipt schema for execution or promotion.

Before any Wave 2 replacement launch, rehash the original JSON at the path
above and require the exact SHA-256 recorded here. If the original artifact is
unavailable or differs, the replacement launch remains on hold until a
byte-identical durable copy or stronger provenance evidence is supplied.
