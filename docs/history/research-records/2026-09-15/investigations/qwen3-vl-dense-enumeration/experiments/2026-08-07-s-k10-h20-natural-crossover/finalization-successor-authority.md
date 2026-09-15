# Post-execution finalization successor authority

Status: `AUTHORIZED ONCE — CPU-only, two-slot finalization repair`

This authority exists only because all three preGPU-v5 GPU shards completed
successfully and the first formal CPU finalization exposed two deterministic,
operator-neutral consumer seams. It does not reopen, rewrite, or supersede the
GPU execution authority. The parent remains the immutable preGPU-v5 receipt:

- raw SHA-256:
  `7e4adff6e272dfaad8ad5bb5656c9fc0baf7f2992d3561e37edfb613991e57ec`;
- semantic self SHA-256:
  `c349258554a2e308c5a4242157eb59cbb6793cdc7476c2a3a65719b845ae4571`.

The immutable execution set is exactly:

1. `shard-000`, `gt:2299:29`, physical GPU 0, result self SHA-256
   `945bfdc40f35656a6a06af2d9c9e9d93a265033d05b2b7356330b7df6375d69f`;
2. `shard-001`, `gt:13348:14`, physical GPU 1, result self SHA-256
   `59866d4561541fbf0a3a0b6306ffac0cd280a61467de3256df0c17443f046a60`;
3. `shard-002`, `gt:16228:15`, physical GPU 7, result self SHA-256
   `480edb0e3f1d59f0ae6968fbd9e8d642a5e054497bcfb19923b3f741df659ec7`.

## Exact repair allowance

One write-once post-execution finalization receipt may authorize exactly two
fixed binding slots and no generic override map:

1. `source_files.crossover_finalizer` may advance from the parent-sealed
   finalizer only to:
   - validate declared file versus directory references with the same
     deterministic inventory identity used by the parent sealer; and
   - accept the runner-recorded `pre_gpu_receipt_sha256` only when it equals
     the recomputed raw SHA-256 of the immutable parent receipt.
2. `test_files.crossover_finalizer_test` may advance only to cover those two
   consumer repairs and the successor-receipt fail-closed contract.

The successor receipt must bind the parent receipt and plan by path, raw hash,
and semantic self hash; all four documents for each of the three shards by
path, raw hash, and semantic self hash; the old and final live hashes for both
authorized slots; every other parent source/test binding as unchanged; this
authority document; the complete execution root; and the absent fresh
`evidence-v4` root. It must be canonical, self-hashed, write-once, CPU-only,
model-free, and no-training. The finalizer must record both the parent and the
successor receipt in the resulting evidence.

No endpoint, parser, owner-match, denominator, event, cell, tau, utility,
operator, prompt, token, wrapper, checkpoint, or runtime trajectory semantics
may change. No GPU rerun, A3, P4, training, production behavior, promotion,
stage, commit, or push is authorized.

## Stop rule

The successor route gets one sealed attempt after the final code and tests are
independently accepted. Stop with `HOLD` and return the decision to the user if
any third drift appears, any shard artifact changes, the evidence root exists
before sealing, the final blocked-to-complete probe does not close exactly the
two named seams, or independent review finds a P0 in this successor boundary.
