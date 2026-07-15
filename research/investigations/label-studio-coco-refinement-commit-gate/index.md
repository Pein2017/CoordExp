# Label Studio COCO Refinement Commit Gate

## Status

**Hold lifted for the bounded redesign on 2026-07-15.** The original
synchronous one-row whole-file Commit contract was not feasible on the selected
full train split. The user subsequently approved accumulating durable Drafts
across several images and asynchronously publishing one same-split batch in the
background without blocking later annotation.

This note records a negative feasibility result, not current product behavior
or a replacement contract. The active authority remains
`openspec/changes/label-studio-coco-refinement/`.

## Evidence Boundary

- Checkout: `/data/CoordExp`, Git base
  `c06c188349cece5d922358baaa6ae507adce3203`.
- Source: `public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl`,
  117,266 rows and 849,947 boxes, SHA-256
  `d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a`.
- Working JSONL size after the full probe: 116,111,237 bytes.
- Raw receipts are retained under
  `outputs/label_studio_coco_refinement/probes-v2/`:
  `receipt.json`, `post_patch_receipt.json`, and `code_states.json`.
- The full benchmark used `store.py` SHA-256
  `b546a1f06b659a64c9955837f74f8e2fe92f50d749ee1d32befda685ff550ca7`.
  The later store patch changed retry-identity validation only; its bounded
  post-patch SHA-256 was
  `436d34456c4d5665eecdf687dba1dca8f394d2cea507f13632d7001148b81075`.

## Result

| Rows | Bootstrap | One-row Commit | Gate result |
| ---: | ---: | ---: | --- |
| 1,000 | 0.389 s | 0.137 s | within 2 s |
| 10,000 | 3.418 s | 0.992 s | within 2 s |
| 117,266 | 36.311 s | 12.145 s | exceeds 5 s hard stop |

Only one full-size Commit was executed. The hard-stop rule prevented repeated
samples, so no full-size p95 distribution is claimed. A final-patch bounded
1,000-row Commit took 0.109 s; identical retry and metadata-conflict behavior
also passed.

All 13 named durability cut points passed on a bounded 100-row slice across two
independent reopens. Pre-replacement cuts rolled back to generation 0;
post-replacement cuts recovered generation 1. Source hashes stayed unchanged.

## Mechanism

The current Commit path performs three full-file I/O passes around one edited
row:

1. parse and validate every row while computing the candidate hash;
2. parse the source again while writing the complete temporary JSONL;
3. read and hash the temporary JSONL before rename and fsync.

This makes latency approximately linear in split size and puts the full train
split well beyond the approved threshold.

## Approved Resolution

The active OpenSpec now owns the replacement contract:

1. retain one monolithic ordinary `working.norm.jsonl` as the complete
   last-terminal-generation authority;
2. collect several exact durable Draft snapshots into one immutable,
   all-or-nothing same-split batch;
3. durably enqueue and return without waiting for whole-file publication, while
   one background worker rewrites and atomically publishes the complete next
   generation;
4. preserve any newer Draft made after enqueue and never report queue acceptance
   as committed success;
5. remove the redundant full-file parse/rehash passes and separately measure
   foreground enqueue versus background total/amortized latency, RSS, and
   recovery.

Delta/database authority and sharded JSONL remain unapproved alternatives. The
12.145-second result remains valid negative evidence for the superseded
synchronous path, not a benchmark claim about the new batch implementation.

## Async Batch Follow-up Evidence

The replacement core was exercised under explicit code-state receipts in
`outputs/label_studio_coco_refinement/async-batch-v1/`.

Bounded measurements under intermediate store hash `578309999d73...` changed
ten rows in one generation while preserving an untouched sentinel row byte for
byte:

| Rows | Bootstrap | Durable enqueue | Background worker | Amortized/member |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.443 s | 2.749 ms | 0.155 s | 15.5 ms |
| 10,000 | 3.545 s | 3.105 ms | 1.117 s | 111.7 ms |

One full 117,266-row run was completed under earlier store hash `5cc77532703a...`:
bootstrap took 39.579 seconds, durable enqueue took 2.538 milliseconds, and the
ten-member background publication took 12.935 seconds. It produced one
generation, changed all ten captured rows, preserved the sentinel row, used a
116,112,684-byte working JSONL and 17,838-byte queue, and retained managed image
links to the shared image root. This is pre-final background evidence, not a
claim that the final source hash was run at full size. Later source changes were
restricted to existing/active enqueue-receipt reconciliation and tests; the
candidate/process/publication path was not rerun at full size.

The exact final store hash
`77ac5fa04dba619b10f783506d8e598e1ff62160932a98fa021b3f0179569d39`
was independently exercised at the receipt boundary. While the transaction
lock was held, exact and different-batch retries returned the existing
`Queued`/`Running` identity in under one millisecond per pair without appending
queue bytes. After a durable journal terminal with a missing queue-terminal
projection and a released lock, both returned the same identity as
`Reconciling` in 1.500 milliseconds. All 14 matrix assertions passed.

The canonical `train.norm.jsonl` hash remained
`d64edc553bdc4d725cb9c3a504f369a9799e8bdde20c8fef0787a09cec33c16a`;
the image root and three sampled JPEG identities/hashes were unchanged. These
receipts attest the pure store/locking boundary. Browser/HTTP Draft saving,
navigation while a worker runs, an independently killed OS worker, and a final
hash full-size rerun remain explicitly unattested.
