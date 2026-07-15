# Label Studio COCO Refinement Commit Gate

## Status

**Hold as of 2026-07-15.** The approved whole-file Commit contract is not
feasible with the first canonical implementation on the selected full train
split. Do not continue into deeper Label Studio UI or service integration until
the storage design is explicitly re-approved.

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

## Decision Required

Before implementation resumes, choose and record one authority shape:

1. retain immediate monolithic `working.norm.jsonl` authority and approve a
   bounded redesign/profiling attempt that removes redundant passes, accepting
   that a 126 MiB rewrite plus fsync may still miss 2 seconds;
2. make a small row-level journal or delta store the immediate authority and
   project ordinary JSONL periodically or on export;
3. use ordinary chunked/sharded JSONL as immediate authority with a manifest and
   materialize a monolithic export when needed.

Options 2 and 3 change the approved immediate-output contract and therefore
require an OpenSpec revision plus user approval.
