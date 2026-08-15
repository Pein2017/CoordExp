# Task 4 report — guarded dual-GPU entry and continuation gate

## Scope

Task 4 now has a value-only leaf config and a guarded one-image entry at
`scripts/research/run_human13_all_hf_shared_surface_vertical.py`.  The entry
is dry-run by default.  A live caller must provide `--execute`, explicit
`--user-model-gpu-authority`, two distinct suitable cards (GPU 0 training /
GPU 1 HF fp32/SDPA batch-one audit), and a newly confirmed-absent output root.
The entry exposes only injected Task-1/2/3 service seams; it does not load a
model, reserve a card, write an output root, or execute the one-image update.

The audit analyzer delegates prediction ordering, duplicate exclusion, and
one-to-one owner matching to the canonical Human-13 K-union analyzer.  It
reports per-RP source/proposal owner IDs, H gains, G losses, M gains, net
unique delta, duplicate/unmatched/malformed burdens, cap stops, rows, and
tokens.  The continuation receipt requires the exact H-positive/net-positive
RP-1.0 result, zero own-G loss under both audits, and non-increasing duplicate,
malformed, and cap burdens.  Full-panel admission remains hash-bound to a
passing one-image terminal and is model/GPU inert.

All injected failure paths attempt private proposal cleanup and Source
reproduction/rollback before closing borrowed sessions.  Retry, fallback, and
checkpoint promotion are forbidden in the resource and terminal receipts.

The fail-closed correction bundle also joins prebuilt resources to exact GPU 0
training/GPU 1 audit roles; validates the manifest image and its parent binding
envelope, source checkpoint/adapter/embedding paths and hashes,
tokenizer/prompt/panel/image provenance, canonical matcher identity, output
RP/image/provenance/stop fields, clean-HF-greedy metadata, distinct Source and
proposal checkpoint payload identities, and disjoint G/H/M owner strata before
update admission.  Partial session opens and writer failures close/clean
borrowed state, with one rollback attempt and no retry/fallback.  Full-panel
admission requires an issued in-process sealed passing terminal, not an
arbitrary matching hash or forged receipt.

## Verification evidence

Focused Task-4 tests pass with CPU/value doubles:

```text
25 passed — tests/research/test_run_human13_all_hf_shared_surface_vertical.py
```

The terminal seal is issued only by the guarded one-image path; a direct
attempt to seal a value without its private issuer is rejected, and full-panel
admission requires that issued in-process seal.

The default CLI was exercised against the leaf config and produced a terminal
receipt with zero model loads, forwards, backwards, optimizer steps, GPU
allocations, network actions, and output creations.  No live HF/model/GPU,
network, checkpoint, or output action was performed.

Task 4.6 remains intentionally unchecked pending the requested independent
bounded prelaunch review.  Task 5 owns any future live-state reservation and
execution decision.

The reviewer’s bounded Task-1/2/3/live-model/evaluator set (the focused suite
plus seven adjacent files) passes 254 tests after the two final provenance
regressions (252 before those two tests).  A broader nine-file Task-1/2/3 and
live-model/evaluator run passes 278 tests including the focused suite.  Both
sets use CPU/value doubles only.  Compileall, Ruff, strict OpenSpec
validation, Serena diagnostics, and Pyright (`0 errors, 0 warnings, 0
informations`) are clean.  No model/GPU/network action was used for any
diagnostic.
