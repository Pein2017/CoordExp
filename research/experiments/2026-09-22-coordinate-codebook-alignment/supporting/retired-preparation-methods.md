# Retired preparation implementation details

Architecture intake, 2026-09-25. This records implementation choices, not new
execution or acceptance. The [accepted result](../lead-results.md), candidate
report, original configs and output receipts retain scientific authority.

The conditional-LR builder copied the frozen nominal config, scaled only the
five LR groups (language/vision/aligner adapters, selected token embeddings and
coordinate codebook) by 0.3 or 3 with Decimal multiplication, and changed the run
name. All other resolved values were checked against nominal. The packet fixed
seed1729, 512 updates, effective batch8 and checkpoints32/128/512. Evaluation
preserved sorted image IDs, empty prefixes, greedy decoding, RP1.0, token cap3084
and teacher metrics. A prepared packet did not prove that an LR arm executed.

The seed-repeat builder changed only run name and seed to2718, retained the
32-image fit cohort, source cache, batch8 and512updates, and checked rank-major
schedule equivalence between one-rank and eight-rank execution. Its three
checkpoints used the same evaluation policy. The accepted result establishes
which repeat completed; the builder itself was not evidence of completion.

The loader bridge compared historical Mixin assembly against live source
assembly on the fixed three-image qualification selection, without loading the
expanded codebook. It recorded parameter and selected-row identities, per-case
logit differences and actual model/vision calls. It deliberately had no parity
gate. Diagnostic differences were not a passed composition certificate.

These one-time implementations are recoverable at Git commit
`d4763fd048f1e651ca6067045e5f6d56798cdc07` under
`probes/training_set_completion/coordinate_codebook_alignment/`:
`production_lr_prepare.py`, `production_repeat_prepare.py`, `runtime_bridge.py`.
They are not supported commands in the current tree. Current composition and
evaluation implementations, independent readers, configs, output data and
required receipt-bound source captures remain. No original receipt was changed.
