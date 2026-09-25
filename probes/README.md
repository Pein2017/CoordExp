# Retained research operators

Scientific decisions belong to [research](../research/index.md), not these files.
The small implementation set is explicit; there is no universal trainer, profile
registry or compatibility alias for a retired experiment.

| Capability | Entry and contract |
|---|---|
| Output-QP | `python -m probes.output_qp --capture <six-array.npz> --output <fresh.json>`; selected-output-row minimum-Frobenius solve with exhaustive supplied-state FP32 certificate |
| Readout norm | `python -m probes.readout_norm --input <explicit-arrays.json> --output <fresh.json>`; effective OUTPUT-row lower-median norm scaling, selected logits only |
| Saved row evaluator | `src.eval.saved_rows`; explicit raw/case/reference-bank inputs, class-agnostic matching, validity and recurrence separately |

QP NPZ fields are hidden_states, target_ids, route_token_ids, base_route_logits,
top_ids and top_logits. No pickle, model loading or image/panel selection occurs.
The certificate is for the supplied states, not native generation or transfer.
Norm JSON fields are effective_output_rows, scores and token_ids. FP64 factors
multiply selected FP32 scores; input embeddings and nonselected columns remain
unchanged. Tied and untied model selection remains the caller's responsibility.

Pure functions support CPU tests. CLI qualification binds clean Git commit/tree,
required source files and input bytes before work and again before publication.
`--source-receipt` accepts only the exact current qualification identity and same
input; legacy receipts fail closed. A new qualification is not a recovered
historical run or an execution grant. External artifacts are not rewritten.

The former finite-panel producers, model-specific convenience loaders, stage
controllers and repair/closeout chains are no longer maintained. Their useful
results and exact historical recovery points are in the existing catalog.
Core training still retains its consumed DoRA/loss/packing mechanisms, regardless
of scientific novelty. New implementation requires an actual current question or
consumer, not another broad default directory.
