# Recurrence dynamics

This family investigates the onset, persistence and conditional control of
autoregressive repetition: historical content, spatial/rotary positions, readable
KV history and native continuations. Numerical recurrence, owner recurrence and
physical recovery remain different outcomes. Current interpretation belongs to
the [history question](../../research/questions/history-repetition-stopping.md),
not this implementation map.

Profiles include `recurrence_census`, `recurrence_spatial`, `recurrence_mass`,
native history reading and the later row/phase/readback modules. A frozen
single-case profile is not a general-purpose recipe for arbitrary images.
Independent raw-data readbacks are retained instead of being replaced by calls
to the producer's own final summary.

## Shared operations, explicit conditions

`src.qwen.saved_prefix` preserves the old EOS-then-pad companion extension.
`src.qwen.native_row_scores` has a different exact-history companion rule and
performs inference-only scoring. Their conditioning policies were not merged.
The fixed compact-role convention is not a general parser for arbitrary-length
category text. `src.eval.numerical_recurrence` preserves pairwise near-equality
without treating it as transitive or as physical-owner identity.

Native request planning, exact histories, generation, tensor identity and token
spans are owned by their existing `src` domains. The mature tied/untied loader
and saved panel/receipt adapter live in [model profiles](../model_profiles/README.md).
Do not import a different experiment's runner merely to obtain these operations.

## Use and evidence

Select a profile through its owning unit and exact input records. For example,
`python -m probes.recurrence_dynamics.recurrence_book_first_revisit.run --help`
describes one frozen entry; it does not authorize its historical job. Preserved
protocols and raw receipts keep their original paths and source hashes. New
execution must bind current producers and shared dependencies separately.

Family tests and `tests/qwen/test_saved_native_operations.py` cover CPU contracts.
They do not establish full-model mask, position or native-greedy numerical parity.
