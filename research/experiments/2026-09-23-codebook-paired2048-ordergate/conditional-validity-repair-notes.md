# Conditional-order gate execution notes

Status: implementation in progress; no new model entry yet.

## Frozen initial write allowlist (before implementation edits)

- `src/config/models.py`: add an explicit, zero-default conditional-order-gate config while preserving legacy hinge defaults/readers.
- `src/losses/conditional_order_gate.py`: implement the distinct FP32 coordinate-family valid-mass term, causal predecessor checks, and box/segment telemetry.
- `src/losses/runner.py`: wire configuration, batch and planned-step reductions, global denominator and telemetry to the maintained objective.
- `src/losses/__init__.py`: export the new term only if the maintained consumer requires it.
- `tests/losses/test_conditional_order_gate.py`, `tests/losses/test_runner.py`, `tests/config/test_train_config.py`: nearest caller, mutation and schema checks.
- `probes/training_set_completion/coordinate_codebook_alignment/paired2048_train.py` and its `test_paired2048_train.py`: only actual-entry objective accounting/receipt compatibility needed for the fixed rerun.
- `probes/training_set_completion/coordinate_codebook_alignment/three_loss_checks.py` and its focused test: the existing real trainer hook hardcodes the historical hinge name/reference; parameterize only its third term so it independently checks the new configured gate while retaining the historical caller unchanged.
- `probes/training_set_completion/coordinate_codebook_alignment/three_loss_train.py`: pass an explicit expected-weight mapping into its existing trainer hook; old calls retain the old mapping.
- `configs/research/paired2048-ordergate-early.yaml`, `configs/research/paired2048-ordergate-late.yaml`: explicit new scientific leaves, if the maintained launcher consumes YAML.
- This notes file, new execution-specific JSON receipts under the ordergate output root, and new official YAML inference leaves if needed for the fresh checkpoints.

No historical config, accepted output bytes, common data/cache, shared trainer/pipeline, model architecture or inference semantics are in this allowlist. A newly necessary seam will be recorded here before editing it and escalated if it changes the declared meaning.

The user-fixed objective is CE1 + type gate0.2 + conditional order gate0.2; historical raw-axis hinge0 and Gaussian0 in new configs only. The new term is `logsumexp(z_C)-logsumexp(z_{bin>prefix})` at x2/y2, using the actual same-object consumed x1/y1 token. The archived old run remains evidence for its executed mean-hinge recipe.
