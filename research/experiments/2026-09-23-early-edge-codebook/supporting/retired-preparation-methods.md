# Retired early-edge packet preparation

Architecture intake, 2026-09-25, not an experiment or protocol amendment. The
[accepted result](../lead-results.md) and frozen protocol retain authority.

The one-time builder derived its config from the accepted three-loss packet.
Its model changes were `early_patch_edges` mode and projection seed1729; the
codebook-projection optimizer group used LR2e-5 and zero weight decay. It changed
run/output locations and checked that other config fields were unchanged.
Saved late three-loss epoch8/epoch16 cells were comparators for early
epoch8/epoch16 conditions at planned updates492/984. The original packet and
eight-rank sequential amendment remain distinct.

Preparation checked the existing492-micro-step cache and exact fingerprint,
refused occupied run/queue-state paths, verified inherited inputs, and captured
sources before execution. It did not itself establish training success. The
closed, unpromoted study keeps its independent reducers, payload checks,
architecture implementation, original configs and output receipts.

The retired builder is recoverable at Git commit
`d4763fd048f1e651ca6067045e5f6d56798cdc07`, path
`probes/training_set_completion/coordinate_codebook_alignment/early_edge_prepare.py`.
It is no longer a current launch command. No original record, frozen dataset,
payload or receipt binding was rewritten.
