## Context
CoordExp uses a JSON-only assistant output with `<|coord_k|>` tokens embedded as JSON strings. Stage-1 is responsible for teaching the model to reliably emit valid JSON and to place probability mass correctly over the ordered coord bins at coordinate slots.

The current codebase already contains coord-token utilities such as coord-vocab masking and expectation decoding, and an auxiliary loss path (L1/GIoU) that expectation-decodes coord logits into continuous coordinates.

This change introduces an alternative Stage-1 supervision path that is strictly token-level:
- no expectation decoding required for training,
- no box-level L1/GIoU dependence,
- supervision is applied only at coord-token slots using (softCE + W1).

## Goals / Non-Goals
Goals:
- Provide a stable Stage-1 loss recipe: CE on non-coord tokens + coord-gated `softCE+W1` on coord tokens.
- Keep the loss packing-compatible and numerically stable under bf16.
- Keep behaviour opt-in and backward compatible.
- Produce a new retrained Stage-1 checkpoint with the updated recipe (operationally tracked).

Non-goals:
- No Stage-2 / matching changes.
- No changes to JSON schema, prompts, or chat template.

## Decisions
- Decision: Coord-vocab gating is mandatory for coord loss positions.
  - Rationale: Stage-1 removes full-vocab CE at coord positions; gating prevents coord slots from behaving like text slots.
- Decision: Use Gaussian-kernel unimodal soft targets `q(k)` and supervise with `softCE + W1(CDF)`.
  - Rationale: keeps targets stable and ordered; W1(CDF) penalizes “mass far away” and is less brittle than hard CE.
- Decision: Mask coord positions out of the base CE loss when this mode is enabled.
  - Rationale: avoid double-supervision with conflicting gradients (full-vocab CE vs coord-gated soft labels).
- Decision: Keep the implementation **single-forward** (no extra model forward).
  - Rationale: the codebase already supports adding auxiliary losses from `outputs.logits` in `Trainer.compute_loss` mixins.
  - Plan: pass labels with coord targets masked to `ignore_index` into the native CE path, and compute coord `softCE+W1` from the same forward logits.

## Alternatives considered
- Keep expectation decode + L1/GIoU in Stage-1:
  - Pros: directly optimizes continuous geometry early.
  - Cons: introduces additional decoding/loss complexity; interacts with multi-peak coord distributions; not aligned with token-only Stage-2 plan.
- Use hard one-hot CE on coord tokens:
  - Pros: simplest.
  - Cons: too brittle under quantization and noisy labels; provides no notion of ordered distance between bins.

## Risks / Trade-offs
- Risk: Masking coord tokens from CE can reduce “token hit rate” early if coord distributions are too diffuse.
  - Mitigation: keep a unimodal label with adjustable width; optionally warm up with wider labels.
- Risk: W1(CDF) is O(K) per token and may add cost under long packing.
  - Mitigation: keep K fixed at 1000; implement vectorized CDF and absolute diff; optionally subsample coord positions for logging (not loss).
- Risk: Using the native CE path requires correct label masking under the model’s shift rule.
  - Mitigation: mask coord targets in the unshifted `labels` tensor (so after the model’s internal shift, coord targets are ignored), and compute coord losses against `labels[:, 1:]` using `logits[:, :-1]` (same alignment used by existing aux loss mixins).

## Migration Plan
- Add the new mode behind an explicit config flag; existing runs remain unchanged.
- Provide one new Stage-1 retrain YAML that enables the mode and uses placeholders.
- Retrain Stage-1 and compare stability/parse-rate/coord-token metrics against the previous baseline.

## Open Questions
- Should the unimodal target be defined in bin-index space or normalized [0,1] space (equivalent up to scaling), and what kernel family is preferred (Gaussian vs triangular)?
- Should `line` geometry be included in Stage-1 retrain by default or kept disabled until downstream tools fully support it?
