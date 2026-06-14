---
name: loss-numerics-sanity
description: "Use when designing, reviewing, debugging, or interpreting ML loss functions, custom objectives, auxiliary losses, soft labels, masking, weighting, gradient accumulation, distributed training, mixed precision, or early training curves. Triggers include suspected loss-scaling mistakes, NaN/Inf risk, silent zeroed terms, normal-looking but wrong metrics, launch-health checks, and requests to decide whether loss/metrics are expected."
---

# Loss Numerics Sanity

Run this skill as an alertness gate for objective math and training telemetry. The goal is to prevent plausible-looking training from hiding wrong supervision, wrong scaling, unstable numerics, or misleading metrics.

Default posture: loss curves are evidence, not proof. Verify the contract, scalar math, and effective contribution before interpreting model behavior.

## First Classify The Situation

Choose the smallest applicable path:

- **New or changed objective**: prove formula, target support, masking, weighting, dtype, and logging before trusting training.
- **Launch-health check**: verify the intended loss terms are active, finite, moving, and contributing at expected scale.
- **Metric/loss anomaly**: separate instrumentation bugs, data/mask bugs, optimization instability, and genuine model behavior.
- **Production interpretation**: require artifact-backed scope labels and avoid claiming quality from training loss alone.

If this overlaps with a repo-specific audit skill, use this skill for the numeric/objective questions and that skill for repository contracts, provenance, or eval semantics.

## Minimum Evidence Set

Collect these before giving a confident verdict:

- authored config and resolved/runtime config for objective weights, enabled terms, precision, batch size, gradient accumulation, and distributed world size;
- one real example through target creation: raw target, tokenized or tensorized target, mask, ignore index, sidecars, and special-token handling;
- one collated batch: shapes, dtype, device, valid-count denominators, padding, mask density, target counts, and per-rank consistency when distributed;
- one deterministic tiny-logit or tiny-tensor probe for every nontrivial formula, with expected scalar values computed independently;
- live telemetry: raw term losses, weighted/effective losses, total loss, gradient norm, learning rate, step, accumulation count, finite checks, and memory/utilization;
- artifact handles: log path, run directory, resolved config, manifest, checkpoint or step range, dataset slice, and scope label.

Do not rely on a single aggregate `loss` if custom terms exist.

## Numeric Invariants

Check these explicitly:

- **Finite values**: loss, logits, probabilities, gradients, and optimizer states must not contain NaN or Inf.
- **Denominators**: every average must divide by the intended count, not by padded tokens, world size twice, local batch only, or all tokens when only a subset is supervised.
- **Weights**: log both raw loss and weighted contribution; a term can look healthy while having zero or excessive effective weight.
- **Scale**: compare each term to its expected initial magnitude; verify soft-label cross entropy, KL, MSE, IoU, unlikelihood, entropy, and auxiliary terms are not mixed without intended normalization.
- **Masking**: confirm ignore-index, padding, EOS/BOS, prompt tokens, assistant tokens, object separators, and special tokens match the intended supervision.
- **Support**: ensure soft labels, constrained vocabularies, coordinate bins, class sets, negative samples, and allowed-token masks include the teacher target unless the design explicitly excludes it.
- **Reduction**: check `sum`, `mean`, per-token, per-object, per-sequence, per-image, and per-batch reductions. State which unit the loss optimizes.
- **Distributed math**: verify DDP all-reduce, gradient accumulation, packed batches, dropped/padded repeats, and partial accumulation windows do not change the effective objective silently.
- **Precision**: identify fp32 islands needed for exp/log/softmax/sigmoid/logsumexp, variance, distances, large negative masks, or tiny probabilities under fp16/bf16.
- **Gradient path**: verify intended tensors require grad and monitoring-only computations are not accidentally optimized, detached, or used as the differentiable term.

## Alert Patterns

Treat these as high-risk until explained:

- loss decreases but the custom term is disabled, zero-weighted, detached, or absent from logs;
- total loss looks normal while a new term is orders of magnitude larger or smaller than expected;
- gradient norm collapses to zero, spikes repeatedly, or changes sharply when adding a supposedly small auxiliary term;
- token accuracy rises while task-specific target probability, valid mass, or coordinate/error telemetry does not;
- training loss improves while eval validity, parser success, or rollout format collapses;
- soft-label entropy, target peak probability, support radius, or allowed-token mass is constant at an impossible value;
- coordinate or class targets report high accuracy before the model could plausibly learn, suggesting target leakage or mask shift;
- metrics are logged only on rank 0 while loss is computed on all ranks without clear aggregation semantics;
- per-step denominators vary with packing, object count, sequence length, image count, or truncation in a way the report does not acknowledge;
- checkpoint/eval artifacts exist but cannot be tied to the resolved config and exact objective version.

## Probe Sequence

Use this order for objective changes or suspicious curves:

1. **Static contract probe**: inspect resolved config and code path. Confirm intended terms are enabled and legacy keys cannot silently override them.
2. **Single-example probe**: pass one real sample through preprocessing and loss target construction. Print or inspect the target count, mask, and special-token boundary.
3. **Tiny formula probe**: construct logits/tensors where the expected answer is hand-computable. Test exact scalar, shape, dtype, and reduction.
4. **Batch probe**: run one collated batch with the real model or tiny stand-in. Log raw terms, weighted terms, valid counts, and finite checks.
5. **Tiny train probe**: run enough steps to see repeated exposure, not just one optimizer step. Check loss trend, gradient norm, LR, and term-specific movement.
6. **Production launch gate**: only then interpret long-run curves. Continue to watch early steps, first eval, and first checkpoint.

When time is limited, do not skip the tiny formula probe for new math. It catches scaling bugs that training curves often hide.

## Interpreting Early Training

Healthy early launch usually shows:

- active intended terms with nonzero valid counts;
- finite total and component losses;
- raw and weighted losses in plausible ranges;
- gradient norms high at first if expected, then settling without repeated explosion;
- learning rate schedule matching warmup/decay expectations;
- task-specific probabilities, top-k accuracy, valid mass, or error telemetry moving in the right direction;
- no contradiction between loss movement and artifact/parser/schema health.

Do not over-claim. Early training can prove optimization is alive and numerically sane; it cannot prove final task quality unless eval artifacts support that claim.

## Reporting Format

Use concise evidence-first output:

```text
Verdict
Scope
Evidence
Loss/Metric Trend
Numerics And Scaling
Masking/Target Contract
Distributed/Precision Notes
Warnings
Next Gate
Confidence
```

Severity guidance:

- `P0`: objective is almost certainly wrong, corrupt, or not the one intended.
- `P1`: plausible silent objective, masking, scaling, dtype, or distributed mismatch.
- `P2`: observability, reproducibility, or interpretation weakness that could hide future bugs.
- `P3`: cleanup or better diagnostics.

## Common Fix Directions

Prefer fixes that make future mistakes harder:

- add deterministic unit tests for loss formulas and reductions;
- log raw and weighted terms, valid counts, support statistics, finite checks, and per-rank or aggregate semantics;
- assert impossible states: zero valid targets, teacher outside support, all-masked batches, empty soft-label support, nonfinite tensors;
- isolate numerically sensitive operations in fp32;
- make denominators and reduction units explicit in code and metric names;
- store objective version, resolved config, and metric definitions in run artifacts;
- add launch-health checks that fail fast before expensive training when the objective is inactive or malformed.
