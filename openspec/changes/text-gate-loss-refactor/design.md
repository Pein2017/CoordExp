## Context

Stage-2 training currently computes and logs losses through multiple partially overlapping pathways:
- A YAML-declared (or defaulted) objective/diagnostics module pipeline,
- Trainer-local loss computation (with precomputed module outputs), and
- Separate Stage-1-style aux loss paths (e.g., coord softCE/W1) that use their own config surface.

This has created:
- Multiple sources of truth for “effective” loss weights (flat typed knobs, pipeline module configs, module-level aliases, and context fallbacks),
- Silent misconfiguration risk in rollout-matching pipelines (missing strict per-module config validation), and
- Objective drift where configured knobs do not change the objective (`text_gate_weight` is currently a no-op).

Constraints:
- YAML-first: no new CLI flags.
- Preserve geometry invariants: never drop/reorder coords; do not resize in training (`do_resize=false`).
- Maintain Qwen3-VL template compatibility.
- Upstream HF model internals are off-limits.

## Goals / Non-Goals

**Goals:**
- Implement `text_gate` as a first-class `coord_reg` sub-term to penalize coord-vocab probability mass at text positions (`type=struct|desc`), respecting rollout-context FP-neutral masking semantics.
- Refactor loss/weight ownership into a single unified registry + pipeline contract so:
  - module names are centralized,
  - allowed config keys are strict and centralized,
  - effective weights are derived from one place, and
  - trainers do not bypass module implementations with duplicated math.
- Remove all backwards-compat aliases for loss weight keys (config and runtime).
- Enforce strict per-module config validation for both Stage-2 (`stage2_ab.pipeline`) and rollout-aligned (`rollout_matching.pipeline`) pipelines.
- Emit only canonical registry-derived `loss/<component>` metric keys for objective components (remove trainer-specific loss aliases).

**Non-Goals:**
- Rewriting the underlying detection evaluator or COCO metric implementation.
- Introducing a new model head for gate terms (gate terms must be derived from token logits).
- Changing tokenizer vocab, coord token encoding, or prompt templates.
- Full Stage-1 loss refactor (Stage-1 can be migrated in a follow-up once Stage-2 and rollout-aligned are stabilized).

## Decisions

1) **Make pipelines explicit (no implicit defaults)**

Decision:
- Require `stage2_ab.pipeline` for `custom.trainer_variant=stage2_two_channel`.
- Require `rollout_matching.pipeline` for `custom.trainer_variant=stage2_rollout_aligned`.

Rationale:
- Eliminates hidden “default manifest” behavior that is currently split across schema defaults, `src/sft.py` manifest building, and trainer local fallbacks.
- Makes loss weight changes auditable from a single YAML diff.

Alternative considered:
- Keep default pipeline manifests and keep flat knobs.
  - Rejected: still multiple sources of truth; higher chance of drift and accidental enablement.

2) **Single registry owns module names, config schemas, and module resolution**

Decision:
- Centralize the objective/diagnostics module registry in one Python module (shared by Stage-2 and rollout-aligned trainers).
- The registry owns:
  - allowed objective module names,
  - allowed diagnostics module names,
  - per-module typed config schemas / allowed keys,
  - module resolution (name -> module runner).

Rationale:
- Today module names + config schemas are duplicated across:
  - Stage-2 schema (`src/config/schema.py`),
  - rollout-matching schema (`src/config/rollout_matching_schema.py`),
  - the runtime pipeline (`src/trainers/teacher_forcing/objective_pipeline.py`).
- Centralization prevents drift and makes adding/modifying modules a single-edit task.

3) **No backwards-compat for config keys**

Decision:
- Remove aliases such as:
  - `bbox_smoothl1_weight` inside module configs (use `smoothl1_weight` only),
  - `coord_soft_ce_weight` / `coord_w1_weight` (use `soft_ce_weight` / `w1_weight` only),
  - token CE aliases (`fn_desc_ce_weight`, `matched_prefix_struct_ce_weight`).
- Remove flat Stage-2 objective knobs (`stage2_ab.desc_ce_weight`, `stage2_ab.fmt_struct_ce_weight`, etc.) in favor of pipeline module configs.

Rationale:
- Alias support multiplies codepaths and makes the effective objective non-obvious.
- Breaking changes are acceptable by request; the goal is maintainability and auditability, not compatibility.

4) **Implement `text_gate` via stable logit-derived p_coord(t)**

Decision:
- Define `p_coord(t)` at a token position `t` from logits:
  - `S_all(t) = logsumexp(logits_full[t, :])`
  - `S_coord(t) = logsumexp(logits_full[t, coord_vocab_ids])`
  - `p_coord(t) = exp(S_coord - S_all)`
- Define:
  - `coord_gate(t) = -log(p_coord(t) + eps)` at `type=coord` positions,
  - `text_gate(t) = -log(1 - p_coord(t) + eps)` at `type=struct|desc` positions.
- Compute gate terms only when the corresponding weight is non-zero.

Rationale:
- Avoids adding any new model heads and matches the OpenSpec gate definitions.
- Can reuse the existing `coord_vocab_gate_loss` helper to compute `-log(p_coord)` robustly via logsumexp differences.

5) **Masking and context semantics are owned by unified registry helpers**

Decision:
- `text_gate` and `coord_gate` MUST share the same context-specific masks as CE/geometry:
  - rollout-context FP spans excluded,
  - matched-prefix `desc` excluded where `CE_desc=0`,
  - EOS handling remains unchanged (gate terms may ignore EOS).

Rationale:
- Gate terms must not re-introduce supervision leakage into FP spans or disabled desc spans.
- Ensures `text_gate` does not fight the CE masking contract.

6) **Sparse metric emission for rollout-only monitors**

Decision:
- Rollout-only monitor keys (notably `rollout/*` scalars and rollout timing keys) are emitted only when rollout was executed for the step.
- Placeholder metrics that are consistently `0.0` (e.g., `time/mask_build_s` in trainers that do not measure it) are omitted.

Rationale:
- A-only Stage-2 runs currently emit constant-zero rollout monitors due to decode configuration being present even when no rollout occurs.
- Constant metrics add noise and make it harder to monitor training regressions; sparse emission makes dashboards actionable without requiring per-run filtering.

## Risks / Trade-offs

- [Large breaking config churn] → Provide an explicit migration checklist + update all canonical configs in-repo; keep strict parser errors actionable (full dotted paths, allowed keys listed).
- [Performance overhead for gate terms] → Compute gate terms only when weights are non-zero; implement via logsumexp differences with stable clamping.
- [Metric key breakage for dashboards] → Update docs to the new canonical key set and provide a one-time mapping note in the change description/tasks.
- [Incorrect masking semantics] → Add targeted unit tests for rollout-context FP-neutral gating and for text-position token-type gating; include packed multi-segment tests.
