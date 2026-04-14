## Context

CoordExp’s canonical public contract is still `bbox_2d: [x1, y1, x2, y2]`.
That contract is embedded in raw JSONL, builder metadata, standardized
inference artifacts, visualization, and detection evaluation.

The original bbox-format commit tried to switch only some model-facing surfaces
and left prompt, parser, and loss behavior inconsistent. A later redesign draft
then widened too aggressively into Stage-2 and multiple bbox loss families.

V1 is intentionally smaller:

- Stage-1 and standalone inference are the only surfaces in scope,
- external/model-independent geometry remains canonical `xyxy`,
- model-facing Stage-1 bbox slots are reparameterized internally to
  center-plus-log-size,
- coord-token supervision is pure CE plus positive gating,
- regression-style bbox losses are deferred.

This gives us a fast path to answer the actual research question: does a
center-log-size parameterization help under minimal loss complexity?

## Goals / Non-Goals

**Goals**

- Add an opt-in model-facing Stage-1 bbox parameterization
  `bbox_2d: [cx, cy, u(w), u(h)]`, where `u(*)` is the shared log-size
  expression.
- Keep canonical `xyxy` on raw JSONL, builder metadata, standardized inference
  artifacts, evaluation, benchmarking, and visualization.
- Implement one shared Stage-1/inference conversion layer between canonical
  `xyxy` and model-facing center-log-size slots.
- Update prompt templates so the model is clearly instructed to emit
  `[cx, cy, u(w), u(h)]`, with `u(*)` defined as the shared log-size
  expression.
- Evaluate the parameterization under pure CE on coord slots, with retained
  gating terms for coord vs non-coord separation.

**Non-Goals**

- No Stage-2 support in V1.
- No decoded-box regression losses in V1:
  - no W1
  - no soft CE
  - no Smooth L1 / Huber bbox regression
  - no bbox geometry loss
  - no bbox size auxiliary
- No raw JSONL schema change.
- No evaluator or visualization schema change.
- No rename of external fields such as `bbox_2d` or `desc`.

## Decisions

### 1. V1 uses a Stage-1/inference model-facing `cxcy_logw_logh` parameterization

Decision:

- `custom.bbox_format` remains the Stage-1 training-side knob.
- `infer.bbox_format` remains the standalone inference-side knob.
- Supported values for V1 are:
  - `xyxy`
  - `cxcy_logw_logh`
- Stage-2 training, rollout-matching, and trainer-driven rollout/eval prompt
  rebuilding are out of scope for `cxcy_logw_logh` in V1 and MUST fail fast if
  they request it.
- `cxcy_logw_logh` means model-facing `bbox_2d` is serialized as:
  - `[cx, cy, lw, lh]`
  where `lw = u(w)` and `lh = u(h)` are normalized log-size slots in the
  shared encoded space.

Rationale:

- This preserves the original configurable-surface idea while making the new
  representation explicit instead of overloading `cxcywh`.

### 2. External/model-independent surfaces remain canonical `xyxy`

Decision:

- The following remain canonical `bbox_2d: [x1, y1, x2, y2]`:
  - raw JSONL
  - in-memory canonical geometry
  - builder metadata
  - standardized inference artifacts
  - benchmarking/evaluation inputs
  - visualization inputs
- Only model-facing rendered targets / prompts / parsed predictions may use
  `cxcy_logw_logh`.

Rationale:

- This keeps downstream tooling stable and avoids a schema migration outside the
  training/inference boundary.

### 3. The serializer uses an internal `xyxy -> cxcy_logw_logh` conversion

Decision:

- The conversion layer operates after canonical geometry is loaded and before
  model-facing dense payload text or target slots are produced.
- The conversion computes normalized center and size from canonical `xyxy`:
  - `cx = (x1 + x2) / 2`
  - `cy = (y1 + y2) / 2`
  - `w = x2 - x1`
  - `h = y2 - y1`
- Width/height then enter a fixed V1 normalized log-size chart with global
  floor `s_min = 1/1024`:
  - `u(s) = (log(max(s, s_min)) - log(s_min)) / -log(s_min)`
  - serialized slots are `[cx, cy, u(w), u(h)]`
- Each serialized slot is then placed onto the existing coord-token carrier
  lattice with:
  - `k = clamp(floor(999 * z + 0.5), 0, 999)` for
    `z in {cx, cy, u(w), u(h)}`
  - emitted coord tokens remain `<|coord_k|>`
- Inverse parsing first decodes the coord-token lattice back into normalized
  slots with:
  - `z_hat = k / 999`
- Inverse decoding for inference/prediction parsing uses:
  - `w_hat = s_min * (1 / s_min) ** u_hat_w`
  - `h_hat = s_min * (1 / s_min) ** u_hat_h`
  - `x1_hat = cx_hat - w_hat / 2`
  - `y1_hat = cy_hat - h_hat / 2`
  - `x2_hat = cx_hat + w_hat / 2`
  - `y2_hat = cy_hat + h_hat / 2`
  - decoded canonical boxes are then clamped/canonicalized with the existing
    shared `xyxy` box validity rules before standardized artifact emission
- External field names do not change:
  - geometry key remains `bbox_2d`
  - `desc` remains unchanged

Rationale:

- This satisfies the user requirement to convert internally to
  `(cx, cy, logw, logh)` while still using the existing coord-token grid.
- Fixing `s_min` in V1 removes a whole class of configuration and audit
  complexity.

Alternatives considered:

- Raw `cxcywh` with linear `w,h`: rejected because it is the wrong chart for
  size.
- Configurable `s_min` in V1: rejected to keep the first experiment narrow.

### 4. Prompt templates describe `[cx, cy, u(w), u(h)]`

Decision:

- Dense prompt variants must render bbox instructions in terms of
  `[cx, cy, u(w), u(h)]` when `bbox_format=cxcy_logw_logh`.
- Prompt wording must define the size expression explicitly as:
  - `u(s) = (log(max(s, s_min)) - log(s_min)) / -log(s_min)`
- Override-style prompt variants must use structured placeholders for:
  - bbox parameterization wording/examples
  - object field order wording/examples
- Missing required placeholders fail fast.

Rationale:

- This avoids the original override drift bug and makes the experimental
  parameterization legible to the model.

### 5. V1 Stage-1 supervision is pure CE plus gating

Decision:

- Standard token CE on non-coord tokens remains unchanged.
- Coord-token positions for the bbox experiment use hard CE only.
- Soft CE and W1 are explicitly disabled in this V1 experiment.
- `custom.coord_soft_ce_w1.enabled` MUST be `true`.
- `ce_weight` MUST be `> 0`.
- `soft_ce_weight` MUST be `0`.
- `w1_weight` MUST be `0`.
- Gating remains enabled to preserve separation between coord tokens and
  non-coord tokens:
  - `gate_weight` continues to parameterize `coord_gate` on coord-token
    positions and encourages coord-vocab mass
  - `text_gate_weight` is added to the Stage-1 surface and parameterizes
    `text_gate` on supervised `struct|desc` positions to discourage coord-vocab
    mass
- `gate_weight` and `text_gate_weight` MUST both be `> 0`.
- Soft-target shaping knobs are compatibility-only in this profile:
  - `temperature` MUST remain `1.0`
  - `target_sigma` MUST remain `2.0`
  - `target_truncate` MUST remain `null`
- `custom.bbox_geo.*` and `custom.bbox_size_aux.*` are rejected in this
  profile.
- Decoded-box regression losses are out of scope.

Rationale:

- This keeps the loss surface minimal so the experiment isolates the effect of
  the parameterization itself.
- It also makes the gating requirement implementable instead of leaving
  coord-vs-text separation implicit.

### 6. Standardized inference artifacts remain canonical while raw payloads stay parsed-best-effort

Decision:

- Standalone inference may prompt for and parse `cxcy_logw_logh`.
- After parsing, standardized prediction objects are canonicalized to pixel
  `xyxy` before writing:
  - `gt_vs_pred.jsonl`
  - `vis_resources/gt_vs_pred.jsonl`
  - visualization/eval-ready copies derived from the canonical standardized
    artifact
- `raw_output_json` remains the parsed best-effort raw payload emitted by the
  shared salvage/parser path rather than a verbatim raw-text mirror.
- Confidence post-op remains `xyxy`-only in V1:
  - `pred_confidence.jsonl`
  - any pipeline step that requires the raw bbox payload to be `(x1, y1, x2,
    y2)` bins
- When `infer.bbox_format=cxcy_logw_logh`, those confidence surfaces MUST fail
  fast instead of silently reinterpreting raw bins.
- Official score-aware evaluation remains allowed:
  - `gt_vs_pred_scored.jsonl`
  - `gt_vs_pred_scored_guarded.jsonl`
  are materialized from the canonical standardized `gt_vs_pred.jsonl` using a
  deterministic constant-score compatibility policy rather than confidence
  reconstruction.
- Canonical raw duplicate-control on already-standardized `xyxy` artifacts may
  remain available:
  - `gt_vs_pred_guarded.jsonl`

Rationale:

- This preserves evaluation stability without pretending the raw-bin
  confidence contract already understands `[cx, cy, u(w), u(h)]`.

### 7. Prompt/cache identity still needs full rendered-sample determinism

Decision:

- Any rendered-sample or encoded-sample cache that depends on model-facing
  dense payload text must include the full dense prompt identity:
  - prompt variant
  - object ordering
  - object field order
  - bbox format
  - coord mode
  - prompt/template hash when prompt text changes without key changes
- The same prompt/template hash MUST also be persisted in reproducibility
  metadata such as `resolved_config.json` and summary outputs.

Rationale:

- Even with a fixed log-size floor, stale cache reuse can still happen if prompt
  formatting changes are not fingerprinted.

## Risks / Trade-offs

- [Risk] The prompt could casually say `log(w), log(h)` while the actual
  serialized values are the normalized `u(w), u(h)` expression. → Mitigation:
  make the prompt wording explicit about the exact `u(*)` expression.
- [Risk] Pure CE may underperform more expressive loss stacks. → Mitigation:
  this is intentional for V1; the purpose is ablation clarity.
- [Risk] Leaving Stage-2 unchanged means the branch does not yet deliver a full
  repo-wide parameterization story. → Mitigation: record that explicitly as a
  V1 boundary.

## Migration Plan

1. Rewrite the OpenSpec change around the Stage-1-only center-log-size scope.
2. Add shared encode/decode helpers between canonical `xyxy` and model-facing
   center-log-size slots.
3. Update Stage-1 dataset rendering and prompt templates.
4. Update Stage-1 coord supervision to the pure-CE-plus-gating profile.
5. Update standalone inference parsing and canonicalized artifact emission, and
   reject confidence post-op for `cxcy_logw_logh` while allowing
   constant-score official-eval compatibility artifacts.
6. Add focused tests and then replay later commits on top of the rewritten
   branch.

## Open Questions

- None for the narrowed V1 scope. The user has decided to keep external `xyxy`
  canonical, use internal center-log-size conversion, limit training to
  Stage-1, and defer regression-style bbox losses.
