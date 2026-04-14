## 1. OpenSpec Narrowing

- [x] 1.1 Rewrite the bbox-parameterization change around the Stage-1-only
  center-log-size experiment.
- [x] 1.2 Remove Stage-2 and regression-loss requirements from this V1 change so
  there is one unambiguous contract source of truth.

## 2. Config And Shared Conversion

- [x] 2.1 Support `custom.bbox_format` and `infer.bbox_format` values:
  `xyxy | cxcy_logw_logh`.
- [x] 2.2 Add shared helpers that convert canonical `xyxy` boxes to/from
  model-facing center-log-size slots using the fixed V1 log-size floor
  `1/1024`.
- [x] 2.3 Make the coord-token carrier contract explicit for the new
  parameterization:
  - quantize each normalized slot with `round(999 * z)`
  - decode generated bins with `k / 999`
  - reconstruct canonical `xyxy` via `cx +/- w/2`, `cy +/- h/2`
- [x] 2.3 Ensure rendered-sample / encoded-sample cache identity captures the
  full dense prompt identity for the new model-facing serialization.

## 3. Prompt And Dataset Integration

- [x] 3.1 Replace fragile override rewriting with structured prompt placeholders
  for bbox parameterization wording and examples.
- [x] 3.2 Update prompt text so `bbox_2d` is described as
  `[cx, cy, u(w), u(h)]` under `cxcy_logw_logh`, with the shared log-size
  expression defined explicitly.
- [x] 3.3 Update Stage-1 dataset rendering / serialization so external
  canonical geometry stays `xyxy` while model-facing bbox slots use
  center-log-size.

## 4. Stage-1 Loss Surface

- [x] 4.1 Keep standard CE on non-coord tokens unchanged.
- [x] 4.2 Update Stage-1 coord-token supervision so the center-log-size
  experiment runs in pure-CE mode:
  - coord CE enabled
  - soft CE disabled
  - W1 disabled
- [x] 4.3 Retain coord/text gating so coord-vocab separation remains enforced
  across coord vs non-coord token families, including an explicit
  `text_gate_weight` surface for supervised `struct|desc` positions.
- [x] 4.4 Defer bbox regression-style losses in this V1 path:
  `bbox_geo`, `bbox_size_aux`, and related decoded-box regression surfaces.
- [x] 4.5 Fail fast on authored non-pure settings for this profile:
  - `soft_ce_weight > 0`
  - `w1_weight > 0`
  - non-default soft-target shaping knobs
  - `bbox_geo` / `bbox_size_aux`
  - Stage-2 or rollout surfaces requesting `cxcy_logw_logh`

## 5. Inference And Canonical Artifacts

- [x] 5.1 Update standalone inference to parse center-log-size predictions and
  canonicalize them to pixel `xyxy` before standardized artifact emission.
- [x] 5.2 Keep `raw_output_json` as the parsed best-effort raw payload while
  standardized/vis artifacts stay canonical `xyxy`.
- [x] 5.3 Reject confidence post-op for `cxcy_logw_logh`, but materialize the
  official scored-eval artifact family from canonical standardized predictions
  using a deterministic constant-score compatibility policy.
- [x] 5.3 Record the resolved prompt identity and bbox parameterization in
  reproducibility artifacts, including a prompt/template hash.

## 6. Verification And Branch Rewrite

- [x] 6.1 Add focused tests covering:
  prompt-template parity,
  center-log-size encode/decode,
  CE-only coord supervision,
  gating behavior,
  fail-fast rejection of non-pure settings,
  canonical inference/eval artifacts,
  fail-fast rejection of confidence post-op,
  constant-score official-eval compatibility for `cxcy_logw_logh`,
  and cache invalidation across prompt formatting changes.
- [x] 6.2 Run targeted validation and replay commits `f16f6a2` and `f30aea7`
  on top of the rewritten branch.
