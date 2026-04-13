## ADDED Requirements

### Requirement: V1 bbox parameterization is Stage-1/inference-facing and preserves canonical external `xyxy`
The system SHALL preserve canonical `bbox_2d: [x1, y1, x2, y2]` on all
model-independent surfaces while allowing Stage-1 and standalone inference to
use an internal model-facing bbox parameterization.

Normative behavior:

- the supported model-facing bbox parameterizations for this V1 change SHALL be
  exactly:
  - `xyxy`
  - `center_log_size`
- Stage-1 training SHALL resolve the parameterization from `custom.bbox_format`,
- standalone inference SHALL resolve the parameterization from
  `infer.bbox_format`,
- Stage-2 training, rollout-matching, and trainer-driven rollout/eval prompt
  rebuilding MUST reject `center_log_size` in this V1 change,
- raw JSONL, builder metadata, standardized prediction artifacts, evaluation
  inputs, benchmarking inputs, and visualization inputs MUST remain canonical
  `xyxy`,
- `bbox_format` applies only to `bbox_2d`,
- external field names such as `bbox_2d` and `desc` MUST remain unchanged,
- `poly` behavior remains unchanged.

#### Scenario: External artifacts remain canonical `xyxy`
- **WHEN** a run uses `bbox_format=center_log_size`
- **THEN** raw JSONL and standardized evaluation/visualization artifacts remain
  canonical `bbox_2d: [x1, y1, x2, y2]`
- **AND** no downstream evaluator schema migration is required.

#### Scenario: Stage-2 rejects `center_log_size`
- **WHEN** a Stage-2 or rollout-driven config requests
  `custom.bbox_format=center_log_size`
- **THEN** config validation fails fast with guidance that V1 is limited to
  Stage-1 and standalone inference.

### Requirement: `center_log_size` is produced through a fixed internal conversion layer
When model-facing `bbox_2d` uses `center_log_size`, the runtime SHALL derive it
from canonical `xyxy` through a shared internal conversion layer.

Normative behavior:

- the conversion layer MUST preserve external field names:
  - geometry key remains `bbox_2d`
  - non-geometry fields such as `desc` remain unchanged
- for a canonical normalized box `[x1, y1, x2, y2]`, the runtime MUST derive:
  - `cx = (x1 + x2) / 2`
  - `cy = (y1 + y2) / 2`
  - `w = x2 - x1`
  - `h = y2 - y1`
- width and height MUST then enter the fixed V1 normalized log-size chart with
  global floor `s_min = 1/1024`:
  - `u(s) = (log(max(s, s_min)) - log(s_min)) / -log(s_min)`
- model-facing `center_log_size` serialization MUST use:
  - `[cx, cy, u(w), u(h)]`
- each model-facing normalized slot `z` in `{cx, cy, u(w), u(h)}` MUST be
  quantized onto the coord-token lattice with:
  - `k = clamp(floor(999 * z + 0.5), 0, 999)`
  - emitted coord tokens remain `<|coord_k|>`
- inverse decoding for parsing/inference MUST use:
  - `z_hat = k / 999`
  - `w_hat = s_min * (1 / s_min) ** u_hat_w`
  - `h_hat = s_min * (1 / s_min) ** u_hat_h`
  - `x1_hat = cx_hat - w_hat / 2`
  - `y1_hat = cy_hat - h_hat / 2`
  - `x2_hat = cx_hat + w_hat / 2`
  - `y2_hat = cy_hat + h_hat / 2`
  - decoded canonical boxes MUST then be clamped/canonicalized with the shared
    repo `xyxy` validity rules before standardized artifact emission
- shared encode/decode helpers MUST be used wherever model-facing bbox slots are
  rendered or parsed.

#### Scenario: Stage-1 renders center-log-size without mutating canonical geometry
- **WHEN** a Stage-1 sample resolves `custom.bbox_format=center_log_size`
- **THEN** model-facing bbox slots are rendered as center-log-size
- **AND** the underlying canonical geometry remains `xyxy`.

#### Scenario: Inference decodes center-log-size back to canonical geometry
- **WHEN** standalone inference resolves `infer.bbox_format=center_log_size`
- **THEN** parsed model-facing bbox slots are decoded through the shared inverse
  helper
- **AND** standardized predictions are canonicalized to `xyxy`.

#### Scenario: Stage-1 targets stay on the coord-token lattice
- **WHEN** Stage-1 renders `center_log_size` bbox slots
- **THEN** each slot is emitted through the existing `<|coord_k|>` vocabulary
- **AND** no raw decimal bbox serialization is introduced.

### Requirement: Dense cache identity remains sensitive to model-facing bbox rendering
The system SHALL ensure that any rendered-sample or encoded-sample cache keyed
on model-facing dense payloads includes the full resolved prompt identity for
the V1 bbox experiment.

Normative behavior:

- cache identity MUST include, or be equivalent to a normalized tuple/hash over:
  - prompt variant
  - ordering policy
  - object field order
  - bbox format
  - coord mode
  - prompt/template hash when prompt text changes without key changes
- changing any one of those controls MUST invalidate or bypass stale cached
  rendered samples,
- bbox-format-only fingerprinting is insufficient.

#### Scenario: Changing bbox parameterization invalidates cache identity
- **WHEN** rendered dense samples switch between `xyxy` and `center_log_size`
- **THEN** the cache identity changes
- **AND** stale rendered samples are not reused.

#### Scenario: Prompt text drift invalidates cache identity
- **WHEN** prompt wording changes without a prompt key change
- **THEN** the prompt/template hash changes
- **AND** stale rendered samples are not reused.

#### Scenario: Object field order invalidates cache identity
- **WHEN** object field order changes for otherwise identical dense prompts
- **THEN** the cache identity changes
- **AND** stale rendered samples are not reused.
