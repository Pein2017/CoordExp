## Context

This change introduces infrastructure for a targeted ablation: per-object field order in assistant JSON for detection training with AR multimodal models.

Current behavior is baseline-safe but hardcoded (`desc -> geometry`). The implementation surface is split across stage-1 and stage-2 builders. Without a unified knob, we cannot run clean comparisons with reproducible config-only switches.

Key constraints:
- YAML-first, no new CLI flags.
- Keep object instance ordering unchanged.
- Preserve geometry invariants and Qwen chat-template compatibility.
- Keep default behavior identical to current runs.

## Goals / Non-Goals

**Goals**
- Add one config knob (`custom.object_field_order`) to control per-object field order.
- Apply the knob uniformly across stage-1 and stage-2 training serialization paths.
- Align prompt wording with configured field order.
- Keep strict fail-fast validation and reproducibility.

**Non-Goals**
- Choosing a "best" ordering in this change (this is infrastructure only).
- Changing object sequence policy (`custom.object_ordering`).
- Changing matching/loss algorithms or geometry semantics.
- Introducing compatibility aliases for undocumented legacy key names.

## Decisions

### 1) Single source of truth: `custom.object_field_order`

Decision:
- Add `custom.object_field_order` with values:
  - `desc_first` (default),
  - `geometry_first`.
  - `geometry_first` means "emit the existing geometry key (`bbox_2d` or `poly`) before `desc` within each object payload."
  - It does not introduce a new field named `geometry`.
  - Terminology note: this generalizes the direction doc's "bbox-first" wording; for bbox-only payloads, `geometry_first == bbox_first`.

Rationale:
- Minimal config surface and unambiguous ablation control.
- Default preserves exact baseline behavior.

### 2) Scope: field order only, not object order

Decision:
- `custom.object_ordering` remains responsible for object instance order (`sorted` / `random`).
- `custom.object_field_order` controls only key order inside each object payload.

Rationale:
- Prevent confounding two independent ablation dimensions.

### 3) Stage-1 and Stage-2 must share ordering contract

Decision:
- Stage-1 `JSONLinesBuilder` serialization follows `custom.object_field_order`.
- Stage-2 Channel-A payload building and Channel-B FN append serialization follow the same setting.
- The rendered assistant message JSON text used by chat-template encoding MUST reflect the same per-object field order as structured payloads.

Rationale:
- Avoid training target mismatch between stages and ensure apples-to-apples comparisons.

### 4) Prompt wording follows configured field order

Decision:
- Dense prompts should explicitly request `geometry_first` layout when configured.
- Object ordering instructions remain unchanged and continue to come from `custom.object_ordering`.

Rationale:
- Instruction-target alignment improves ablation validity for AR behavior.

### 5) Parser/matching stays behaviorally stable

Decision:
- Rollout strict parsing and matching keep current semantics.
- Field order variation is accepted as schema-equivalent object content.

Rationale:
- Keep change isolated to serialization/instruction surface, not matching logic.

## Data Flow Impact

### Stage-1

`config -> CustomConfig -> BaseCaptionDataset/FusionCaptionDataset -> JSONLinesBuilder -> assistant text/messages -> template.encode -> trainer`

Changed component:
- `JSONLinesBuilder` insertion order between the existing geometry key (`bbox_2d`/`poly`) and `desc`.

### Stage-2 Channel-A

`sample assistant_payload -> stage2_ab_training payload reconstruction -> assistant_text -> teacher-forced encode`

Changed component:
- payload dict construction order for `{desc, bbox_2d}` (current stage-2 bbox-only path), extensible to `{desc, poly}` when enabled.

### Stage-2 Channel-B

`rollout parse -> unmatched GT -> serialize_append_fragment -> Y_train -> teacher-forced encode`

Changed component:
- `serialize_append_fragment` per-object dict field order between the existing geometry key and `desc`.

Unchanged:
- object key numbering (`object_{n}`), predicted object appearance order, matching policy, loss decomposition.
- assistant-output field set remains `desc` + exactly one geometry key; optional JSONL metadata like `poly_points` is not emitted into assistant outputs.

## Compatibility and Failure Modes

- Unknown `custom.object_field_order` values fail fast at config load.
- Missing value defaults to `desc_first`.
- If stage-specific serialization paths diverge from configured order, unit tests fail.
- Existing configs without this key are unaffected.

## Verification Strategy

1. Config validation tests:
- accepted values parse,
- invalid values fail with actionable errors.

2. Stage-1 serialization tests:
- same sample serialized under `desc_first` vs `geometry_first`,
- diff confirms only intra-object field order changes.

3. Stage-2 serialization tests:
- Channel-A constructed payload honors selected order,
- Channel-B FN append fragment honors selected order.

4. Prompt tests:
- dense prompt text explicitly reflects selected field order,
- object-ordering phrases remain unchanged.

5. Regression safety:
- existing baseline tests continue passing under default (`desc_first`).
