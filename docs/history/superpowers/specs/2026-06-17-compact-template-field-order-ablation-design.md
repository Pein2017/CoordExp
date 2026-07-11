# Compact Template Field-Order Ablation Design

## Objective

Adopt a two-knob compact detection-template contract for the bbox-first final
ablation while preserving standard Stage-1 SFT, sorted object ordering, static
packing, and strict artifact provenance.

## Contract

The compact contract has two authored axes:

- `detection_template.id` or the active Stage-1 SFT equivalent
  `custom.detection_template_id` selects template family, closure tokens,
  newline behavior, strict parser family, structural token rows, prompt wrapper
  pattern, and artifact template identity.
- `custom.object_field_order` selects semantic segment order inside each object
  row:
  - `desc_first`: object-ref segment before box segment.
  - `geometry_first`: box segment before object-ref segment.

The requested rich bbox-first row is:

```text
<|box_start|><|coord_x1|><|coord_y1|><|coord_x2|><|coord_y2|><|box_end|><|object_ref_start|>{desc}<|object_ref_end|>
```

`custom.object_ordering` remains independent and the final ablation pins it to
`sorted`.

## Template Family

The implementation should support this compact family:

- `compact`
- `compact_object_closed`
- `compact_box_closed`
- `compact_object_box_closed`
- `compact_object_box_closed_lines`

Newline behavior remains registry-owned by the line template id.  Do not add a
free `row_separator` config knob.

## Data Flow

```text
Stage-1 SFT YAML
-> ConfigLoader / CustomConfig validation
-> prompt resolver + JSONLinesBuilder / dense-caption surface
-> compact renderer using template id x object_field_order
-> tokenizer/chat template
-> encoded-sample cache and static packing fingerprint
-> standard SFT trainer
-> inference prompt/parser policy with the same axes
-> gt_vs_pred materialization with normalized objects
-> mAP over normalized predictions, duplication severity reported separately
```

## Acceptance Boundary

Implementation is not approved by this document.  The next state after this
planning packet is `ready for user approval`.

## Required Evidence After Implementation

- Unit tests covering render/parse for every compact template id x
  `desc_first` / `geometry_first`.
- Prompt hash tests showing field-order and template-id changes.
- Encoded-sample cache and static-packing fingerprint tests over both axes and
  `global_max_length`.
- Config-load tests for the paired production and smoke ablation configs.
- Two tiny smoke runs after implementation approval, one desc-first and one
  bbox-first, with artifact roots and parse/drop counters recorded.
