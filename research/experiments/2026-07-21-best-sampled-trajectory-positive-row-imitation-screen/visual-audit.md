# Training-Event Visual Audit

## Scope

Six training events were inspected on enlarged crops before the real training
smoke. The audit deliberately covered:

- one high-gain sampled route;
- one crowded same-category scene;
- one geometry match just above the 0.75 Intersection over Union threshold;
- three unique-category rows whose entity identity is retained while their
  coordinate sites receive zero gradient.

The corrected visual artifact is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/visual-audit-v2-absolute-pixels/`

The assembly receipt stores bounding boxes in absolute image-pixel
coordinates. The earlier `visual-audit-v1` draft incorrectly rescaled them as
normalized 0-1000 coordinates and is explicitly marked as superseded.

## Findings

| Image | Category | Intersection over Union | Contract decision | Visual finding |
|---:|---|---:|---|---|
| 55232 | umbrella | 0.935 | train full row | Both boxes identify the same umbrella with tight, compatible extent. |
| 536486 | sheep | 0.770 | train full row | In a crowded flock, both boxes bind the same central sheep; the sampled box is slightly inset but remains instance-specific. |
| 555271 | car | 0.751 | train full row | Both boxes bind the same partially occluded left-edge pickup truck. The sampled box is narrower, but it does not switch to the adjacent car. |
| 102446 | couch | 0.521 | train schema and description only | The sampled row identifies the real couch but covers only its left portion; the official box uses a much broader extent. Coordinate masking is necessary. |
| 166998 | refrigerator | 0.596 | train schema and description only | The sampled row identifies the right-edge refrigerator but extends the box upward over its full visible column, whereas the official box starts lower. This is partly an annotation-extent ambiguity, so coordinate supervision would be unsafe. |
| 190052 | microwave | 0.653 | train schema and description only | The microwave identity is correct, but the sampled box extends left beyond the official appliance boundary. Coordinate masking is necessary. |

## Decision

The sampled-path admission rule is visually consistent with its intended use:

- trusted geometry rows provide complete-row supervision;
- unique-category low-overlap rows provide only schema and description
  supervision;
- uncertain geometry does not receive coordinate-token gradients.

This audit supports proceeding to the one-event real training smoke. It does
not establish dataset-wide label correctness or generalization.
