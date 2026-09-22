# Lane C qualification handoff to recurrence-spatial

Lane B may use this existing native qualification as a source-entry parity
reference, subject to verifying that its target is the same original
untied417044 source boundary and policy.

- Qualification: `qualification/qualification.json`
- State: `untied-417044-failure`
- Split-aware source identity: `coco2017_train_000000417044`
- Boundary: `native[:source_row.end]`, prefix length 169 generated tokens;
  saved `next_row` starts at the same boundary
- Accepted source trace:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/runtime/untied-original/refined-03/trace.json`
- Source trace SHA256: `1b75265af561e712e74a74f455ae484dde17468fb5e76367b5cb32b6838295d6`
- Source trace offset/batch: offset 169, batch index 3
- Batch1/4 maximum logit delta: `3.147125244140625e-05`
- Batch1/8 maximum logit delta: `3.528594970703125e-05`
- Batch1/source-trace delta: `7.62939453125e-06`
- Batch8/source-trace delta: `2.288818359375e-05`
- Fixed tolerance: `2e-4`; positions identical (`0`); dtype/media/grid match
- Media SHA256: `66b4ed305acb5aa1c198309d4ce0ec6ed5edb25563ae8e9c2b04574ef18bda62`
- Image grid: `[1, 54, 72]`

This receipt has no spatial transform and does not establish Lane B’s
free-continuation or spatial-location result; it only supplies the already
qualified original native entry identity.
