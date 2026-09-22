# Eight-coordinate BBox Supervision

## Question

Does explicitly supervising all four clockwise corners of an axis-aligned box
improve the model's localization understanding compared with the existing
`x1 y1 x2 y2` target?

The candidate target is:

```text
x1 y1 x2 y1 x2 y2 x1 y2
TL       TR       BR       BL
```

The representation is intentionally redundant. The loader and inference parser
therefore reject an eight-coordinate box unless it is exactly the canonical
clockwise expansion of the same axis-aligned `xyxy` box.

## Experiment Contract

- Candidate intervention: `object_box_closed_quad_clockwise` with eight
  coordinate tokens per object.
- Sorted candidate: `geo_sorted_xy`, ordered by top-left `(x1, y1)`.
- Random candidate: the existing deterministic, run-seeded random ordering.
- Compatibility: legacy `geo_sorted` retains its historical `(y1, x1)` order.
- Primary evidence: matched COCO bbox evaluation, with particular attention to
  AP75 and size-stratified AP in addition to overall AP/AR.
- Required integrity evidence: valid output rate, malformed-box rate,
  duplicate/missing-object behavior, token truncation, and realized object
  order receipts.
- Strongest alternative explanation: any change may come from doubled
  coordinate-token supervision weight, output redundancy, or sorting changes,
  rather than better geometric understanding.

The old `geo_sorted` four-coordinate run and the new `geo_sorted_xy`
eight-coordinate run differ in both representation and ordering. That is a
compound comparison. A decision-grade representation claim requires a matched
four-coordinate `geo_sorted_xy` control. The random arm can isolate the
representation change if its baseline uses the same random-order seed policy.
The current user-owned priority is to run the eight-coordinate arms first, so
no four-coordinate `geo_sorted_xy` control is prepared in this worktree.

## Configuration Fidelity

Production inherits the existing r16/a32, EBS24, pure-CE plus typegate-0.2,
warmup-0.1, eight-epoch baselines for the matching sorted and random arms. The
two-step smokes inherit their corresponding typegate smoke baselines. The
candidate profiles preserve the baseline model settings other than the
explicit base checkpoint, plus the adapter, packing, loss, optimizer,
scheduler, runtime, evaluation, and checkpoint settings. Their resolved diffs
are limited to:

- `run.name`;
- base checkpoint path;
- train/eval JSONL paths;
- assistant coordinate format;
- object ordering;
- system and user prompts.

The user-owned base checkpoint override is
`/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
It is not byte-identical to the older `coordexp` checkpoint,
so checkpoint identity is treated as an explicit experiment input rather than
an interchangeable path alias.

Both prompts explicitly require the axis-aligned clockwise sequence
`x1 y1 x2 y1 x2 y2 x1 y2`; neither contains the legacy four-coordinate
instruction.

## Data Provenance

Derived root:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_quad_clockwise_xy_sorted
```

The source is the current four-coordinate COCO root
`rescale_32_1024_bbox_len12000`. The derivation is fresh-output only and records
source/output SHA-256 hashes in `pipeline_manifest.json`.

| Split | Records | Objects | Max objects | Output SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 117,266 | 849,951 | 90 | `858c0ac2c172a76c9400486f19d3e910332bb934ba2c24590af7790f573ad76d` |
| val | 4,952 | 36,491 | 62 | `1ea48807c850282e16a217ce0cfd6ebfeb2640d0307695e9aa1ecdb11ab81d24` |

## 12k Length Gate

The full census measures image tokens plus prompt plus supervised assistant
response. Batched chat tokenization was checked against the production encoder
on 16 samples per split.

| Split | Maximum | Mean | Median | p95 | p99 | Over 12k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,695 | 1,548.42 | 1,521 | 1,749 | 1,891 | 0 |
| val | 2,346 | 1,549.93 | 1,521 | 1,753 | 1,916 | 0 |

Conclusion: no object-count cap is needed for the 12k length budget. The
90-object maximum train example is also the maximum-length example.

## Launch Boundary

Data preprocessing, the length census, and the two-step sorted smoke are
complete. The sorted production run was launched and reached finite optimizer
step 2,765 of 5,529, but it failed during the following full-eval metric gather.
The failure was not an over-length pack or model-forward OOM. The default NCCL
`all_gather_object` used separate size and pickle-payload collectives that
became phase-skewed across ranks. Reconstructing the step-2,765 metric report
produced a 683-byte pickle whose first eight bytes, misread as an integer size
and multiplied by eight ranks, predict exactly 84.072762 GiB.

The rank-report path now uses one fixed 65,592-byte CPU/Gloo frame per logical
report. Its header carries sequence, kind, step, split, rank, world size,
payload length, and CRC; validation happens only after the sole collective.
Payloads are bounded to 64 KiB and the control group has a 120-second timeout.

Verification receipts:

- 45 focused runtime/eval/pipeline tests passed, including a real eight-rank
  Gloo test that forbids `all_gather_object`, asserts one fixed-width collective
  per report, covers scalar/gradient/metrics/loss-denominator reports, and
  fails all ranks closed on kind, oversize, and serialization mismatches.
- A four-rank default-NCCL to dedicated-Gloo stress smoke completed 512 report
  rounds with zero CUDA bytes allocated by the report path.
- The final four-GPU model smoke completed two finite EBS24 steps, one
  checkpoint, and the entire 4,952-example / 701-pack eval. Its `run.json` is
  `completed`, the eval loss is finite, and no terminal error is recorded. The
  run is
  `outputs/smoke/coord_pure_ce_typegate/qwen3_vl_2b_desc_first_geo_sorted_xy_quad_clockwise_single_frame_full_eval_4gpu`.

The failed production run did not save optimizer, scheduler, RNG, sampler, or
trainer continuation state, so it cannot be resumed exactly and is not a
completed eight-epoch model. Its old failed artifacts were deleted after the
above facts were distilled here. At that point, no eight-card production
restart or val-200 rollout was authorized. A later user-authorized fresh run is
recorded below.

The infrastructure now supports exact step-boundary continuation for newly
written checkpoints through explicit `--resume-from`. This does not recover the
deleted historical step-1659 run: its checkpoint predates the trainer-state
payload. The representative four-rank control/resume receipt and measured
checkpoint cost are recorded in
`2026-08-02-infra-optimization-handoff.md`, with a machine-readable comparison
receipt beside it. Cache ownership/admission remains the next optimization
package. The original diagnostic checkpoints predate final full-payload
attestation and are retained as immutable evidence, not as future resume
sources.

Promotion of a model-quality claim from the sorted arm remains scoped as a
compound representation-plus-ordering experiment unless a matched control is
added later.

## Completed Sorted Production and Val-200 Evaluation

The fresh sorted production run on `test-train` completed all 5,529 of 5,529
optimizer steps at `2026-08-04T00:52:38.254093+00:00`. Its terminal state is
`completed`, the final finite status is `finite`, and the final optimizer update
status is `applied`. The run took approximately 16 hours 41 minutes from the
recorded creation time. The final `step-5529` checkpoint was copied back to the
matching local output path, and all 16 files matched their remote SHA-256
digests.

The user-authorized val-200 rollout used the HF backend on eight GPUs with batch
size 4 per worker, `max_new_tokens: 3084`, repetition penalty 1.10,
temperature 0, and the exact trained eight-coordinate prompt. All 200 rows
decoded successfully with `im_end`; there were zero truncations, parser
failures, dropped predictions, and score failures. The scored artifact contains
1,407 predictions for 1,600 ground-truth objects.

The primary geometry evaluator rasterizes each predicted clockwise
quadrilateral and each rectangular GT object as COCO segmentation masks and
uses `iouType="segm"`. It therefore measures mask IoU directly and does not
replace a quadrilateral with its enclosing bounding box. The legacy bbox
metrics remain only as a secondary compatibility diagnostic.

| Metric | Val-200 result |
| --- | ---: |
| Quadrilateral mask AP | 0.430679 |
| Quadrilateral mask AP50 | 0.595571 |
| Quadrilateral mask AP75 | 0.461091 |
| Quadrilateral mask AP small | 0.126981 |
| Quadrilateral mask AP medium | 0.280590 |
| Quadrilateral mask AP large | 0.567920 |
| Quadrilateral mask AR100 | 0.522518 |

The mask and compatibility bbox metrics are numerically equal on this slice
because 1,405 of 1,407 decoded polygons are exact axis-aligned rectangles. The
remaining two differ from a canonical rectangle by at most one coordinate bin
out of 1,000. This equality is a property of the model outputs, not an evaluator
fallback: a focused regression test uses a skewed diamond whose enclosing bbox
matches the GT exactly while its mask IoU is 0.5.

Evidence:

- inference config:
  `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_xy_quad_clockwise_pure_ce_typegate_dora_r16a32_step5529_val200_hf.yaml`;
- inference run:
  `outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-xy-quad-clockwise-pure-ce-typegate-dora-r16a32-step5529-val200-hf-rp1p10-maxnew3084`;
- scored JSONL SHA-256:
  `afb855fa6a65a1e02d79fb0ecc330daeb8295b2dda835157b3b4c2e7411c685a`;
- evaluation metrics SHA-256:
  `ee481368eb1ebf25fd7bb3cfabc541fe2df56460bb987bd810b40e6f1180bfe7`.

This result establishes that the eight-coordinate sorted training, HF rollout,
quad parsing, and direct mask-IoU evaluation path work end to end. It does not
yet establish a causal advantage over four-coordinate supervision because the
representation and ordering changed together and no matched control has been
run.
