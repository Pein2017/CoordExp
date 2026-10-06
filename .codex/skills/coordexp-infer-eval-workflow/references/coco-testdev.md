# COCO test-dev benchmark brief

Use only for an explicitly requested official benchmark or historical test-dev
reproduction. This brief is neither execution nor upload authorization. Resolve
the selected checkout's actual input, inference, scoring and export consumers;
a historical config file does not prove its launcher still exists.

## Distinct outcomes

A local validation score, a validated ready-to-submit export, and an official
server result are different deliverables. Record the selected checkpoint and
composition, effective configuration, input identity, infer/export artifacts,
closest matched validation result, and official submission response or exact
external blocker in the owning research unit/result. Do not open a progress
ledger or invent a score when credentials, access or submission are unavailable.
Handoff the ready artifact and remaining action when that is the actual endpoint.

## Verify the path before use

Confirm whether the current input consumer accepts label-free test metadata and
how it represents absent annotations; do not invent inline objects or evaluate
missing test labels as negatives. Reuse maintained preparation/export code where
supported. Any missing implementation belongs to a separately scoped code task,
not a command copied from this document.

Keep checkpoint composition, prompt, ordering, resolution, backend, generation
and scoring as close as possible between the validation anchor and test run;
record unavoidable differences. An exporter must bind image IDs, official category
IDs, scores, source dimensions and its exact bbox transform. Check its output on
validation data against a supported export route before trusting a test export.
Local evaluator sidecars are not automatically official submission files.

The historical recipe used offline resize with image factor32, pixel budget
32×32×1024 and minimum32×32×4; it projected resized-image predictions back to
original test image dimensions. These are source-time assumptions, not current
model defaults. Preserve the actual transform and avoid converting twice.

Verify the current official server format/account procedure only when submission
is requested. Preserve its response, submission identifier and returned
AP/AP50/AP75/APs/APm/APl with scope and caveats. A failed upload must not erase a
valid ready-to-submit artifact or be worked around with an unverified exporter.

## Historical scope

The former root handoff and legacy runbook contained planning/commands, not an
executed official result. Their implementation assumptions included now-absent
`scripts/run_infer.py`, `scripts/postop_confidence.py` and
`scripts/export_coco_submission.py`. Original text is Git
`a550899623be64fd9c8019522bb972d0217dd634` at
`COCO_TEST_BENCHMARK_HANDOFF.md` and this reference directory's former
`historical-coco-testdev.md`; it is not a current launch recipe.
