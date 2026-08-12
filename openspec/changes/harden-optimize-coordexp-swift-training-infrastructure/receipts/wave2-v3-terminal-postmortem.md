# Wave 2 v3 terminal postmortem

Date: 2026-08-10 UTC

This record preserves the interpretation and launch boundary of the sole Wave 2
v3 GPU attempt. It does not modify or supersede the immutable JSON artifacts.

## Immutable artifacts

- Plan: `wave2-v3-plan.json`
  - file SHA-256:
    `141b4294dacae69f39907303d3ae75c4c4182d8da7625b399280ed99ed66187d`
  - authenticated body SHA-256:
    `03834e309357dace9d7b51a6c51186b14b09bea8953b3c3d12e82609cff1a49d`
- Attempt marker: `wave2-v3-attempt-marker.json`
  - file SHA-256:
    `8d1d16404814e95fa5cd9ced4062f9fd6e051ef870bde208d1f7e886b1231d48`
- Terminal receipt: `wave2-v3-terminal-receipt.json`
  - file SHA-256:
    `781838ada548c9c0d9db4dacf1b43b6bc0e767303fa2906006c95a3ce70e6958`
  - terminal status: `failed`
  - failure code: `qwen.parity.clean_failed`
- No publication-failure sidecar was produced.

The marker consumes the only v3 attempt. There is no retry, sample switch,
threshold edit, receipt rewrite, or v4 authorization.

## Accepted executed evidence

- The exact plan, marker, and receipt validate against one another.
- All 138 supervised semantic rows align. Every row's complete 152,670-value
  logit vector is byte-identical between packed-primary, packed-repeat, and the
  streaming-separate reference.
- Total-loss delta is `9.5367431640625e-7`; every mandatory per-term scalar,
  planned-step denominator field, semantic-atom key, and finite-value gate
  passes.
- Packed-primary and packed-repeat are byte-identical over all 589 gradient
  tensors and 20,062,208 compared values. The fixed repeat measurability gate
  passes with maximum absolute difference `0`.
- The real attention proof passes for all 28 expected Qwen text layers, each on
  deterministic FlashAttention varlen with boundaries `[0,1436,2822]` and the
  24 vision events classified separately.
- The boundary-only negative changes supervised logits by up to `35.125` and
  total loss by more than `2.48`; detection does not rely on gradients.
- Host RSS HWM (`8.83 GB`), CUDA reserved HWM (`6.34 GB`), and sampled device
  HWM (`6.89 GB`) remain below the frozen ceilings.

## Retained rejected diagnostic

Both packed arms fail the predeclared packed-versus-streaming gradient
comparison on the same three tensors:

- shared special-token delta: maximum absolute difference `0.2236328125`;
- layer 14 self-attention q-projection DoRA magnitude: `0.0061143041`;
- layer 15 counterpart: `0.0064807832`.

All 589 rows are present, finite, and correctly typed. The two packed arms are
identical, ruling out stale gradients or run-to-run nondeterminism. The separate
arm uses the production two-forward/two-immediate-backward cadence. Independent
reviews found no denominator, `no_sync`, comparator, coverage, or artifact-
binding defect.

The strongest explanation is deterministic BF16 reduction-order sensitivity
between one packed backward and two different-shape streaming backwards. Of the
589 rows, 196 LoRA-A tensors are exact-zero in both arms because the cold-start
LoRA-B tensors are zero; they provide presence coverage but no numerical
discrimination. The result does not prove exact Jacobian equality, and it also
does not demonstrate segment leakage when the complete supervised forward is
byte-identical and the negative control is decisive.

## User-owned disposition

After reviewing the immutable result, the user selected the narrow release
gate:

- accept only the demonstrated forward/logit, objective/denominator,
  boundary-isolation, and all-layer FA2 semantics;
- retain the cross-shape gradient comparison as a failed diagnostic rather than
  a release gate;
- do not widen or reinterpret the frozen band and do not rerun Wave 2;
- release Wave 3 only after the separate rich-failure preflight-evidence bug is
  fixed and independently audited.

## Evidence limitation and follow-up

The terminal receipt preserves phase-boundary and HWM measurements but its
comparison-stage failure serialization omitted the earlier exact GPU-idle
preflight sample subtree. Contemporaneous independent launch audits and command
output established an idle `cuda:0` before launch, but this Markdown record is a
post-run archival transcription, not a replacement for missing immutable JSON
fields. The serializer/validator is repaired for future receipts only; the v3
receipt remains untouched.
