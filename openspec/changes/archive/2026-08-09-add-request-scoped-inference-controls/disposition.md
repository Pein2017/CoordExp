# Disposition

Status: `superseded_abandoned_without_spec_sync`

The user selected retirement on 2026-08-09. Tasks `2.3`, `5.2`, and `5.4`
remain intentionally unchecked:

- The old request-scoped sampled execution path is no longer present in the
  current inference architecture, so its availability gate was not completed.
- The durable CUDA attestation at
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-next-row-probability-transition-and-causal-source-trace/runtime-attestation-current-v1/request-scoped-sampling-three-policy-cuda.json`
  is negative for the cross-cardinality requirement rather than acceptance
  evidence.
- The required current independent audit closure was not obtained for the old
  implementation.

The delta specs assume the superseded Hugging Face-only implementation and
include calibration-instance details that are incompatible with the current
first-class Hugging Face and vLLM architecture. They are therefore archived
with `--skip-specs`. A future need for request-scoped sampling or assistant
continuation must start as a new change against the current backend contracts;
this archive does not restore the removed implementation or claim that its
runtime gate passed.
