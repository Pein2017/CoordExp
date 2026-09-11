## Why

CoordExp research inference cannot currently distinguish request-owned sampled
decoding from batch- or process-owned randomness, nor can it continue inside an
already-open assistant response without risking a changed chat boundary. Those
gaps can silently invalidate paired experiments even when generated outputs and
run-level manifests look plausible, so the stable inference boundary needs
small reusable controls before research orchestration is implemented.

## What Changes

- Extend the backend-neutral decode request with an explicit tagged generation
  policy for greedy or sampled decoding and an optional request-owned sampling
  seed. Greedy decoding remains the production default.
- Require sampled execution to use request-scoped random-number-generator state
  whose output is invariant to compatible batch scheduling and request order
  when results are joined by request identity.
- Return an immutable per-request execution receipt with each decode result so
  executed generation arguments, sampling seed, prompt identity, backend
  identity, stop reason, and output identity remain structurally bound.
- Add an optional assistant-continuation prompt seam that appends canonical text
  inside the current open assistant turn, retokenizes the complete prompt, and
  records exact byte, token, span, and fingerprint evidence.
- Preserve ordinary prompt construction and deterministic greedy inference when
  the new controls are absent.
- Add installed-runtime attestation for four-request sampled batches, including
  same-seed replay, reversed scheduling, distinct request-to-generator seed
  mapping, and receipt binding. Scientific output-diversity acceptance remains
  owned by the frozen research calibration panel. A source-only or fake-model
  test cannot establish these runtime semantics.
- Add the user-authorized, score-preserving custom Hugging Face sampling branch
  inside the existing batched backend because the installed stock sampling
  path does not accept one random-number generator per request. The branch
  replaces only categorical token selection, reuses the installed generation
  preparation/processors/stopping/cache path, and remains unavailable until its
  four-request executable attestation passes.
- Sanitize and receipt the complete effective generation profile so model-owned
  defaults such as top-k sampling cannot silently alter the declared
  temperature-plus-nucleus-sampling protocol.
- Keep experiment-specific cohort selection, tile and masked-full-canvas
  construction, seed derivation namespaces, arm scheduling, calibration,
  merging, matching, non-maximum suppression, metrics, and scientific decision
  thresholds outside the stable inference interface.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-infras-infer-backend-trace`: Add request-scoped greedy or sampled
  generation policy, request-owned sampling seeds, schedule-invariance
  requirements, and result-bound decode execution receipts.
- `coordexp-infras-infer-prompt-parsing`: Add a generic open-assistant
  continuation contract with full-prompt retokenization, exact continuation
  evidence, forbidden-boundary validation, and no-continuation compatibility.

## Impact

- `src/inference/backend.py` and its tests gain typed request generation controls,
  request-scoped random-number-generator execution, and immutable execution
  receipts.
- `src/inference/prompt.py` and its tests gain the optional assistant-
  continuation input and exact prompt-boundary evidence.
- The existing public inference configuration, production presets, raw and
  scored evaluator row schemas, greedy defaults, Hugging Face backend ownership,
  no-resize image semantics, and per-device `generation.batch_size` semantics
  remain unchanged.
- Research runners under `scripts/research/` and reusable analysis code under
  `src/analysis/` may consume these controls later, but that orchestration is not
  part of this stable compatibility change and this proposal alone does not make
  the spatial-scope experiment execution-ready.
