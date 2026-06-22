## Why

The compact detection-template work made closure and separator behavior explicit,
but it also stated too strongly that `detection_template.id` is the only authored
serialization source.  The bbox-first ablation needs a second stable axis:
`custom.object_field_order` should select desc-first versus geometry-first row
layout while the semantic template id continues to select the wrapper family.

This lets us compare bbox-first and desc-first with rich special tokens, sorted
ordering, standard Stage-1 SFT, static packing, and the new chat template without
inventing a parallel compact-format surface.

## What Changes

- Revise the single-source statement:
  - `detection_template.id` remains the authored source for template family,
    closure tokens, row separator, structural token rows, strict parser family,
    prompt wrapper pattern, and artifact template id.
  - `custom.object_field_order` remains the authored source for per-object
    semantic field order and now also controls compact row field order:
    `desc_first` versus `geometry_first`.
- Extend compact template ids to cover the full closure family:
  - `compact`
  - `compact_object_closed`
  - `compact_box_closed`
  - `compact_object_box_closed`
  - `compact_object_box_closed_lines`
- Keep newline behavior registry-based, not a free authored separator knob.
  The existing line variant remains a semantic template id.
- Define compact rendering as the product of two resolved axes:
  - `desc_first`: object-ref segment before box segment.
  - `geometry_first`: box segment before object-ref segment.
- Require strict parsing, prompt examples, target rendering, token-role spans,
  packing/cache fingerprints, and inference/eval artifact provenance to consume
  both resolved axes.
- Keep object instance ordering independent:
  - paired production ablations use `custom.object_ordering: sorted`.
  - `custom.object_field_order` MUST NOT change object instance sequence.
- Add planning for a bbox-first/desc-first final ablation pair using:
  - standard Stage-1 SFT,
  - `custom.detection_template_id: compact_object_box_closed` or the resolved
    equivalent template surface used by the active Stage-1 SFT path,
  - `custom.object_field_order: desc_first` versus `geometry_first`,
  - static packing with `global_max_length: 12000`,
  - LLM-only trainability,
  - checkpoint
    `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`,
  - sorted object ordering.
- Do not implement or launch the bbox-first training run until this proposal,
  specs, and roadmap are reviewed and explicitly approved.

## Capabilities

### New Capabilities

- `compact-template-field-order-ablation`: Defines the two-knob compact
  rendering contract, the closure-family template ids, and the paired
  bbox-first/desc-first ablation contract.

### Modified Capabilities

- `object-field-ordering`: Extend `custom.object_field_order` from JSON object
  key order to compact row field order while preserving object instance order.
- `dataset-prompt-variants`: Include compact template id and object field order
  together in compact prompt examples and prompt hashes.
- `stage1-detection-objectives`: Require standard Stage-1 SFT and compact
  detection teacher-forcing surfaces to render and cache targets from the two
  resolved axes without migrating standard SFT into rollout-only training.
- `token_embeddings_adapter`: Clarify that compact token-row adaptation depends on template
  structural tokens, not on desc-first versus geometry-first ordering.
- `encoded-training-cache`: Require encoded-sample cache fingerprints to differ
  when either compact template id or object field order changes.
- `packing-dataset`: Require static packing fingerprints to include both axes
  and the global packing length.
- `inference-engine`: Persist and consume both axes when materializing compact
  generated text into normalized predictions.
- `shared-inference-runtime`: Carry both axes in prompt/parser provenance and
  backend parity metadata.
- `detection-evaluator`: Require post-hoc mAP preflight to validate both axes as
  artifact provenance while scoring normalized geometry unchanged.
- `training-config-hierarchy`: Keep the ablation configs YAML-first and
  auditable through existing config sections, without new CLI flags.

## Impact

- Affected OpenSpec/docs surfaces include the existing
  `detection-template-variants` planning language, object-field-ordering,
  prompt variants, Stage-1 objective, packing/cache, inference, shared runtime,
  evaluator, and config-hierarchy specs.
- Affected code surfaces include `src/detection/template_contracts.py`,
  `src/detection/template.py`, `src/common/detection_sequence.py`,
  `src/config/prompts.py`, `src/config/schema.py`, `src/config/loader.py`,
  `src/datasets/builders/jsonlines.py`, `src/datasets/dense_caption.py`,
  `src/detection/packing.py`, `src/sft.py`, inference/eval artifact
  materialization, and their tests.
- Correctness risk: if render and parse use different axes, bbox-first output
  can be silently dropped or mis-scored.  Mitigation is strict roundtrip tests
  over template id x field order.
- Reproducibility risk: if caches omit either axis, desc-first and bbox-first
  runs can reuse stale encodings.  Mitigation is fingerprint tests over template
  id, object field order, prompt hash, and `global_max_length`.
- Eval-validity risk: mAP should compare normalized predictions, not raw wrapper
  strings.  Mitigation is artifact preflight that records both axes and keeps
  metric computation on normalized `gt`/`pred` objects.
