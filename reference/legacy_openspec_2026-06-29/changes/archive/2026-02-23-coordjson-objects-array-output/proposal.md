## Why

CoordExp currently trains and evaluates dense detection/captioning by asking the model to emit an “assistant JSON” structure keyed by `object_{n}` (e.g., `"object_1": {...}`), with geometry represented as quoted coord-token strings (e.g., `"<|coord_123|>"`). This structure is suboptimal for autoregressive VLMs (e.g., Qwen-VL family) for two research reasons:

- The `object_{n}` keyed dict shape adds avoidable structural tokens and introduces key-order pitfalls (e.g., appearance vs numeric ordering), complicating controlled ordering ablations.
- Quoted coord-token strings add punctuation tokens and make `<|coord_k|>` less “atomic” in the emitted sequence. For the intended detection research setting, we want the model to emit bare `<|coord_k|>` coord tokens in geometry arrays so they behave like dedicated discrete coordinate symbols.
- The current format is unnecessarily expensive in **token budget**, especially for dense scenes and poly supervision:
  - `"object_{n}"` repeats a long key per instance; the overhead grows with the number of objects and with the number of digits in `n`.
  - Quoting every coordinate as a JSON string adds repeated punctuation characters around each `<|coord_k|>` literal; this overhead scales with the number of coordinates (`O(#coords)`), making it particularly wasteful for `poly`.
  - The extra structural tokens compete directly with semantic capacity (geometry + `desc`) under fixed `max_new_tokens` / packing budgets, increasing truncation frequency and reducing the number of fully-formed records available for matching and supervision.

This change introduces a model-facing CoordJSON format (coord tokens as bare literals) and a deterministic conversion step into strict RFC JSON used by the existing training/inference pipeline. The goals are to (a) reduce structural token overhead so more budget goes to geometry + `desc`, and (b) enable clean ablations without rewriting Hungarian matching/eval/loss code.

## What Changes

- **BREAKING**: Assistant output container changes from top-level `"object_{n}"` keys to a single top-level object containing an `"objects"` array: `{"objects": [...]}`
- **BREAKING**: Assistant outputs become **CoordJSON** (JSON-like DSL) where geometry arrays (`bbox_2d` / `poly`) contain bare coord-token literals (e.g., `<|coord_12|>`) rather than quoted JSON strings.
- Introduce a CoordJSON → strict RFC JSON transpiler that converts coord-token literals to integer coordinate bins (0–999) and produces valid JSON for downstream `json.loads`.
- Tighten strictness split:
  - Cooked SFT/GT data MUST fail-fast on any contract violation (record-level).
  - Model rollout predictions use salvage parsing: drop invalid/truncated records, keep valid prefix.
- Remove `repeat_terminate` / max-object gating: rely on sequence length truncation and salvage parsing for partial tails.
- Update documentation (`progress/full_idea.md`, data contract docs) and tooling (chat-template inspection / eval scripts) to reflect the new container and CoordJSON semantics.

## Capabilities

### New Capabilities
- `coordjson-output-format`: Define CoordJSON (model-facing) and strict JSON (pipeline-facing) representations, the coord-token literal rules, and the CoordJSON→strict JSON transpiler contract (strict vs salvage modes).

### Modified Capabilities
- `object-field-ordering`: Replace `object_{n}` keyed assumptions with `objects[]` array semantics; keep `custom.object_field_order ∈ {desc_first, geometry_first}` for ablation and require deterministic canonical serialization in emitted assistant text.
- `rollout-matching-sft`: Update parsing/validation to consume `{"objects": [...]}` and to run through the CoordJSON transpiler; update salvage behavior and remove repeat-terminate references.
- `stage2-ab-training`: Update stage-2 Channel-B assumptions to use `objects[]` indices (pred_index) instead of `object_{n}` keys; remove repeat-terminate semantics and keep existing matching/reordered-GT logic.

## Impact

- Data cooking / templates: assistant target rendering and deterministic serialization.
- Parsing: strict GT validation and salvage rollout parsing via CoordJSON transpilation.
- Stage-2 (Channel-B): list-based Hungarian matching inputs and stable `pred_index` alignment.
- Eval / visualization / inspection tooling: iterate `objects[]` instead of `object_{n}`.
- Documentation: revise examples, invariants, and failure-mode expectations around the new format.
