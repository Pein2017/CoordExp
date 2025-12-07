# Design: Coord Token Modules

## Context
- JSONL contract today: geometry fields are numeric pixel coords; templates normalize to norm1000 and the text path may emit `emit_norm` converted numbers.
- Research goal: run training where the assistant text already contains `<|coord_k|>` tokens and the raw JSONL geometry may be stored as those tokens. We still need numeric forms for geometry losses and eval.
- We must not break the existing numeric path; instead add a gated “coord-token mode.”
- Token range decision: coord vocab is currently expanded to `<|coord_0|>` .. `<|coord_999|>` (no 1000 by default; optional if needed).

## Proposed Components (under `src/`)
1) **coord_tokens/codec.py**
   - Bidirectional mapping: token string ↔ int k ↔ normalized float k/1000.
   - Helper to detect tokenized coord lists and convert to numeric tensors.
   - Mask builder to identify coord token ids for CE masking / logit restriction.

2) **coord_tokens/validator.py**
   - Lightweight checks for tokenized geometry arrays in JSONL objects.
   - Ensures width/height metadata present to recover pixels when needed.
   - Reuses existing geometry bounds where possible; adds token-aware branch.

3) **coord_tokens/template_adapter.py**
   - Toggle to bypass template `normalize_bbox` when data already quantized to norm1000 tokens.
   - Keeps the standard template path for numeric datasets; only activates via config flag (e.g., `coord_tokens.enabled: true`).

4) **tools/convert_to_coord_tokens.py** (or `scripts/coord_tokens/convert.py`)
   - Offline converter: numeric bbox/poly/line → coord-token arrays in assistant text and/or geometry fields, preserving a numeric copy for losses.
   - Deterministic rounding rule: round(x/width*1000) consistent with current pipeline.

5) **loss helpers** (extend existing loss module or new `coord_tokens/loss.py`)
   - Expectation decoder over coord-token logits.
   - Shared util for CE masking and CoordExp/GIoU numeric targets derived from tokens.

## Control Flow (coord-token mode)
- Loader reads JSONL; validator accepts tokenized geometries; codec produces numeric tensors for losses.
- Template adapter skips bbox normalization, assuming incoming coords are already norm1000 tokens; text path keeps tokens unchanged.
- Loss path uses codec outputs to compute CE/CoordExp/GIoU.
- Reverse mapping at inference: token → k → pixel using current image size (post-resize if any).

## Backward Compatibility
- Default remains numeric; coord-token mode is opt-in via config flag and template selection.
- Converter is offline; existing numeric datasets keep working unchanged.

## Open Questions
- Should geometry fields in JSONL remain numeric while assistant text uses tokens, or allow token geometry too? (Plan: support both; prefer keeping numeric copy for loss reliability.)
- Where to register the template toggle (config schema vs runtime flag)?
