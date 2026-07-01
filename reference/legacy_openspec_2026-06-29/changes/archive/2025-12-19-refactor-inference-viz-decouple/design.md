## Context
- Current `vis_tools/vis_coordexp.py` interleaves generation, JSON repair/parsing, scaling, visualization, and eval prep. That creates duplicated coord/token logic and makes inference outputs dependent on visualization side-effects.
- Model outputs 0–999 coord tokens (or ints) in raw text; absolute pixel coords must be derived via width/height. Some generations are malformed/truncated, so per-line robustness is required.
- Evaluation (COCO) already consumes parsed JSONL and needs bbox–poly compatibility (bbox GT vs poly pred). Scores are fixed to 1.0 (greedy decoding); categories remain alias-free and single-image per record.

## Goals
- Stageable pipeline: inference produces the canonical JSONL (mandatory parsed preds, coord_mode=norm1000); evaluation and visualization operate purely on that output without rerunning the model.
- Single source of truth for coord-token decode/encode and norm↔pixel scaling, wired through `src/common/geometry`/`src/common/schemas`; avoid double-scaling via explicit coord_mode metadata.
- Robust per-line JSON emission with aggressive repair and error fields so large jobs never halt on bad generations; retain flawed raw outputs for debugging even when preds are empty.
- Preserve bbox–poly IoU feasibility (line→bbox, bbox→segm helper) and alias-free category mapping; unknown desc bucket remains default.

## Non-Goals
- No change to model architecture or decoding strategy beyond existing decoding params.
- No introduction of confidence estimation; scores stay constant 1.0.
- No multi-image prompting support; single-image contract holds.

- Introduce `coord_mode` field fixed to `norm1000` in inference JSONL (model output is always 0–999); `pixel` may be accepted for future-proofing but not emitted by current runner.
- Split coord logic into two layers: (1) token/text ↔ int 0–999; (2) size-aware scaling/clamp + geometry helpers (bbox/poly/line, line→bbox, bbox→segm), exported via `src/common/geometry`.
- Inference CLI writes one JSON object per line: `{image_id, images, width, height, coord_mode, raw_output, preds, error?, meta}`; parsed preds are mandatory and carry `score: 1.0`; malformed generations still emit a line with `preds: []` and the raw text retained.
- Visualization consumes the inference JSONL (parsed preds preferred; fallback reparsing) and never loads the model.
- Detection evaluator switches to the shared coord module (no direct dependency on `vis_tools`), honoring coord_mode to avoid double scaling, and keeps bbox–poly matching via segmentation fallback; invalid geometries are dropped with counters but recorded in diagnostics.
- Preserve alias-free categories and constant scores; input order defines output order when scores tie.

- Additional CLI could confuse users; mitigate with docs and consistent naming.
- coord_mode mismatch could cause double scaling; mitigate by enforcing norm1000 in the runner, warning on missing/unsupported modes.
- Moving parsing logic risks subtle regressions; mitigate with fixtures/tests covering malformed JSON, degenerate geometries, and bbox–poly IoU paths; retain raw invalid shapes for debugging.

## Open Questions
- Do we need an option to also emit pixel coords alongside norm1000 for debugging (default remains norm1000-only output)?
  No
- Should visualization downsample images or rely on existing sizes? (out of scope unless needed.)
  Rely on existing sizes.