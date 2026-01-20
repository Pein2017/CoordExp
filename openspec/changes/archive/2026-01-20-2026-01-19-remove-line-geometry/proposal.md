# Change: Remove `line` Geometry Support (BBox/Poly Only)

## Why
- `line` geometry is not supported by the stage_2 rollout-matching trainer (only `bbox_2d|poly` are parsed/matched), but the repo-level prompts/docs currently mention `line`, which can silently degrade training by encouraging unsupported outputs.
- Removing `line` reduces surface area, ambiguity, and maintenance cost, and keeps the research stack aligned with what is actually trained/evaluated today.

## What Changes
- **Contract + prompting**: remove `line` / `line_points` from the documented JSONL contract and from all prompts/examples.
- **Runtime schemas + geometry utilities**: remove the `line` geometry kind and any polyline helpers; keep only `bbox_2d` and `poly`.
- **Preprocessing / augmentation**: remove polyline handling and any `skip_if_line` behaviors; encountering `line` in inputs becomes a clear validation failure (or explicit drop with counter where appropriate).
- **Inference / evaluation / visualization**: accept and emit only `bbox_2d|poly`. Any `line` objects in predictions are treated as invalid geometry and excluded.

## Impact
- **BREAKING**: any JSONL (GT or predictions) containing `line` must be converted to `bbox_2d` or `poly` offline before use.
- Affected areas: `src/common/*`, `src/datasets/*`, `src/eval/*`, `src/infer/*`, `vis_tools/*`, `docs/*`, `tests/*`, and OpenSpec specs.

