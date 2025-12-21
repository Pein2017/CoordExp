# Change: Add coord-token-safe augmentation path

## Why
- Coord-token datasets currently crash in augmentation because geometry is parsed as strings; we disable augmentation as a workaround.
- We need augmentation to work identically for `<|coord_*|>` geometries without altering downstream chat/template behaviour.

## What Changes
- Add a coord-token-aware pre/post adapter around augmentation that converts tokens → ints before ops and ints → tokens after.
- Gate by `custom.coord_tokens.enabled` and reuse existing codec (range fixed to 0–999).
- Preserve existing numeric behaviour and defaults when coord tokens are off.

## Impact
- Affected specs: coord-token-mode (new augmentation requirements).
- Affected code: dataset preprocessors/augmentation, geometry utils (conversion helpers), small tests/smokes.
