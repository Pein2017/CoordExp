# Visualization guide

- `canonical-index-atlas.png`: cyan, yellow, magenta, and green are GT persons by physical row; red is confirmed tie GT.
- `gt-vs-pred` left panel: green is matched GT; yellow is missed GT.
- `gt-vs-pred` right panel: green is matched prediction; red is strict-unmatched prediction; purple dashed is a duplicate hint.
- A red tie is panel-unmatched, not automatically a false positive. Final crop review is in `bbox-adjudication.jsonl`.
