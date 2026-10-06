# LVIS annotation visualization

`visualize_lvis.py` displays original LVIS annotations and their COCO images.
It is useful for inspecting visible masks and bounding boxes before conversion.
It does not visualize a processed CoordExp JSONL or model predictions.

Run from `public_data/`:

```bash
conda run -n ms python vis_tools/visualize_lvis.py --help
conda run -n ms python vis_tools/visualize_lvis.py \
  --split val --mode both --num_samples 2 --save \
  --output_dir /absolute/path/to/an-owned-visualization-directory
```

Inputs are `lvis/raw/annotations/lvis_v1_<split>.json` and extracted images in
`lvis/raw/images/<split>2017/`. Without `--save`, matplotlib needs a usable
display. `--seed` makes sampling repeatable; the source and CLI own defaults.

Polygon display is intentionally limited: it draws only the first polygon part
of an instance and caps the number of displayed polygons. A picture is therefore
not a complete segmentation export or proof of the converter's hull/cap policy.
See [LVIS geometry modes](../lvis/README.md) for those conversion decisions.
