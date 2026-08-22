# Operational entrypoints

Use module entrypoints for training and inference:

- `python -m src.train --config <config>`
- `python -m src.infer --config <config>`

The remaining operational scripts are Python modules, not shell wrappers:

- `python -m scripts.evaluate_detection --help`
- `python -m scripts.visualize_detection --help`

Historical experiments are documentation-only under `research/`; they are not
part of the default runtime or test surface.
