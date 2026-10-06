# Public-data checks

From the repository root, the retained converter runner is:

```bash
bash public_data/tests/run_tests.sh
```

The script selects the `ms` Python interpreter and exercises LVIS presorting,
annotation/geometry conversion, polygon caps, COCO conversion/validation and
the pipeline factory. Its source owns the exact test selection.

For a focused fixture check, use the owning test directly, for example:

```bash
conda run -n ms python -m pytest -q public_data/tests/test_poly_cap.py
```

Some LVIS checks consume `public_data/lvis/raw/annotations/lvis_v1_train.json`
and skip when it is absent; they do not need downloaded images. Synthetic
polygon-cap checks use temporary fixtures. Converter checks are CPU evidence;
they do not qualify full-dataset regeneration or model quality.

The [data contract](../../docs/data/contract.md) owns stored geometry and image
resolution semantics. Dataset preparation and the separate full processing
boundary are described in [the public-data README](../README.md).
