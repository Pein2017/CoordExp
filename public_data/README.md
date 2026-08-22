# COCO and LVIS preparation

The retained public-data surface prepares COCO 2017 and LVIS v1 only. All
commands run from the repository root and write only to explicitly selected
dataset/output roots.

```bash
./public_data/run.sh coco download
./public_data/run.sh coco convert
./public_data/run.sh coco all --preset rescale_32_1024_bbox
./public_data/run.sh lvis download
./public_data/run.sh lvis convert
./public_data/run.sh lvis all --preset rescale_32_768_bbox
python -m public_data.scripts.build_coco_views --help
python -m public_data.provenance --help
```

The shared pipeline produces pixel-space `*.jsonl`, norm1000 integer
`*.norm.jsonl`, and Qwen coordinate-token `*.coord.jsonl`. Provenance manifests
hash model-facing JSONL only; image trees are never part of routine checksums.
