# CoordExp research workset

Current research authority is the registered `research-probes` checkout. This
repository retains the maintained train/infer/evaluation core, two reusable
research operators and distilled scientific knowledge, not every historical run.

Start at [research/index.md](research/index.md) for current questions and decisions,
[probes/README.md](probes/README.md) for retained operators, or
[docs/README.md](docs/README.md) for engineering interfaces.

## Maintained entries

| Task | Entry |
|---|---|
| Training | `python -m src.train --config <current-yaml>` |
| Inference | `python -m src.infer --config <current-yaml>` |
| Saved detection evaluation | `python -m scripts.evaluate_detection --help` |
| Saved visualization | `python -m scripts.visualize_detection --help` |
| Explicit-state output QP | `python -m probes.output_qp --help` |
| Output-only norm rescaling | `python -m probes.readout_norm --help` |
| Knowledge integrity | `python -m scripts.check_research_knowledge check` |

`src/` owns common execution, geometry, losses, artifacts and integrity.
`probes/` owns the two explicit numerical methods, not a runner registry.
`tests/` protects retained contracts; tests for retired capabilities retire with
them. `configs/coordexp_infras` contains current production/qualification inputs.
External datasets and outputs are not a cleanup target.

New decision-bearing training and operator qualification require clean Git source
identity. Old receipts lacking a verifiable current source gate cannot resume;
requalification is a new run, not an old hash replacement. Exact detailed history
is recovered through the [catalog](research/experiments/catalog.jsonl), not an
archive in HEAD. No GPU/model replay is implied by the CPU test suite.

Run offline CPU checks with CUDA hidden, one-thread math libraries and
`python -B -m pytest -q -p no:cacheprovider`. Configuration and real model resources
must be selected explicitly before any separately authorized model execution.

Public-data recovery is a maintained exception to historical factory retirement:
[the recovery contracts](manifests/public_data_provenance/README.md) bind current
COCO inputs, their minimal annotation delta and a tested raw-ZIP restore path.
