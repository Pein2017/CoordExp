# Lane B artifact handoff

All output paths below are under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-visual-instance-binding`.

Qualification command shape (`qualification-01`, devices 4–7; primary evidence):

```text
python -m probes.training_set_completion.visual_instance_binding.runtime --mode qualify --campaign qualification-01 --state-id <state> --device cuda:<gpu> --output <root>/qualification-01/<state>
```

The retained `scientific-01` command shape (`states/`, devices 4–7) is audit-only
protocol debt and is excluded from all primary denominators and claims:

```text
python -m probes.training_set_completion.visual_instance_binding.runtime --mode state --campaign scientific-01 --state-id <state> --device cuda:<gpu> --output <root>/states/<state>
```

States: `tied-885-first-revisit`, `untied-885-first-revisit`, `tied-14038-first-revisit`, and `tied-14038-control-before-row7`. Qualification and scientific logs are in `logs/qualification-01/` and `logs/scientific/`; all eight receipts are `candidate_complete`.

The exact producer, loader, request, source, admission, budget, wall-limit, image, raw, trace, and receipt bytes used before model load are copied under each state’s `source-snapshot/` and bound by its `source-snapshot.json` and `launch.json`.

Key closure artifacts:

```text
candidate-manifest-v2.json
response-matrices-v2.json
reduction-v2.json
cpu-acceptance-v2.json
wall-start.json
```

The v2 CPU closure compares qualification-01 against the audit-only duplicate
execution across all 64 saved rows and records `max_abs_delta=0.0`. The old
unversioned closure files remain byte-for-byte superseded audit records. CPU
closure commands are recorded in `candidate-manifest-v2.json` and `results.md`.
No release, retry, or further model call was run.
