# Lane C artifact index

Unit: `2026-09-19-recurrence-conditional-mass`.

Owning research record: `/data/CoordExp/.worktrees/research-probes/research/experiments/2026-09-19-recurrence-conditional-mass/results.md`.
The full initial scientific result remains at `results.md` in this output
root. Its pre-CPU-update version is preserved as `results.pre-cpu-update.md`.

The mechanical native qualification is bound to the accepted
`untied-417044-failure` selection and uses the mature untied loader:

- `qualification/qualification.json`
- `qualification/run-manifest.json`
- `qualification/self-consistency.json`
- `qualification-prebatch4/` preserves the corrected-boundary qualification
  before the native bs4 source-score parity check.
- `qualification-prechunk4/` preserves the corrected-boundary bs4-qualified
  record before the scientific batch-8 stream was frozen.
- `qualification-preboundary/` preserves the earlier before-source-boundary
  mechanical attempt; it is excluded from all Lane C denominators.

The final scientific run consumed Lane A's frozen panel and sources by exact
path and hash. It completed 45/45 states with 256 draws each (11,520 total);
the qualification is not included in that denominator. The primary artifacts
are:

- `draws.jsonl` — all raw token sequences, token log-probabilities, stream
  seeds/offsets, conditioning prefixes and event metadata.
- `run-manifest.json` and `run-progress.json` — bound panel/source hashes,
  state IDs, draw counts and runtime cost.
- `sampler-entry.json` — first counted batch-8 native sampler/readback receipt.
- `state-bindings.json` — split-aware image identity, condition, source and
  panel-stratum bindings for every state.
- `reduction.json` — standalone CPU replay output.
- `results.md` — initial scientific reduction, per-state q/intervals, greedy
  event audit, strata and interpretation limits.
- `results.pre-cpu-update.md` — byte-preserved result before the uncertainty
  wording and owning-record closeout update.
- `launch-failures.json` — preserved failed-closed attempts; none contributed
  scientific draws.

CPU acceptance commands:

```text
python -m probes.training_set_completion.recurrence_mass.reduce --self-test
python -m probes.training_set_completion.recurrence_mass.self_consistency --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-conditional-mass/qualification/self-consistency.json
python -m py_compile probes/training_set_completion/recurrence_mass/run.py probes/training_set_completion/recurrence_mass/reduce.py probes/training_set_completion/recurrence_mass/self_consistency.py
```

The parent reducer must replay every raw token sequence in `draws.jsonl`, with
all 256 attempts per completed state in the denominator. Its image-level
summary uses the bound split-aware `source_example_id`; numeric image IDs are
retained as labels only. Root acceptance remains pending independent replay of
the raw draws, panel/source hashes and strata. Seven prospective proxy states
have a distinct next-row description and an empty same-description historical
union, making q structurally zero under the frozen conditioning; this is
recorded in `results.md` and must not be read as model suppression. Raw draw
records omit `source_policy`; use `state-bindings.json` plus the frozen panel
and source manifests for the canonical policy/source crosswalk.
