# Verification Matrix

Use this when a CoordExp routing task turns into an edit, audit proof, or handoff.

## By Surface

- Python implementation: `python -m py_compile <touched.py>` plus targeted `python -m pytest <test>`.
- YAML/config leaf: parse or resolve the YAML and run the smallest config/schema test.
- OpenSpec/stable contract: validate the exact active change or stable specs strictly when they changed.
- Infer/eval artifact: check `resolved_config.json`, `summary.json`, `metrics.json`, raw/scored artifact names, coordinate surface, and scope label.
- `public_data`: use the provenance-manifest test and JSONL checksum contract.
- Docs-only: run `git diff --check` and link every changed behavior claim to current docs/spec/code.
- Commit hygiene: inspect dirty state, stage only intentional paths, and run `git diff --cached --check`.

## Evidence Handles To Report

- docs/spec file;
- config key and resolved config path;
- symbol or module path;
- artifact root and manifest/summary file;
- metric file and scope label;
- command run and result;
- skipped verification with reason.
