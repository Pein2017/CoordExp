# Artifact index — 2026-09-18-untied-active-readout-geometry

- `result.json`: bounded result and status; `terminal.json`: cost and ended-job witness. Candidate is not root acceptance.
- `static/static.json`, `static/static-weights.pt`: effective input/output rows, deltas, normalization gain, static spectra and identities.
- `events.json`: frozen source-order panel, exact prefixes/roles/tokens and source bindings.
- `native-captures/{tied,untied}-original/receipt.json`, `weights.pt`, per-event `.pt`/`.pt.reduced.pt`: full-vocabulary logits, pre/postnorm head input,28-layer current-position residual/attention/MLP vectors and exact replay checks.
- `summarize.py`, `summary.json`, `layer-summary.json`: CPU reducer and compact per-event/layer summaries. `reconstruction-recheck.json`: earlier independent replay.
- `runtime.json`, `native-captures/*.exit`: cost/exit evidence. No full KV or attention-matrix archive.
