# Artifact index — 2026-09-18-untied-highconfidence18-natural

- `result.json`: bounded result and status; `terminal.json`: cost and ended-job witness. Candidate is not root acceptance.
- `panel.json`, `sources/`, `shared-gate.json`, `input-audit.json`: model/data/runtime identity and real-entry gate.
- `runtime/{condition}/{group}/raw.json`, `trace.json`, `receipt.json`: all580 outputs, tokens/text/EOS/cap, per-step shadow winners/top2/logsumexp and raw chosen scores, exact source/companion identities.
- `reduction.json`: parsed boxes, owner G/L/retention, all annotation views, errors/endpoints, secondary scores, bootstrap and shadow accounting. `reduce.py` reproduces it; `reduction-recheck.json` is JSON-exact.
- `physical-review/events.json`, `final-review.json`, image/context PNGs:30 bounded events; UNKNOWN and HOLD preserved.
- `verification.json`: exact checked bindings, source snapshot resolutions, exits/PIDs. `integrated-terminal.json`: four-unit candidate.
- `weights/`: effective E/U and norm factors, saved once per producer; detailed selected states/logits are owned by B. No comprehensive KV/attention archive.
- `closeout.py`, `write_records.py`: CPU integration, no GPU calls. `main-*.log/.exit`, qualification logs and original runtime-handoff.json retain execution history; terminal supersedes the old live handoff.
