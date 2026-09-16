## Verification receipt

Date: 2026-08-28 UTC

Candidate state: uncommitted worktree changes on `/data/CoordExp` `main`; no push, archive, installation, or Codex session mutation was performed.

### Interface and static gates

- Pre-change RED: four focused tests failed for the intended missing route metrics, CLI flags, duplicate-key rejection, and ambiguity fields.
- `conda run -n ms pytest -q`: 11 passed.
- `conda run -n ms ruff check .`: passed.
- `conda run -n ms ruff format --check .`: 9 files already formatted.
- `conda run -n ms python -m compileall -q codex_usage_ledger tests`: passed.
- Serena/Pyright diagnostics for `attempts.py`, `cli.py`, `pricing.py`, and `tests/test_ledger.py`: no diagnostics.
- Skill `quick_validate.py`: `Skill is valid!`.
- `python -m pip wheel . --no-deps`: built `codex_usage_ledger-0.2.0-py3-none-any.whl` (21,097 bytes) in a temporary directory without installing it.
- `openspec validate improve-codex-usage-ledger-decision-summary --type change --strict --no-interactive`: valid.

### Production-shaped offline audit

Window: inclusive 2026-08-15 through 2026-08-28 over `/data/CoordExp/.codex/sessions`, subagents only, `followup_aware` proxy, local GPT-5.6 Standard price snapshot.

- 559 rollout files seen; 510 records emitted; 0 parse errors.
- 510 fully priced sessions; 512 known route segments; 0 unpriced segments.
- 13 model-effort route pairs with wall-time, measured-token, billable-token, and estimated-cost distributions.
- Price receipt SHA-256 matched the selected TOML: `cb3c1e5da544b43d76f72af812c009ce1742598267191118a7646e185de4f74b`.
- Compact summary: 41,001 bytes; full summary over the same 510 records: 612,092 bytes; reduction: 93.30%.
- Compact and full runs emitted byte-identical detailed JSONL, SHA-256 `e97cbe21164e018aa63aa42f2f2cd98f6bcef16181088618e2223c9f2303baaa`.
- Compact summary SHA-256: `1eaad6d3500fe6b1b4a2f019f8b2e358b77da73c0f31d05ad2259bc8f42a7deb`.
- Full summary SHA-256: `8072c8de2cdb058866af9e610b8ec6704874ce6f414a3cf4c4fcdeeca71978de`.
- A 5-file real-session full-summary smoke restored both legacy arrays and produced five strict-outcome template rows.

Raw rollout-derived reports remain temporary under `/tmp` and are not added to Git because they may contain session detail. This receipt records only aggregate counts, sizes, and hashes.

### Claim boundary

This verifies offline extraction, summary compatibility mode, price binding, fail-closed inputs, and decision-field production on the observed session snapshot. It does not establish task comparability, semantic acceptance, invoice accuracy, or a global model ranking; strict cost per accepted task still requires explicit outcomes.
