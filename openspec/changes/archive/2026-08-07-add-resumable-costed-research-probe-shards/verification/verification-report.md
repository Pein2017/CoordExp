# Verification report

## Accepted fixed point

- Infrastructure tests: `75 passed`.
- Current unchanged active-unit downstream regressions: `48 passed` (`8`
  merger, `10` census-v3, `30` analyzer).
- Ruff, compileall, `git diff --check`, and strict OpenSpec validation passed;
  strict OpenSpec validation reported `24 passed, 0 failed` after main-spec
  synchronization.
- CPU deterministic fresh-process interruption equivalence passed for all eight
  canonical legacy receipts; the unchanged merger accepted `220` records.
- The final single-GPU v7 mechanics run bound adapter
  `9e2005bf127956ecfec496fdf6a652e3f11d34ad58e859da19c361de65111033`
  and worker
  `400f85905791015dabb1cf22788b0e03ac5ab4922db125eb24410aaae53dc79b`.
  Parent-observed external `SIGTERM` returned `-15` after one durable context;
  a separately invoked exact-identity continuation returned `0`, ran only the
  missing context, preserved the first record bytes, materialized the bounded
  terminal, and released GPU 1.
- Independent Sol/high standards and intent-contract audits both returned
  `PASS` with no P0/P1 blockers after the materializer boundary repair and v7
  re-smoke.

## Compatibility and provenance

- `source-bindings-v4.json`, `compatibility-gate-v4.json`, and
  `nonmutation-v4.json` bind the latest external active-unit evolution. The
  consumer and merger identities used by the adapter remain unchanged; the
  active owner expanded materializer/analyzer coverage from 41 to 48 passing
  tests during this work.
- No file was edited, staged, merged, cherry-picked, or committed in the active
  `research-probes` worktree by this change. Old sealed roots and failed or
  superseded v1-v6 mechanics roots remain preserved and are not recovered or
  reinterpreted.
- Main specs were synchronized at the user's corrected direction before
  archive: journal diagnostics were added to the existing execution-evidence
  spec, and the natural-boundary support-shards capability was created.

## Claim boundary

These receipts establish resumability, identity, durable publication,
deterministic scheduling/materialization, compatibility, and process mechanics.
They do not establish support prevalence, a model mechanism, recovered v3/v4
results, or a scientific outcome. No push, merge, or cherry-pick is authorized
or performed.
