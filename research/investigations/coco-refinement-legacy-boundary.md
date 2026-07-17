# Standalone COCO Refinement Legacy Boundary

## Status

This receipt freezes the rollback boundary accepted for the standalone
`coco-refinement` change on 2026-07-17. It is provenance, not current product
authority. The active contract is
`openspec/changes/coco-refinement/`.

## Git Boundary

- Parent legacy implementation baseline:
  `38141540b126d0364c6a017ccef607d3ed918918` (`Advance managed ROI source pin`).
- Parent standalone OpenSpec baseline:
  `5f0bec2a4e3e76f4123b9c93e9323f44c01687df`.
- Nested Label Studio repository HEAD:
  `4fd750eafae199711de99ef9c0f04c66fefa8fb7`.
- Local nested archive branch:
  `codex/archive-label-studio-coco-refinement`, pointing at that exact nested
  HEAD.
- The parent repository does not track `label-studio/` as a Git tree entry;
  the nested identity is therefore recorded explicitly instead of being
  inferred from a parent submodule pointer.

The superseded `label-studio-coco-refinement` OpenSpec remains unarchived at
28/37 completed tasks. Its checked implementation and receipts remain useful
historical evidence, but its unfinished tasks are not acceptance evidence for
the standalone editor.

## Preserved Runtime Boundary

- The legacy process still owns numeric loopback port 8080 as PID `1240251`
  using the nested repository's `serve_coordexp_refinement` command.
- A 2026-07-17 read-only request observed HTTP 502 while the process and socket
  remained present. No restart, repair, state migration, or disposition is
  implied by this receipt.
- The standalone service must use a different port, runtime root, SQLite file,
  and Draft namespace until explicit user acceptance.

## Cypress and Untracked Harness Boundary

`git diff --exit-code -- web/apps/labelstudio-e2e/cypress.config.ts` passed in
the nested repository, proving that the tracked Cypress configuration matches
`4fd750eafae199711de99ef9c0f04c66fefa8fb7`.

The following unfinished legacy-only harness files are untracked in the nested
repository and must remain outside every commit:

- `label_studio/coordexp_refinement/tests/browser_e2e_fixture.py`
- `web/apps/labelstudio-e2e/biome.json`
- `web/apps/labelstudio-e2e/src/e2e/coordexp-managed.cy.ts`

AgentGuard rejected their deletion. They are deliberately retained as local
residue until the user runs the manual cleanup command recorded by final task
5.3. Agents must not retry deletion, work around the guard, or stage them.

## Protected Data Boundary

The standalone runtime starts from the exact max_len12000 train/validation
sources and shared image roots. It does not import Label Studio Drafts, mutate
the source JSONL, copy source images, or write into the legacy runtime root.
Stopping the standalone service before acceptance is the complete rollback;
it does not require changing the legacy repository, process, or source data.
