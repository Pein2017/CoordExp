# research-base-v3 receipt (task 7.3)

- Tag object: `52815950b5aa26becf2b4cb69c16d57bb3612519`
- Peeled commit: `b7c9ae4407f7d4d03c7eef4f917b7b26af049b7f` (`docs(openspec): record src/ entropy audit (report-only)`)
- Created: 2026-08-28T04:09:34Z on `research-probes` at `/data/CoordExp/.worktrees/research-probes`
- Message: research-base-v3: post-reclaim research main (reclaim-research-probes-lifecycle; 86 scaffold scripts + 48 test modules deleted, 30 lanes retired; replay anchor for deleted producers remains research-base-v2)

## What it carries relative to research-base-v2 (`8dac2d041`)

- `git diff --stat research-base-v2..research-base-v3 -- scripts tests configs | tail -1`: 134 files changed, 72807 deletions(-)
- `scripts/research/*.py`: 315 → 229; `tests/research` modules: 230 → 185
- Lifecycle docs rewritten (`docs/BRANCH_AND_WORKTREE_POLICY.md`, `docs/AGENT_INDEX.md`, `docs/PROJECT_CONTEXT.md`, `openspec/config.yaml`); five completed changes archived; one requirement synced into `coordexp-swift-research-probe-admission`.
- Research records: N/K factorial unit returned; four idle investigations under `research/archive/`; five root orphans under `docs/history/root-orphans/`.
- CPU baseline at close (from wave-5 batch 2): 85 failed / 4815 passed / 2 skipped, zero new failures vs the 88-failure pre-change baseline (3 failures disappeared with a deleted test module).

## Worktrees at tag time

```
/data/CoordExp                                                     29e368144 [main]
/data/CoordExp/.worktrees/CoordExp-swift                           8d12eab28 [coordexp-swift]
/data/CoordExp/.worktrees/codex-rtk-correctness-first              38b30ebc1 [codex/rtk-correctness-first]
/data/CoordExp/.worktrees/codex-wake-me-up-event-monitor           8dfb8102a [codex/wake-me-up-event-monitor]
/data/CoordExp/.worktrees/image2299-mechanism-microscope           60a0b25a1 [codex/image2299-mechanism-microscope]
/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation  477b376a3 (detached HEAD)
/data/CoordExp/.worktrees/research-probe-infras                    c5867df0d [research-probe-infras] locked
/data/CoordExp/.worktrees/research-probes                          b7c9ae440 [research-probes] locked
```

Tags: 18 `probe-final/*`, 14 `archive/*`, research-base-v1/v2/v3. Local branches: 8.
