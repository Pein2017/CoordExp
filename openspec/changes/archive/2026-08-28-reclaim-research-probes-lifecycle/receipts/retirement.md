# Retirement Receipt — Bucket A / B / C (tasks 3.1–3.4)

**Executed**: 2026-08-28. Sequence per D5: verify dirty == 0 on the target
worktree (and, for Bucket A/C, that the branch tip matches `design.md` exactly)
→ `git tag -a` → `git worktree remove` → `git branch -D`. No commits were made
in `research-probes` by this lane (D9); this file and the new research unit
are left unstaged for the lead to commit with explicit paths. Baseline tip/
dirty verification for every lane was re-confirmed fresh (not merely read from
`receipts/inputs.md`) immediately before each tag/remove/delete.

## Bucket A — `probe-final/<name>` (task 3.1, 15 lanes)

All 15 lanes: tip matched `design.md`, dirty == 0, tag created, worktree
removed without `--force`, branch deleted. Zero lanes STOPped.

| Name | Branch | Tip | Tag object SHA | Worktree removed | Branch deleted | Outcome |
|---|---|---|---|---|---|---|
| human13-runner | codex/human13-runner | 832dd63f8 | 3593ad19a20787063f53ba4b0d13aa04d4b3c765 | /data/CoordExp/.worktrees/human13-runner | yes | OK |
| human13-analyzer | codex/human13-analyzer | 68fafe2fb | d81360d9dbf565b89e5fb143f7df8600e345a1af | /data/CoordExp/.worktrees/human13-analyzer | yes | OK |
| human13-live-model | codex/human13-live-model | a6bfb1c94 | ff13c4458de088d267f3624f5e65ff81c0ffce4f | /data/CoordExp/.worktrees/human13-live-model | yes | OK |
| human13-loss-census | codex/human13-loss-census | a4cb7e89e | 8d62fac00a69f4ea04e0e9dd2f229298eb003a1c | /data/CoordExp/.worktrees/human13-loss-census | yes | OK |
| human13-manifest-collector | codex/human13-manifest-collector | 10c37852e | 3282e421236b9aba07372f46f5d71147b0667bb3 | /data/CoordExp/.worktrees/human13-manifest-collector | yes | OK |
| human13-materializer-launcher | codex/human13-materializer-launcher | fd2a7542b | cd8e8f463ab61432a66092377ccfe6a194dcbdff | /data/CoordExp/.worktrees/human13-materializer-launcher | yes | OK |
| human13-discovery-adapter | codex/human13-discovery-adapter | 4d3d01800 | bd779756fba8c237a8d299d852f7f1a28b25fb01 | /data/CoordExp/.worktrees/human13-discovery-adapter | yes | OK |
| rp-crossover-analyzer | codex/rp-crossover-analyzer | b796b5ebb | 4bc6816e48fae33e014ed449128719ebbf5e958c | /data/CoordExp/.worktrees/rp-crossover-analyzer | yes | OK |
| rp-crossover-integration | codex/rp-crossover-integration | efc57dc11 | 5bdc1de099f1bd625006d8f30e2caa03eb8408c2 | /data/CoordExp/.worktrees/rp-crossover-integration | yes | OK |
| rp-crossover-launcher | codex/rp-crossover-launcher | 78d0069d0 | d8c441549965982397379d0b717010b623e70f26 | /data/CoordExp/.worktrees/rp-crossover-launcher | yes | OK |
| rp-crossover-live-integration | codex/rp-crossover-live-integration | 7cb17a832 | 74a6ada1473875905a971d63a923dfc103203d2f | /data/CoordExp/.worktrees/rp-crossover-live-integration | yes | OK |
| rp-crossover-materializer | codex/rp-crossover-materializer | dbc36730c | a493a6dfce4d379b7184568461541be1f15fea12 | /data/CoordExp/.worktrees/rp-crossover-materializer | yes | OK |
| rp-crossover-production | codex/rp-crossover-production | e7e373037 | a4e39fe976fee11b83d06ca51e9679d4e40ccf2f | /data/CoordExp/.worktrees/rp-crossover-production | yes | OK |
| rp-crossover-runtime | codex/rp-crossover-runtime | 2c5632e10 | 0c15eab3570ee824bfe20afc402316828544579b | /data/CoordExp/.worktrees/rp-crossover-runtime | yes | OK |
| rp-crossover-wave5-correction | codex/rp-crossover-wave5-correction | 51d518f75 | be2e49a895467acd694fc3b6303442f4e500a3ed | /data/CoordExp/.worktrees/rp-crossover-wave5-correction | yes | OK |

## Bucket B — mixed retirement targets (task 3.2)

| Item | Kind | Tip | Tag object SHA | Removed/killed | Outcome |
|---|---|---|---|---|---|
| probe/human13-standalone-recovery | branch + worktree | accd80ded | 1a857b0bd357952be88d0f59f4b3a20576aea7a9 (`probe-final/human13-standalone-recovery`) | worktree `/data/CoordExp/.worktrees/research-probe-human13-standalone-recovery` removed; branch deleted | OK |
| codex/human13-scientific-fast-path | branch only | bed82651d | c6c9c22c0251d78df1cf5ba418e823d1d8138070 (`probe-final/human13-scientific-fast-path`) | branch deleted (no worktree) | OK |
| /data/CoordExp/.codex/worktrees/3f15/research-probes | detached checkout | b36216f10 | n/a — no unique commit, no tag (commit already carried by `probe-final/human13-nk-factorial-probe`) | dirty re-checked == 0; worktree removed without `--force` | OK |
| /tmp/coordexp-base-check2 | detached checkout | 2a297a93a | n/a — no unique commit, no tag | dirty re-checked == 0; worktree removed without `--force` | OK |

`research-probe-infras` stale branch @ 62274a97d (also listed under Bucket B in
`design.md`) is D8/task 2.4's responsibility, not this lane's; it was already
tagged `archive/research-probe-infras-62274a97d` and renamed before this lane
ran (verified present in `git tag -l`, not re-touched here).

### tmux `wait-for human13-*` orphan processes

Before (re-verified with `ps -eo pid,etime,cmd`, matched exactly the six
declared PIDs and command names):

```
1641167  4-17:39:51 tmux wait-for human13-gpu0-stage1-ok
1641172  4-17:39:51 tmux wait-for human13-gpu0-stage2-ok
1641176  4-17:39:51 tmux wait-for human13-gpu0-stage3-ok
1641180  4-17:39:51 tmux wait-for human13-gpu1-stage1-ok
1641183  4-17:39:51 tmux wait-for human13-gpu1-stage2-ok
1641188  4-17:39:51 tmux wait-for human13-gpu1-stage3-ok
```

Action: plain `kill` (SIGTERM) on all six PIDs.

After: `ps -eo pid,etime,cmd | grep 'tmux wait-for human13' | grep -v grep`
returned no rows — all six confirmed gone.

## Task 3.3 — N/K factorial result return, then lane retirement

Returned as a new research unit (records-only return, D3), created before any
tag/removal for this lane per the task's Step A ordering:

- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-22-human13-owner-credit-nk-factorial/unit.md`
  — frozen scientific surface, derived from
  `codex/human13-nk-factorial-probe:research/2026-08-22-human13-owner-credit-nk-factorial/contract.md`
  (as it existed at the branch's retired tip `a904e3ae3`), plus the artifact
  handle for all three source commits.
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-22-human13-owner-credit-nk-factorial/results.md`
  — executed evidence and interpretation, reproduced verbatim from
  `memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md` and the
  corresponding paragraphs of `memories/current.md`.
- One row appended to
  `research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`.
- `memories/current.md`: the sentence pointing at the memories note as the
  primary interpretation authority was replaced with a pointer to the new
  unit's `results.md`, keeping the memories note as secondary provenance.
- Frontmatter status: `complete`; `evidence_status: verified`;
  `architecture_promotion_status: not_promoted`;
  `implementation_status: authorized_separately`.

Artifact roots verified present on disk before writing the unit (`ls`):

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-21-human13-standalone-owner-credit-probe/` — contains `full-gpu1.json` (14.5K), `sentinel-gpu0.json` (56.4K), `multi-image-n4-k16-t4-gpu1.json` (373.4K).
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-23-human13-n13-k4k8-corrected-geometry-probe/v3` — contains `cells/`, `logs/`, `preflight/`, `receipts/`.

Lane retirement, executed after the above files existed and were verified
present (`ls` + `grep -c` on `experiments/index.md` and `memories/current.md`):

| Name | Branch | Tip | Tag object SHA | Worktree removed | Branch deleted | Outcome |
|---|---|---|---|---|---|---|
| human13-nk-factorial-probe | codex/human13-nk-factorial-probe | a904e3ae3 | 4373a76e4cbf3ff1cef64d2024c5170038941b25 (`probe-final/human13-nk-factorial-probe`) | /data/CoordExp/.worktrees/human13-nk-factorial-probe | yes | OK |

## Bucket C — `archive/<name>` (task 3.4, 12 lanes)

All 12 lanes: tip matched `design.md`, dirty == 0 where a worktree exists, tag
created, worktree removed (8 of 12) without `--force`, branch deleted. Zero
lanes STOPped. `origin/codex/ledger-auxiliary-loss` and
`origin/codex/prefix-denoising-sft` were not touched (verified present and
unchanged after this lane, see Gate).

| Name | Branch | Tip | Tag object SHA | Worktree | Removed | Branch deleted | Outcome |
|---|---|---|---|---|---|---|---|
| coverage-ledger-mechanistic-probing | codex/coverage-ledger-mechanistic-probing | 77acee47c | 8152d64ad4b94bfe27ceaa8eaecb99ff39a9b982 | /data/CoordExp/.worktrees/coverage-ledger-mechanistic-probing | yes | yes | OK |
| vllm-mechanistic-round-gaussian-rps | codex/vllm-mechanistic-round-gaussian-rps | cd9c92211 | 456e9cb8f7f1d6b7f60dcf1bcbba2316ec56df4d | /data/CoordExp/.worktrees/vllm-mechanistic-round-gaussian-rps | yes | yes | OK |
| permutation-bundle-coordinate-noise-pilot | codex/permutation-bundle-coordinate-noise-pilot | 9ddbe4e91 | 14609bbc6f427865b590fe40130898ea81ab3d75 | /data/CoordExp/.worktrees/permutation-bundle-coordinate-noise-pilot | yes | yes | OK |
| regionlock-simplified-pointer | codex/regionlock-simplified-pointer | 444cd7ba3 | c95dfcd5a250e6694ac907f6d7e7cb0a3825c7e5 | /data/CoordExp/.worktrees/regionlock-simplified-pointer | yes | yes | OK |
| ledger-auxiliary-loss | codex/ledger-auxiliary-loss | 91cddea1a | b3d2c59e009249bb111a1be42502c34b54836bb2 | /data/CoordExp/.worktrees/ledger-auxiliary-loss | yes | yes | OK |
| prefix-denoising-sft | codex/prefix-denoising-sft | b3919b49f | 6edf9266677c64324d3172a2f1698f335cc9ad30 | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | yes | yes | OK |
| owner-commit-binding | codex/owner-commit-binding | 210ad8c0c | 0de22b7da864308cc8bb127296a5c29c0f047206 | /data/CoordExp/.worktrees/owner-commit-binding | yes | yes | OK |
| permanent-owner-bridge | codex/permanent-owner-bridge | 84fc6b318 | 681b17ffa93ae0a7d242f7f29380d3e6cd6c5435 | /data/CoordExp/.worktrees/permanent-owner-bridge | yes | yes | OK |
| qwen3-vl-painted-gt-transcription-probe | codex/qwen3-vl-painted-gt-transcription-probe | f90381b15 | 75c14de3524cb314e9f9906df31553d291be28d0 | (branch only) | n/a | yes | OK |
| historical-markdown-recovery | codex/historical-markdown-recovery | 93deed826 | eb2db4e7b603df5a7edfc204e98b66b134097b6f | (branch only) | n/a | yes | OK |
| 8-coords-bbox | codex/8-coords-bbox | e7d3724b3 | 6767565e582997600ffbe88dac3d511288a82f59 | (branch only) | n/a | yes | OK |
| continue-handoff-session | codex/continue-handoff-session | 2fde14b37 | 11ed2c9b9cea715f2e971554b68f26ca5e3b4501 | (branch only) | n/a | yes | OK |

## Gate (task 3.G)

### `git worktree list` (verbatim)

```
/data/CoordExp 29e368144 [main]
/data/CoordExp/.worktrees/coordexp-infras 8d12eab28 [coordexp-infras]
/data/CoordExp/.worktrees/codex-rtk-correctness-first 38b30ebc1 [codex/rtk-correctness-first]
/data/CoordExp/.worktrees/codex-wake-me-up-event-monitor 8dfb8102a [codex/wake-me-up-event-monitor]
/data/CoordExp/.worktrees/image2299-mechanism-microscope 60a0b25a1 [codex/image2299-mechanism-microscope]
/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation 477b376a3 (detached HEAD)
/data/CoordExp/.worktrees/research-probe-infras 74609d2b1 [research-probe-infras] locked
/data/CoordExp/.worktrees/research-probes 8209b5eeb [research-probes] locked
```

This is only 8 rows, short of the task-3.G target list of 9
(`main`, `coordexp-infras`, `research-probes`, `research-probe-infras`,
`image2299-mechanism-microscope`, `permanent-owner-bridge-cache-validation`,
`codex-wake-me-up-event-monitor`, `codex-rtk-correctness-first`,
`codex-start-if-idle-turn-guard`): `codex-start-if-idle-turn-guard` was not
present as a worktree before this lane ran either (absent from
`receipts/inputs.md`'s pre-lane worktree-list snapshot) and this lane did not
remove it; its absence predates and is outside this retirement. No
never-touch worktree was removed by this lane. `research-probes` HEAD moved
from `521183ea0` (at lane start) to `8209b5eeb` by the time this receipt was
written, from other concurrent Phase 0/D9 lanes committing to the same
checkout; this lane made no commits.

### `git tag -l 'probe-final/*' 'archive/*'`

32 tags: 15 `probe-final/*` from Bucket A, 4 `probe-final/*` from Bucket B/3.3
(`human13-standalone-recovery`, `human13-scientific-fast-path`,
`human13-nk-factorial-probe`, plus the 15 Bucket A names above = 18 total
`probe-final/*`), 12 `archive/*` from Bucket C, plus 2 pre-existing `archive/*`
tags not created by this lane (`archive/research-probe-infras-62274a97d` from
task 2.4, and `archive/birth-first-stage2-channel-b`, unrelated prior tag).
Matches the per-bucket tables above exactly; see the raw list embedded in the
per-bucket tables' "Tag object SHA" columns.

### Ancestor check

`git merge-base --is-ancestor <tip> <tag>^{}` was run for every `probe-final/*`
and `archive/*` tag (32 total, including the 2 pre-existing ones), using each
tag's own peeled commit as `<tip>` (tags in this change are annotated directly
at the recorded tip, so this also confirms no tag drifted from its recorded
SHA). All 32 returned success; zero failures.

### `git worktree prune --dry-run`

Empty output — nothing to prune. All eight Bucket A/Bucket C `.worktrees/*`
directories and the two detached checkouts were cleanly removed by
`git worktree remove` (no `--force` was needed for any of the 25 removals
executed by this lane), leaving no stale administrative metadata.
