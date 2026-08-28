## 1. Freeze inputs and baseline

- [ ] 1.1 Record in `receipts/inputs.md`: `research-probes` HEAD SHA, `git status --short | wc -l == 0`, the three retirement buckets with tip SHAs exactly as listed in `design.md`, and a fresh `git -C <wt> status --short | wc -l` per worktree target. Any non-zero count removes that target from its bucket and is reported.
- [ ] 1.2 Run `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest tests/research tests/artifacts -q -p no:cacheprovider` and store the failure set (test node ids, not counts) and runtime in `receipts/test-baseline.md`.

## 2. Lifecycle and routing documents (Phase 0)

- [ ] 2.1 Rewrite the "Research-probe routing" section of `docs/BRANCH_AND_WORKTREE_POLICY.md` to D1–D5 and D8: fork from `research-probes` HEAD into `probe/<direction>` at `.worktrees/<direction>`; no mid-life sync obligation; records-only return with the hand-merged shared routers named; code promotion default experiment-local with either route allowed; `research-base-vN` as post-mechanics-merge milestone; the tag-then-remove retirement sequence with `probe-final/` vs `archive/` namespaces; `research-base-v2` as the replay anchor for scaffolding deleted by this change. Delete the "before the first such tag exists" fallback text and the `probe/<ticket>` wording.
- [ ] 2.2 Apply the same rule to `openspec/config.yaml` context (lines naming the newest-tag rule), `docs/AGENT_INDEX.md` "Current research-probe route", and `docs/PROJECT_CONTEXT.md`; state the integration lane as permanent everywhere it is described.
- [ ] 2.3 Write `receipts/research-base-v2.md`: tag object `960c627c6`, peeled commit `8dac2d041`, tag message, what it carries relative to v1 (execution-root receipt selector), and that it is the last tag containing every `scripts/research/` file before this change's deletions.
- [ ] 2.4 From `/data/CoordExp/.worktrees/research-probe-infras`: `git tag -a archive/research-probe-infras-62274a97d 62274a97d`, `git branch -D research-probe-infras`, `git branch -m codex/research-probe-infra-foundation research-probe-infras`; record the three commands and resulting `git worktree list --porcelain` lines in `receipts/integration-lane.md`.
- [ ] 2.5 Archive the five completed changes (`establish-research-probes-baseline-v1`, `fix-integration-receipt-root-binding`, `add-human13-k-union-greedy-overfit-probe`, `add-human13-on-policy-first-bottleneck-successor`, `add-human13-row-contrast-geometry-preservation-successor`) with `openspec archive`; leave the three in-flight changes alone.
- [ ] 2.G Gate: `openspec validate --all`; `grep -rn 'probe/<ticket>\|Before the first such tag\|newest annotated' docs openspec/config.yaml AGENTS.md` returns nothing; lead reads the diff; lead commits with explicit paths.

## 3. Retire dead lanes (Phase 1; user-approved 2026-08-28)

- [ ] 3.1 Bucket A (15 lanes): for each, re-check dirty == 0, `git tag -a probe-final/<name> <tip> -m "probe-final: <name>; all content returned to research-probes"`, `git worktree remove /data/CoordExp/.worktrees/<name>`, `git branch -D codex/<name>`. Receipt table (name, tip, tag SHA, removed path) in `receipts/retirement.md`.
- [ ] 3.2 Bucket B, non-factorial: `probe-final/human13-standalone-recovery` at accd80ded then remove worktree `research-probe-human13-standalone-recovery` and branch `probe/human13-standalone-recovery`; `probe-final/human13-scientific-fast-path` at bed82651d then `git branch -D`; `git worktree remove` the Codex-app detached checkout `/data/CoordExp/.codex/worktrees/3f15/research-probes` and `/tmp/coordexp-base-check2` (no unique commits); re-verify with `ps` then `kill` the six `tmux wait-for human13-*` PIDs. Append to the receipt.
- [ ] 3.3 Return the N/K factorial result before retiring its lane: create `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-22-human13-owner-credit-nk-factorial/` with `unit.md` (from `codex/human13-nk-factorial-probe:research/2026-08-22-human13-owner-credit-nk-factorial/contract.md`) and `results.md` (from `memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md`), binding artifact roots under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-2{1,3}-human13-*` and commits `32bc918d4`, `b36216f10`, `a904e3ae3`; add the row to `experiments/index.md`; point `memories/current.md` at the unit; follow the research-graph unit frontmatter contract. Then `probe-final/human13-nk-factorial-probe` at a904e3ae3, remove worktree `human13-nk-factorial-probe`, `git branch -D codex/human13-nk-factorial-probe`.
- [ ] 3.4 Bucket C (12 lanes): `git tag -a archive/<name> <tip> -m "archive: untriaged pre-baseline lane; content not returned"`, `git worktree remove` for the 8 with worktrees (`geometry-aware-denoising-sft` for `prefix-denoising-sft`), `git branch -D codex/<name>`. Remote refs untouched. Append to the receipt.
- [ ] 3.G Gate: `git worktree list` shows only `main`, `CoordExp-swift`, `research-probes`, `research-probe-infras`, `image2299-mechanism-microscope`, `permanent-owner-bridge-cache-validation`, `codex-wake-me-up-event-monitor`, `codex-rtk-correctness-first`, `codex-start-if-idle-turn-guard`; `git tag -l 'probe-final/*' 'archive/*'` matches the receipt; every retired tip is reachable (`git merge-base --is-ancestor <tip> <tag>^{}`); lead commits `receipts/` and the new research unit.

## 4. Records hygiene (Phase 2a)

- [ ] 4.1 `git mv` the four zero-inbound root reports and `research/2026-07-29-iterative-forced-continue-extreme-capacity.md` to `docs/history/root-orphans/`; add a two-line `README.md` there naming their origin commits.
- [ ] 4.2 `git mv` `research/investigations/{pi-lightweight-worker-ablation,coordexp-swift-physical-length,autoregressive-binding-template-study,docs-vs-mcp-authority}` to `research/archive/<name>/`; update `research/investigations/index.md`, `research/archive/index.md`, and every `docs/catalog.yaml` `path:` that pointed at them; leave `progress/` and `reference/` untouched.
- [ ] 4.3 After confirming with `ps` that no process has cwd or open files in them, delete ignored residue: `.pi-worker/`, `.serena/logs/`, `$evidence_dir/`, `temp/`. Record freed size in `receipts/residue.md`. `.serena/cache/` is deleted by the lead at close (task 7.1) because a Serena server for this checkout is live during execution.
- [ ] 4.G Gate: a one-off check that every `path:` in `docs/catalog.yaml` and every relative Markdown link in `research/index.md`, `research/investigations/index.md`, `research/archive/index.md` resolves; `grep -rn` for each moved basename outside `docs/history/` and `research/archive/` finds only intentional references; lead commits.

## 5. `scripts/research` entropy reclamation (Phase 2b)

- [ ] 5.1 Build the classified reference ledger for all 315 scripts: for each script, hits split into production import (`src/`, other `scripts/`), dedicated test, research-record citation (`research/`, `memories/`), docs citation, config citation, and hits inside `/data/CoordExp/.worktrees/image2299-mechanism-microscope` (tracked and untracked). Assign class per D6. Write `receipts/scripts-entropy-ledger.md` with one evidence record per delete candidate and the keep list with its reason class.
- [ ] 5.2 Batch 1, zero-reference scripts: delete end to end (script, any config or fixture cited only by it). Run the narrow tests for neighbouring modules, then the baseline command; diff the failure set against `receipts/test-baseline.md`. Commit with explicit paths; message names the ledger batch.
- [ ] 5.3 Batch 2, test-only scripts: delete script, its dedicated test module(s), and configs/fixtures cited only by them. Same verification and commit form.
- [ ] 5.4 Report, do not delete: scripts whose only citation is a closed research record or a docs page. List them in the ledger as HOLD with the citing path so the user can decide whether to rewrite the citation to `research-base-v2` later.
- [ ] 5.5 Residue search for every deleted basename across the tree and the image2299 worktree; `git diff --check`; final failure-set diff; net reduction (files, lines, tests, configs) recorded in the ledger.
- [ ] 5.G Gate: lead reviews the ledger and `git diff --stat research-base-v2..HEAD -- scripts tests configs`; admission tests green; no HOLD item deleted.

## 6. `src/` entropy audit (Phase 2b′, report-only until approved)

- [ ] 6.1 Audit every `src/` module, starting from the 60+ files unique to `research-probes` vs `main` (`git diff --name-only main...research-probes -- src`), for the nine candidate classes in `reclaim-code-entropy`; consumer search covers `src/`, `scripts/`, `tests/`, configs, research replay citations, and the image2299 worktree. Write `receipts/src-entropy-audit.md` ranked by confidence, risk, and net reduction, with silent-corruption surfaces marked report-only.
- [ ] 6.2 Present the audit to the user; execute only user-named cuts, in D6 batch-and-test form, appending to the same receipt.

## 7. Close (Phases 3–4)

- [ ] 7.1 From the integration lane: `git merge --ff-only research-probes`; if refused, `git merge research-probes` and record why in `receipts/integration-lane.md`.
- [ ] 7.2 Refresh `memories/current.md` continuity pointer for the lifecycle change; confirm `research/index.md` and `docs/AGENT_INDEX.md` describe the executed lifecycle, not the designed one.
- [ ] 7.3 `git tag -a research-base-v3 <final reviewed commit> -m "research-base-v3: post-reclaim research main"`; write `receipts/research-base-v3.md` (tag object, peeled commit, net reduction summary, worktree list).
- [ ] 7.G Gate: `openspec validate --all`; `openspec verify`-style review of this change against its tasks; user approval to archive this change.
