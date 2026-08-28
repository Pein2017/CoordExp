## Context

See `proposal.md` for motivation. Facts frozen on 2026-08-28 (all verified with `git`, not from documents):

- `research-probes` HEAD `f775ff2c3`, working tree clean (0 dirty). Tags `research-base-v1` → `258868571`, `research-base-v2` → `8dac2d041` (1 commit behind HEAD, no governance record anywhere). `probe-final/*`: none.
- `research-probes` and `main` diverged at `5f1680d4e` (2026-07-12): 382 vs 156 commits, 60+ `src/` files differ. Neither production line is a research base or a merge source; unchanged by this change.
- Integration lane `.worktrees/research-probe-infras` is on `codex/research-probe-infra-foundation` @ `74609d2b1`, a strict ancestor of `research-probes` (2 behind). A stale unmounted branch literally named `research-probe-infras` @ `62274a97d` (2 old AGENTS syncs, 240 behind) collides with the directory name. Docs contradict on whether the lane retires after merge.
- Admission owner `src/artifacts/research_probe_admission.py` binds worktree root + full commit + clean status and revalidates at one pre-model choke point (44 CPU-only tests across `tests/artifacts/`, `tests/research/`). It does not consult tags.
- `scripts/research/`: 315 tracked scripts. Coarse reference scan over `src/ scripts/ tests/ research/ docs/`: 38 referenced nowhere, 87 referenced by exactly one place (typically their own test), 190 referenced by ≥2 places (research-record replay citations and shared mechanics such as `human13_*`, `build_human13_k_union_manifest` at 37 refs).
- Live direction worktree `image2299-mechanism-microscope` (`codex/image2299-mechanism-microscope`, 75 dirty, active processes): 32 new probe-local scripts, no shared `src/` modification, 40 research records not yet returned. It is the reference specimen of the intended fork model and is out of scope for retirement.
- Retirement candidates (every one verified `git status --short | wc -l == 0`):
  - Bucket A, `probe-final/<name>` (0 files absent from `research-probes`): `codex/human13-runner` 832dd63f8, `codex/human13-analyzer` 68fafe2fb, `codex/human13-live-model` a6bfb1c94, `codex/human13-loss-census` a4cb7e89e, `codex/human13-manifest-collector` 10c37852e, `codex/human13-materializer-launcher` fd2a7542b, `codex/human13-discovery-adapter` 4d3d01800, `codex/rp-crossover-analyzer` b796b5ebb, `codex/rp-crossover-integration` efc57dc11, `codex/rp-crossover-launcher` 78d0069d0, `codex/rp-crossover-live-integration` 7cb17a832, `codex/rp-crossover-materializer` dbc36730c, `codex/rp-crossover-production` e7e373037, `codex/rp-crossover-runtime` 2c5632e10, `codex/rp-crossover-wave5-correction` 51d518f75. Worktrees at `/data/CoordExp/.worktrees/<name>`.
  - Bucket B: `codex/human13-nk-factorial-probe` a904e3ae3 (result closed via `memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md`; `contract.md` and code never returned; 6 orphan `tmux wait-for human13-gpu{0,1}-stage{1,2,3}-ok` processes, PIDs 1641167 1641172 1641176 1641180 1641183 1641188, ~4d17h); `probe/human13-standalone-recovery` accd80ded (byte-identical re-commit of the same code, worktree `research-probe-human13-standalone-recovery`); `codex/human13-scientific-fast-path` bed82651d (branch only, route retired); stale branch `research-probe-infras` 62274a97d; Codex-app detached worktree `/data/CoordExp/.codex/worktrees/3f15/research-probes` @ b36216f10 (commit already on the nk-factorial branch); detached `/tmp/coordexp-base-check2` @ 2a297a93a.
  - Bucket C, `archive/<name>` (65–8890 files absent from `research-probes`, untriaged): `codex/coverage-ledger-mechanistic-probing` 77acee47c, `codex/vllm-mechanistic-round-gaussian-rps` cd9c92211, `codex/permutation-bundle-coordinate-noise-pilot` 9ddbe4e91, `codex/regionlock-simplified-pointer` 444cd7ba3, `codex/ledger-auxiliary-loss` 91cddea1a (has `origin/` upstream), `codex/prefix-denoising-sft` b3919b49f (worktree `geometry-aware-denoising-sft`, has `origin/` upstream), `codex/owner-commit-binding` 210ad8c0c, `codex/permanent-owner-bridge` 84fc6b318, and branch-only `codex/qwen3-vl-painted-gt-transcription-probe` f90381b15, `codex/historical-markdown-recovery` 93deed826, `codex/8-coords-bbox` e7d3724b3, `codex/continue-handoff-session` 2fde14b37.
  - Not touched: `image2299-mechanism-microscope`, `permanent-owner-bridge-cache-validation` (detached HOLD, dirty 1), `/data/CoordExp` `main` (dirty 8, not ours), `CoordExp-swift`, `codex-wake-me-up-event-monitor`, `codex-rtk-correctness-first`, `codex-start-if-idle-turn-guard`, all `origin/*` refs.
- Records: `research/archive/` and `research/mechanisms/` hold only an `index.md`; four root-level reports (`2026-06-23_autoregressive_binding_template_study_synthesis_report.md`, `COCO_TEST_BENCHMARK_HANDOFF.md`, `self-distillation.md`, `大文件同步.md`) and `research/2026-07-29-iterative-forced-continue-extreme-capacity.md` have zero inbound references; investigations `pi-lightweight-worker-ablation`, `coordexp-swift-physical-length`, `autoregressive-binding-template-study`, `docs-vs-mcp-authority` have no commits since 2026-07-23 or earlier. `docs/catalog.yaml` (123 entries, all paths resolve) must follow any move. `progress/` (231) and `reference/` (1038) are inherited from `main` and cited by 171 docs: keep.
- `memories/` is the documented continuity layer (`memories/README.md`: "`research/` owns research interpretation and executed evidence"); the research-flow closeout gate already orders result → experiment router → decision/compass → `memories/current.md`. No new memory rule is needed; the N/K factorial case simply skipped the gate.
- Artifact root `/data/CoordExp/outputs/research/` (82 GB) is the repository-root, cross-worktree evidence root; it is not part of any worktree tree and is untouched.

## Goals / Non-Goals

**Goals:**

- One research main, many direction-level forks; each lifecycle step has one owner and one mechanical form (a git command or a checklist line), executed once for real in this change.
- Every deletion or retirement is reversible from a tag or a commit; no evidence is lost.
- Net reduction in files, tests, concepts, and worktrees that the research main must keep coherent, measured and reported.

**Non-Goals:**

- No merge, rebase, or reconciliation between `research-probes` and `main`/`coordexp-swift`.
- No change to admission, artifact, config, or loss behavior; no new lifecycle tooling or CLI.
- No retirement of `image2299-mechanism-microscope`, no triage of Bucket C content, no off-host backup claims, no GC or reflog expiry.
- No `src/` deletion without a separate user approval of the audit report.

## Decisions

**D1. Fork from `research-probes` HEAD; tags are milestones, not gates.**
The admission owner already binds commit and clean status; the tag rule only added ceremony, and `research-base-v2` was cut without any of the v1 governance apparatus, which shows the ceremony does not survive contact with real use. `research-base-vN` is cut after a reusable-mechanics merge into `research-probes` (not after records-only returns) and recorded in a one-paragraph receipt. Alternative rejected: keep "newest tag only" and automate tagging — adds a tool for a rule that has no consumer.

**D2. Direction-level forks.**
A fork is a research direction (weeks, many units, e.g. image2299), not a ticket. Branch `probe/<direction>`, directory `.worktrees/<direction>`. Existing `codex/`-prefixed direction branches are not renamed. A direction worktree does not sync from `research-probes` during its life unless it needs a new mechanic (`git merge research-probes` on demand); records-only returns from other directions never affect it.

**D3. Records-only return.**
Return = `git checkout probe/<direction> -- research/<unit-dirs>` plus hand-merge of shared routers (`experiments/index.md`, `compass.md`, `research/index.md`, decisions), committed on `research-probes`. Shared routers are append-merged by hand because a direction worktree edits them with divergent content (image2299 already diverges on all three). Alternative rejected: `git merge` then delete code — imports the scaffolding the policy says to leave behind and creates conflict noise in 300+ files.

**D4. Code promotion path is unrestricted; default is experiment-local.**
Either `probe → research-probes` or `probe → research-probe-infras → research-probes` is acceptable; the agent chooses per case. The policy states only the default and the "real second consumer" condition already in force.

**D5. Retirement representation: annotated tag, then remove.**
`probe-final/<name>` marks a lane whose lifecycle completed (all content returned, or its result returned by this change). `archive/<name>` marks an untriaged pre-baseline lane whose content is preserved but not returned. Sequence per lane: verify `git -C <wt> status --short` is empty → `git tag -a <ns>/<name> <tip> -m "<one line>"` → `git worktree remove <wt>` → `git branch -D <branch>`. Recovery is `git branch <branch> <tag>^{}` + `git worktree add`. Remote-tracking refs are never touched. Alternative rejected: keep branches, remove only worktrees — leaves the ref list as noisy as before and nobody reads 40 branches.

**D6. Delete scaffolding from HEAD; the tag is the archive.**
Per `reclaim-code-entropy`: a scanner produces candidates; only consumer, ownership, history, and verification evidence justify deletion. Candidate classes for `scripts/research/`:
- *zero-reference* (no import, test, doc, or record cites it) → delete;
- *test-only* (only its dedicated test cites it) → delete script and test, plus configs cited only by them;
- *record-cited* (a `research/` unit names it as replay entry or evidence producer) → **keep**; it is an evidence obligation, not entropy;
- *shared mechanic* (imported by another production script or `src/`) → keep.
Consumer search covers `research-probes` **and** the live `image2299-mechanism-microscope` working tree (tracked and untracked), since that direction imports shared modules. Each batch carries the compact evidence record (`candidate / evidence / cut / tradeoff / verify`) in `receipts/scripts-entropy-ledger.md`, one commit per batch, narrow tests between batches. Replay of any deleted producer binds to `research-base-v2`, which is the last tag containing every file; the policy records this. Alternative rejected: `git mv` to `scripts/research/archive/` — same tree size, every citing path changes, and nothing is actually removed from the maintenance surface.

**D7. `src/` is audit-first.**
The same evidence format produces a ranked report over the 60+ `src/` files unique to `research-probes` and any other zero-consumer surface. Silent-corruption surfaces (masking, supervision positions, loss accounting, parity) are reported but never cut without the user naming them. Cuts execute only after user approval, in the same batch-and-test form as D6.

**D8. Integration lane is permanent and named after its directory.**
Sequence: `git tag -a archive/research-probe-infras-62274a97d 62274a97d` → `git branch -D research-probe-infras` → `git branch -m codex/research-probe-infra-foundation research-probe-infras` (executed from the lane's own worktree). After this change's final commit the lane is fast-forwarded (`git merge --ff-only research-probes`); if a fast-forward is impossible because the lane moved, a normal merge is used and recorded. This closes the "retired after merge" vs "never a retirement target" contradiction in favour of permanence and updates the HOLD row from `establish-research-probes-baseline-v1` task 1.4 without editing that (to be archived) change.

**D9. Concurrent-agent commit protocol inside one worktree.**
Multiple subagents edit the same `research-probes` checkout. Rules: stage by explicit path only (`git add -- <paths>`), never `git add .`/`-A`; at most one lane commits at a time — the scripts-entropy lane (D6) commits its own batches; docs, records, and retirement lanes leave edits unstaged and the lead commits them with explicit paths; on `index.lock` contention, wait and retry rather than delete the lock; no lane touches paths outside its declared write surface.

**D10. Verification is failure-set based.**
Before any deletion, record the CPU-only baseline (`CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest tests/research tests/artifacts -q -p no:cacheprovider`) as a failure set; after each batch compare failure sets, never pass counts. The 44 admission tests must remain green throughout.

## Risks / Trade-offs

- [A deleted script is a replay entry that the coarse scan missed] → record-cited class is decided by a full-text search over `research/`, `docs/`, `memories/`, and both worktrees before each batch; the deletion commit message names the batch ledger; recovery is `git checkout research-base-v2 -- <path>`.
- [Worktree removal discards uncommitted work] → dirty count re-verified immediately before each `worktree remove`; any non-zero count stops that lane and is reported.
- [`git worktree remove` refuses because of untracked/ignored files] → use `--force` only after the dirty re-check passes; ignored caches are not evidence.
- [Killing `tmux wait-for` orphans affects a live job] → the six PIDs are 4d17h old `wait-for` blockers whose worktree has no other process; verified again with `ps` before `kill`.
- [Deleting `.serena/cache` under a running Serena process for this worktree] → check `ps` for `serena` with cwd `research-probes`; cache is regenerable, logs are not evidence.
- [Two agents race on the index] → D9; the lead serialises commits.
- [Fast-forward of the integration lane fails] → D8 fallback merge, recorded in the receipt.
- [`docs/catalog.yaml` paths break after moves] → gate runs a path-resolution check over every `path:` entry.
- [Deleting the stale `research-probe-infras` branch contradicts the frozen HOLD in baseline-v1] → tag first (evidence preserved), and this design records the supersession explicitly.
- [Remote branches `origin/codex/ledger-auxiliary-loss`, `origin/codex/prefix-denoising-sft` diverge from the archived tips] → local tag captures the local tip; remotes are untouched and out of scope.

## Migration Plan

Execution order is the task list: freeze → docs → retirement → records → scripts entropy → `src/` audit → close. Each wave ends with a gate the lead reviews. Rollback: every tree change is a commit (`git revert`); every ref retirement is recoverable from its tag; deleted ignored residue is regenerable. No step depends on a remote or an off-host resource.
