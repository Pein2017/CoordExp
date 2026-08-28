# Artifact reclaim manifest (task 8.4)

Executed at 2026-08-28T04:30:35Z, HEAD `b3d3be9b9e7102f7559fb4b225036cf7648556a2`.
Scope per user approval 2026-08-28: reclaim only the UNBOUND "neither a path
citation nor a same-named unit anywhere" subset (41 dirs / ~3.53 GB) and the
BOUND-ARCHIVED class (~1.0 GB) from
`receipts/artifact-root-inventory.md`. Never BOUND-COMPLETE, BOUND-ACTIVE,
`/data/CoordExp/outputs/research-probe-infras/`, or anything the inventory
marked KEEP. The "bound-by-path in image2299" 5.2 GB subset and the
"bound-by-naming-convention only" ~11.9 GB subset are explicitly out of scope
and were not touched or re-checked here.

## Method (independent re-verification, run now, not reused from the inventory)

For each of the 43 candidate roots (41 UNBOUND + 2 BOUND-ARCHIVED):

- (a) `grep -rl -F '<full root path>' research memories docs openspec
  /data/CoordExp/.worktrees/image2299-mechanism-microscope --exclude-dir=.git
  --exclude-dir=outputs` — must be empty for UNBOUND; for BOUND-ARCHIVED must
  hit only under `research/archive/**`, `docs/history/**`,
  `openspec/changes/archive/**`, or this change's own receipts.
- (b) same grep on the root's basename alone (units often cite by name, and a
  path split across markdown lines defeats the single-line grep in (a)); a
  hit inside an active (non-archived) `research/investigations/**` unit or
  `memories/current.md` disqualifies the root.
- (c) `lsof +D <root>` and `ps -eo pid,cmd | grep -F '<root>'` — must be
  empty.
- (d) confirm the path is under `/data/CoordExp/outputs/research/` and is
  not a symlink (`test -L`).

**This re-verification found the inventory's classification was wrong for 3
of the 43 candidates** (all caught by the basename check in (b), because the
original inventory's literal full-path grep missed citations that wrap across
markdown lines):

1. `qwen3-vl-dense-enumeration/2026-07-21-exact-greedy-terminal-rescue-training-screen/`
   — genuinely cited (line-wrapped) by a complete-status, non-archived
   research unit's results.md.
2. `qwen3-vl-dense-enumeration/2026-07-17-next-row-probability-transition-and-causal-source-trace/`
   — the inventory's BOUND-ARCHIVED call rested only on a peripheral hit in
   an archived openspec disposition.md; the real, primary binding is a
   complete-status, non-archived unit in *this* worktree's own
   `research/investigations/` tree. This root is actually BOUND-COMPLETE,
   outside the approved reclaim scope.
3. `pi-lightweight-worker-ablation/` — archived in this worktree's
   `research/` tree, but the live, not-yet-returned
   `image2299-mechanism-microscope` worktree still holds its own
   un-archived copy of the same unit, citing the same shared
   `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/`
   artifact root as evidence.

A fourth root (`logs`, 8K/1 file) was skipped as an unresolved ambiguous
basename match (the word "logs" collides with unrelated prose) rather than
force a judgment call; no literal path citation was found for it anywhere.

All other 39 UNBOUND roots re-verified clean: no path or basename hit apart
from this change's own planning/receipt files, empty lsof/ps, confirmed
under `outputs/research/`, none are symlinks.

## Table — all 43 candidate roots, decision, and outcome

| root | size | newest mtime | files | class | verification (a/b/c/d) | decision | reason |
|---|---:|---|---:|---|---|---|---|
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-30-iterative-forced-continue-exact-native` | 1.7G | 2026-07-30 | 508 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-candidates-six-context-smoke-v2` | 576M | 2026-08-01 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-candidates-final` | 497M | 2026-08-01 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-iterative-forced-continue-extreme-capacity` | 171M | 2026-07-29 | 10 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal full-path hit anywhere. Basename hit in `docs/history/root-orphans/README.md` line 6 is a coincidental name collision with an unrelated retired research *note* (`research/2026-07-29-iterative-forced-continue-extreme-capacity.md`, moved to that root-orphans catalog, commit e9614de25) — not a citation of this artifact directory; also self-hits in this change's own design.md/tasks.md/scripts-entropy-ledger.md/artifact-root-inventory.md (this task's own planning and receipt files, not a research unit). Not disqualifying per rule (no hit inside an active research/investigations/** unit or memories/current.md). lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-exact-greedy-terminal-rescue-training-screen` | 84M | 2026-07-21 | 116 | UNBOUND | (a) clean (b) DISQUALIFYING HIT (c) clean (d) confirmed | **SKIP** | SKIPPED-NOW-BOUND: literal path cited (line-wrapped across two markdown lines, missed by the original single-line full-path grep) in `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-exact-greedy-prefix-uncovered-object-row-training-screen/results.md` (status: complete), as `collector-shards-k128-v1/` and `smoke-image-34855-v2/` evidence subpaths. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-shard-gt7511-22-root-v1` | 70M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-scores-uncached-timing-ctx7511-26-v1` | 46M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-shard-gt7511-26-root-v1` | 46M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-shard-b2-before-v1` | 44M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-shard-b2-after-v1` | 44M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-30-four-checkpoint-human-refined12-max3084` | 40M | 2026-07-30 | 33 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-30-three-checkpoint-coordinate-confidence-visualization-v1` | 16M | 2026-07-30 | 13 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-cohorts-v2-final` | 13M | 2026-08-01 | 5 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-summary-uncached-reviewed-v3` | 6.5M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-control-review-final-r6` | 3.2M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-control-review-final-r5` | 3.2M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-controls-final-r4` | 3.2M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-controls-final-r3` | 3.2M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-stop-rule6-final-r2` | 2.8M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-contexts-v2-stop-rule6-final` | 2.8M | 2026-08-01 | 3 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-summary-uncached-merged-v1` | 1.9M | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-summary-uncached-timing-ctx7511-26-v1` | 432K | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-input-plan-final` | 344K | 2026-08-01 | 4 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-input-plan-six-context-smoke-v2` | 236K | 2026-08-01 | 4 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-input-plan-six-context-smoke-v1` | 236K | 2026-08-01 | 4 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-permutation-bundle-k16-val200` | 232K | 2026-07-29 | 9 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-permutation-bundle-k16-val200-max3084` | 48K | 2026-07-29 | 8 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_final_20260821T094017Z` | 28K | 2026-08-21 | 5 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v5` | 12K | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v6` | 12K | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_final_20260821T091729Z` | 28K | 2026-08-21 | 5 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_genuine_20260821T093409Z` | 28K | 2026-08-21 | 5 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v3` | 12K | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v4` | 12K | 2026-08-02 | 2 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-02-sorted-owner-basin-task4-control-attestation-uncached-reviewed-v2` | 8.0K | 2026-08-02 | 1 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_shape_fix_science_20260821T102033Z` | 28K | 2026-08-21 | 7 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_shape_fix_direct_20260821T102202Z` | 28K | 2026-08-21 | 7 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_shape_fix_pane_20260821T102430Z` | 28K | 2026-08-21 | 7 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_shape_fix_run_20260821T101824Z` | 28K | 2026-08-21 | 7 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/durable-human13_streaming_shape_fix_20260821T101410Z` | 28K | 2026-08-21 | 7 | UNBOUND | (a) clean (b) clean (c) clean (d) confirmed | **DELETE** | Re-verified UNBOUND: no literal path or basename hit anywhere in research/memories/docs/openspec/image2299 except this change's own `artifact-root-inventory.md` (the audit receipt, not a research unit); lsof/ps clean; confirmed under `outputs/research/`, not a symlink. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/logs` | 8.0K | 2026-08-27 | 1 | UNBOUND | (a) clean (b) ambiguous (see reason) (c) clean (d) confirmed | **SKIP** | Ambiguous, not disqualified by a real citation: basename "logs" is a common English word and matches dozens of unrelated active-unit files by coincidence (e.g. "training logs"); zero literal full-path citations of this directory found anywhere. Skipped out of caution rather than force a judgment call on a mechanical check; directory is 8K/1 file so the cost of skipping is negligible. |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-next-row-probability-transition-and-causal-source-trace` | 131M | 2026-07-17 | 419 | BOUND-ARCHIVED | (a) clean-for-class (b) DISQUALIFYING HIT (c) clean (d) confirmed | **SKIP** | SKIPPED-NOW-BOUND: the original BOUND-ARCHIVED call came from a literal-path grep that only caught a peripheral hit in an archived openspec disposition.md. Independent re-verification found the artifact root is the primary evidence path (line-wrapped citations) of `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-17-next-row-probability-transition-and-causal-source-trace/{unit.md,results.md}` — a LIVE (non-archived) unit with `status: complete`, plus its experiments/index.md, topic index.md, overview.md, and a cross-referencing sibling unit. This is properly BOUND-COMPLETE, not BOUND-ARCHIVED, and is out of the approved reclaim scope. |
| `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation` | 877M | 2026-07-23 | 78693 | BOUND-ARCHIVED | (a) clean-for-class (b) DISQUALIFYING HIT (c) clean (d) confirmed | **SKIP** | SKIPPED-NOW-BOUND: this worktree (research-probes) archived the unit to `research/archive/pi-lightweight-worker-ablation/` (commit 4b4ac8d24), which is why the inventory called it BOUND-ARCHIVED here. But the live, not-yet-returned `image2299-mechanism-microscope` worktree still carries its own un-archived copy at `research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/{unit.md,results.md}` (status: complete) that literally cites the same shared filesystem artifact root `/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/` as its evidence store. Deleting it would destroy evidence a still-live direction depends on. |

## Totals

- Candidates in scope: 41 UNBOUND + 2 BOUND-ARCHIVED = 43 roots.
- DELETE: 39 roots (all UNBOUND), ~3.20 GiB (3,438,753,740 bytes measured via
  `du -sb`; ~3.36 GB decimal) before deletion.
- SKIP (SKIPPED-NOW-BOUND): 3 roots — see table for citing paths
  (`2026-07-21-exact-greedy-terminal-rescue-training-screen`,
  `2026-07-17-next-row-probability-transition-and-causal-source-trace`,
  `pi-lightweight-worker-ablation`).
- SKIP (ambiguous, precautionary): 1 root (`logs`, 8K).
- BOUND-ARCHIVED class net result: **0 roots deleted** — both candidates in
  this class were found to be re-bound on independent re-verification.

## Not reclaimed (outside approved classes)

Per task 8.4 step 1's explicit scope wording ("the inventory says 41 dirs /
~3.53 GB"), the following inventory-UNBOUND roots are **not** part of the 41
dirs named for this reclaim and were not touched or independently
re-verified beyond what the inventory already recorded:

- `/data/CoordExp/outputs/research/qwen3-vl-native-text-coordinate-val200/`
  (1.6G, top-level Table 1 UNBOUND — "zero citers found").
- `/data/CoordExp/.worktrees/research-probes/outputs/third_party/` (42M,
  Table 3 UNBOUND — "no hit... found").

These are named here per step 5 ("never widen scope") for the user to name
separately if reclamation is wanted.

## Deletion outcome

All 39 DELETE-decision roots were removed with `rm -rf --` and each verified
with `test ! -e '<path>'` immediately after (loop reported `FAIL=0`, no
`FAILED-STILL-EXISTS` lines). The 4 SKIP-decision roots
(`2026-07-21-exact-greedy-terminal-rescue-training-screen`, `logs`,
`2026-07-17-next-row-probability-transition-and-causal-source-trace`,
`pi-lightweight-worker-ablation`) were re-checked present after the deletion
pass and confirmed untouched.

- `df -h /data` before: `/dev/nvme0n1p1  3.5T  1.6T  1.7T  49% /data`
- `df -h /data` after: `/dev/nvme0n1p1  3.5T  1.6T  1.7T  49% /data`
- `df -h` shows no visible change: on a 3.5T filesystem with 1.7T free, `df -h`
  rounds to ~100 GB increments, so a ~3.2 GiB reclaim is below its display
  resolution. The freed amount is instead the precise pre-deletion `du -sb`
  sum of the 39 deleted roots: **3,438,753,740 bytes ≈ 3.20 GiB (3.44 GB
  decimal)**.
- Remaining top-level entries under
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/`: 143 (down
  from the inventory's 170; some of the difference is the 39 removed here,
  the rest reflects new experiment artifacts landed by ongoing active runs
  between the 2026-08-28 inventory audit and this deletion pass — this
  worktree's `qwen3-vl-dense-enumeration/` output tree is still live).


## Addendum 2026-08-28 — user-named roots (approval: "批准清理")

| Root | Size | Re-verification | Decision |
|---|---|---|---|
| `/data/CoordExp/.worktrees/research-probes/outputs/third_party` | 43136871 bytes (42M, 62 files) | no citation in research/memories/docs/openspec/tests/src/scripts/configs or the image2299 worktree; no process/lsof; not a symlink | DELETED |
| `/data/CoordExp/outputs/research/qwen3-vl-native-text-coordinate-val200` | 1.6G, 445 files | **cited by the live provenance investigation** `research/investigations/coordexp-experiment-knowledge-handoff/{08_coordinate_objective_and_decode_negatives.md,11_historical_result_registry.md,claims.tsv,source_coverage.tsv}` — the inventory's UNBOUND call was wrong (it matched the path, not the basename) | KEPT (bound); reclaim only if the user retires that registry's claims |
