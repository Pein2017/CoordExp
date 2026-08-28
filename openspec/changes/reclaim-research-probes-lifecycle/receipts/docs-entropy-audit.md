# Docs / progress / reference / memories entropy audit (task 6.3, report-only)

Audited at HEAD `9f06b47889c8e815e4da5a2c0a33f7d468a71cc5`, 2026-08-28. No files
were edited, moved, or deleted to produce this receipt. Per user steer
2026-08-28 ("大量的内容是可以清理的(文档/artifacts/*.py)"), this receipt covers the
documents side; `receipts/artifact-root-inventory.md` covers the artifact-root
(`outputs/`) side.

## Method

"Current document" = a `docs/catalog.yaml` entry with status
`canonical`/`canonical-router`/`draft`/`supplemental` (31 of 123 catalog
entries), OR any file under `research/**` except `research/archive/**`, OR
`AGENTS.md`/`README.md`, OR any file under `openspec/specs/**`. A citation
only counts if the citing file is itself current by this test; a hit from
another legacy/historical file (including `docs/catalog.yaml` itself, which
is a manifest, not a citer) does not count. For each non-current catalog
entry, inbound references were resolved as `grep -rl -F <full-path>` over
`docs research openspec/specs AGENTS.md README.md`, plus a basename-context
pass (`grep -n <basename>`, kept only when the matched line also names the
parent directory, or when the basename is long/distinctive) to catch
markdown links that drop the full path. Bare generic basenames (`README.md`)
otherwise produce false positives (any doc that mentions any README) and were
suppressed unless directory-qualified.

## Summary table (counts and size per bucket)

| bucket | files | size | non-current / weakly-cited | current citer count |
|---|---:|---:|---:|---|
| `docs/catalog.yaml` entries | 123 tracked paths (91 unique non-current after 1 duplicate catalog listing) | n/a (catalog is a manifest) | 91 non-current | 3 with zero current citer; 65 with exactly the `docs/history/README.md`-style router citation or exactly 1 real citer |
| `docs/history/` | 894 files | 21M | all (out of catalog scope by design; authority = `non_normative_doc_history`) | destination dir, not source; not ranked for deletion |
| `progress/` | 231 files | 43M | 231 (entire tree is `legacy_evidence_archive_status: deprecated_read_only_migrate_to_research`) | 21 distinct current-doc citers total (of ~172 docs+research files that mention `progress/` at all, only 21 are themselves current; 177 are legacy-status docs or `docs/history/` snapshots; 1 is `docs/catalog.yaml` itself) |
| `reference/` | 1038 tracked files | 16M (`reference/legacy_src/` = 9.1M of it) | n/a, KEEP by design | 0 external imports/links found anywhere in `src scripts tests docs research` |
| `memories/notes/` | 33 files | 200K | 30 not linked from `memories/current.md` or any `research/` file | 3 linked from `current.md` (secondary-provenance pattern); 2 linked from a `research/` unit |

## Ranked candidates (top 30 by net reduction, highest confidence first)

Sizes are dominated by one binary artifact directory; everything else in this
audit is KB-scale text. Ranking mixes size and confidence as instructed —
the single largest item swamps the rest, then the ranking is effectively a
confidence ordering over small files.

[1] [high confidence / low risk] `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/` (curated image sample bank, 36M of `progress/diagnostics/artifacts/`'s 41M, 45 files)
evidence: only `.../et_rmp_rp_sample_bank_2026-04-29/README.md` (the manifest) is named by any current document (`research/investigations/coordexp-experiment-knowledge-handoff/11_historical_result_registry.md`, `claims.tsv`, via the generic phrase "copied diagnostic artifact summaries under `progress/diagnostics/artifacts/...`"). The 6 per-sample subdirectories (`benefit_121/`, `benefit_158/`, `hurt_025/`, `hurt_178/`, `hurt_061/`, `benefit_010/`, each holding `rp110_review/rp115_review/rp118_review` image sets, 5-7M each) are never named by path or basename anywhere in `docs research openspec/specs AGENTS.md README.md`.
cut: the 6 per-sample image subdirectories (keep `README.md`/manifest only) — the single largest lever in this entire audit.
tradeoff: loses the raw side-by-side visual comparison images behind a 2026-04-29 historical diagnostic; the prose finding is already distilled into `11_historical_result_registry.md` RESULT-006 and `progress/benchmarks/2026-04-21_mixed_objective_sota_checkpoint_probe.md`.
verify: `grep -rn 'et_rmp_rp_sample_bank' docs research AGENTS.md README.md` returns only the manifest-generic mention above; net reduction ~36M / 45 files.

[2] [high confidence / low risk] `progress/` subdirs with no real (non-manifest) current citer: `progress/audits/` (12 files, 200K), `progress/directions/` (9 files, 284K), `progress/explorations/` (10 files, 464K), `progress/handoffs/` (2 files, 16K)
evidence: every file in these 4 subdirs is reached by exactly one current citer, `research/investigations/coordexp-experiment-knowledge-handoff/source_coverage.tsv` — a completed provenance ledger (the investigation's own frontmatter status is `historical-synthesis`/`complete`; its `disposition` column already marks these progress rows `archive_historical`) — or by `research/ideas/prefix-denoising-sft/*` (3-5 files each), which is itself scheduled to move under a different task in this same change (4.2) and is not a load-bearing dependency on these specific `progress/` paths. No prose in a current document quotes or depends on any individual file in these 4 subdirs.
cut: all files under the 4 subdirs (964K total, 33 files).
tradeoff: loses read-only legacy evidence already superseded per `docs/catalog.yaml` status and already indexed as archived in `source_coverage.tsv`; `git log` / `research-base-v2` remain the recovery path per this change's own D6 convention.
verify: `grep -rl -F progress/audits progress/directions progress/explorations progress/handoffs` over `docs research openspec/specs AGENTS.md README.md`, excluding `source_coverage.tsv` and `research/ideas/prefix-denoising-sft/`, returns nothing; net reduction ~964K / 33 files.

[3] [high confidence / low risk] `docs/training/LVIS.md` (`historical-reference`, 13170B, last commit 2026-07-11)
evidence: zero current citers (path and basename both absent from `docs research openspec/specs AGENTS.md README.md` outside itself); also absent from any ambiguous/basename-context hit.
cut: file, end to end; no catalog cross-reference to update besides its own catalog row.
tradeoff: none observed; LVIS-specific historical training notes with no current reader.
verify: `grep -rln 'LVIS.md\|training/LVIS' docs research openspec AGENTS.md README.md` returns only `docs/catalog.yaml` and the file itself.

[4] [high confidence / low risk] `docs/history/training/STAGE1_ET_RMP_CE.md` (`superseded`, 1418B, last commit 2026-06-15) and `docs/history/training/STAGE2_DESIGN.md` (`legacy-direction`, 1399B, last commit 2026-06-15)
evidence: zero current citers for either path or basename.
cut: both files (already under `docs/history/`, so this is a delete-in-place, not a move).
tradeoff: none observed; both are already-superseded design snapshots.
verify: same grep pattern as [3]; net reduction ~2.8K / 2 files.

[5] [medium confidence / low risk] `progress/pretrain/` (3 files, 32K) — real citers exist (`12_legacy_design_lineage.md`, `claims.tsv`) but citations are table-cell evidence references, not structural dependencies
evidence: `progress/pretrain/stage1_foundation.md` is cited from a specific claims-registry row; the other 2 files in the subdir are not.
cut: the 2 uncited files only (`progress/pretrain/README.md` router + 1 more); keep `stage1_foundation.md`.
tradeoff: loses the subdir's own router page; low value since the subdir would then hold 1 file.
verify: `grep -rn 'progress/pretrain' research/investigations/coordexp-experiment-knowledge-handoff/*.md` to confirm only `stage1_foundation.md` is named.

[6]-[30] [medium confidence / low-medium risk] the remaining 65 `docs/catalog.yaml` non-current entries and ~194 `progress/` files with exactly 1 current citer, where that citer is `research/investigations/coordexp-experiment-knowledge-handoff/source_coverage.tsv` (a completed, `historical-synthesis`-status provenance manifest whose own `disposition` column already reads `archive_historical` for these rows) rather than a load-bearing prose dependency. Full per-path table is in the Appendix below (91 rows: path, status, last commit, size, current-citer count). MOVE-TO-HISTORY is the appropriate tag for the ~65 `docs/` entries still living under their original `docs/<topic>/` directories (their status field already says legacy/historical; the physical location has not caught up) rather than DELETE, because unlike `progress/`, several of these (`docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`, `docs/eval/COCO_TEST_SUBMISSION.md`, `docs/data/VISUAL_GENOME.md`) are still cited by 2-6 current documents (mostly the same `coordexp-experiment-knowledge-handoff` capsule series) and are evidence obligations, not entropy, per the `record-cited` class in `reclaim-code-entropy` D6 — tag these specific ones KEEP-FLAGGED, not delete or move.
evidence: see Appendix table; `current citers` column distinguishes them.
cut: candidate is a physical move `docs/<x> -> docs/history/<x>` for the true-legacy set (citer count 0-1, citer = a historical-synthesis document), no cut for the KEEP-FLAGGED set.
tradeoff: none for the move (status already says legacy; only the physical location is stale); KEEP-FLAGGED set stays because deleting a still-cited evidence source removes a claim's traceability.
verify: after any move, `docs/catalog.yaml` `path:` fields must be updated (existing gate at task 4.G) and the same current-citer grep re-run against the new path.

## `docs/history/` (context only, not ranked for deletion)

21M / 894 files. Per `docs/catalog.yaml` authority block this directory is
`non_normative_doc_history`, the acknowledged destination for moves like [3]
above — it is the sink, not a source, of this audit. Breakdown for context:
`worktree-cleanup/` 7.2M, `research-intake/` 5.9M, `superpowers/` 3.5M,
`worktree-union/` 3.2M, `architecture/` 692K, `root-orphans/` 96K (already
populated ahead of task 4.1's move), `training/` 12K.

## `progress/` detail

Of 231 tracked files (43M), only 21 distinct current documents mention
`progress/` at all (vs. 177 legacy-status docs / `docs/history/` snapshot
copies, and 1 hit that is `docs/catalog.yaml` itself, out of ~172 docs+research
files that reference the string `progress/` anywhere — this is the source of
the design.md "~171 docs" figure). All 21 current citers are:
`docs/AGENT_ENGINEERING_CONSTITUTION.md`, `docs/AGENT_INDEX.md`,
`docs/PROJECT_CONTEXT.md`, `docs/README.md`, `docs/standards/REPO_HYGIENE.md`
(governance mentions of the directory as a policy subject, not per-file
citations), 6 files under `research/ideas/prefix-denoising-sft/` (this
investigation is itself a candidate for `research/archive/` under a different
task in this change, 4.2 — not touched here, but its citations should not be
read as durable), and 10 files under
`research/investigations/coordexp-experiment-knowledge-handoff/` (whose own
frontmatter status is `historical-synthesis` for the capsules and `complete`
for the intake audit; this is the investigation whose purpose was precisely to
distill `progress/` into `research/` and record the disposition of each
source file — its citations are a completed provenance record, not an
ongoing dependency).

Per-subdir: every one of `progress/{audits,benchmarks,diagnostics,
diagnostics/artifacts,directions,explorations,pretrain,handoffs}` has at
least the mechanical `source_coverage.tsv` hit, so literally zero subdirs
have an absolute-zero current-citer count. Excluding that manifest and the
soon-to-archive `prefix-denoising-sft` citations, 4 of 8 subdirs (`audits`,
`directions`, `explorations`, `handoffs` — 964K, 33 files) have zero real
citation; `benchmarks` (356K, 8 real citers across the capsule series),
`diagnostics` minus `artifacts` (~1M, 6 real citers), and `diagnostics/
artifacts` (41M, but only manifest-level naming, see [1] above) retain
real evidence citations and are KEEP-FLAGGED at the subdir level even though
most individual files inside them are still single-manifest-cited (see
Appendix).

## `reference/` detail

1038 tracked files, 16M, `reference/legacy_src/` = 9.1M of it (recovery
quarantine, KEEP per design.md — not evaluated for deletion here).
`grep -rn 'reference/' src scripts tests docs research --include='*.py'
--include='*.md' | grep -v '^docs/history'` returns 17 lines; all 17 are
substring false positives (`preference/`, `package_reference/`, code comments
using `reference/self` or `reference/neighborhood` as compound identifiers) —
zero of them are a real path reference into the top-level `reference/`
directory. Tag: KEEP-FLAGGED, whole tree, exactly as the baseline design
states; this audit found no new evidence to revisit that.

## `memories/notes/` detail (33 files, 200K)

| note | date | size (B) | linked from `current.md`? | linked from a `research/` file? | disposition |
|---|---|---:|---|---|---|
| `019f4a19-...-comprehensive-recap.md` | 2026-07-22 | 38428 | N | N | KEEP-FLAGGED (low confidence) — largest note (34% of the bucket by size); broad retrospective with no single-unit successor found by name-matching; needs a human read against `research/investigations/qwen3-vl-dense-enumeration/index.md` before any move |
| `019f4a19-...-native-prefix-state-and-causal-boundaries.md` | 2026-07-22 | 5569 | N | N | KEEP-FLAGGED — weak (0.40) topical overlap with `experiments/2026-07-15-prefix-state-phrase-geometry-factorial`, not a confirmed duplicate |
| `019f4a19-...-proposal-bridge-and-workflow-reset.md` | 2026-07-22 | 4880 | N | N | KEEP-FLAGGED — no matching unit found |
| `019f4a19-...-treatment-gates-and-completion-study.md` | 2026-07-22 | 6705 | N | N | KEEP-FLAGGED — weak (0.40) overlap with `experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment`, not confirmed |
| `2026-07-22-constant-dose-image-breadth-screen-checkpoint.md` | 2026-07-22 | 6453 | N | N | MOVE-TO-HISTORY (sediment) — `experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md` owns this result |
| `2026-07-22-physical-owner-duplication-goal-checkpoint.md` | 2026-07-23 | 2867 | N | N | MOVE-TO-HISTORY (sediment) — `experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/` owns this |
| `2026-07-22-physical-owner-duplication-treatment-decision.md` | 2026-07-23 | 1509 | N | N | MOVE-TO-HISTORY (sediment) — same unit as above |
| `2026-07-22-pi-stage0-network-failure-checkpoint.md` | 2026-07-23 | 2232 | N | N | MOVE-TO-HISTORY (sediment) — `research/archive/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/` owns this (unit already archived under task 4.2's bucket, separately) |
| `2026-07-22-pi-stage0-readiness-checkpoint.md` | 2026-07-23 | 2999 | N | N | MOVE-TO-HISTORY (sediment) — same archived unit |
| `2026-07-22-project-memory-design.md` | 2026-07-22 | 2358 | N | N | KEEP-FLAGGED — documents the rejected event-sourced-memory design; not a research result, so it fails the sediment test even though `memories/README.md` now encodes the final design; preserves rejected reasoning per that README's own instruction |
| `2026-07-22-set-transition-treatment-authorization.md` | 2026-07-22 | 3597 | N | N | KEEP-FLAGGED — no confirmed matching unit |
| `2026-07-22-state-banks-v1-rejected-by-fixed-point-audit.md` | 2026-07-22 | 1916 | N | N | KEEP-FLAGGED — cited once from `experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen` artifact-root binding grep (task 6.4), but not confirmed as the same rejected-design content; needs a read |
| `2026-07-23-codex-pi-research-probe-forks.md` | 2026-07-23 | 2811 | N | N | KEEP-FLAGGED — process/tooling note, no research unit match |
| `2026-07-23-independent-advanced-model-review-gate.md` | 2026-07-23 | 1506 | N | N | KEEP-FLAGGED — no match |
| `2026-07-23-pi-default-cli-context-parity.md` | 2026-07-23 | 1062 | N | N | KEEP-FLAGGED — no match |
| `2026-07-23-pi-stage0-proxy9090-rerun-result.md` | 2026-07-23 | 2284 | N | N | MOVE-TO-HISTORY (sediment) — same archived pi-lightweight-worker-ablation unit as above |
| `2026-07-23-pi-stateful-rpc-thread-pilot.md` | 2026-07-23 | 1950 | N | N | MOVE-TO-HISTORY (sediment) — `research/archive/pi-lightweight-worker-ablation/experiments/2026-07-23-stateful-rpc-thread-pilot/` owns this |
| `2026-07-23-set-level-supervision-route-correction.md` | 2026-07-23 | 2242 | N | N | KEEP-FLAGGED — no match |
| `2026-07-23-vllm-sampled-only-panel-complete.md` | 2026-07-23 | 2443 | N | N | KEEP-FLAGGED — weak overlap only |
| `2026-07-24-primary-predicate-design-scope-fork.md` | 2026-07-25 | 3773 | N | N | KEEP-FLAGGED — no match |
| `2026-07-24-set-level-multiparadigm-execution-boundary.md` | 2026-07-25 | 5181 | N | N | KEEP-FLAGGED — no match |
| `2026-07-25-existing-checkpoint-phase-zero-plan.md` | 2026-07-25 | 2149 | N | N | MOVE-TO-HISTORY candidate (0.40 overlap) — likely superseded by `experiments/2026-07-25-existing-checkpoint-transition-mechanism-decomposition/`; needs a 1-line confirmation read |
| `2026-07-25-prefix-local-long-training-and-matched-eval.md` | 2026-07-25 | 3725 | N | Y (`experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/unit.md`) | KEEP-FLAGGED — cited as prior context from a current unit, not superseded |
| `2026-07-25-transition-mechanism-phase-zero-results.md` | 2026-07-25 | 4607 | N | N | MOVE-TO-HISTORY candidate (0.40 overlap, same successor as above) |
| `2026-07-25-transition-step36-matched-transfer-evaluation.md` | 2026-07-25 | 3304 | N | Y (`experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/unit.md`) | KEEP-FLAGGED — same as the prefix-local note |
| `2026-07-26-dense-enumeration-architecture-reset.md` | 2026-07-26 | 2478 | N | N | KEEP-FLAGGED — durable user-steering/decision record ("stop automatic geometry-training..."), not a restated research result; no matching unit |
| `2026-08-12-human13-k-union-execution-result.md` | 2026-08-12 | 2237 | Y | N | KEEP (secondary provenance per `current.md`'s own convention, even though `experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/` covers the same ground) |
| `2026-08-12-human13-k-union-planning-checkpoint.md` | 2026-08-12 | 3608 | N | N | MOVE-TO-HISTORY (sediment) — planning note for the same now-complete unit above, not itself linked |
| `2026-08-13-human13-on-policy-first-bottleneck-result.md` | 2026-08-13 | 1057 | Y | N | KEEP (secondary provenance, same pattern) |
| `2026-08-14-scalable-k-trajectory-successor-direction.md` | 2026-08-20 | 5601 | N | Y (`experiments/2026-08-16-human13-all-hf-shared-surface-trajectory-credit-vertical/handoff-task4.6-task5.md`) | KEEP-FLAGGED — cited as prior context, not confirmed superseded |
| `2026-08-15-all-hf-shared-surface-successor.md` | 2026-08-24 | 1997 | N | N | MOVE-TO-HISTORY (sediment) — `experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/` and `.../2026-08-16-.../` own this ground (0.75 name overlap) |
| `2026-08-15-k-trajectory-parity-closure.md` | 2026-08-15 | 888 | N | N | MOVE-TO-HISTORY candidate — likely closure note for the same k-trajectory vertical; weak name overlap only, needs a read |
| `2026-08-24-human13-n13-k4-k8-factorial-result.md` | 2026-08-24 | 3185 | Y | Y (`experiments/2026-08-22-human13-owner-credit-nk-factorial/results.md`) | KEEP (secondary provenance, explicit per `current.md`) |

Sediment count (MOVE-TO-HISTORY, confident): 9 of 33 notes (the 2
physical-owner-duplication notes, 2 constant-dose/goal notes collapse to 3
distinct, 4 pi-lightweight-worker-ablation notes, 1 human13-k-union-planning
note, 1 all-hf-shared-surface note — see table). 3 more are lower-confidence
MOVE-TO-HISTORY candidates pending a 1-line read (phase-zero-plan,
phase-zero-results, k-trajectory-parity-closure). 4 notes are explicit
secondary-provenance KEEP entries per `current.md`'s own established
pattern. The remaining ~17 have no confirmed successor unit and stay
KEEP-FLAGGED; `memories/README.md` explicitly instructs preserving rejected
reasoning over pruning it speculatively, so no-match notes are not treated as
sediment.

## Smallest check per bucket

- `docs/catalog.yaml`: the path-resolution check already specified at task
  4.G (`grep -rn` for each moved basename outside `docs/history/` and
  `research/archive/`) is the smallest decisive check; it fails if a moved
  doc still had a live citer this audit missed.
- `progress/`: `grep -rl -F progress/ docs research | grep -v '^docs/history' | wc -l` should collapse from ~172 to ~21 once the 4 zero-real-citer subdirs are moved and the manifest citation is discounted; a residual count above 21 means a citer was missed.
- `reference/`: `grep -rn 'reference/' src scripts tests docs research` (excluding `docs/history`) should keep returning only substring false positives; a real path hit means the KEEP-FLAGGED status must be revisited.
- `memories/notes/`: `grep -rl -F <note-filename> memories/current.md research` is the per-note check already used above; any future note should pass through the research-flow closeout gate (result -> unit -> `memories/current.md`) before being written, per design.md's existing observation that the N/K factorial case simply skipped that gate.

## Appendix: all 91 non-current `docs/catalog.yaml` entries (deduplicated)

| path | status | last commit | size (B) | current citers |
|---|---|---|---:|---:|
| `progress/explorations/2026-05-15_training_infrastructure_architecture_decisions.md` | legacy-reference | 2026-05-17 | 131710 | 1 |
| `progress/diagnostics/2026-06-12_autoregressive_duplication_causal_chain_synthesis.md` | legacy-router | 2026-06-15 | 80522 | 1 |
| `progress/directions/stage2_emish_set_supervision_v1.md` | legacy-direction | 2026-06-15 | 58599 | 1 |
| `progress/directions/prefix_denoising_sft_v1.md` | legacy-branch-provenance | 2026-06-15 | 56945 | 4 |
| `progress/explorations/2026-06-12_coord_repel_stage1_sft_design_decisions.md` | legacy-branch-provenance | 2026-06-15 | 54922 | 1 |
| `progress/audits/2026-05-03_type_schema_architecture_audit.md` | legacy-audit | 2026-05-03 | 46701 | 1 |
| `progress/directions/2026-06-07_segment_aware_packing_infra.md` | legacy-draft | 2026-06-15 | 34954 | 1 |
| `progress/diagnostics/2026-03-12_stage2_triage_posterior_coco1024_train_dynamics.md` | legacy-diagnostic | 2026-06-15 | 33518 | 1 |
| `progress/directions/2026-06-05_row_conditioned_visual_coverage.md` | legacy-reference | 2026-06-15 | 33084 | 1 |
| `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md` | legacy-branch-provenance | 2026-06-15 | 29554 | 3 |
| `progress/directions/full_idea_v5.md` | legacy-reference | 2026-06-15 | 28600 | 1 |
| `docs/training/STAGE1_OBJECTIVE.md` | historical-reference | 2026-07-11 | 27906 | 2 |
| `progress/directions/full_idea_v3.md` | legacy-reference | 2026-06-15 | 27326 | 1 |
| `progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md` | legacy-diagnostic | 2026-04-23 | 24919 | 1 |
| `progress/diagnostics/2026-06-12_fn_guidance_and_coord_basin_synthesis.md` | legacy-router | 2026-06-15 | 23785 | 1 |
| `progress/diagnostics/2026-04-13_duplication_collapse_final_analysis.md` | legacy-mechanism-evidence | 2026-04-23 | 22994 | 1 |
| `progress/diagnostics/2026-03-09_stage2_ul_capture_highres1024.md` | legacy-diagnostic | 2026-06-15 | 21868 | 1 |
| `progress/diagnostics/2026-03-05_stage2_near_duplication.md` | legacy-diagnostic | 2026-06-15 | 21431 | 1 |
| `progress/diagnostics/2026-02-17_stage2_b_ratio_085_instability.md` | legacy-diagnostic | 2026-06-15 | 21370 | 1 |
| `progress/diagnostics/2026-06-12_pre_onset_duplication_precursor_synthesis.md` | legacy-router | 2026-06-15 | 21346 | 1 |
| `progress/diagnostics/2026-06-03_candidate_field_cardinality_representative8192_analysis.md` | legacy-diagnostic | 2026-06-04 | 20456 | 1 |
| `docs/training/METRICS.md` | historical-reference | 2026-07-11 | 20256 | 3 |
| `progress/diagnostics/2026-04-22_raw_text_decode_bias_mechanism_findings.md` | legacy-reference | 2026-04-23 | 19837 | 1 |
| `progress/directions/full_idea_v4.md` | legacy-superseded | 2026-06-15 | 19667 | 1 |
| `progress/audits/2026-02-25_stage2_channel_a_coord_loss.md` | legacy-audit | 2026-04-23 | 18854 | 1 |
| `progress/diagnostics/2026-03-16_stage2_2b_stage1_vs_aonly_prefix_fn_hypotheses_plan.md` | legacy-supporting-reference | 2026-04-23 | 18548 | 1 |
| `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md` | legacy-decision | 2026-06-20 | 15692 | 2 |
| `progress/diagnostics/2026-03-26_stage2_small_object_duplication_crowded_deep_dive.md` | legacy-supporting-reference | 2026-04-23 | 15228 | 3 |
| `progress/diagnostics/2026-03-26_stage2_small_object_duplication_offline_harness_findings.md` | legacy-supporting-reference | 2026-04-23 | 15129 | 1 |
| `progress/audits/2026-06-01_detection_scene_cleanup_validation.md` | legacy-provenance | 2026-06-01 | 15082 | 1 |
| `progress/diagnostics/2026-02-25_stage2_channel_a_visual_audit.md` | legacy-diagnostic | 2026-04-23 | 14971 | 1 |
| `progress/explorations/2026-05-31_grid_anchor_pending_record.md` | legacy-reference | 2026-05-31 | 14639 | 1 |
| `progress/diagnostics/README.md` | legacy-router | 2026-07-07 | 13364 | 1 |
| `docs/training/LVIS.md` | historical-reference | 2026-07-11 | 13170 | 0 |
| `progress/diagnostics/2026-03-26_stage2_small_object_duplication_offline_synthesis.md` | legacy-mechanism-evidence | 2026-04-23 | 12594 | 6 |
| `progress/diagnostics/2026-06-12_diagnostics_consolidation_summary.md` | legacy-router | 2026-06-15 | 12414 | 1 |
| `progress/pretrain/stage1_foundation.md` | legacy-reference | 2026-03-09 | 12022 | 3 |
| `progress/diagnostics/2026-02-21_stage2_channel_a_coord_gate.md` | legacy-diagnostic | 2026-06-15 | 11325 | 1 |
| `progress/directions/stage2_clean_prefix_v2.md` | legacy-reference | 2026-06-15 | 10813 | 1 |
| `progress/diagnostics/2026-02-22_stage2_softctx_discretization_vs_stage1_bbox.md` | legacy-diagnostic | 2026-04-23 | 10604 | 1 |
| `progress/benchmarks/2026-03-11_stage2_oracle_k_first200.md` | legacy-benchmark | 2026-04-23 | 10537 | 5 |
| `progress/diagnostics/2026-03-24_stage2_pseudo_positive_k4_coord_only_findings.md` | legacy-diagnostic | 2026-04-23 | 10414 | 1 |
| `progress/diagnostics/2026-06-04_prefix_state_transition_phase_a3_1_analysis.md` | legacy-diagnostic | 2026-06-04 | 10387 | 1 |
| `progress/audits/2026-01-22_stage1_softce_logging.md` | legacy-audit | 2026-04-23 | 9272 | 1 |
| `progress/diagnostics/2026-04-22_stage2_birth_first_channel_b_decision_study.md` | legacy-decision-evidence | 2026-05-14 | 9063 | 1 |
| `progress/diagnostics/2026-03-17_stage2_2b_fn_factor_results.md` | legacy-diagnostic | 2026-04-23 | 8980 | 1 |
| `progress/diagnostics/2026-04-21_raw_text_coordinate_mechanism_findings.md` | legacy-reference | 2026-04-23 | 8977 | 1 |
| `progress/diagnostics/2026-04-15_cxcy_logw_logh_retrained_performance_analysis.md` | legacy-reference | 2026-04-23 | 8930 | 1 |
| `progress/benchmarks/2026-04-21_mixed_objective_sota_checkpoint_probe.md` | legacy-benchmark | 2026-05-14 | 8789 | 4 |
| `progress/pretrain/2026-01-26_stage1_ablation.md` | legacy-reference | 2026-04-23 | 8761 | 1 |
| `progress/diagnostics/2026-04-20_coord_family_basin_and_recall_comparison.md` | legacy-reference | 2026-04-23 | 8643 | 1 |
| `progress/diagnostics/2026-02-25_stage2_channel_a_coord_loss.md` | legacy-diagnostic | 2026-04-23 | 8120 | 1 |
| `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md` | legacy-mechanism-evidence | 2026-04-23 | 8087 | 1 |
| `progress/benchmarks/2026-04-23_stage1_raw_text_vs_coord_token_repetition_penalty_sweep.md` | legacy-benchmark | 2026-05-14 | 7915 | 1 |
| `progress/diagnostics/2026-03-17_stage2_2b_fn_factor_artifact_guide.md` | legacy-supporting-reference | 2026-04-23 | 7797 | 1 |
| `progress/benchmarks/2026-02-26_stage1_training_dynamics_4b.md` | legacy-benchmark | 2026-04-23 | 7743 | 1 |
| `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md` | legacy-superseded-branch-evidence | 2026-06-15 | 7347 | 3 |
| `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md` | legacy-branch-provenance | 2026-06-15 | 7001 | 5 |
| `progress/diagnostics/2026-04-20_raw_text_coord_continuity_probe.md` | legacy-supporting-reference | 2026-04-23 | 6981 | 1 |
| `progress/diagnostics/2026-03-11_visualization_tools_index.md` | legacy-diagnostic | 2026-04-23 | 6947 | 1 |
| `docs/data/VISUAL_GENOME.md` | historical-reference | 2026-07-11 | 6870 | 1 |
| `progress/diagnostics/2026-03-17_stage2_2b_prefix_random_order_followup.md` | legacy-supporting-reference | 2026-04-23 | 6815 | 1 |
| `progress/benchmarks/2026-02-01_stage2_channel_a_infer_eval.md` | legacy-benchmark | 2026-06-15 | 6631 | 4 |
| `progress/diagnostics/2026-06-05_a3_3_post_x1_instance_basin_real_tiny_smoke.md` | legacy-diagnostic-evidence | 2026-06-05 | 6505 | 1 |
| `progress/diagnostics/2026-03-11_gt_overlap_threshold_search.md` | legacy-diagnostic | 2026-04-23 | 6412 | 1 |
| `progress/explorations/2026-03-19_runtime_refactor_architecture_program.md` | legacy-exploration | 2026-04-23 | 5743 | 1 |
| `progress/benchmarks/2026-02-26_stage1_coco80_4b_res_768_vs_1024.md` | legacy-benchmark | 2026-04-23 | 5660 | 1 |
| `progress/benchmarks/2026-02-26_stage1_coco80_temp0_compare.md` | legacy-benchmark | 2026-04-23 | 5658 | 1 |
| `progress/diagnostics/2026-03-11_rollout_duplication_thresholds_ul_vs_ulv2.md` | legacy-diagnostic | 2026-04-23 | 5481 | 1 |
| `progress/diagnostics/2026-05-14_a5_a6_iou_gibbs_softce_negative_result.md` | legacy-concluded-negative | 2026-06-26 | 5009 | 1 |
| `docs/eval/COCO_TEST_SUBMISSION.md` | historical-reference | 2026-07-11 | 4871 | 2 |
| `progress/diagnostics/2026-04-17_cxcywh_quickcheck_val200.md` | legacy-reference | 2026-04-23 | 4783 | 1 |
| `progress/diagnostics/2026-04-20_raw_text_and_coord_family_decision_summary.md` | legacy-supporting-reference | 2026-04-23 | 4684 | 1 |
| `progress/diagnostics/2026-06-04_a3_2_sorted_random_no_newline_smoke_findings.md` | legacy-diagnostic-evidence | 2026-06-05 | 4422 | 1 |
| `progress/explorations/2026-01-26_stage2_infrastructure.md` | legacy-exploration | 2026-04-23 | 4295 | 1 |
| `progress/benchmarks/2026-03-11_stage2_rollout_temperature_refinement.md` | legacy-benchmark | 2026-04-23 | 3875 | 5 |
| `progress/diagnostics/2026-03-25_stage2_small_object_duplication_offline_protocol.md` | legacy-supporting-reference | 2026-04-23 | 3769 | 1 |
| `progress/benchmarks/2026-02-27_stage1_coco_2b_ce_softce_res_768_vs_1024.md` | legacy-benchmark | 2026-04-23 | 3458 | 1 |
| `progress/README.md` | legacy-router | 2026-07-07 | 3205 | 2 |
| `progress/audits/README.md` | legacy-router | 2026-07-07 | 2844 | 1 |
| `progress/explorations/README.md` | legacy-router | 2026-07-07 | 2694 | 1 |
| `progress/benchmarks/README.md` | legacy-router | 2026-07-07 | 2645 | 1 |
| `progress/directions/README.md` | legacy-router | 2026-07-07 | 2598 | 1 |
| `progress/diagnostics/artifacts/README.md` | legacy-router | 2026-06-15 | 2592 | 1 |
| `docs/training/README.md` | historical-router | 2026-07-11 | 2473 | 5 |
| `docs/training/STAGE2_RUNBOOK.md` | historical-reference | 2026-07-11 | 2077 | 6 |
| `docs/history/README.md` | historical-router | 2026-07-12 | 1576 | 7 |
| `docs/history/training/STAGE1_ET_RMP_CE.md` | superseded | 2026-06-15 | 1418 | 0 |
| `docs/history/training/STAGE2_DESIGN.md` | legacy-direction | 2026-06-15 | 1399 | 0 |
| `docs/history/architecture/README.md` | historical-router | 2026-07-11 | 999 | 5 |
| `progress/pretrain/README.md` | legacy-router | 2026-07-07 | 931 | 1 |

## Wave 8 (8.3) — applied 2026-08-28 on `lane/docs`

Executed the DELETE and memories/notes buckets of this audit per user approval
2026-08-28 ("完全同意"). Lane worktree `/data/CoordExp/.worktrees/lane-docs`,
branch `lane/docs`, forked from `research-probes` HEAD `b3d3be9b9`. Before
each deletion, the current-citer grep from the Method section above was
re-run against the live tree (not just trusted from this receipt's original
snapshot).

### Re-verification finding (changed the plan)

The re-run found 3 files in the item-[2] "cut: all files" set that now have
a real current citer that the original audit missed: `research/ideas/
prefix-denoising-sft/{overview.md,discussion.md,draft.md}` (still under
`research/ideas/`, not `research/archive/` — task 4.2 archived four other
investigations but not this one) cite `progress/audits/
2026-06-14_prefix_denoising_sft_v1_audit.md`, `progress/directions/
prefix_denoising_sft_v1.md`, and `progress/explorations/
2026-06-20_docs_progress_okf_upgrade_alignment.md` as primary source
material ("review findings that motivated repair...", "original idea,
design rationale..."), not as a completed-provenance manifest reference
like `source_coverage.tsv`. Per task 8.3's re-verification instruction,
these 3 were skipped and kept (with their catalog entries), leaving
`progress/audits/`, `progress/directions/`, and `progress/explorations/`
each holding exactly the one still-cited file.

### MOVE-TO-HISTORY bucket: resolved to zero git-mv operations

Task 8.3's own move mechanic (`git mv <path> docs/history/<same relative
path under docs/>`) is only definable for a source path already under
`docs/`. Re-checking the audit's Appendix, there are exactly 7 `docs/`-
prefixed non-current catalog paths total: `docs/training/{STAGE1_OBJECTIVE,
METRICS,LVIS,README,STAGE2_RUNBOOK}.md`, `docs/data/VISUAL_GENOME.md`,
`docs/eval/COCO_TEST_SUBMISSION.md`. Of these: `LVIS.md` is DELETE (0
citers, see below); `STAGE1_OBJECTIVE.md`, `METRICS.md`, `STAGE2_RUNBOOK.md`,
`VISUAL_GENOME.md`, `COCO_TEST_SUBMISSION.md` are the 5 the audit explicitly
named KEEP-FLAGGED (real 2-6 citers, evidence obligations). The remaining
one, `docs/training/README.md` (historical-router, 5 real citers), was
neither named KEEP-FLAGGED nor given a per-path MOVE tag by the audit; it
also falls outside the audit's own move criterion ("citer count 0-1") and
is the directory index for the three KEEP-FLAGGED siblings that stay in
place. It was left untouched and is recorded below for the lead rather than
moved on inference. The audit's prose "~65 docs/ entries ... MOVE-TO-HISTORY"
therefore reads, against the real Appendix data, as using "docs/" loosely
for "docs/catalog.yaml entries" in general (including `progress/`-prefixed
ones); those `progress/`-prefixed entries are excluded from this task's
move mechanic because `progress/`'s own catalog authority status is
`deprecated_read_only_migrate_to_research` (destination `research/`, not
`docs/history/`) — a different, out-of-scope disposition. Net result: no
commit 2 (`chore(docs): move legacy catalog entries under docs/history`)
was created; there is nothing to move under 8.3's literal instruction.

### Deleted (DELETE bucket, commit `46ec2ccf7`)



| subdirectory (deleted whole) | files | size |
|---|---:|---:|
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_010/` | 7 | 5.06 MB |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_121/` | 7 | 7.26 MB |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/` | 7 | 7.07 MB |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/` | 7 | 6.00 MB |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/` | 7 | 5.02 MB |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/` | 7 | 5.29 MB |
| **total** | 42 | 35.70 MB |

(README.md, index.json, research_subset.json kept per audit item [1].)

| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_121/rp115_review/vis_0000.png` | 1,848,692 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_121/rp118_canonical.jsonl` | 2,728 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_121/rp118_review/vis_0000.png` | 1,850,469 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/benefit_158_comparison.png` | 1,971,717 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp110_canonical.jsonl` | 932 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp110_review/vis_0000.png` | 1,812,181 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp115_canonical.jsonl` | 996 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp115_review/vis_0000.png` | 1,813,474 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp118_canonical.jsonl` | 1,189 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/benefit_158/rp118_review/vis_0000.png` | 1,816,895 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/hurt_025_comparison.png` | 1,729,914 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp110_canonical.jsonl` | 2,443 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp110_review/vis_0000.png` | 1,519,132 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp115_canonical.jsonl` | 2,510 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp115_review/vis_0000.png` | 1,522,458 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp118_canonical.jsonl` | 2,149 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_025/rp118_review/vis_0000.png` | 1,508,157 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/hurt_061_comparison.png` | 1,464,399 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp110_canonical.jsonl` | 2,496 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp110_review/vis_0000.png` | 1,264,868 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp115_canonical.jsonl` | 2,496 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp115_review/vis_0000.png` | 1,263,945 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp118_canonical.jsonl` | 2,426 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_061/rp118_review/vis_0000.png` | 1,265,876 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/hurt_178_comparison.png` | 1,557,394 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp110_canonical.jsonl` | 3,882 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp110_review/vis_0000.png` | 1,328,049 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp115_canonical.jsonl` | 3,801 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp115_review/vis_0000.png` | 1,330,606 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp118_canonical.jsonl` | 3,830 |
| `progress/diagnostics/artifacts/et_rmp_rp_sample_bank_2026-04-29/hurt_178/rp118_review/vis_0000.png` | 1,315,252 |

### progress/audits (zero-real-citer, 10 files) — 0.14 MB

| path | size (B) |
|---|---:|
| `progress/audits/2026-01-22_stage1_softce_logging.md` | 9,272 |
| `progress/audits/2026-02-25_stage2_channel_a_coord_loss.md` | 18,854 |
| `progress/audits/2026-05-03_type_schema_architecture_audit.md` | 46,701 |
| `progress/audits/2026-05-14-instance-trie-gaussian-post-implementation-audit.md` | 4,089 |
| `progress/audits/2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md` | 5,111 |
| `progress/audits/2026-05-19_openspec_reactivation_audit.md` | 8,633 |
| `progress/audits/2026-05-20_repository_stewardship_audit.md` | 12,556 |
| `progress/audits/2026-05-31_detection_scene_clean_break_archive_checkpoint.md` | 1,111 |
| `progress/audits/2026-05-31_detection_scene_phase0_surface_classification.md` | 18,224 |
| `progress/audits/2026-06-01_detection_scene_cleanup_validation.md` | 15,082 |
| `progress/audits/README.md` | 2,844 |

### progress/directions (zero-real-citer, 8 files) — 0.21 MB

| path | size (B) |
|---|---:|
| `progress/directions/2026-06-05_row_conditioned_visual_coverage.md` | 33,084 |
| `progress/directions/2026-06-07_segment_aware_packing_infra.md` | 34,954 |
| `progress/directions/README.md` | 2,598 |
| `progress/directions/full_idea_v3.md` | 27,326 |
| `progress/directions/full_idea_v4.md` | 19,667 |
| `progress/directions/full_idea_v5.md` | 28,600 |
| `progress/directions/stage2_clean_prefix_v2.md` | 10,813 |
| `progress/directions/stage2_emish_set_supervision_v1.md` | 58,599 |

### progress/explorations (zero-real-citer, 8 files) — 0.41 MB

| path | size (B) |
|---|---:|
| `progress/explorations/2026-01-26_stage2_infrastructure.md` | 4,295 |
| `progress/explorations/2026-03-19_runtime_refactor_architecture_program.md` | 5,743 |
| `progress/explorations/2026-05-15_training_infrastructure_architecture_decisions.md` | 131,710 |
| `progress/explorations/2026-05-19_unified_teacher_forcing_objective_architecture_decisions.md` | 112,511 |
| `progress/explorations/2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md` | 92,102 |
| `progress/explorations/2026-05-31_grid_anchor_pending_record.md` | 14,639 |
| `progress/explorations/2026-06-12_coord_repel_stage1_sft_design_decisions.md` | 54,922 |
| `progress/explorations/2026-06-17_codebase_refactoring_program_kickoff.md` | 16,002 |
| `progress/explorations/README.md` | 2,694 |

### progress/handoffs (whole dir, 2 files) — 0.01 MB

| path | size (B) |
|---|---:|
| `progress/handoffs/2026-06-11-packed-gaussian-sft-retrain.md` | 5,346 |
| `progress/handoffs/README.md` | 766 |

### progress/pretrain (2 zero/manifest-only files; stage1_foundation.md kept) — 0.01 MB

| path | size (B) |
|---|---:|
| `progress/pretrain/2026-01-26_stage1_ablation.md` | 8,761 |
| `progress/pretrain/README.md` | 931 |

### docs/ zero-citer files (3) — 0.02 MB

| path | size (B) |
|---|---:|
| `docs/history/training/STAGE1_ET_RMP_CE.md` | 1,418 |
| `docs/history/training/STAGE2_DESIGN.md` | 1,399 |
| `docs/training/LVIS.md` | 13,170 |

### memories/notes sediment (9) — 0.02 MB

| path | size (B) |
|---|---:|
| `memories/notes/2026-07-22-constant-dose-image-breadth-screen-checkpoint.md` | 6,453 |
| `memories/notes/2026-07-22-physical-owner-duplication-goal-checkpoint.md` | 2,867 |
| `memories/notes/2026-07-22-physical-owner-duplication-treatment-decision.md` | 1,509 |
| `memories/notes/2026-07-22-pi-stage0-network-failure-checkpoint.md` | 2,232 |
| `memories/notes/2026-07-22-pi-stage0-readiness-checkpoint.md` | 2,999 |
| `memories/notes/2026-07-23-pi-stage0-proxy9090-rerun-result.md` | 2,284 |
| `memories/notes/2026-07-23-pi-stateful-rpc-thread-pilot.md` | 1,950 |
| `memories/notes/2026-08-12-human13-k-union-planning-checkpoint.md` | 3,608 |
| `memories/notes/2026-08-15-all-hf-shared-surface-successor.md` | 1,997 |

**Total freed: 38,284,451 bytes (36.51 MB)**

### `docs/catalog.yaml` entries removed (24)

`docs/training/LVIS.md`, `docs/history/training/STAGE1_ET_RMP_CE.md`,
`docs/history/training/STAGE2_DESIGN.md`, all 6 `progress/audits/*` entries
except the kept `2026-06-14_prefix_denoising_sft_v1_audit.md`, all 9
`progress/directions/*` entries except the kept `prefix_denoising_sft_v1.md`,
all 7 `progress/explorations/*` entries except the kept
`2026-06-20_docs_progress_okf_upgrade_alignment.md`, and 2 of 3
`progress/pretrain/*` entries (`README.md`, `2026-01-26_stage1_ablation.md`;
`stage1_foundation.md` kept). Verified: 123 tracked paths -> 99, `yaml.safe_load`
parses, every remaining `path:` resolves (gate script below).

### Skipped items (real current citer found on re-verification; not deleted)

| path | citer | reason |
|---|---|---|
| `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md` | `research/ideas/prefix-denoising-sft/{overview.md,discussion.md}` | primary source citation, not manifest-only |
| `progress/directions/prefix_denoising_sft_v1.md` | `research/ideas/prefix-denoising-sft/{overview.md,discussion.md,draft.md}` | primary source citation |
| `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md` | `research/ideas/prefix-denoising-sft/overview.md` | primary source citation |
| `docs/training/README.md` | itself (5 real citers) not moved | not KEEP-FLAGGED-named nor MOVE-tagged by the audit; router for kept `STAGE1_OBJECTIVE.md`/`METRICS.md`/`STAGE2_RUNBOOK.md`; held for lead, see reasoning above |

### Inbound links rewritten (write-surface files only)

| file | change |
|---|---|
| `docs/training/README.md` | removed `LVIS.md` bullet; removed `../history/training/` bullet (dir now empty/gone) |
| `docs/training/STAGE2_RUNBOOK.md` | removed `../history/training/STAGE2_DESIGN.md` bullet |
| `docs/history/README.md` | removed `training/` bullet (dir now empty/gone) |
| `progress/README.md` | removed `progress/handoffs/` bullet; repointed `directions/`, `audits/`, `explorations/` bullets from their deleted `README.md` routers to the single retained prefix-denoising-sft-cited file in each |

`progress/index.yaml` (a parallel legacy manifest, not named in 8.3's gate
or write-surface enumeration beyond generic `progress/**`) still lists
`path:`-style entries for files deleted in this wave (e.g.
`progress/audits/2026-05-03_type_schema_architecture_audit.md`); left
untouched since it is outside the stated gate and `progress/`'s own
authority status is read-only — flagged here for the lead rather than
edited.

### memories/notes pruned (9, commit `02a21f9dc`)

| note | owning research unit |
|---|---|
| `2026-07-22-constant-dose-image-breadth-screen-checkpoint.md` | `research/investigations/.../experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md` |
| `2026-07-22-physical-owner-duplication-goal-checkpoint.md` | `experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/` |
| `2026-07-22-physical-owner-duplication-treatment-decision.md` | same unit |
| `2026-07-22-pi-stage0-network-failure-checkpoint.md` | `research/archive/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/` |
| `2026-07-22-pi-stage0-readiness-checkpoint.md` | same archived unit |
| `2026-07-23-pi-stage0-proxy9090-rerun-result.md` | same archived unit |
| `2026-07-23-pi-stateful-rpc-thread-pilot.md` | `research/archive/pi-lightweight-worker-ablation/experiments/2026-07-23-stateful-rpc-thread-pilot/` |
| `2026-08-12-human13-k-union-planning-checkpoint.md` | `experiments/2026-08-12-human13-k-union-to-greedy-overfit-screen/` |
| `2026-08-15-all-hf-shared-surface-successor.md` | `experiments/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/` and `.../2026-08-16-.../` |

None of the 9 were linked from `memories/current.md` or any `research/` file
(re-verified: `grep -rln <basename> memories research openspec/specs docs
AGENTS.md README.md` returned nothing for all 9 before deletion), so no
continuity-pointer rewrite in `memories/current.md` was needed.

### Post-merge link fixes for the lead

None. Every citing file this wave found in `research/**` or `openspec/**`
either (a) cites a file that was kept (the 3 prefix-denoising-sft skip
items), or (b) is itself a `docs/history/` frozen snapshot whose broken
links pre-date this change and are not this wave's to fix (see before/after
counts below, which are identical).

### Gate: catalog + link check (before/after)

`docs/catalog.yaml` path resolution: 0 missing paths, both before and after.

Broken-link check scope (`docs/*.md`, `docs/**/README.md`, `README.md`,
`AGENTS.md`, `memories/current.md`, relative links only):

| | count |
|---|---:|
| before (stashed to pre-edit tree, HEAD `b3d3be9b9`) | 306 |
| after (first pass, before fixing 2 new dir-links) | 308 |
| after (final, post `docs/training/README.md` + `docs/history/README.md` fixes) | 306 |

Diff of the final after-list against the before-list: empty (no new
breakage, none of the 306 pre-existing ones — all inside frozen
`docs/history/worktree-*/snapshots/**` — were touched). `git diff --check`:
clean on both commits.

### `git diff --check`

Clean (no whitespace errors) on both commits.

### Commits

- `46ec2ccf7` `chore(docs): delete zero-citer legacy documents` (82 files
  changed: 77 deletions in `progress/`, 3 in `docs/`, 5 edits in
  `docs/catalog.yaml` + 4 link-bearing docs)
- `02a21f9dc` `chore(memories): prune sediment notes owned by research units`
  (9 files deleted)
- No `chore(docs): move legacy catalog entries under docs/history` commit —
  see "MOVE-TO-HISTORY bucket: resolved to zero git-mv operations" above.

### MB freed

36.51 MB total (35.70 MB `et_rmp_rp_sample_bank` subdirs + 0.81 MB
`progress/audits+directions+explorations+handoffs+pretrain` +
0.02 MB `docs/` + 0.02 MB `memories/notes`), computed from git blob sizes
at parent commit `b3d3be9b9` for every path in both commits' diff
`--diff-filter=D` lists.
