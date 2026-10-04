# Canonical repository review

## Actual responsibilities

The desired stable / infrastructure / research split is useful as an ownership target, but it is not an accurate release-status guarantee. At the continuation HEADs (main 52d7dd204, infrastructure f534953c4, research a6f236f38), main and infrastructure have 273 / 157 unique commits; infrastructure and research have 155 / 795. Promotion must be capability-scoped rather than treating the branch tips as a linear release ladder.

Main integrates the training/inference core and also owns substantial COCO and Label Studio annotation/data operations. Its 137 tracked source files coexist with legacy launch/config/test surfaces whose implementations were removed by e1662c2c76d387665c9706ccc75a284ccc2ee682. The integrated core is not automatically qualified against the currently installed dependency and artifact state.

Infrastructure is a comparatively compact product surface for config-first training, packing/cache preparation, exact resume, independently authenticated checkpoint payloads, dynamic HF/composed vLLM inference and direct evaluation. Its source differs substantially from main; its infra-base spec deliberately limits runtime claims and requires live qualification. It is not merely main plus a few unfinished experiments.

Research is the active scientific authority and retains a complete execution core plus source/evidence admission, saved-row evaluation, rollout calibration and experiment-specific probes. research/index.md correctly separates technical qualification from scientific acceptance. The root README had incorrectly limited probes to two numerical methods and its OpenSpec context referenced deleted PROJECT_CONTEXT.md while declaring the infrastructure lane retired. The continuation corrected these local ownership claims, retained the scientific evidence owners and consolidated the three numerical/operator test modules into tests/probes.

## Implemented boundaries

Main retirement is enumerated in retirements.json: 366 exact tracked files, each with the inspected bytes' SHA-256 and a reason. This includes removed-implementation tests/scripts, their shell/tmux/queue callers, a historical Wave-7 machine-receipt test and harness-installation snapshot assertions. Actual training/inference/annotation implementation, data and outputs were not rewritten. Historical code remains in Git; no archive tree was added.

Current command ownership now points to src.train / src.infer and source-owned operational entrypoints. The branch policy no longer treats a temporary web-codex worktree as a permanent architecture owner. OpenSpec no longer directs changes to deleted architecture manuals or a mandatory historical orchestration scheme.

Default test discovery now includes both annotation systems. Importlib collection plus an importable tests package handles duplicate basenames and spawned test workers. The inference test checks actual module resolution instead of absence of ignored cache directories. A storage-layout assertion now matches the existing .local/state ownership contract. External data/application receipt tests have an explicit live_environment marker but remain enabled by default; excluding them is an explicit validation choice, never a successful environment qualification.

Infrastructure implementation is recorded in that checkout's unify-config-document-loading change: one YAML document owner, direct caller migration, preserved train/inference policy differences, independent-copy semantics and new edge-case tests. Its benchmark continuation/live-owner test is now part of default discovery. No implementation was copied between worktrees.

An independently recorded infrastructure repair, align-fa2-capture-kernel-api, updates the packed-varlen capture and fixtures to the installed five-kernel Transformers API. The prior four-kernel assumption existed in both production capture and the test fake. Existing topology/varlen/padding proof remains; unqualified KV-cache execution fails closed. Native/GPU qualification is not implied by this CPU repair.

## Protected data consumers and remaining debt

Two scripts are deliberately retained despite their unsupported old model APIs: scripts/tools/inspect_chat_template.py is called by public_data/run.sh, and scripts/analysis/measure_gt_max_new_tokens.py is called by the LVIS export pipeline. Their rendering and token-budget consumers need a semantic migration before deletion. Public-data provenance also cites configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml in concrete recovery commands. Deleting the old config roots wholesale would threaten recovery, not simplify it.

The existing closed documentation seals were not appended to or replaced. Main retains nine long-lived docs; infrastructure retains five. Prior main history is still recoverable as 796 sealed items. research/ and legacy progress/config material were not mass-archived or rewritten without scientific/data lineage review.

Remaining main debt includes data-bound legacy configurations/diagnostics, other old root-level tests and self-contained historical reducers, two substantial annotation implementations whose semantic differences deserve a separate domain review, and the observed model/dependency/golden-fixture validation failures. The current source implementation is unchanged in main, and the three model/render failures reproduce when isolated from annotation tests.

Infrastructure still has large coherent-but-heavy owners such as training_state, training/session, packing/planner and inference/merge. Size alone is not enough evidence to split them. Smaller duplicate hash helpers also remain; changing receipt code should have a separate identity-aware review rather than a blind global deduplication.

## Research continuation and concurrency

The first pass correctly held research writes: an eight-rank Python/elastic workload and whole-checkout clean-source gates made even documentation edits disruptive. The continuation found no training/test worker or GPU compute application; the current research index recorded the affected units closed and their source holder released. No process was stopped or execution gate weakened.

A write preflight then rejected stale HEAD a1244d801 because concurrent visualization work had been committed as ca7ec3c28. The nine visualization files were preserved. A new baseline also protected five concurrently edited readout-audit knowledge files. The local clarify-research-ownership-and-test-layout change corrects README/OpenSpec ownership and relocates three test modules byte-for-byte into tests/probes, removing broad probes discovery without deleting ignored historical directories. Runtime source, scientific records and released evidence are outside this implementation slice. The five protected knowledge files were subsequently committed by another actor as ca29b8d43, with their protected hashes unchanged at that check. The same parallel work then committed coordinate_readout source/tests and its own updated state as a6f236f38. That later state change was detected and preserved rather than restored to the earlier baseline; all those changes remain outside this upgrade.

Future execution/development separation still deserves its own design. Do not weaken current source gates or replace old receipt hashes to accommodate maintenance. This round creates or modifies no temporary worktree and executes no research model experiment.

## Test architecture debt exposed by execution

Research's same 41 relocated operator/knowledge cases pass before and after, but the wider retained suite timed out with failures. A reproduced evidence-bound test depends on a missing artifact in a historical temporary worktree. Portable synthetic contracts and source-bound evidence replay need distinct ownership and explicit selection; missing historical evidence must not be treated as a successful unit test. No test was deleted merely to remove that failure.

Infrastructure's FA2 repair passes all 29 focused contracts. The full suite has 2079 passes, 34 failures and 5 skips. Remaining failures include CPU-hidden launcher mapping, incomplete session doubles and dependency/payload snapshots. Configuration artifacts/errors match all 26 frozen before observations, but that is not a proof that every broader failure predates this change. Separate local structural acceptance from branch runtime/release qualification.

## Proposed future capability movements

| Candidate | Recommended owner | Reason and next action |
|---|---|---|
| Infrastructure cache preparation, exact resume and authenticated checkpoint consumers | Infrastructure develops; main receives tested integrated capability slices | Strong reusable execution contracts. Promote with config migration, negative tests and a matching runtime witness, not a wholesale merge. Worth a dedicated next round. |
| Qualified dynamic HF/composed vLLM inference and bounded child supervision | Infrastructure first, then main when qualified | Runtime identities and receipts are source-bound. Preserve separate numerical/runtime admission and regenerate qualification under explicit GPU authority; do not copy historical receipts. |
| Generic research source/evidence mechanics, saved-artifact parsing and reusable calibration primitives | Infrastructure only after separating scientific policy | Potential reuse exists in src/artifacts/{git_identity,evidence_journal}, src/eval/saved_rows and src/rollout_calibration. First compare existing infrastructure counterparts and real callers; retain study-specific owner metrics, release policy and state banks in research. A focused extraction review is worthwhile, not automatic promotion. |
| COCO/Label Studio annotation and data recovery | Main / data-operation owners | These support human-maintained datasets, not a generic training runtime. Do not move them to research just because they are experiment-adjacent. Review duplicated domain mechanics independently. |
| Historical scientific conclusions currently reachable from main | Research catalog and question owners after distillation | Preserve exact Git/data provenance, retain only knowledge not already captured, then retire duplicate narratives. No bulk archive or file transfer in this round. |
| Live research run admission versus editable development | Explicit execution/development ownership | Whole-checkout cleanliness currently blocks concurrent maintenance. Plan the separation without modifying active source or treating dirty code as qualified. |

## Acceptance scope

This is an implemented structural change, not a claim that every branch is releasable or every legacy asset has been eliminated. validation.md records positive evidence and failed/limited gates. All changes made by this upgrade remain uncommitted; unrelated .codex changes and concurrent research work are excluded. The initial research hold, its revalidated release, and remaining source/receipt/dependency disagreements are recorded rather than hidden by skipped assertions or rewritten provenance.
