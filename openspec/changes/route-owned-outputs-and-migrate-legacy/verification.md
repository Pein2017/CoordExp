# Output ownership migration — verified partial completion, remaining HOLD

Date: 2026-09-30. Owner change: `route-owned-outputs-and-migrate-legacy`.

## Decision and actual execution

The selected migration and routing/tooling changes passed their bounded checks.
**Global root-output cleanup is NOT complete and MUST NOT be reported as accepted.**
The generic root layout checker still returns exit 1 with 217 findings. This change
remains open and unarchived for the explicit HOLDs below. No commit, staging, push,
GPU/model execution, process termination, dataset mutation or environment upgrade
was performed. Unrelated dirty source remains byte-identical to the captured baseline.

Main is `73822b0030934d28784b72a8a5325387b2f5cd41`; Research Probes is
`19af7b15db17a861ae73310b377b9002d889e0df`. Other existing worktrees were not synchronized.
The newly created `/data/CoordExp/.worktrees/main-runs` is detached at that exact main
commit, locked for asset retention, and has a clean index/worktree with ignored outputs.
This proves storage ownership, NOT current-source integration or model/environment
qualification. Its pinned source does not contain the still-uncommitted new operators.

## Why the wrong placement happened

Main's actual worktree policy told operators to launch official runs from repository
root, making relative outputs shared-root writes. The existing worker skill requested
message/report files and receipts without a placement distinction. Sampled real sessions
showed a main-produced audit script written under root outputs (session
`01a0e2e2-ef29-7291-9045-f631930359c1`, line 854), a Research Probes compatibility script
written there (session `01a0eec2-93f9-7242-b1bd-8e2a92cf584b`, lines 220/258), and a worker
sending root-output Markdown via `--message` (session
`01a0f0cc-423f-7773-a1ee-0a2d6c848424`, lines 1317/1561). See inventory.json for exact
producer cwd and scope. Session records were accessible and were not changed.

Fixed the existing main/research storage and worktree guides, root agent entry and shared
lead-worker skill. Human research interpretation goes to its question/unit; disposable
messages to task scratch; machine receipts to owner outputs. Root outputs is documented as
a selected shared-asset store. Existing path-bound research constants remain unchanged
and are not advertised as new-run defaults. These are routing instructions, not a new
universal filesystem enforcement framework. Other pinned worktrees retain their own
committed source/instructions until separately integrated.

A final instruction scan updated the main README/storage examples and the research-flow
contract to resolve outputs through the owning worktree. Current inference/evaluation
commands now live in the existing inference/eval Skill; docs/eval retains short routes,
artifact compatibility, and metric owners. The two annotation runbooks label their
existing root paths as project-specific bindings; their live services and stored workspaces
were left unchanged. The accepted val200 input remains its exact historical root locator.

## Completed batches

| Batch | Before -> after | Actual disposition |
|---|---|---|
| Main-produced COCO/LVIS review assets | `outputs/research/coco-lvis-proxy-exploration/` -> `.worktrees/main-runs/outputs/coco-lvis-proxy-exploration/` | 73 selected regular runtime files, 74,351,507 bytes, copied exclusively, SHA-verified and original retired individually. No whole-directory move. |
| Audit source | Two loose output scripts -> main `public_data/scripts/audit_coco_lvis_proxy.py` and `coco_lvis_pair_stats.py` | Parameterized maintained operators, explicit fresh output, no import-time processing. Original exact bytes retained in task rollback; not falsely described as committed Git recovery. |
| Research notes | Four original Markdown records -> Research Probes `research/questions/physical-evaluation.md` | Distilled finite-review definitions, positive/counterevidence and separate v2 export boundaries. No parallel historical report package. Original exact bytes retained in task rollback. |
| HF compatibility | Root `runtime-optimization/2026-09-29-vllm-dora/hf-compat/` -> same suffix below Research Probes outputs | Four result/log files SHA-verified as owner copies. **Originals retained HOLD** because parent final-acceptance.json binds absolute paths/hashes. Extracted scoped source is `probes/runtime_compat.py`; old probe.py remains. |
| Deliberately shared checkpoint | Original start-loss `instance_margin-order17/checkpoints/step-256` -> `outputs/shared/checkpoints/start-loss-instance-margin-order17-step256/` | Six payload files (88,738,129 bytes) plus unchanged run/config evidence copied; outer provenance.json records sharing decision. **Original retained HOLD**, no whole benchmark promotion or consumer cutover. |

The manifest records **91 per-file transfers**: 73 runtime moves, six source/report
rollback transfers, four HF owner copies and eight checkpoint/provenance copies. A final
independent read verified all 91 destination hashes, absence of the 79 retired originals,
identity of the 12 retained originals, all seven extraction source/target identities and
five pre-existing dirty-file hashes. No symlinks substitute for retained copies.

The selected checkpoint's real recorded producer is the start-loss worktree at
`01941e687d7ea197df0667701bad5e8e40235652`, with repository state **dirty** and recorded
execution-relevant-change count zero. That historical state is preserved, not relabeled
as clean/current main. Its five component files additionally match the payload manifest's
relative roots, sizes and hashes. The base model, tokenizer and data remain external
inputs: this is not an independent full-model or dataset backup. adapter/README.md is
hash-bound generated package metadata, not a handwritten research report; it remains an
explicit compatibility exception and is not concealed from the generic checker.

One checkpoint preflight initially stopped on an assumed component key before any shared
checkpoint directory was created. Earlier recorded transfers remained intact. Only that
preflight/copy stage was retried using the actual `special_token_embedding_delta` plus
`relative_root` schema; no old checkpoint or receipt bytes changed.

## Current validation

| Check | Result and scope |
|---|---|
| Main data-audit tests | 10 passed; malformed labels/hash/identity/notes, no clobber, read-only input behavior, geometry/crowd/denominators. |
| Research compatibility tests | 3 passed; import safety, explicit-input checks and no overwrite. No model load. |
| Saved-result parity at NEW location | Exact aggregate dict equality for six relations / 192 reviewed images plus 32 separate scene-inference records. |
| CPU check-inputs on real research inputs | Passed; input hashes recorded and no output file created. Not checkpoint/native qualification. |
| New owner output layout | Fresh check: main-runs 73 files and Research Probes 94 files; both pass with zero findings/symlinks. |
| New-run instruction routing | Generic README/eval/research-flow examples resolve to the owning worktree root; current annotation roots are named exceptions. |
| Eval workflow routing | Current run commands and the historical test-dev recipe are in the inference/eval Skill; docs/eval pages route to the Skill or retain contract/interpretation ownership. |
| Research knowledge checker | Passed: 329 catalog entries, 324 distilled, five current, 165 claim references, no errors. Existing catalog semantics preserved. |
| OpenSpec strict validation | Passed with supported `skip_specs: true`: no training/checkpoint/source-identity/scientific behavior contract changed; existing spec owners remain authoritative. |
| Main/research git diff --check | Passed; no staged paths and all existing worktree HEADs unchanged. |
| Entire root output layout | **NOT PASSED: 128,582 files, 217 source-or-prose findings; 125 symlinks not followed.** This is remaining debt, not a successful cleanup claim. |
| WebCodex advisory closeout | **fail / unexpected_tool_failures**: three unexpected historical tool failures, two categorized actionable; zero active Jobs, zero workspace conflicts and no secret-like paths. The advisory ledger is not fully green; the successful direct post-migration checks above do not erase prior failure events. Dirty workspace/hygiene remains intentionally visible because no staging/commit/cleanup was authorized. |

Complete command logs are in root and Research Probes
`.local/scratch/route-owned-outputs-and-migrate-legacy/validation/`.
`migration.json` is the finite per-file integrity/recovery map; `inventory.json` contains
before/after metadata and every initial loose-source/Markdown disposition;
`reference-check.json` records bounded consumer checks, not an exhaustive future-user proof.
No full raw-data pair-statistics pass, GPU generation/backward, full training or benchmark
was rerun. New source remains uncommitted/untracked where newly added.

## Remaining HOLDs and required evidence

The 217 generic findings consist of **141 generated package cards** (including the
new selected copy), **57 research Markdown records**, **10 frozen-source/capture files**,
**eight legacy scripts** and **one extracted-but-retained HF script**. Initial 222 source/
Markdown candidates had six originals retired; adding the selected package card yields217.

- Active root research runs / annotation state: a fresh 2026-09-30 process check found
  Gate A PID903507 holding `coco_refinement/.writer.lock` on listener 127.0.0.1:53662,
  Label Studio PID2447154 serving on 127.0.0.1:8080, historical-review server PID1116924
  serving from a `research/` candidate directory on port 8766, and a detached `prod` tmux
  session with tee processes holding old training/rollout logs. None was stopped. The
  annotation workspaces remain path-bound HOLDs; stop them only if an authorized path
  migration needs it. Active research/log holders still prevent retirement of their files.
- HF results and original selected checkpoint: immutable/current consumers still bind old
  paths and hashes. Require a versioned consumer migration and acceptance of both old/new
  identities before retirement; never edit frozen receipts in place.
- Five remaining COCO/LVIS one-off generators, three infra launchers, frozen captures and
  recent research reports: resolve retained executable dependency closure or exact Git
  recovery and the actual scientific question/unit owner. Current useful audit entrypoints
  are documented; old loose generators are historical HOLD, not supported launch routes.
  The four distilled notes do not imply every human report was reconciled.
- Other old runtime families, environment/package backups and empty skeletons: resolve the
  per-run producer/owner, intended retained asset, current path consumers and independent
  recovery or verified regeneration. Names like archive/research/prod do not prove sharing
  or authorize blanket relocation. Per-root evidence and suggested targets are in inventory.

The verified rollback copies under root `.local/scratch/.../originals/` must remain while
replacement source/doc changes are uncommitted. To recover a retired original, look up
its recorded destination/rollback path, verify its hash, then copy only to an absent original
path. Do not overwrite a path another writer recreated. Rollback is same-host task
protection, not an independent backup. Main-runs stays locked until its contents and
consumers are explicitly resolved; no automatic teardown on OpenSpec closure.

## Physical directories named `outputs`

A repository-wide directory-name sweep found 18 paths named `outputs`. Only three are
CoordExp run-output roots: `/data/CoordExp/outputs`, `main-runs/outputs`, and
`research-probes/outputs`; these are the roots covered by the layout check above. The
other 15 are empty Label Studio web-output directories, Baidu sync staging/manifests,
task-local rollback bytes, prior maintenance snapshots, a Pi subagent artifact directory,
or an `outputs` path used by a historical worktree config. They have separate owners and
were left untouched; their names alone do not make them CoordExp run roots. The Baidu
download staging tree alone contains 26,616 files / 21,058,297,117 bytes and needs its
sync owner's retention decision before any cleanup.

## Changed source/document files

Main checkout (plus this change's proposal/design/tasks/inventory/reference/migration/verification):
- `AGENTS.md`
- `docs/AGENT_INDEX.md`
- `docs/OUTPUT_STORAGE_POLICY.md`
- `docs/BRANCH_AND_WORKTREE_POLICY.md`
- `.codex/skills/lead-worker/SKILL.md`
- `.codex/skills/research-flow/references/research-graph-contract.md`
- `README.md`
- `docs/eval/WORKFLOW.md`
- `docs/eval/README.md`
- `docs/eval/CONTRACT.md`
- `docs/eval/COCO_TEST_SUBMISSION.md`
- `.codex/skills/coordexp-infer-eval-workflow/SKILL.md`
- `.codex/skills/coordexp-infer-eval-workflow/references/current-workflow.md`
- `.codex/skills/coordexp-infer-eval-workflow/references/historical-coco-testdev.md`
- `docs/data/COCO_REFINEMENT_RUNBOOK.md`
- `docs/data/COCO_REFINEMENT_STANDALONE_RUNBOOK.md`
- `public_data/README.md`
- `public_data/scripts/audit_coco_lvis_proxy.py`
- `public_data/scripts/coco_lvis_pair_stats.py`
- `public_data/tests/test_audit_coco_lvis_proxy.py`

Canonical Research Probes:
- `docs/OUTPUT_STORAGE_POLICY.md`
- `docs/BRANCH_AND_WORKTREE_POLICY.md`
- `docs/eval/README.md`
- `docs/eval/WORKFLOW.md`
- `docs/eval/CONTRACT.md`
- `docs/eval/INTERPRETATION.md`
- `probes/README.md`
- `research/index.md`
- `research/questions/physical-evaluation.md`
- `probes/runtime_compat.py`
- `probes/tests/test_runtime_compat.py`

Pre-existing dirty shared AGENTS, same-host transport reference, runtime-performance skill,
v2 exporter and its test were preserved byte-for-byte. The start-loss worktree's existing
dirty result record is untouched; coordexp-infras and research-probes-web-codex remain clean.

## Complete top-level root census after this pass

Initial census: 29 roots, 128,652 regular files, 252,776,876,923 logical bytes.
Final census: 30 roots, 128,582 regular files, 252,791,314,203 logical bytes.
The selected shared real copy increases retained bytes; this is ownership repair, **not a
claimed disk-space saving**. Existing root families remain HOLD unless a selected leaf is
explicitly identified. Full per-root owner/producer/ref/holder/recovery evidence is in inventory.

| Root child | Regular files | Logical GB | Disposition |
|---|---:|---:|---|
| `_baidu_filename_mapping` | 2 | 0.000081 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `_probe_labelstudio_bootstrap_1k_20260716` | 20 | 0.013304 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `archive` | 305 | 1.923538 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `async-batch-probe-10k-100-fix-20260716` | 9 | 0.021837 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `async-batch-probe-10k-100-fix-final-20260716` | 9 | 0.021837 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `attestations` | 2 | 0.000006 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `bench` | 1,579 | 0.844212 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `benchmarks` | 24 | 0.000233 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `cache_logs` | 1 | 0.000007 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `coco_refinement` | 114 | 0.695368 | HOLD_ACTIVE_HOLDER |
| `codex-audit-async-batch-tiny` | 9 | 0.000037 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `coordexp_swift` | 396 | 0.127482 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `environment-upgrades` | 150 | 5.499522 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `eval` | 494 | 0.144178 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `git-hygiene-owner-scale-20260914-CYqDZ1` | 12 | 0.000628 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `infer` | 8,057 | 6.635544 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `infra_base` | 4,499 | 72.150600 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `label_studio_coco_refinement` | 414 | 5.493634 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `label_studio_reviews` | 51 | 0.004604 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `launch_logs` | 3 | 0.000025 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `oracle_k` | 293 | 0.208597 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `prod` | 79 | 0.611811 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `research` | 111,641 | 158.256291 | HOLD_ACTIVE_HOLDER |
| `research-probe-forks` | 115 | 0.040066 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `research-probe-infras` | 255 | 0.002483 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `research-probe-infras-final-smoke-flykAS` | 4 | 0.000652 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `research-probe-infras-validation-gHpGim` | 4 | 0.000652 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `runtime-optimization` | 32 | 0.005270 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |
| `shared` | 9 | 0.088814 | SELECTED_SHARED_CHECKPOINT_ONLY |
| `stage1_2b_base` | 0 | 0.000000 | HOLD_LEGACY_OWNER_OR_PATH_RECOVERY |

## 2026-09-30 user-directed closeout update

The user authorized absorption of the remaining research notes, and deletion of their output copies after preserving exact source identities and verifying owner coverage. The 56 research records (414,700 bytes) were mapped into their existing owners: self-prefix/online correction 9, hidden-human pilot 2, rollout row-credit 21, online row-credit 15, and iterative-positive recovery 9. The owner map and per-file original path/size/SHA-256 are recorded in `inventory.json`; each entry is now `ABSORBED_AND_RETIRED` with its destination, retirement time, and absence receipt. The two unique omissions found in the record audit were added to the self-prefix owner: the LOCAL duplicate/termination objective diagnosis, with causal limits, and the exact paired1-02 owner-serialization `TypeError` disposition.

Before retirement, all 56 files were regular non-symlink files and matched their recorded byte counts and SHA-256 values. No live process cwd/root/fd referenced the `hidden-human-annotation-recovery` source tree, and a tracked maintained-source/docs path scan found no consumers. After unlink, all 56 exact paths were absent. The research knowledge checker passes after the owner edits and removals: 329 catalog entries, 324 distilled, five current, 165 claim references, zero errors. No other output code was executed.

Inventory record 1 was corrected from `human_record` to `generated_package_card`: the native-adapter README is a PEFT model-card template with no research result. It remains on `HOLD_GENERATED_PACKAGE_COMPATIBILITY` until its producing package/checkpoint identity is reconciled.

Fresh output-layout check after this pass: shared root has 128,526 files and 161 source/prose findings (exit 1); Research Probes has 141 files and passes. The 56 retired notes account for the entire decrease from 217 to 161 findings. The exact remaining generated cards, frozen source captures and scripts remain governed by their individual inventory dispositions. The root is not accepted as clean; active annotation roots, current run dependencies, and unresolved package/legacy holds remain. `openspec validate route-owned-outputs-and-migrate-legacy --strict --json`, `git diff --check`, and the inference/eval skill validator pass.

The user also authorized metadata-only retirement of frozen source/script artifacts, with no rerun or reusable copy. Ten frozen captures and nine one-off scripts (340,935 bytes) were hash-verified against `inventory.json`, had no matching live cwd/root/fd/argv holder or maintained source/doc path reference, then were removed by exact path. Their original paths, sizes, hashes and deletion receipts remain in the inventory; no executable source copies were made. The frozen preparation note is linked to the hidden-human recovery unit and pilot result.

For `outputs/infer`, current maintained source/research/OpenSpec references named seven immediate run roots. Those seven remain intact. The other 222 immediate roots had no current path reference or process holder; 4,328 regular files (5,795,785,484 bytes) were hash-recorded and removed. The family now has 3,729 regular files in the seven referenced roots. An empty-directory pass first stopped on a residual `vis_resources` child; a fresh check found no symlinks or data there, and only empty directories were subsequently pruned. No symlink target was followed or deleted. Full file receipts and the retained-root list are in `inventory.json`.


## 2026-09-30 exact step-256 README restoration

At 2026-09-30T20:41:54+00:00, the exact 5294-byte `adapter/README.md` bound by the frozen step-256 inference payload manifest was restored by exclusive no-clobber creation from the same adapter `model_card.json` `content_utf8` field. The wrapper is 5684 bytes, SHA-256 `fc9e4f71f3885fd42a808c5ee953d1c2b36f3ce1f7a94a717cfaaca1a3f84b19`, declares `original_name: README.md`, and its embedded bytes have SHA-256 `2860abd12bc4395ca0cb2fd2af9582eb6c46a7e8e8525d0f6871b56a831af4bd`. This establishes lossless metadata recovery, not independent authorship/custody.

Before restoration the README path was absent and the other four listed payloads matched their manifest sizes and SHA-256 values. After restoration, all five payloads match; `inference_payload_manifest.json` remains byte-for-byte unchanged at SHA-256 `1fdd43eb9e94f21379b1494ca6ee51f65046b68f718787970ac8a5ce3ce420a1`, and the wrapper and four pre-existing payloads are unchanged. The shared selected-copy path remains absent.

`inventory.json` retains the historical `MIGRATED_TO_JSON_METADATA` retirement event and records the later restoration at `/artifact_retirements/2/files/4`; `/loose_source_and_markdown/8` now records the recovered source and protection from re-retirement while this manifest consumes it, retaining its compatibility hold. `migration.json` `/transfers/83` retains the historical copy status and now records that its shared destination is absent.

The exact before/after payload hashes, source and manifest identities, metadata owner hashes, and worktree checks are in `/data/CoordExp/.worktrees/research-probes/outputs/research/hidden-human-annotation-recovery/2026-09-30/recall-with-error-floor-01/lead-anchor-review-01/anchor-restoration-receipt.json`. Research Probes remained at HEAD `15851aaaf13fa3e16c67e2cf16cf5107cced5b5d` with a clean tracked/untracked status before and after. No model, GPU, native binding retry, device/resource query, evaluator/hidden-truth access, or full output-layout scan was performed; this is dependency recovery only, with no runtime release.

## 2026-10-01 fresh output-layout recheck: manifest-bound adapter cards remain held

The fresh read-only layout command returned exit 1: `/data/CoordExp/outputs` has 42,060 files, 102 symlinks not followed, and one `source_or_prose_in_outputs` finding; `/data/CoordExp/.worktrees/main-runs/outputs` has 73 files and passes; Research Probes outputs has 2,209 files and 12 findings. All 13 findings are `adapter/README.md` package metadata.

The shared-root finding, `/data/CoordExp/outputs/infra_base/start-loss-benchmark-20260928/train/instance_margin-order17/checkpoints/step-256/adapter/README.md`, is 5,294 bytes with SHA-256 `2860abd12bc4395ca0cb2fd2af9582eb6c46a7e8e8525d0f6871b56a831af4bd`. The unchanged `inference_payload_manifest.json` (136,464 bytes; SHA-256 `1fdd43eb9e94f21379b1494ca6ee51f65046b68f718787970ac8a5ce3ce420a1`) binds that exact relative path, size, and digest. Its sibling generated `model_card.json` remains 5,684 bytes with SHA-256 `fc9e4f71f3885fd42a808c5ee953d1c2b36f3ce1f7a94a717cfaaca1a3f84b19`. This is the exact restored payload already recorded at `inventory.json` `/loose_source_and_markdown/8`; deleting it would break the frozen payload manifest.

The 12 Research Probes findings are also sealed package files, not loose research prose. Each run below binds four files—`control/checkpoint-{0,1}/adapter/README.md` and `treatment/checkpoint-{0,1}/adapter/README.md`—at 5,294 bytes and SHA-256 `a741199073be194a22e03b5aa5e51580dec81281b384e92514f5afc4b1afb594`:

- `outputs/research/hidden-human-annotation-recovery/2026-09-30/recall-with-error-floor-01/native-identity-paired1-03/worker-native-terminal-manifest-01.json`, SHA-256 `5b7c72e2e95dfc1f0a480b5f81daf410fde7f709e741c64476c88881062dd87c`.
- `outputs/research/hidden-human-annotation-recovery/2026-10-01/chain-mass-restoration-01/native-paired1-01/worker-terminal-manifest-01.json`, SHA-256 `a5b13688381742c28865ffbd63e98b944f9f48a889948c0e4e4516b9beee8a58`.
- `outputs/research/hidden-human-annotation-recovery/2026-10-01/chain-mass-restoration-01/native-paired1-replication-01/worker-terminal-manifest.json`, SHA-256 `71939b1355296b65699895eeac72d0e7b3b52d231437b0745151e1fe59a277c0`.

Disposition: preserve these 13 exact files and their sealed manifests; do not rewrite or retire them in place. The output-layout scanner does not yet encode this generated, manifest-bound metadata exception, so the roots still fail its suffix rule. OpenSpec tasks 4.2 and 4.4 remain unchecked; overall cleanup/migration is not closed. No files were deleted, moved, or rewritten during this recheck.

## 2026-10-01 Research Probes worktree-local archival copy

The scoped research migration receipt at `/data/CoordExp/.worktrees/research-probes/outputs/coordination/artifact-migrations/2026-10-01/research-probes-01/migration-receipt.json` has SHA-256 `1fe7f03092d3f0b2febd42b291d61526141fe6cce4583115f28087dd7e9ab0ef` and status `VERIFIED_COPY_WITH_EVALUATOR_PAYLOAD_HOLDS`. Its copy manifest SHA-256 is `4a807c1fc17b4a090f8985a668b7e656fb03573455826a0fcae6fb78a7b6fab5`, routing SHA-256 is `85551f364eed304520718f644f1e779ccf9215a8deaf119d1cc613f3e6880976`, and sealed migration manifest SHA-256 is `9901626d0b7a5dee4393dab9ca793290268192eed8ba9844b779bbc8e293cd76`. These digests were independently rechecked. The receipt records 30,004 regular files / 13,150,106,783 bytes copied with independent inodes, source bytes and sealed path bindings unchanged, zero additional GPU hours, and future writes routed to `/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/`. Its declared seven evaluator truth/partition payload holds remain untouched; this audit read the summary receipt and file digests only, not the per-file manifest contents or copied payloads.

A fresh full layout check after the copy reports: shared `/data/CoordExp/outputs` 42,060 files / 1 finding; main-owned outputs 73 files / pass; Research Probes outputs 32,232 files / 24 findings. The 24 Research Probes findings are the 12 original sealed `adapter/README.md` files recorded above plus their 12 copied counterparts under `physical-fn-recovery`; the copy preserves rather than retires the original sealed paths. The layout checker still flags these manifest-bound package cards, so the copy does not close the package-metadata policy HOLD. Generic future output routing is already worktree-local under `docs/OUTPUT_STORAGE_POLICY.md`; OpenSpec tasks 4.2 and 4.4 remain unchecked, and the global migration remains open.

## 2026-10-01 current tracked-file literal-reference scan

Read-only `git grep -I -l -F` checks covered the current tracked worktree files for the literal shared root, Research Probes root, main-runs root, and new `physical-fn-recovery` root. Counts below are matching files, not lines. The new root appears in only two Research Probes records (`research/experiments/2026-09-29-self-prefix-known-fn/results.md` and `state.json`); the current storage policy already routes future branch-owned outputs to the owning physical worktree.

| Current worktree | HEAD | Shared root | Research Probes root | main-runs root | physical-fn-recovery root |
|---|---|---:|---:|---:|---:|
| `/data/CoordExp` | `7189768ac` | 671 | 2 | 2 | 0 |
| `.webcodex-managed-worktrees/research-probes-012001b3` | `0943f0fb1` | 30 | 0 | 0 | 0 |
| `.worktrees/coordexp-infras` | `bfd276d0b` | 54 | 0 | 0 | 0 |
| `.worktrees/main-runs` | `73822b003` | 666 | 1 | 0 | 0 |
| `.worktrees/research-probes` | `c46ad0d51` | 52 | 4 | 1 | 2 |
| `.worktrees/research-probes-web-codex` | `6ab709bc4` | 52 | 0 | 0 | 0 |
| `.worktrees/start-loss-benchmark` | `bf1993aef` | 57 | 0 | 0 | 0 |

These are literal tracked-file matches and include historical documentation, records, configs, and code; they do not prove that every match is a live consumer. They do show that blanket shared-root retirement is unsupported without per-owner closure. The scan did not read output payloads or inspect the seven evaluator truth/partition files. Existing owner-backed holds remain; no path or manifest retirement is authorized by this scan.

The copy receipt records an empty Research Probes tracked status before and after its copy at 2026-10-01T16:58:38Z. At this later verification, `git status` shows `research/experiments/2026-09-29-self-prefix-known-fn/results.md` and `state.json` modified; this output-migration task did not edit either file and leaves them untouched. Current knowledge-check output remains 329 entries / 324 distilled / five current / 165 claim references / zero errors, and OpenSpec strict validation passes. No paths were staged or committed.

## 2026-10-01 17:36 UTC final acceptance check

The required full output-layout check was rerun with details. Shared outputs contain 42,060 files and 102 untraversed symlinks, with one finding at the frozen step-256 `adapter/README.md`; main-runs has 73 files and passes; Research Probes has 32,232 files and 24 findings, all `adapter/README.md` files. These are the exact manifest-bound package metadata holds documented above: 12 original Research Probes cards plus 12 copied counterparts, and the shared step-256 card. The current tracked-file reference scan, recorded above, was also reviewed; its historical and config matches do not establish blanket retirement safety.

The research knowledge checker passes (329 entries, 324 distilled, five current, 165 claim references, zero errors). OpenSpec strict validation passes after this record and task update. The existing modified self-prefix `results.md` and `state.json` remain untouched; nothing was staged or committed. Task 4.4 is complete because the fresh checks passed and each exact remaining layout finding has an explicit manifest-backed HOLD. Task 4.2 remains open for unresolved per-family legacy owner/path and consumer dispositions; the change remains unarchived.

## 2026-10-02 follow-up: shared payload repair and launch routing

The cross-chat closeout found the selected shared checkpoint's
`payload/adapter/README.md` absent although its unchanged payload manifest still
required it. The original step-256 README and the shared `model_card.json`
embedded bytes agreed at 5,294 bytes / SHA-256
`2860abd12bc4395ca0cb2fd2af9582eb6c46a7e8e8525d0f6871b56a831af4bd`.
Exclusive no-clobber creation restored only the missing shared file. Before
repair, exactly one of five manifest payloads failed (absent README); afterward,
all five matched their declared size and SHA-256. The manifest, JSON wrapper and
original README remained unchanged. No model or evaluator was run.

Machine evidence is worktree-local under
`/data/CoordExp/.worktrees/main-runs/outputs/maintenance/output-storage-closeout-20261002/`:
`shared-payload-repair.json`, `layout-check.json`, `root-census.json`, and
`readme-routing-check.json`. These receipts supplement historical observations;
they do not rewrite sealed producer records or authorize original retirement.

The post-repair layout checker exited 1: root outputs had 42,061 files and 102
untraversed symlinks, with two README findings; Research Probes had 32,232 files
and 24 README findings; main-runs passed. The new root finding is the restored
shared copy of the same manifest-bound package metadata, with the same
compatibility HOLD. All 26 findings must retain their sealed bytes. The scanner
still uses a suffix rule; its nonzero result does not indicate a corrupt repaired
payload. All seven checkouts have zero Git-tracked paths under `outputs/`.

The metadata-only root census totals 59,431,973,668 logical bytes across 30
children. Largest families are `infra_base` (29.42 GB), `research` (15.45 GB),
`environment-upgrades` (5.50 GB), and `label_studio_coco_refinement` (5.49 GB).
These are current counts, not new shared-asset classifications. Research Probes'
13.15 GB migration remains a verified copy with seven evaluator payload holds;
it has not retired the original paths.

README's training quick start now explicitly enters `main-runs`; branch-owned
runs enter their own physical checkout. A no-launch config/resolver check
reproduced the old root-output destination and verified the documented main-runs
and branch destinations. Gate A's fixed annotation namespace and containment
remain unchanged pending the user's shared-state versus service-migration choice.

The user's earlier hold on the output/OpenSpec commit batch remains in force.
The 30.3 MB inventory remains in place pending its evidence-retention decision.
No staging, commit, push, service restart, worktree removal or bulk retirement
was performed. Task 4.2 stays open; this change is not archived. The existing
main-runs skill deletions and Research Probes record edits were left untouched.


## Latest user ruling and usefulness audit — current closeout

This section supersedes older completion/HOLD and cutover statements above.
Tasks 4.2 and 4.4 remain open. The user explicitly defers annotation relocation
to the next public-data migration. Gate A stays at
`/data/CoordExp/outputs/coco_refinement/gate-a-20260717/`; its existing root
annotation payloads are not removed or moved in this round. Published COCO
views already remain in `public_data/coco`. No Gate A restart or GPU run occurred.

A move into main-runs is not usefulness qualification. The earlier 32-group,
9,900-entry, 37,360,578,672-logical-byte move is recorded in
`unheld-family-moves.json` as a physical operation only, not final retention
acceptance. Most moved content consists of models, data, logs and evidence;
there are no regular `.py`/`.sh` files or executable regular files in those
selected groups. Opaque patches/source archives were not established code-free.
Unknown legacy families must not become a permanent generic output collection.

### Accepted retention and actual retirement

Small maintained provenance is in root `manifests/runtime_assets/`. Root shared
outputs retain only the selected package bytes below, not their donor runs.

| Selected asset | Concrete purpose / limitation |
|---|---|
| `shared/checkpoints/start-loss-instance-margin-order17-step256/payload` | Existing Research Probes native anchor; exact generated README restored without changing its sealed manifest. CPU real-anchor parser passes: 588 keys / 196 targets. |
| `shared/checkpoints/untied-axis001-step2444/payload` | Current hidden-human recovery probe and Research Probes asset input; all five native manifest members verified. |
| `shared/checkpoints/four-coordinate-xy-step2444/payload` | Current tied-coordinate/Image2299 input; four original components verified. Original native-manifest absence and unknown producer commit are explicitly preserved. |
| `shared/checkpoints/untied-illegal-mass001-step2444/payload` | Exact baseline for the named completed start-loss comparison; preserve inference weights and exact resolved config. It is a different model from untied-axis001. This does not release the old comparison launcher. |
| `shared/wheels/flash-attn-2.8.4-cp312-cu129-sm80/` | Locally built SM80/CUDA12.9 wheel; its native library matches the installed binary SHA-256 exactly. Fresh binary parity is not a new GPU qualification. |

The current root shared census has 27 regular files / 579,942,306 logical bytes.
Weights and retained package metadata were independently hash-checked. Their
original selected checkpoint directories were retired after fresh holder checks;
obsolete training states were not promoted. Existing signed/frozen producer
fields are not relabeled with the storage checkout's HEAD.

Actual discarded cache/training-state file sizes total **31,451,514,094 bytes**:
27,030,707,102 bytes of unused wheel downloads, expired smoke materializations,
packing/compile caches and derived debug execution models;
2,290,972,788 bytes of nonshared training state from the selected A/B checkpoint
originals; and 2,129,834,204 bytes of illegal-mass baseline training state.
These are logical file sizes, not a `df` physical-space claim. Exact hashes,
identities, absence and holder checks are in the corresponding JSON receipts.

Main and main-runs historical source captures each lost 54 exact `.py`/`.sh`
copies, for 108 executable historical copies removed. Clean Git blobs remain
recoverable; 60 historical Markdown records in each tree remain. Their large
machine identity manifests moved to the current maintenance evidence owner.
The empty `outputs/stage1_2b_base` directory tree was pruned. No compatibility
symlinks or historical executable backups were created.

### Current consumers and annotation state

New training/inference destinations resolve paths before rejecting root outputs
and descendants, including symlink/dirname escapes. The same bounded guard is
present in all five active worktrees. The focused main train/infer checks passed
88 tests; branch-bound destination probes passed. Shared checkpoints remain inputs.
Research Probes' hidden-human checkpoint and two worktrees' native anchor test
locators now use the selected shared packages. The named CPU parser passed.
Start-loss's output ROOT is worktree-local; its two continuation tests passed.
Its old frozen data/template/source-run bindings still need fresh qualification
before a future rerun; a path edit is not runtime release.

The old Label Studio listener exited normally. Exactly two pending drafts were
exported to `public_data/coco/annotation_drafts/retired-label-studio-20261002/drafts.jsonl`
with small Git-managed identity in `manifests/annotation_drafts/`. Independent
SQLite integrity and full selected draft/task/linked-annotation row equality
passed; draft bytes have SHA-256
`dcdbdde5ca82d486147a4bff7374e96768a7275b15f8f0521afd0a55aabe8bfe`.
They are pending, neither published ground truth nor imported into Gate A.
Gate A remains accepting writes with its writer lock held, 122,218 tasks,
5 drafts and 506 mutations at the checked snapshot. Its published views and
counters were preserved. The focused annotation checks passed 78 tests plus
two Label Studio contract tests; this is not a claim that the unrelated extra
public-data smoke-file failure in the broader suite was fixed.

### Remaining dispositions, not accepted legacy retention

- Annotation trees and live Gate A: **user-deferred relocation**; preserve this
  round. The retired service must not be restarted.
- Research Probes physical-fn-recovery: current worktree owner is valid, but the
  earlier 30,004-file / 13,150,106,783-byte full copy is not automatically the
  minimum scientific retention. Its seven evaluator truth/partition holds were
  not read or hashed. Source duplicates and exact evidence scope remain open.
- Old benchmark/eval/oracle/prod/recursive-detection/PVci families: audit useful
  conclusions and critical evidence, then retire unneeded bytes. The nine
  recursive-detection config references resolve to legacy `src.analysis` imports,
  whose implementations exist only under `reference/legacy_src`; these are not
  working maintained consumers and do not justify whole-family retention.
- Main-runs' remaining infer/bench/infra_base and the completed start-loss bundle:
  preserve only actual current inputs, explicitly needed final weights and key
  evaluation evidence. The question whether a named old experiment needs its
  complete trajectory for mechanism analysis is pending the user's answer.
- Root research/archive/probe families with live review/log holders: reconcile
  the concrete consumer and owner before retirement. Ambiguous research
  producers and unrelated dirty edits are untouched.

Full per-file evidence remains in the current task's maintenance output directory;
`inventory.json/current_closeout/receipts` binds its exact paths, sizes and hashes.
This is evidence for this maintenance pass, not a historical binary dumping ground.
No staging, commits, pushes, worktree removal, scientific reruns or annotation
relocation were performed. The OpenSpec change remains unarchived and partial.


### Fresh final checks for this candidate

The corrected layout scanner handles each nested native payload manifest instead
of checking only the scan root. Its focused RED/GREEN nested-card and symlink
regressions pass; the complete affected artifact suite passes **52 tests**.
The actual shared-root command passes: 27 files, zero findings. The broader
metadata-only scan also passes root outputs (32,170 files, 97 untraversed links)
and main-runs (8,476 files, five untraversed links). Research Probes still has
26 README findings across 32,253 files: these are not covered by the narrowly
qualified native-payload exception and must not be deleted blindly. Layout
passing is not usefulness acceptance. Details are in `final-layout-check.json`.

OpenSpec strict validation and `git diff --check` pass. Every inspected checkout
has an empty staged set. Main-runs' 88 pre-existing skill deletions remain;
Research Probes' unrelated scientific results/state edits remain untouched.
Gate A's original process is still live. Its on-disk health receipt reports
ready/accepting-writes/lock-held, but has a historical `updated_at`; it is not a
fresh live-health measurement. Unauthenticated live state requests correctly
return 401. No editor session was created and no service state was changed.


## Retained worktree ruling and completed temporary-worktree retirement

The user retains exactly main/root, research-probes, research-probes-web-codex
and coordexp-infras. This supersedes the earlier main-runs lifecycle decision.
The temporary main-runs/start-loss-benchmark and detached WebCodex checkouts are
removed. Local pilot/benchmark branches are deleted only after their commits
are reachable from retained branches. No remote branch/ref or transcript was
modified. The canonical Research Probes lock remains.

The benchmark's eight commits fast-forwarded into coordexp-infras; output routing
is committed at `e68169b0a`. Its loss/continuation checks passed 28 tests and
branch config/inference checks passed 43. The old pilot's seven commits are
historical parents of Research Probes via an explicit ours merge; unused probe
code was not restored. Its exact bounded scientific result and current COCO/LVIS
evidence locator are committed at `14e45168a`.

Necessary benchmark weights/evaluations/input identities and compact old
endpoint/runtime-acceptance evidence reached purpose-named retained owners.
Important frozen configs/receipts are outside outputs; raw datasets, predictions
and model bytes use the actual owning worktree's outputs. All prior temporary
source edits match retained source owners, and unrelated dirty file hashes are
unchanged. Force worktree removal discarded only redundant tracked cleanup
state and bytecode/pytest caches after asset qualification and fresh no-holder
checks; the main-runs lock was explicitly unlocked under this lifecycle ruling.

The typo `output/` (83 files / 82,206,073 bytes), `output_remote`,
`output_remote_DEPRECATED_20260604`, temporary output remnants and benchmark
preparation cache are gone. The latter batch retired 17,404 entries /
11,150,346,182 logical bytes; important compact endpoints were retained first.
These sizes are not a physical disk-space claim. Source configs pointing into
old configs/infer are unsupported by current loaders; they did not justify
retaining the typo model. Closed prefix-denoising records now explicitly mark
the old full baseline as retired and preserve the confounded-comparison limit.
The obsolete Label Studio exporter with a removed worktree default was removed;
Gate A and deferred annotation roots were not relocated or restarted.

The exact machine retirement/transfer/preflight records now live in
`/data/CoordExp/.worktrees/coordexp-infras/outputs/maintenance/output-storage-closeout-20261002/`.
Older paths above describe historical captures, not current evidence locators.
The small current inventory and selected runtime manifests point at this retained
maintenance owner; the full machine inventory bytes are unchanged. This is one
finite maintenance packet, not a legacy runtime collection. Global tasks4.2/4.4
stay open for remaining root research duplication/package qualifications and the
explicit annotation relocation deferral; worktree retirement itself is complete.

### Root temporary-directory and template closeout

Root temp, tmp/.tmp, monitor_dumps, tb and disposable caches are retired.
The temp batch removed 33,012 entries / 28,503,887,939 logical bytes after
current-holder checks and unsupported historical-binding qualification; no
Gate A process referenced it. Five dataset symlinks were unlinked without
following targets. The cache/log batch removed 158 entries / 20,772,589 bytes.
Together with the typo and temporary-output batches, this worktree-retirement
turn removed 50,657 entries / 39,757,212,783 logical bytes. This is separate
from the earlier 31,451,514,094-byte retirement above.

Only 72 unique otherwise-unrecoverable historical Markdown records survive
in flat Git-managed docs/history/retired-temp-reports-20261002, under the
user's explicit Markdown exception. Generated cards and records already
reachable from retained Git history were not duplicated. Source paths and
hashes remain in temp-markdown-retention.json; no executable/config/model
archive was created.

Four live infrastructure output declarations are corrected at 5bea60f.
A pre-change failing template test and fresh 390 passing config/inference
tests verify all 11 actual train/smoke/infer loaders and run-directory
resolvers. Inference binds relative paths to declaring config directories;
training output resolution uses launch cwd. Model/data/training semantics
and sealed historical config bytes are unchanged.

Current per-file receipts are hashed by inventory.json. Only this bounded
branch/worktree retirement is complete; broader global migration tasks4.2/4.4
remain open and the prior partial OpenSpec commit hold remains in force.

The Research Probes and Web output guards are now committed at 55f74a7 and
87ec0fe. Each checkout passed 132 focused config/inference checks, including
shared root, parent-directory and symlink rejection plus branch-local admission.
Captured dirty-file byte hashes remain identical; unrelated changes are not
staged. Empty cypress/dist screenshot trees (seven directories, zero files)
were retired after exact-holder/runtime-use checks. Other unresolved manual
tools and historical notes remain outside this qualified deletion set.


## 2026-10-02: qualified retirement of the remaining non-annotation run payloads

The four root `research-probe-*` directories, obsolete dense/eight-coordinate
runs, root hidden-recovery duplicates, the older hidden-recovery worktree copy,
invalidated root archive and obsolete runtime-optimization attempts are gone.
Eight retirement receipts total33,741 entries /16,617,545,089 logical bytes;
these follow-up totals exclude all previous cleanup totals. Exact receipt hashes
and current identities are recorded by `inventory.json.current_closeout.remaining_legacy_followup_20261002`
and the finite maintenance packet's `legacy-output-placement-final.json`.

Required inputs and named result/runtime payloads have their physical Research
Probes or infra owners; small configs/results/acceptance records are Git-managed
outside outputs. No generic legacy destination or old-path alias was created.
Seven evaluator-private files were moved opaquely, preserving inode identities;
no evaluator content was opened or hashed. The original sealed copy manifest
remains unchanged. Current checkpoint readers rebase only the retired selected
anchor, with unchanged original hashes; wrong hashes still fail before model load.
Research Probes and Web each passed13 focused CPU checks, and the canonical
knowledge checker passed329 entries /165 references /zero errors. No GPU run.

Current commits: infra `c129be609`, Research Probes `97491fbff`, Web `3b86e757b`,
main bounded retirement record `9f2db590f`. All four indexes are empty; the four
intended worktrees/branches and the canonical lock remain. Sixteen unrelated
preflight dirty-file hashes match. Native tmux pipe maintenance preserved the
three live shell panes and removed only confirmed dead-pane pipe waiters.

Annotation relocation is still user-deferred. Root `outputs/research` contains
only the existing8766 annotation review page's two files (1,267,082 bytes), with
its original server PID/cwd and HTTP200 verified. Gate A and retired Label Studio
pending annotation payloads remain as previously authorized. Root non-annotation
run cleanup passes its scoped checks, but tasks4.2/4.4 remain open for the explicit
annotation deferral and remaining package-metadata qualification. This partial
OpenSpec bundle remains uncommitted and unarchived.

Fresh `openspec validate route-owned-outputs-and-migrate-legacy --strict --json` passes (one change, zero errors). This validates the bounded change record, not completion of deferred annotation relocation.
