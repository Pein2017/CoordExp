# Upgrade acceptance

Date: 2026-09-12. Implementation owner: this task; independent frozen-design reviewer: Astra high. User authorized implementation after the review, one GPU at a time and 60 cumulative GPU minutes. Canonical integration was completed after verifying that the active research task released affected executable dependencies.

Evidence root: `/data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912`. All relative evidence paths below resolve there. Code base: `ef6d44d1196a7043deeef90cefc90136bce9859f`; candidate checkout: `/data/CoordExp/.worktrees/research-probes-upgrade-20260912`.

## Implemented scope and critical judgment

The upstream comparison justified a shared input composition owner and fixed Source256 parameter binding. It did not justify a new trainer, profile registry, trajectory schema, dtype change or universal diagnostic reducer. Four config-conversion consumers now share one implementation; DORA, Logit Lens and Human13 reuse the input planner. Six admitted selective-preservation profiles share their fixed DoRA binding. Cohorts, request policy, targets, objectives, reduction, DDP placement and model lifetime remain profile-owned.

The new direction-local inspection entry selects ordered IDs or a count/seed cohort before frontend loading and publishes generation inputs and optional annotated target spans. It uses the existing processor and native materializer, loads no weights, and does not replace strict Source256 admission. The three hash-bound historical trainers were retained byte for byte.

Production Python diff is **449 added / 251 removed** (net +198, including the new inspection entry and boundary validation); Python tests add 496 lines. This reduces duplicated ownership and repeated operations, but does not reduce total source lines. Detailed accounting: `code-diff-counts.json`.

## CPU behavior and falsification

- Two real fixture rows have exactly equal old/new normalized prompt IDs, media identity, grid, full targets and physical supervised/ignored spans. Both profiles' normalized row-record digest is `8adfe41b0bb254ec1010a2ffc936421e18c84e71ec10565cb2dc46d2ff72a32e`. Old code is preserved in `input-baseline/baseline-source.tar` with archive digest `c840054b84ecbbac843ca560c7c953a936ad23c8526746a1f6de9d823f301b6f`.
- Full Source256 CE dictionaries and real strict two-row preflight outputs agree, including action counts 119 and 29. The strict gate was not relaxed. See `source256-inputs-before.json`, `source256-inputs-after.json`, and `source256-migration-receipt.json`.
- Root replay of the real input-contract suite: 9 passed. Tests discriminate prompt edits, target ordering, wrong media and lazy/materialized execution. Root's ordered-selection mutation failed as intended and the restored implementation passed.
- The actual inspection CLI initially exposed a strict-JSON tuple publication failure. Native grids were changed to JSON lists, then the actual CLI and 16 inspection tests passed; the real-entry test rejects model loading. Evidence: `inspection-publication-red.log`, `inspection-green.log`, `inspection-two-rows.json`.
- Source256 migration checks: 70 passed plus one real processor preflight. Caller-level binding fixture covers exact ordered selection, errors, frozen complement, gradients and one update; two deliberate regressions were detected. All six fresh profile preparation snapshots were exercised. See `source256-final-tests.log`, `source256-mutation-sensitivity.log`, `source256-preflight-reuse-test.log`.
- Input-owner consumer checks: 104 passed. Fresh Logit/Human13 identities and DORA snapshots include changed shared sources; modified-helper tests discriminate source identity changes. Seven inherited input packets contained no Python source pins; all 25 recorded staged source entries matched their historical hashes (`profile-source-admission.json`).

The identical broad command before and after the change was:

```bash
python -m pytest -q tests/qwen tests/inference tests/templates tests/adapters \
  probes/dora_owner_learning/tests probes/logit_lens/tests probes/human13/tests
```

Unchanged base: **971 passed, 14 failed**, exit 1. Integrated candidate: **1,014 passed, the same 14 failed**, exit 1, with no new failing test IDs. Full failure lists, logs and timings are in `baseline-tests.json`, `integrated-tests.json`, `baseline-pytest.log` and `integrated-pytest.log`. Existing failures concern historical source-gate evidence, retained artifact paths/receipts and renderer snapshots. The suite is not represented as wholly green, and no baseline checks were weakened or unrelated failures repaired.

## Measured preparation cost

Two real rows, one separately measured frontend load, three warm repetitions in each independent old/new process. The benchmark preserves singleton native materialization in both arms. Times below are medians with observed ranges, in milliseconds.

| Profile/path | Old | New |
| --- | ---: | ---: |
| Source256 planning | 9.724 [9.651, 10.456] | 6.326 [6.209, 7.016] |
| Source256 planning + materialization | 50.298 [49.078, 50.856] | 46.271 [46.113, 47.207] |
| Logit composed inspection planning | 10.158 [10.114, 10.973] | 6.704 [6.305, 6.906] |
| Logit composed inspection total | 43.158 [42.789, 45.832] | 41.337 [40.179, 43.814] |

Per two-row preparation, rendering falls **6 → 2**, image planning **4 → 2**, and native materialization remains **2 → 2**. Source256 total preparation is about 8% lower in this small observation. Filesystem caches were not cleared and arms were not interleaved. The Logit annotated-target baseline composes its generation path with the existing Source256 target encoder; it is not evidence of a historical Logit CE execution. These timings support reduced CPU preparation work, with no GPU-throughput or scientific-quality claim. Exact measurements, identities and capture script: `input-baseline/comparison.json` and `input-baseline/capture_inputs.py`.

## Real-model acceptance

**Lead-accepted mechanical runtime slice.** The task-local harness uses Source FP32/SDPA, existing adapter/embedding composition, one selected real image, exact annotated replay and one existing profile-owned update. Limits are four model/image forwards, one pixel materialization, no natural generation and no checkpoint-format change. Exact old-record/new-input replay and existing conditional scorers are the acceptance consumer.

Run 01 stopped before model load because the checked-in historical embedding source-gate document no longer matched its accepted digest. It used no model forwards and is conservatively charged 0.100365 GPU minutes for 6.021871 seconds wall time. The failure is retained. Exact accepted document and receipt bytes were recovered from Git object `b5fb15015bb6183b8b98cf6278da2ff952299db8` under the task-local output directory; both expected digests and the unchanged CPU gate validator passed. Run 02 used this explicit restored root and completed successfully (exit 0). No historical file, expected digest or validation rule was edited. See `runtime-acceptance/restored-source-gate/restoration-receipt.json` and the two launch plans.

Run 02 completed in **43.519 seconds** wall time, with **37.402 seconds** in its GPU region on one NVIDIA A100 80GB. Exactly one model load, one native pixel materialization, four model/image forwards, one update and zero generated tokens were recorded. Peak allocation was 25,604,762,112 bytes and peak reservation 27,355,250,688 bytes; peak RSS was 11,056,320 KiB. Cumulative conservative cost across both attempts is **0.825686 GPU minutes** of the authorized 60. GPU work is finished.

Archived/current native tensors and zero-update logits and scores were exact. All 588 selected tensors (18,006,016 scalars) changed after the existing profile update; the complete frozen hash and tensor versions stayed unchanged, with no frozen gradients. Existing conditional branch target log-probability changed from −2.764309 to −2.488897 on this mechanical fixture; this is not a model-quality or owner-recovery result. No save-adjacent implementation changed, so checkpoint round-trip work was unnecessary.

The cold CPU consumer reloaded saved FP32 arrays through `route_access.score_logits` and `branch_bridge.summarize_logits`. Old/current logits and same-backend token scores, saved NumPy branch summaries and all non-logprob fields were exact. An initial overstrict CPU/GPU score-JSON equality check failed: maximum per-token log-softmax differences were 1.192093e−6 before and 1.072884e−6 after the update. These reductions pass the installed PyTorch FP32 comparison tolerance; sums and means are also recomputed from their token values. The original failure and quantified deltas are retained. This device comparison does not relax old/new parity on either backend.

The lead independently inspected the harness and receipts, replayed the no-model cold consumer successfully, and checked all 89 recorded source hashes against both staged and current bytes. Runtime evidence and commands: `runtime-acceptance/launch-plan-02.json`, `runtime-acceptance/run-02/receipt.json`, `runtime-acceptance/run-02/cold-consumer-receipt.json`, `runtime-acceptance/candidate-summary.json`; root replay log: `runtime-acceptance/run-02.lead-cold-consumer.log`.

The complete executed selection → planning → exact replay → profile-owned update → paired-diagnostics example is preserved as `runtime-acceptance/run-02/executed-script.py`, bound by SHA256 `64befcd5074fd4f62aba458ea19757918d0230e22b0dd73b76bb1a8466e41c3e`. It is a task-local acceptance example rather than a new repository runner. Its saved results can be rechecked without GPU work or output mutation:

```bash
CUDA_VISIBLE_DEVICES='' python /data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912/runtime-acceptance/verify_saved_consumers.py \
  --source /data/CoordExp/.worktrees/research-probes \
  --baseline /data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912/input-baseline \
  --run /data/CoordExp/outputs/research/upgrade-research-probe-workflow/20260912/runtime-acceptance/run-02 \
  --check-only
```

## Acceptance and integration disposition

Design: **lead-accepted** after one independent audit (`review.md`). CPU implementation and measured preparation claims: **lead-accepted** within the evidence above. Real-model acceptance: **lead-accepted**. Standards verdict: no new regression relative to the retained 14-failure baseline; input/span, source identity and frozen-update invariants passed. Intent verdict: the accepted shared/profile boundaries, small-cohort inspection and reduced preparation work are delivered; no unmeasured model-throughput gain is claimed. Strict OpenSpec validation and range whitespace checks passed before the documentation commit. The 25 changed/new Python files are frozen in `implementation-freeze.json` and were checked before the real-model launch.

Implementation commits: `a66ec668c` (shared input planning and six-profile binding) and `bd8dcee` (cohort inspection). Their staged patches and validation records are retained as `commit-01.patch`, `commit-02.patch` and the receipts above. The documentation/OpenSpec batch records this acceptance.

Canonical integration: **completed by fast-forward at 2026-09-12T09:32:32Z**, from `ef6d44d1196a7043deeef90cefc90136bce9859f` to `5904e8ec244932a62b14e3d93c53ebdcf83bb9f1`. The candidate still contained the latest target without divergence; its 25 Python hashes remained equal to the accepted implementation. All **47** current excluded dirty files retained exact bytes and status, and the canonical worktree lock remained intact. See `canonical-pre-merge.json`, `canonical-merge.log`, and `canonical-merge-receipt.json`.

At the pre-integration check, canonical HEAD remained `ef6d44d1196a7043deeef90cefc90136bce9859f`, the canonical lock remained present, and the candidate changed paths did not overlap any initial excluded dirty file. Of the 34 initial dirty files, 26 were byte-identical and eight had continued live changes in the active task. They were neither staged nor restored here; see `pre-integration-isolation.json`. The three protected producer hashes remained exact. The App list still reported the research task active; no release evidence was available.


The release witness was the original task's completed turn `01a09430-b272-7ee3-87c6-3cddcbb3b6f8` at 09:25:54.327Z, followed by independent read-only inspection of its final worker settlement, experiment closeout and live processes. Its acceptance file reports `driver_complete=true`, all eight GPUs released and `no_more_model_work=true`; the closeout prohibits further fit/decode. The lead checked both exact file digests and the original final message, with no newer task turn before merge. Monitor `9c22c918-b543-4b0f-b7fa-1fe6dc6319d2` supplied a lifecycle wake only; the release judgment used those concrete artifacts. Evidence: `merge-wake-decision.json`, `release-worktree-inventory.json`, and the source acceptance/closeout bindings in `canonical-pre-merge.json`.

Post-merge validation in canonical: **35 passed**, exit 0, including real input parity, inspection CLI, strict Source256 preflight, binding and source-identity checks (`canonical-entry-tests.log`). The saved-result CPU consumer also passed from canonical (`runtime-acceptance/canonical-cold-consumer.log`), checking all 89 current relative source paths against their recorded hashes and retained staged bytes. Its task-local reader now supports the explicit `--source` relocation; the original executed scripts and receipts remain unchanged. No additional GPU work was performed.

Temporary worktree retirement: **completed at 2026-09-12T09:37:35.521629+00:00** after navigator contexts released the directory, clean/merged checks passed, and all 89 staged execution sources were verified. The local candidate branch remains as a replay anchor. External acceptance artifacts, original executed scripts, raw logs and receipts are preserved. The documented cold-reader command now uses canonical and validates explicit relative source relocation; supplying the archived pre-upgrade source fails closed because required executed files are absent. See `worktree-retirement.json` and `runtime-acceptance/canonical-cold-consumer-wrong-source.log`.
