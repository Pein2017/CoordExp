# Closed: no coverage benefit from the frozen pre-row auxiliary screen

**Lead accepted the completed primary16 execution and terminal consumers.** On the frozen18-image/570-label cohort, adding the training-only pre-row class/joint-box auxiliary at weight.1 did not improve ordinary greedy annotated coverage over the matched fixed-bank baseline. A16 covered233 owners; B16 covered225; the shared anchor covered247. Both arms trained from fresh anchor/optimizers for exactly16 updates. No further run or unit is authorized.

| Primary result | Anchor | A16 existing objective | B16 plus auxiliary |
|---|---:|---:|---:|
| Matched annotation owners /570 | 247 | 233 | 225 |
| Annotation-relative FN | 323 | 337 | 345 |
| Valid rows | 391 | 757 | 840 |
| Annotation-relative F1@IoU.5 | .51405 | .35117 | .31915 |
| Preserved / gained / lost vs anchor | 247 / 0 / 0 | 188 / 45 / 59 | 189 / 36 / 58 |
| Geometry-invalid rows | 5 | 946 | 263 |
| Literal valid / complete repeats | 10 / 10 | 213 / 1111 | 338 / 580 |
| EOS / token caps | 18 / 0 | 14 / 4 | 16 / 2 |
| Malformed rows | 0 | 4 | 2 |
| Conditional current target correct | 0/8 | 2/8 | 2/8 |

Direct endpoint owner sets contain207 shared,26 A-only and18 B-only assignments. B's net loss of8 therefore includes exchanges in both directions. These are evaluator assignments, not verified physical losses or recoveries. B reduces invalid geometry, total complete repeats, length caps and generated tokens; it increases valid repeats and annotation-relative unmatched rows. Unmatched rows remain unknown relative to annotations, not automatic hallucinations. The same conditional successes occur in both endpoints: actual row342 and synthetic row102; there is no additional B success.

The auxiliary was connected: all580 eligible rows per update were captured, the real opener autograd guard passed, head gradients were nonzero, and generator gradients changed while paired first-update base losses/logits matched. Base losses fell A .62416->.17763 and B .62416->.17582; B auxiliary loss fell2.50077->1.09217. This is a completed negative finite-dose outcome with mixed generation burdens, not an implementation failure. Falling replay/auxiliary losses did not yield better free-generation coverage.

**Claim boundary:** this is one fixed-bank16-update screen on the same18 training images, with568 directly supervised owners and570 evaluated annotations. It does not establish held-out generalization, physical-FN recovery, unique causal mechanism or the failure of every auxiliary design. No checkpoint selection, extra dose or retuning was performed. Scientific user acceptance is not asserted.

Authoritative final acceptance: `outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/native-16-01/lead-acceptance-01.json`; immutable primary raw evidence and expansion paths are in the same root. [unit.md](unit.md) retains the frozen scientific contract; [state.json](state.json) records closure. The following sections preserve preparation, failed-attempt and qualification history; the final section records primary execution and limits.

## Source and implementation

Dispatch source: `3cd1ee54717c5d34fb8e8609a485ee98365a48d2`. Worker submission HEAD: `d28eddc2b2e933ea01622bbc8aa7dc05bac07848` (lead protocol/binding/packaging commits). The worker submitted owned uncommitted changes. The lead commits the accepted implementation and records together; `cpu-05/source-identity.json` binds the final clean commit and exact maintained source closure. No push.

Changed maintained paths: `probes/pre_row_detection_aux/{__init__,__main__,bank,objective,experiment}.py`, `probes/online_row_credit.py`, `probes/online_row_credit_owner.py`, `tests/probes/test_pre_row_detection_aux.py`, and focused additions in `tests/probes/test_online_row_credit_owner.py`. Records: this file and `state.json`. The obsolete owned untracked flat module was removed after moving its content. No other worktree, prior output/receipt, shared checkpoint, protocol, catalog, index or `/external` surface was changed.

One differentiable final-norm hook selects the consumed opener states, retains autograd and releases the hook after one call. Existing detached inspection semantics are unchanged. A calls the original row objective. B adds the frozen auxiliary outside that same call. The shared forward seam is explicit `replay_device_type="cpu"` for tiny fixtures; its default remains CUDA.

Backbone optimizers, 16 fixed-bank updates, branch weights and clipping match the frozen recipe. The FP32 affine head has 84,009 parameters (41 x 2048 weight plus 41 bias). Initialization uses the isolated CPU default generator inside `fork_rng(devices=[])`, seed92711, PyTorch Linear reset_parameters: Kaiming uniform a=sqrt(5), bias uniform +/-1/sqrt(fan_in). CPU and CUDA shared randomness remain unchanged by head construction. Initial tensor identities are in the qualification/inventory. The separate AdamW/head clip never enters the backbone clip denominator. Head state is a sibling `training-head.pt`, absent from ordinary deployed exports and inference. All per-parameter backbone/head norms, preclip norms, base components, auxiliary CE/L1/GIoU, row IDs and saturation counts are retained without a positive-gradient efficacy gate.

## Inputs and exact inventory

Inputs resolve exclusively through predecessor `state.json` -> `lower_lr_followup.arms.constant.run_output`. The accepted v0 raw freeze, credit0/producer0 receipt bindings and accepted evidence hashes were checked. Current maintained `completion_credit` equals all18 saved plans. Records already have `arm="greedy"`; no legacy projection is required and no raw bytes were rewritten. Full-label snapshot/source whitelist and inherited policy bindings are revalidated through the existing verifier. Named shared step256 anchor manifest SHA is `1fdd43eb9e94f21379b1494ca6ee51f65046b68f718787970ac8a5ce3ce420a1`.

Immutable bank: `cpu-01/bank.json`, SHA256 `41ffaf9956187ac33f57ec43a71a29300cbf190f7cf5d05a2439665b5bf5c54f`. Final conditional selection: `conditional-selection-01.json`, SHA256 `3cedbb1dd9ca976c214a998476061d48890004d20b88a3523c78f6c443a3fca6`. Both remain at this unit output root. The preliminary35-case pool remains in the bank; runtime consumes the explicit lead selection.

| Positive kind | Scheduled | Aux eligible | Prefix-conflict excluded | Excluded weight |
|---|---:|---:|---:|---:|
| trace M | 0 | 0 | 0 | 0 |
| CHAIN B | 331 | 331 | 0 | 0 |
| relocated M | 237 | 237 | 0 | 0 |
| redirect | 12 | 12 | 0 | 0 |

Each full-bank update runs18 trace +18 CHAIN +12 redirect forwards. All237 original M rows relocate because every image has a CHAIN branch; trace legal/schema/geometry losses remain. There are580 distinct image/prefix groups, 568 directly supervised owners and570 evaluation owners. Inherited `same_category_supported` omissions `(4134,-99)` and `(7511,-167)` remain in evaluation. They are not auxiliary exclusions. Total eligible positive weight is29.748135903989592 before per-image normalization; per-image coefficients/denominators and exact rows remain inspectable in the inventory and bank.

Class vocabulary is the37 sorted exact GT strings, with no background or EOS label. The per-class GT counts and full list are retained in `cpu-03/inventory-02.json`. B uses GT boxes/1000 and the protocol CE/log37 + coordinate mean L1 + .5*(1-GIoU), image-positive-weight normalization, all18-image mean and coefficient.1. A is a new fixed-bank baseline; it is not a historical online-refresh reproduction.

Conditional rows, in frozen order: actual `[0,101,342,571]` (2 CHAIN B, 2 redirects), synthetic `[1,102,296,523]` on the same four images with different targets. Ordinary greedy/EOS, max64 new tokens; no masks, custom stops or later-row rescue. Only the currently open row is scored; the supplied opener earns no discovery credit. Raw continuations are retained, and histories are reported separately rather than as a causal history contrast.

## Verification and artifact locators

Prior broad command: `OWNER_CPU_SCRATCH=<unit-output>/cpu-tests-scratch python -m pytest -q tests/probes/test_online_row_credit.py tests/probes/test_online_full_label_region.py tests/probes/test_pre_row_detection_aux.py tests/probes/test_online_row_credit_owner.py`: exit0, **109 passed**,260.16s. Fresh real-input preparation of both packets, immutable bank readback, compileall, Git diff --check, AST/knowledge checks and anchor-manifest verification: exit0. The owner rejects the unreleased candidate before launch.

Checks falsify opener position leakage, detached auxiliary gradients, whole-box conflicts, baseline loss/gradient changes when aux is off or excluded, branch/image normalization, saturated-box nonfinite loss/gradient, combined clipping, head/deployment serialization mixing, bank/selection/target identity drift, wrong current-row rescue, missing source qualification, unreleased/drifted stages, conditional token overruns, first-failure handling and existing timeout/owned-descendant cleanup.

Intermediate failures are preserved: the first package test reused a freed comparison graph (fixed the test); the first legacy regression exposed2 missing-input_ids mocks under automatic device inference (repaired the seam to explicit CPU opt-in with CUDA default). Both original counterexamples passed, followed by the full109 final checks.

All evidence is under `outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/`: `cpu-03/checks-02.json` (commands/actual exits, prior failures, final packet hashes), `cpu-03/inventory-02.json` (58 original input locators/hashes, raw producer, row/class/weight inventory), `cpu-03/knowledge-diff-01.json`, `tests-final-02.log`, `final-checks-01.json`, `prepare-final-04.log`, `prepare-final-05.log`. Earlier CPU01/CPU02 artifacts remain unchanged.

Runtime metadata: {"peft": "0.21.1", "safetensors": "0.8.0", "torch": "2.13.0+cu129", "transformers": "5.17.0", "vllm": "0.29.0+cu129"}. Bare python preserves the selected ms wrapper. No native identity/parity is inferred from CPU fixtures.

## Lead consumer repairs and fresh verification

Lead reproduced two acceptance blockers in the initial CPU03 consumer package. CPU04 repairs only `experiment.py`, the narrow owner stage seam and the two assigned test files. The base objective, opener/autograd path, 48-job schedule, DDP coefficients, optimizers/clipping, head export, frozen inputs, protocol and budgets are unchanged.

The owner checks each distributed stage output is absent immediately before launch, in addition to its existing initial fresh-stage check. Runtime release still validates exact release/source/input/runtime/output ownership; ranks admit the shared parent and exclusively create their own `rank-N` directory. CPU tests traverse both actual run/evaluate consumers to the first model/device boundary (mocked to stop), including all eight legal rank arrivals and duplicate-rank rejection. The actual owner fixture rejects a stale stage present initially or inserted immediately before launch. No generic coordinator or historical-mode change was introduced.

Evaluation uses the existing ordinary shard/hash reader without its historical tree finalizer. This unit freezes every payload JSON except exactly root `frozen.json` and root `readback.json`; the latter is the declared derived receipt. Subsequent consumers verify the identical frozen payload plus the receipt status, bank, counts and frozen-manifest identity. The historical `frozen_records` function is byte-identical to HEAD. The actual `eval-readback` CLI followed by the actual offline consumer succeeds with CPU fake persisted 18-record/8-shard outputs; only metric computation is stubbed. Missing/changed payload, extra/nested JSON, missing manifest and changed readback receipts reject. Both finalizer and offline writes remain exclusive.

RED/GREEN evidence under `cpu-04/`: `red-consumers-01.log` exit1 (four regressions; owner fixture was then refined to inject once), `red-offline-02.log` exit1 (actual offline frozen-manifest counterexample), `red-stale-stage-03.log` exit1 (actual owner admits late stale stage before repair), `green-consumers-01.log` exit0 (four consumer tests). Affected existing/new tests: `OWNER_CPU_SCRATCH=<unit-output>/cpu-tests-scratch python -m pytest -q tests/probes/test_pre_row_detection_aux.py tests/probes/test_online_row_credit_owner.py`, exit0, **33 passed in24.26s**. These recheck29 previous tests and add4 consumer tests. The80 untouched base/region tests retain their earlier exit0 evidence within the109-test suite; no claim of113 fresh tests is made.

Fresh16-update and1-update real-input preparations, immutable-bank readback, compileall, Git diff --check, packet/input/cost/initializer comparison, historical-reader comparison and unreleased owner rejection all exit0. `cpu-04/checks-01.json` records exits and packet hashes; `cpu-04/knowledge-diff-01.json` records the four repair path identities and retained inventory. Both CPU04 packets preserve CPU03 input identities, exact11 stages, runtime, initializer, schedule and budgets. CPU03 packet hashes still match their immutable receipts; CPU01 bank bytes and original lead harness/receipt remain unchanged. No native calls or launches occurred.

## Historical primary16 proposal and qualification bounds

Historical primary16 proposal: `cpu-05/native-candidate-16/{qualification.json,argv.json,release-candidate.json}`. It is unreleased and needs a new source binding after the repaired one-update qualification succeeds. The serial owner has11 exact stages: zero evaluate/readback; A run/readback; B run/readback; A evaluate/readback; B evaluate/readback; joint offline. It binds bank, selection, runtime, frozen source, outputs, doses and endpoint counts separately from historical online schedules. Historical paired1/full-label validation, process identity, timeout and cleanup paths remain intact.

Exact two-arm training commands are below. The existing owner substitutes `LEAD_RELEASE_SHA256` once from its validated immutable release; this primary16 proposal has no release.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module probes.pre_row_detection_aux run --bank /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-01/bank.json --bank-sha256 41ffaf9956187ac33f57ec43a71a29300cbf190f7cf5d05a2439665b5bf5c54f --output /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-05/native-candidate-16/A --arm A --updates 16 --release /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-05/native-candidate-16/lead-release.json --release-sha256 LEAD_RELEASE_SHA256
```

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module probes.pre_row_detection_aux run --bank /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-01/bank.json --bank-sha256 41ffaf9956187ac33f57ec43a71a29300cbf190f7cf5d05a2439665b5bf5c54f --output /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-05/native-candidate-16/B --arm B --updates 16 --release /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-05/native-candidate-16/lead-release.json --release-sha256 LEAD_RELEASE_SHA256
```

The complete exact evaluation/readback/offline commands are in `argv.json`. Proposed owner invocation:

```bash
python -m probes.online_row_credit_owner --root /data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-03/pre-row-detection-aux-12/cpu-05/native-candidate-16 --release-sha256 LEAD_RELEASE_SHA256 --mode pre-row-aux --updates 16
```

The original separate real-entry qualification used `cpu-05/qualification-candidate-1/argv.json`, both arms one full-bank update on8 ranks, fresh anchor/optimizers, normal endpoint exports/readback, same head-free native evaluation/conditional consumer and existing cleanup. That released invocation failed as recorded below. The replacement preserves these bounds and requires its own immutable release.

| Bound | Proposed qualification1 | Primary16 |
|---|---:|---:|
| HF training forwards, both arms | 96 | 1,536 |
| HF input tokens | 162,008 | 2,592,128 |
| HF visual tokens | 94,976 | 1,519,616 |
| Primary/qualification evaluation requests | 78 | 78 |
| Evaluation generated-token ceiling | 168,072 | 168,072 |
| Whole owner wall, including checks/export/cleanup | 1,800s | 5,400s |
| Cleanup reserve | 30s | 30s |

Max training context3725, selected-logit rows628, selected hidden rows63 at width2048. Largest selected vocabulary tensor is95,876,760 elements:191,753,520 BF16 bytes or383,507,040 FP32 bytes. The head parameter/gradient/Adam estimate is1,344,144 bytes. Empty-history context ceiling4456; conditional ceiling1545. Proposed aggregate RSS160GiB and artifacts8GiB are conservative, unmeasured estimates; they are not observed resource results. No readback model forwards are proposed.

## Lead acceptance

The lead reproduced the two original failures before repair, then directly reran the real rank-arrival/duplicate check, eval-readback-to-offline consumer, and initial/late stale-stage owner rejection: **3 passed, exit0**,8.83s; `lead-consumer-green-01.log`. CPU04 retains the worker's **33 affected checks** and RED/GREEN evidence; the **80 unchanged base/region checks** reuse the prior109-test receipt. These are distinct checks, not a new113-test full-suite invocation.

The bounded independent trainer review found no blocker: A/B share the base objective; the differentiable opener hook preserves the graph; all48 jobs use the original image coefficient and one final synchronized backward per rank; head SUM/8 plus the18 image denominators implements the frozen auxiliary mean. Optimizer clipping, RNG and head/export separation remain distinct. Real Qwen/NCCL behavior is still unmeasured.

## First native qualification failure

The CPU05 release bound source `a34d83a5f4168d5bdfbf16356c8c979e98854d17` and release SHA256 `2a8c5eba93d9c43cec1e2346dc72fdc128d8b6d079fa7d021779d172cc70b7eb`. One owner invocation returned exit1; anchor evaluation returned exit1 with `vLLM changed exact prompt tokens: 14038:greedy:0`; ten later stages were skipped. There were zero training forwards, endpoint exports, readbacks and offline results. The 18 persisted ordinary records contain 3,724 generated tokens but no complete-rank receipts; they are incomplete-stage evidence, not a qualified benchmark. Conditional generated tokens were not persisted and remain unmeasured.

Owner elapsed time was 72.990874s including cleanup/finalization, or 0.162202 allocated-slot GPU-hours. Sampled descendant aggregate RSS peaked at 37.945988GiB (15s samples, excludes owner; instantaneous peak unmeasured). Root files totaled 764,204 bytes at terminal inspection. External shell/interpreter wall was not separately timed. RSS160GiB/artifacts8GiB remained planning estimates rather than enforced gates. Terminal cleanup reports no owned-live, unconfirmed or unresolved issued stage, and no unfinished watcher; the worker checked all 26 recorded PID/start-tick identities twice without a matching non-zombie process.

Immutable raw evidence: `cpu-05/qualification-candidate-1/{terminal.json,stage-0-zero-evaluate-issued.json,stage-0-zero-evaluate-start.json,stage-0-zero-evaluate-terminal.json,stage-0-zero-evaluate.log,owner-native-01.log}`. Projected exit/resource/cleanup evidence: `worker-native-terminal-01.json` in that same root. Failed artifacts and historical source receipts remain unchanged.

## Historical CPU repair acceptance and replacement qualification

The caller had supplied already-expanded HF bank prompt IDs as `generate_exact`'s unexpanded chat. vLLM expanded the image placeholder again. The repair only tokenizes the maintained `NativeRequest.chat_text` with `add_special_tokens=False`. Image identity, frozen extension, expected expanded prompt, decoder, targets, losses, branch schedule, optimizer/head/export behavior and costs are unchanged. The shared exact-prefix guard is unchanged.

Worker CPU RED reproduced the same strict-prefix failure through actual `evaluate` and installed placeholder processing; GREEN traversed all eight rank partitions sequentially and actual 18+8 readback with model/device computation stubbed. A corrupted processed token still fails before a completion receipt. The affected package plus two existing shared exact-prefix/score-guard tests passed: **18 tests, exit0, 13.14s**. All eight frozen prefixes also matched in the real-input CPU projection. Evidence: `cpu-06/{checks-01.json,knowledge-diff-01.json,final-check-01.json,exact-prefix-cpu-projection-01.json,red-exact-entry-01.log,green-exact-entry-01.log,affected-tests-01.log}`.

The lead inspected the changed caller and unchanged native guard, obtained a focused independent no-blocker review, and reran only the actual evaluation/readback/corruption regression: **1 passed, exit0, 10.43s**, `lead-exact-prefix-green-01.log`. Unchanged earlier checks are reused; no full-suite rerun or native success is claimed. The accepted source and records are committed together; `cpu-07/source-identity.json` and `cpu-07/lead-acceptance.json` bind the clean repaired source. CPU06 proposals remain immutable.

The replacement `cpu-07/qualification-candidate-1` preserves the eleven-stage two-arm one-update packet: 96 HF forwards, 162,008 HF input tokens, 94,976 visual tokens, 78 evaluation requests capped at 168,072 generated tokens, eight ranks, 1,800s whole-owner ceiling including 30s cleanup reserve. A separate exact release permits one invocation only; no autonomous native retry, extra model readback, warmup, primary16 execution or scientific retuning. The worker retains technical execution/check/repair ownership within that grant.

Native successful exact-prefix evaluation, Qwen/PEFT auxiliary autograd, uneven-rank DDP/head reductions, head-free endpoint export, endpoint inference and final readback/offline remain unqualified. The failure reached none of the training paths. CPU acceptance establishes the tested caller contract, not numerical parity, efficacy or physical-FN recovery.

## Accepted one-update native qualification and primary16 boundary

CPU07 used clean source `faee6831e4c0f89b620de79a616dfee6f6263cb6` and release SHA256 `9dc6df92eae796d660cb37dd28972ea17e0c82e9c2bc059447ece53cdfd85e9c`. All11 owner stages and the external invocation returned exit0. Both arms completed8 ranks and one full-bank update:96 HF forwards,162,008 input tokens and94,976 visual tokens total. Losses, generator/head norms and auxiliary components were finite. B delivered all580 eligible rows exactly once across30 positive branches; A delivered none. Base losses and base-logit hashes matched on all48 initialization pairs. B's native opener autograd guard passed, head gradients were nonzero, and589/590 generator norm entries changed; isolated native auxiliary gradient vectors were not separately measured. Separate clipping and head-free endpoint export executed; the B head is saved separately.

Each of zero/A1/B1 has18 ordinary records and8 frozen conditional records with complete8-rank receipts. Endpoint identities match their arm exports and the zero anchor. Both ordinary comparisons retain the same570 annotation identities and identical zero scores. Conditional scoring considers only the completed current row beginning at character zero; later continuations do not rescue a target. All24 current rows parsed but none matched exact description plus IoU>=.5. For example, B/image2299/row101 produced person `[72,415,154,918]` against `[3,298,100,552]` (IoU about.062). No scoring/identity blocker was found.

| One-update observation | Anchor | A existing objective | B plus auxiliary |
|---|---:|---:|---:|
| Matched annotation owners | 247 | 232 | 236 |
| Annotation-relative FN | 323 | 338 | 334 |
| Valid rows | 391 | 381 | 354 |
| F1@IoU.5 | .51405 | .48791 | .51082 |
| Malformed rows | 0 | 137 | 0 |
| Literal valid repeats | 10 | 26 | 3 |
| Newly gained / lost anchor owners | — | 14 / 29 | 10 / 21 |
| Conditional current target correct | 0/8 | 0/8 | 0/8 |

These are bounded one-update observations, not primary16 benefit or physical-FN recovery. B avoids the observed A malformed/repetition burden but still loses anchor coverage. Unmatched predictions remain annotation-relative unknown.

External owner wall was443.470425s (0.985490 allocated-slot GPU-hours), including startup through teardown; internal meter443.344702s, cleanup completed442.988719s. Sampled descendant aggregate RSS peaked44.029648GiB (15s samples, excludes owner; instantaneous peak unknown). Artifacts at worker inspection totaled204,513,006 bytes including endpoint exports/control/log files, excluding shared model assets. All54 ordinary outputs ended EOS; all24 conditional continuations reached64 tokens. Retained inference tokens totaled13,591 across78 requests. Counts exclude engine-internal initialization/kernel work. Terminal cleanup has no owned-live/unconfirmed/unresolved stage or unfinished watcher;100 PID/start-tick identities were rechecked without matching non-zombie processes.

Evidence: `cpu-07/qualification-candidate-1/{worker-candidate-01.json,worker-qualification-checks-01.json,worker-external-exit-01.json,terminal.json,offline-results.json}` plus complete per-rank updates/exports/evaluation receipts. Lead acceptance: `cpu-07/lead-native-acceptance-01.json`. Lead checked terminal consumer evidence identity and ordinary denominators/results, inspected native gradient summaries, and used one focused independent checkpoint/current-row-consumer review. Worker technical checks were reused; no duplicate evaluator/model invocation or test suite was issued.

The original primary16 contrast is released only by `native-16-01/lead-release.json`: fresh anchor/optimizer per arm,16 updates each,1536 HF forwards,2,592,128 input tokens,1,519,616 visual tokens,78 evaluation requests capped at168,072 generated tokens. The owner runs11 serial stages on8 ranks under5400s including30s cleanup reserve. Scaling the entire measured training-stage wall16-fold gives a planning estimate around2618s including other stages; this is not a guaranteed bound. RSS160GiB and artifacts8GiB remain planning estimates, not enforced gates. Measured one-update memory does not prove an instantaneous or long-run peak.

Primary source/packet binding preserves every execution source byte from the accepted qualification; the intervening commit updates research records only. The packet uses current repaired test/runtime bindings. Primary16 starts afresh rather than continuing A1/B1 or selecting a checkpoint. No head-only proxy, perfect-F1 gate, retuning, retry, extra readback or dose extension. General numerical parity, repeated-run determinism and physical-FN efficacy remain outside this qualification. Stop after the frozen primary result and cleanup; no next unit is scheduled.

## Completed primary16: provenance, technical acceptance and limits

The sole primary owner used clean execution commit `e5b57f9292697529ed2720f0f2c45c704e19f28e`, with all152 execution source files byte-identical to the qualified `faee6831e4c0f89b620de79a616dfee6f6263cb6`. Release SHA256 was `68173f0a697b371dd2eef9eb517e1a6c83958b4b7889e5a4bd702a434d0ecfd0`. Bank/selector identities remain the frozen CPU01/final selection. CPU07 was launch qualification only; both primary arms and all primary zero/endpoint evaluations were fresh. Actual owner exit0 and all11 stage exits0; no source repair, native retry, separate stage launch, extra evaluator/model readback or test-suite rerun occurred.

Each arm completed16 updates on8 ranks and768 forwards with the qualified uneven schedule. B captured9280 auxiliary rows total,580 per update, with no degenerate auxiliary boxes. All recorded numeric components and151040 backbone/256 head per-tensor norm entries were finite. Final B auxiliary terms were CE .217859 + L1 .383188 + weighted GIoU .491120. Relative to update1, CE fell from1.357245 and the GIoU term from.836776, while L1 increased from.306748: combined auxiliary-loss improvement is not uniform coordinate improvement. All gradient components and update summaries are retained; isolated auxiliary-only native gradients and postclip norms were not recorded. Both checkpoint16 exports are head-free; the B head remains separately serialized.

Endpoint/checkpoint identity, shared zero and all570 annotation identities were checked at the final consumer. Each zero/A16/B16 pool contains18 ordinary and8 frozen conditional records with8 complete rank receipts. Both conditional improvements concern the same current rows,1/4 actual and1/4 synthetic. All first rows parsed; supplied openers earn no discovery credit, later rows do not rescue a failed current row, and differing actual/synthetic targets prohibit a causal history comparison.

Primary wall was1014.409026s (16.91min), including external startup through teardown;2.254242 allocated-slot GPU-hours. The5400s ceiling was respected. Sampled descendant aggregate RSS peaked44.438248GiB (15s samples excluding owner; instantaneous peak unknown). Artifacts totaled402,496,671 bytes at worker inspection, excluding shared assets. Maximum training allocator allocated/reserved GiB were A25.348/26.951 and B25.350/26.953; native inference and driver allocations are excluded. RSS160GiB/artifacts8GiB remained planning estimates, not enforced gates.

Logical counts were1536 HF forwards,2,592,128 input tokens,1,519,616 visual tokens and78 evaluation requests. Retained generated tokens were31,585: ordinary zero/A/B3724/16070/10388 and conditional512/451/440. Engine-internal initialization, generation kernel/vision work and transient memory were not independently metered. Cleanup reports no owned-live/unconfirmed/unresolved stage or unfinished watcher; the worker rechecked108 PID/start-tick identities without matching non-zombie processes. No live job or follow-on remains.

Machine expansion paths under `native-16-01/`: `worker-candidate-01.json` binds evidence and actual checks; `worker-training-checks-01.json` holds all update summaries; `worker-terminal-checks-01.json` holds evaluation/resources/cleanup; `worker-shared-gradient-check-01.json` holds connection evidence; `terminal.json` and `worker-external-exit-01.json` preserve actual status. Full per-rank updates, head-free endpoint identities, raw evaluation payloads, readback/frozen receipts, evaluator gate and `offline-results.json` remain unchanged. Lead checked six final-consumer receipt hashes, directly recomputed A/B owner-set counts, and used one focused independent checkpoint/denominator/selector review. Unchanged worker technical checks were reused; no evaluator/model/test rerun was issued.

**Interpretation:** an entirely detached or unexecuted auxiliary is inconsistent with the accepted wiring evidence. Why the connected auxiliary failed to improve free rollout remains unidentified. Head-local fitting/conditional feature changes without useful rollout transfer and shared-generator interference are both compatible with the observations. The coordinate L1 regression and different repetition/geometry burdens do not isolate either mechanism. Gradient strength, strict point-box targets and class/box objectives were combined, so this screen cannot assign the outcome to one component. Any future component/gradient-matched contrast needs a new user-owned question and authorization; none is scheduled here.

**Closure:** technical execution accepted; frozen scientific contrast negative for auxiliary coverage benefit, with mixed burdens. All historical failures, qualification and primary evidence remain immutable. All grants are consumed. Stop after this record and user report; no further model call, visual audit, retuning, dose extension, checkpoint selection or next unit is authorized.
