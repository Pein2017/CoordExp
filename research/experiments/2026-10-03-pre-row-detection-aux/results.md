# CPU preparation accepted: pre-row detection auxiliary

The lead accepts CPU preparation after repairing and rechecking both consumer blockers. Native execution remains HOLD; no model load, production forward, GPU job, training or new generation was issued. This acceptance covers CPU implementation and preparation only. Native qualification and scientific benefit remain unmeasured.

Protocol and scientific authority remain in [unit.md](unit.md). The worker has stopped after the repaired candidate; the lead owns these acceptance records and any later release.

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

## Exact proposed packet and qualification

Final16 packet: `cpu-05/native-candidate-16/{qualification.json,argv.json,release-candidate.json}`. The serial owner has11 exact stages: zero evaluate/readback; A run/readback; B run/readback; A evaluate/readback; B evaluate/readback; joint offline. It binds bank, selection, runtime, frozen source, outputs, doses and endpoint counts separately from historical online schedules. Historical paired1/full-label validation, process identity, timeout and cleanup paths remain intact.

Exact two-arm training commands are below. The existing owner substitutes `LEAD_RELEASE_SHA256` once from its validated immutable release; no lead-release file currently exists.

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

Proposed separate real-entry qualification: `cpu-05/qualification-candidate-1/argv.json`, both arms one full-bank update on8 ranks, fresh anchor/optimizers, normal endpoint exports/readback, same head-free native evaluation/conditional consumer and existing cleanup. This is a separately bounded qualification proposal, not dose extension or authority to run extra observations. It needs its own exact lead release.

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

## Remaining HOLD and job state

The source-qualified CPU05 proposals supersede CPU04 for future release; CPU01/CPU03/CPU04 evidence remains immutable. Both proposals retain `native_released=false` and `native_qualified=false`. Their clean source identity and final artifact hashes are bound by `cpu-05/source-identity.json` and `cpu-05/lead-acceptance.json`. The exact owner release is absent; no native stage is scheduled. The 1800s qualification and 5400s primary ceilings remain proposals, not measured costs or launch authority.

Native DDP/uneven-rank synchronization, auxiliary transfer through the real Qwen/PEFT norm, vLLM endpoint reload/conditional exact-history parity, whole-owner RSS/wall/artifact use and numerical export behavior remain unmeasured. Tiny causal/gradient and process fixtures close implementation errors, not native parity or research efficacy. Head learning is not the primary outcome; annotation-relative improvements do not establish physical-FN recovery.

Job state: CPU preparation lead-accepted; worker stopped, no model/GPU/native job live or issued. A separate exact release is required for the one-update real-entry qualification; primary16 must remain unreleased until its native execution risks and measured costs are resolved.
