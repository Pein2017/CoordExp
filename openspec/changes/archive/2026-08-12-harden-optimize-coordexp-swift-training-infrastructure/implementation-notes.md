# Implementation Notes: harden-optimize-coordexp-swift-training-infrastructure

## Current production-path disposition — 2026-08-11

The user selected Wave 7 -> Wave 8 -> Wave 9 as the shortest production-trust
path and ended the optional Wave 3-6 performance campaign before a fresh
successor marker was consumed. Wave 3 closes without an efficiency claim. Wave
4 retains the current correctness-tested eval hydration behavior without a
startup/resource promotion claim. Wave 5 retains the synchronous input-provider
default without a provider-performance claim. Wave 6 remains pending for future
design/matched training, with its CPU research packet descriptive and
`source_order_next_fit` unchanged.

Omitted Wave 3-6 measurements are not passes and MUST NOT be backfilled from
shared-load or CPU-only evidence. The isolated, unexecuted Wave 6
matched-training controller was removed rather than retained as an active
launch surface; immutable Wave 6 r2 research artifacts remain untouched.

## Wave 0 fixed point — 2026-08-09

### Checkout and authority

- Worktree: `/data/CoordExp/.worktrees/CoordExp-swift`.
- Branch: `coordexp-swift`.
- Inspected HEAD: `23a01061a4c7a8571399a572886d75e2c02286f7`
  (`docs: sync compressed CoordExp agent guidance`, 2026-08-08). This commit
  changes agent guidance, not the accepted training implementation.
- Accepted implementation commit:
  `2b0a2165a88499f0314572c5a73c9a308a990154` (`Streamline CoordExp-Swift
  training infrastructure`). Acceptance record:
  `e89d20587a971fa7849aa00edaf6a19fd5c4baac`.
- The synced/archived predecessor change is present at
  `openspec/changes/archive/2026-08-06-streamline-coordexp-swift-base-infrastructure`
  with 11 files and 29/29 checked tasks. Its active-path deletions, stable-spec
  additions, archive additions, and the one `src/qwen/encoding.py` comment-path
  update are pre-existing user-authorized planning/archive work and are
  preserved as task-owned dirty state.
- This follow-on change is the only active OpenSpec change. Reviews, the
  predecessor archive, and these notes are evidence; live stable specs and the
  active change artifacts own the contract.

### Dirty-state inventory at the fixed point

The checkout is intentionally dirty. `git status --porcelain=v1
--untracked-files=all` showed only these classes:

1. deletion of the predecessor active change;
2. additive/synced edits to five stable specs;
3. the archive copy of that predecessor change;
4. the active `harden-optimize-coordexp-swift-training-infrastructure` change;
5. a one-line reference-path edit in `src/qwen/encoding.py` from the active
   predecessor path to its archive path.

No existing source implementation change was attributed to Wave 0 before its
first failing tests. No reset, clean, broad stage, cache mutation, dependency
change, or process termination was performed.

### Live process, GPU, cache, and artifact state

Snapshot time: `2026-08-09T05:03:58+00:00`.

- Eight NVIDIA A100 80 GB PCIe devices are present. GPU 2 used 20,873 MiB at
  39%; GPU 4 used 21,857 MiB at 100%; GPU 5 used 39,185 MiB at 47%; GPU 6 used
  43,247 MiB at 36%; GPU 7 used 19,207 MiB at 100%. GPUs 0, 1, and 3 were idle
  in that exact snapshot. Compute PIDs were visible through `nvidia-smi`, but
  several process names/PIDs were not visible in this process namespace. This
  is a concrete eight-rank operational conflict, so no GPU launch was made.
- Visible long-lived Python services were unrelated host infrastructure
  (`doh_dns_proxy.py`, a local HTTP server, and Label Studio). None was changed.
- `COORDEXP_SWIFT_PACK_CACHE_ROOT` and `COORDEXP_SWIFT_ARTIFACT_ROOT` were
  unset. The resolved default shared cache root is
  `.cache/coordexp_swift/packing`: 61 GiB, 240 files at depth <= 3. It is
  read-only evidence for Waves 0-8.
- Existing artifact evidence includes `outputs/prod/coordexp_swift` (14 GiB,
  499 files at depth <= 3) and `outputs/smoke/production_mimic` (1.1 GiB, 575
  files at depth <= 3). New measurements use dedicated temporary roots.

## Stable spec identities

The implementation authority is commit `2b0a2165...`; the following hashes bind
the live, synced stable-spec contents at the Wave 0 fixed point. They are
uncommitted because the user asked to sync/archive first and then begin this
apply pass; the last committed owner is listed separately so neither identity
is mistaken for the other.

| Stable spec | Live SHA-256 | Last committed owner |
|---|---|---|
| `coordexp-swift-config-runtime` | `59fb01944ed94fe5a7289573e84d6c5848ee4158a5735e54d4b1b7a14a84d04e` | `5f1680d4e1e1c891b503e651f8528f67ab520255` |
| `coordexp-swift-pack-cache-semantic-identity` | `72c19ae5b1b3e912fe82e8c6d731588646a7cd49ae44ae383efbbe58a644e118` | `dd424ccf159a758629774af95edcb47aeabc71e8` |
| `coordexp-swift-packing-forward` | `359e377a432fc8c89536740ee9b69fbacaad7b0466192b43301321c46731f2ce` | `dd424ccf159a758629774af95edcb47aeabc71e8` |
| `coordexp-swift-supervision-losses` | `4f61fad196904dc8e77b9a7ac1af077e0f4a9a154a8efac702d82eb174077311` | `5f1680d4e1e1c891b503e651f8528f67ab520255` |
| `coordexp-swift-training-artifacts` | `98551e8c9b6b77cd9b28719e519b0961d1bb55cb48154ee72a7cc5193dff3a79` | `5f1680d4e1e1c891b503e651f8528f67ab520255` |

## Owner and failing-evidence map

Every behavior-changing wave starts from an interface-level failing test or an
executed negative probe. Source reading alone cannot close a task.

| Slice | Current owner surfaces | Required failing test/probe anchor |
|---|---|---|
| Wave 0 provenance/receipts | `src/artifacts/run_writer.py`, `src/training/pipeline.py`, new bounded provenance/resource helpers | `tests/artifacts/test_provenance.py`, additive/failure cases in `tests/artifacts/test_run_artifacts.py`, pipeline failure-phase tests in `tests/training/test_pipeline_assembly.py` |
| Wave 1 cache determinants/admission | `src/training/pack_cache.py`, cache resolution in `src/training/pipeline.py`, strict config in `src/config/{models,resolve,loader}.py` | mutation cases in `tests/training/test_pack_cache.py`; fail-before-model cases in `tests/training/test_pipeline_pack_cache_rebuild.py` and `test_pipeline_assembly.py` |
| Wave 2 all-layer FA2/parity | `src/qwen/{fa2,forward,positions}.py`, packed boundaries in `src/packing` | negative set/count/boundary tests in `tests/qwen/test_fa2.py` and `test_forward.py`; smallest real-model packed-vs-separate forward/backward probe plus corrupted-boundary control |
| Wave 3 zero-weight loss graph | `src/losses/{runner,token_type_gate,context,normalizers}.py`, trainer loss call | diagnostic/loss/gradient tests in `tests/losses/test_runner.py` and `test_context_and_terms.py`; production-shaped memory/time probe |
| Wave 4 selective eval hydration | cache manifest/loaders in `src/training/pack_cache.py`, `src/eval/forward.py`, pipeline eval partitioning | ordinal/exact-cover/corruption cases in `tests/eval/test_forward_eval.py`, `tests/training/test_pack_cache.py`, and pipeline assembly; full-vs-selective hydration probe |
| Wave 5 input provider | `src/training/forward_input_provider.py`, `supervised_trainer.py`, provider resolution/binding in pipeline/config | exact lifecycle/boundedness tests in `tests/training/test_forward_input_provider.py` and `test_supervised_trainer.py`; eight-rank three-arm paired probe |
| Wave 6 packing policies | `src/packing/{planner,supervision}.py`, micro-step construction/cache identity | plan replay, each-once, fixed-row-order tests in `tests/packing/test_planner.py`, `test_supervision.py`, and cache tests; CPU planner comparison then matched training probe |
| Wave 7 exact resume | `src/artifacts/checkpoints.py`, `run_writer.py`, training schedule/runtime/pipeline; no current exact-resume owner exists | new manifest/admission/interruption tests under `tests/artifacts` and `tests/training`; uninterrupted-vs-resumed executed probe |
| Wave 8 dependency/backend freeze | strict config resolution, new provenance collector, `src/qwen/runtime_loading.py` and FA2 selection | drift/unavailable/backend rejection tests in config/artifact/Qwen suites; rerun accepted compatibility matrix |

Checkpoint publication remains owned by `src/artifacts/checkpoints.py` and
`tests/artifacts/test_checkpoint_writer.py`. Existing run-file compatibility is
owned by `RunWriter` and `tests/artifacts/test_run_artifacts.py`. Exact resume
will be additive rather than overloading the inference-minimal checkpoint.

## Imported dependency baseline — 2026-08-09

Collected from the required `ms` Conda environment. Actual runtime dependencies
were imported to bind their selected module/binary; ms-swift was resolved by
import spec without importing `swift`, because it is a reference-only checkout
for this training route. No install, upgrade, cache write, or dependency
mutation occurred. For wheel-style installs, the `RECORD` digest binds the
installed file manifest; the selected/imported origin digest binds the file
chosen by Python. Unavailable metadata is explicit. Paths are provenance and
are not portable configuration.

| Component | Version / runtime | Selected/imported origin SHA-256 | Distribution `RECORD` SHA-256 |
|---|---|---|---|
| ms-swift (`swift`, reference-only; not imported by training) | `4.2.2`; editable source commit `f2797138dba0e224cfff735cd89a528a08d8732a`, clean relevant tracked state | `d345fd8f68077d11730ffe56747a52b1858550e8c21067ed55a1db2f79ab5caf` | unavailable: legacy `ms_swift.egg-info` has no `RECORD` |
| Transformers | `4.57.1` | `4ef5187b5f66c564aa575ddab9ce94342630d40fe874d0464d790c6f6b748647` | `025b8186b05df7f173dff14013042262c7dacdb556e98ee5c56a182c02e42728` |
| FlashAttention | `2.8.3` | `f1833af940ac1124e09dc05dd922308871411035e75b90bf4dbe3a831dc03b50` | `fee009739702fa997fa85c07f134ad54636d042475235515b90afdcfffd299a5` |
| Torch | distribution `2.9.1`; runtime `2.9.1+cu128` | `3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288` | `a8a2b13b3ba6e31a168babb44dcbdda35d3f56583ae1ba9e98ad51ae809cb0e7` |
| Accelerate | `1.10.1` | `68f56baac9b078db4735567649441ef28932a946b400bf8d8c2519b0a5b94b89` | `8c84d01d529cd3d9745936fd74933835e9838e7119a96832a27123e5170500f8` |
| PEFT | `0.17.1` | `efabc3b44d7326eee59c07910e426fa38ee0c6419b773e8f75af87ae61d3fa32` | `258749eb655ec7ee9b6d4f6040b2a34c9207cafc9f3efc35d96559e63460a38c` |
| tokenizers | `0.22.0` | `644e596a052fa1b05272b1c141d1286e1c78c2a3346ecabd17d68b62404d8d84` | `2df0edacccd5e83cc9fde45cc5199701bc97731a676787ada63d02b8ff0f4bf0` |

Additional binary/runtime identity:

- imported `flash_attn_2_cuda` binary:
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`,
  size 997,961,816 bytes, SHA-256
  `8ca052bf2d3f53baa629e22749b9622a95273c5bffb5f06cd24768ef63f65807`;
- Python `3.12.11` from the `ms` environment;
- Torch CUDA runtime `12.8`, cuDNN `91002`, NVIDIA driver `550.54.15`;
- all seven Python distributions resolve as the recorded `pypi_0` builds except
  editable ms-swift, which resolves as `dev_0` and is therefore bound by its
  live Git source identity above.

The selected/imported-origin digest alone is not treated as a full-package proof; the
distribution manifest or editable-source Git identity is part of the accepted
baseline. Later runtime receipts use the same distinction and record an
explicit unavailable reason rather than substituting a package name.

## Measurement contract

The frozen workloads, pack streams, seeds, tolerances, warm-up exclusion,
paired arm ordering, wall-clock scopes, noise bands, repetition cap, resource
ceilings, GPU cost gate, and stop rules are in `measurement-plan.md`. No
decision-bearing candidate run may precede that freeze.

## Wave 0 executed cache baseline — 2026-08-09

The model-free W0-CPU slice ran only against the dedicated temporary root
`/tmp/coordexp-wave0-baseline-FIU33RQX/cache`; the shared production cache was
not a destination. The tracked 256-train/64-eval smoke config resolved to
configuration fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`.
No model was loaded, CUDA was not initialized, and the resulting temporary root
occupied 33,488,129 bytes across four payload/manifest files.

Cold publication receipt:

- entry-to-return duration: `6.50697948038578` seconds; the enclosing Conda
  process wall clock was `14.586` seconds and is reported separately;
- config/provenance resolution: `2.246297236531973` seconds;
- cache preparation/publication/admission totals:
  `2.364432964473963` / `1.0117708519101143` /
  `0.5266132690012455` seconds;
- process high-water: 1,274,712,064 bytes RSS, zero observed read bytes, and
  33,566,720 write bytes; GPU status was `cuda_not_initialized`;
- train fingerprint `c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1`,
  manifest SHA-256
  `636c701efd1cf94af0f88bdb1d86fcef0529d47609c2b48fbf433a7172da7a4e`,
  32 micro-steps;
- eval fingerprint `60330b24519f0931e5c84765874c70e0cb5d54cd472bd6f79fc30977edab626c`,
  manifest SHA-256
  `d8ca6858a56fb895f3ba037bfa27d96104cde6881aded837ec4473834ecc73bf`,
  8 micro-steps.

The subsequent page-cache-warm admission hit the same immutable manifests.
Entry-to-return was `3.4016683101654053` seconds (Conda process wall clock
`11.130070255` seconds), including `2.462923113256693` seconds of
config/provenance resolution and `0.6155218221247196` seconds of payload-level
admission: train `0.5524180121719837`, eval `0.0631038099527359`. Preparation
and publication were explicitly `not_run_cache_hit`. Process high-water was
1,203,798,016 bytes RSS and 12,288 write bytes, with CUDA still uninitialized.

Authoritative cache loading reported the following packing facts. Capacity is
`micro_step_count * global_max_length`; `padding_tokens` is zero, so unused
capacity is headroom rather than materialized padding.

| Split | Real tokens / capacity | Utilization | Pack length min / median / mean / max | Segments | Oversize / singleton |
|---|---:|---:|---:|---:|---:|
| train | 357,559 / 384,000 | 93.1143229167% | 10,718 / 11,139 / 11,173.71875 / 11,793 | exactly 8 in every pack | 0 / 0 |
| eval | 90,054 / 96,000 | 93.80625% | 10,995 / 11,233 / 11,256.75 / 11,609 | exactly 8 in every pack | 0 / 0 |

All segment-length sums equaled their pack input length. These are compatibility
facts for this bounded stream, not evidence that a changed-order planner is
better or safe.

The command emitted the upstream deprecation warning ``torch_dtype` is
deprecated; use `dtype` instead``. Dependency upgrades remain out of scope, so
this is recorded as upstream friction rather than changed in Wave 0.

## Independent Wave 0 audit triage — 2026-08-09

The required external read-only audit returned `HOLD` for Wave 1 on its sampled
pre-instrumentation tree and independently ran the then-current complete pytest
suite (`1126 passed`), all strict OpenSpec validation (`19 passed`), and
`git diff --check`. Its findings are dispositioned against the current owner:

- The provider-arm duplication was valid: the current synchronous provider is
  already CPU-build-then-transfer. Proposal, design, tasks, and the measurement
  contract now use three executable arms: current synchronous reference,
  legacy device-direct, and depth-one overlap.
- The eval chunk-skip warning was valid. Wave 4 preserves modulo assignment and
  treats retained/deserialized state as primary; bytes read and chunk skips are
  measured rather than assumed.
- The production v2 cache fingerprints are invalidated by the pre-existing
  `src/qwen/encoding.py` archive-comment edit. Wave 0 does not reuse or rebuild
  those caches: its bounded temporary fingerprints are recorded above, and the
  one authorized production materialization remains deferred until every
  determinant owner settles in Wave 9.
- The predecessor archive and stable-spec sync are intentionally uncommitted.
  This note already separates the accepted implementation commit from live
  spec-content hashes and last committed spec owners. No commit is inferred
  from implementation authorization.
- Cache determinant gaps, all-layer FA2 count/set proof, zero-weight graph
  retention, pre-model admission, and exact resume are confirmed target defects
  owned by Waves 1, 2, 3, 1, and 7 respectively; they are not silently accepted
  current behavior.
- ms-swift is a reference checkout, not imported by the training route. Its
  exact selected identity is provenance-only; fail-closed runtime admission in
  Wave 8 covers the dependencies the training route actually imports.
- Supported environment-selector sources must be compactly persisted. Wave 0
  records current selector values/sources; Wave 8 later removes or rejects any
  unpersisted semantic override through strict config.

Wave 1 remains closed until current-code standards and intent auditors accept
the completed Wave 0 receipts. The external audit is evidence, not authority
over the newer concurrent implementation.

### Post-correction receipt and GPU preflight

After the reference-only ms-swift and selector-source corrections, a fresh
warm execution against the same immutable temporary cache produced repository
state digest
`9e368e9a50f6d442a34ca8bd74cca72d8236c5cdd9b39a53d90486cc195ba88b`.
It recorded ms-swift as
`reference_only_not_imported_by_training_route` with
`origin_resolution: import_spec_without_import`; all actual dependencies were
`runtime_dependency` with imported-module identities. The cache root source was
`COORDEXP_SWIFT_PACK_CACHE_ROOT`. Preparation and publication were explicitly
`not_run` / `all_cache_hits`; payload admission was `0.7286293059587479`
seconds, entry-to-return was `3.7498904317617416` seconds, and process RSS
high-water was 1,185,640,448 bytes. The enclosing process wall clock was
11.84899316 seconds.

The temporary W0-GPU child config and private harness are outside the repository
at `/tmp/coordexp-wave0-baseline-FIU33RQX/`. Config validation resolved:

- config fingerprint
  `1088934be1d193307b0f11393ade898d615716e8e49ed00f98fe7ba2236fde2a`;
- harness SHA-256
  `38d092cf8a0a2a421b8642a0985dd2dc6dc6301e32b501bc1e7958ebe849f6bf`;
- the same train/eval cache fingerprints recorded above;
- 5 optimizer steps, grad accumulation 3 at world size 8, 120 pack
  presentations, eval at step 3, and one deduplicated final checkpoint event at
  step 5.

At `2026-08-09T05:52:58Z`, GPUs 2-7 still had unrelated allocations between
19,773 and 45,695 MiB (several at 54-100% utilization). This violates the
frozen all-eight-idle gate, so no W0-GPU launch was attempted and no existing
process was disturbed.

### Wave 0 P1 lifecycle and rank-convergence repair decision

The first current-code standards and intent audits kept Wave 1 on `HOLD` and
caused task 1.7 to be reopened. They found two receipt-level defects that a
five-step W0-GPU launch would otherwise make permanent in its baseline:

- the continuous `steady_state` interval included completed-step callbacks and
  intermediate scheduled evaluation/checkpoint work, while final-step
  evaluation happened to fall outside it;
- an ordinary Python exception on a non-main rank during pre-trainer model
  assembly was not guaranteed to converge into rank zero's terminal artifact.

For lifecycle accounting, three interfaces were compared:

1. repeatable pause/resume phase segments provide literal chronology but would
   expand the one-shot phase schema and every consumer;
2. subtracting evaluation time from one continuous interval is smaller but
   remains incorrect because logging and checkpoint callbacks are still inside
   the interval;
3. summing the already reduced all-rank-maximum optimizer-step durations for
   the accepted post-warm-up steps, while timing each scheduled evaluation in
   its own event receipt and aggregating those events, directly matches the
   production callback boundary and does not add per-step `run.json` writes.

Option 3 is the selected Wave 0 seam. Its steady-state duration scope is
`sum_of_accepted_all_rank_max_step_durations`. Evaluation resource values are
process-lifetime high-water counters observed after evaluation, not
evaluation-only incremental peaks. A run is steady-state eligible only when it
records every expected post-warm-up step and no later phase failure invalidates
the receipt. The outer measurement additionally persists one monotonic
entry-to-terminal duration; the enclosing fresh-process wall clock remains the
comparison for launcher/bootstrap/teardown costs outside that owner.

For rank convergence, rank-zero broadcasts cannot report a peer-only failure,
and an unbounded object collective would weaken the existing control-plane
contract. A new distributed framework would be disproportionate. The selected
seam therefore constructs the existing fixed-frame, 64-KiB-bounded Gloo rank
report gatherer immediately after artifact-owner initialization, reuses it for
pre-trainer phase status and `TrainRuntime`, and closes it from the outer owner.
Every rank reports the same phase identity before any phase is accepted; one
rank failure becomes one deterministic common failure and leaves rank zero able
to finalize the active phase.

This convergence contract covers caught rank-local Python exceptions when all
ranks can reach the phase boundary. It does not claim recovery from a killed
process or an exception inside a mismatched/blocking collective; those remain
launcher-timeout and production-supervision concerns. Acceptance requires the
deterministic five-step lifecycle test, a timeout-guarded real two-rank CPU/Gloo
peer-failure probe, focused and full tests, strict OpenSpec/diff checks, and
renewed independent standards and intent verdicts before W0-GPU or Wave 1.

### Pre-P1 Wave 0 code verification snapshot

- focused Wave 0 integration/artifact/runtime set: `131 passed`;
- complete repository pytest suite: `1160 passed`;
- targeted Ruff over all Wave 0 source/tests: passed;
- strict active-change OpenSpec validation: passed;
- strict all-OpenSpec validation: `19 passed, 0 failed`;
- `git diff --check`: passed.

### Post-P1 Wave 0 repair and verification receipt

The lifecycle/rank-convergence repair implements the decision above without a
dependency, cache-format, packing-policy, provider-default, or FA2 change:

- `steady_state` is an end-of-run aggregate whose duration is the sum of only
  applied, finite, post-warm-up all-rank-max step durations. Eligibility
  requires accepted and expected counts to match; a workload with no
  post-warm-up step is explicitly `not_run`.
- Every scheduled evaluation has an all-rank-max duration, per-rank timing and
  resource fields in its logging row, and an aggregate
  `evaluation_execution` summary. Resource fields are labeled
  `process_lifetime_high_water_observed_after_evaluation`.
- The terminal timing annotation is written only after a first atomic terminal
  state is durable. Its boundary is
  `training_entry_to_terminal_state_durable_before_measurement_annotation`, so
  the annotation does not claim to measure its own subsequent atomic write.
- The bounded rank-report gatherer is constructed immediately after the
  artifact-owner handshake, reused by converged base-model/adapter/embedding/
  memory-saver/policy assembly and `TrainRuntime`, and closed by the public
  pipeline's outer `finally`. Cross-rank failure status includes only rank,
  exception type, and stable error code; peer exception text is not copied.

Executed verification on the final repair tree:

- deterministic five-step lifecycle: warm-up steps 1-2, measured steps 3-5,
  intermediate/final evaluation, and a simulated 100-second final checkpoint;
  the steady duration remained exactly the reduced `3 + 4 + 5` step seconds;
- timeout-guarded real two-rank CPU/Gloo peer-failure test executed three
  independent times (`18.371`, `17.256`, and `16.430` seconds), each `1 passed`:
  rank 1 alone failed model assembly, both ranks received the same sanitized
  common error, rank zero durably recorded `status: failed` and terminal
  `model_loading`, and both gatherers/processes closed;
- one intentionally concurrent local composition of that probe with the
  existing eight-rank Gloo test was contaminated by the latter's 15-second
  connection/receive timeout (`160 passed, 1 failed`). No conclusion was taken
  from it. Serial isolation then passed the existing collective set (`2
  passed`) and all remaining focused tests (`159 passed`), for 161 unique
  focused/adjacent tests;
- complete repository pytest suite: `1166 passed`;
- targeted Ruff, Python compilation, strict active-change validation,
  strict all-OpenSpec validation (`19 passed, 0 failed`), and
  `git diff --check`: passed.

A fresh warm CPU execution against the same immutable temporary cache root
returned in `2.9983208626508713` seconds. Configuration/provenance resolution
was `2.0984597206115723` seconds and payload admission was
`0.5758726336061954` seconds; preparation/publication were explicitly
`not_run` / `all_cache_hits`. Train/eval fingerprints and pack counts remained
`c6be15...c7de1` / 32 and `60330b...626c` / 8. Process high-water was
1,185,480,704 RSS bytes, 0 read bytes, and 12,288 write bytes. CUDA remained
uninitialized, no model loaded, ms-swift remained reference-only, the cache
selector recorded the temporary environment source, and the executed dirty
state digest was
`f63f73ce800d8c0f26afa208532365994717e943dc15a89963c7e6852dbfacee`.

At `2026-08-09T06:41:10Z`, GPUs 2-5 and 7 still held approximately 24-43 GiB,
with GPUs 3-5 at 55-96% utilization. The all-eight-idle W0-GPU gate therefore
remained closed; no GPU process or shared cache was touched.

### Post-P1 independent audit disposition

Separate current-tree standards and intent auditors both closed the prior P1
and accepted task 1.7. Neither reported a P0, P1, or P2:

- standards: `PASS` for Wave 0 code/standards acceptance, including the
  additive aggregate schema, terminal-state-first timing annotation, complete
  converged model surface, bounded gatherer reuse/close, and isolated executed
  collective evidence;
- intent: the prior measurement-contract P1 is `CLOSED`; steady-state/eval/
  checkpoint accounting, failure ineligibility, non-secret peer convergence,
  ordering semantics, and the no-upgrade boundary match the frozen intent.

These are code-repair verdicts, not task 1.9's final executed-baseline verdicts.
Both auditors independently retained the same evidence-pending `HOLD`: task
1.8 still requires the frozen temporary-root five-step eight-rank W0-GPU run
after the all-eight-idle preflight, and task 1.9 requires final receipt review
after that run. Wave 1 remains closed and no later-wave implementation has
started.

### W0-GPU compatibility baseline receipt

The final preflight at `2026-08-09T06:59:45+00:00` found all eight selected
NVIDIA A100 80 GB PCIe devices at 3 MiB or less and 0% utilization, with no
compute process. The temporary root occupied 32 MiB before launch, its output
directory did not exist, and the host had 1.6 TiB available. The child config
resolved to
`1088934be1d193307b0f11393ade898d615716e8e49ed00f98fe7ba2236fde2a`.
An executed recursive comparison with the base config proved that its only
resolved changes were `run.name`, `run.artifact_root`, `training.max_steps`,
`eval.forward.steps`, and `checkpoint.steps`, exactly as frozen.

One command-wrapping attempt placed GNU `timeout` outside `conda run` and
returned zero in 0.189 seconds without starting Python or Accelerate. It
created no output directory, left every GPU at 3 MiB/0%, and a CPU-only control
proved that this shell/tool wrapping shape silently skipped the Conda child.
It is an invalid non-observation, not a training or performance result. The
accepted launch put the same 30-minute guard inside the named environment:

```text
COORDEXP_SWIFT_PACK_CACHE_ROOT=/tmp/coordexp-wave0-baseline-FIU33RQX/cache CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n ms timeout --signal=TERM --kill-after=60s 1800s accelerate launch --num_processes 8 /tmp/coordexp-wave0-baseline-FIU33RQX/run_wave0_gpu.py --config /tmp/coordexp-wave0-baseline-FIU33RQX/wave0_gpu_5step.yaml
```

The valid run completed at
`/tmp/coordexp-wave0-baseline-FIU33RQX/outputs/wave0_compatibility_reference_5step`.
Its immutable receipt identities are:

- harness SHA-256
  `38d092cf8a0a2a421b8642a0985dd2dc6dc6301e32b501bc1e7958ebe849f6bf`;
- train cache `c6be15b8d7840524829c70e1fa5729accf600ae366d97fccf32778eca84c7de1`
  with 32 packs, and eval cache
  `60330b24519f0931e5c84765874c70e0cb5d54cd472bd6f79fc30977edab626c`
  with 8 packs, both admitted as immutable v2 payloads;
- repository HEAD `23a01061a4c7a8571399a572886d75e2c02286f7`, dirty execution digest
  `f63f73ce800d8c0f26afa208532365994717e943dc15a89963c7e6852dbfacee`,
  synchronous/default provider, source-order next-fit packing, source-order
  training, fixed `geo_sorted` intra-image rows, disjoint-shard evaluation,
  installed FA2, and resume disabled;
- strict run/resolved-config/log hashes
  `2c5879129e9e156119e38985496e3d7751453509be3da8b8f70280ea22370ef9`,
  `f424b5e7924af113283c0e369da6d22a264cef4d97079c35d7c28d9cea7c9ef7`,
  and `4a2c268b299291d96c43860e57c9265bc9880e1ef1ee3e5a55386f4915a01d95`.

The run completed five applied finite optimizer steps on eight ranks. Each rank
consumed 15 micro-steps, so the run receipt records 15 local packs and the
executed global rank-pack presentation count is 120. One evaluation ran after
step 3, one checkpoint event ran at step 5, and the final alias selects the
atomically committed `checkpoints/step-5` payload. Entry to durable terminal
state was 73.69345488399267 seconds. The measured phases were:

- entry to completed model load: 22.959885 seconds, of which model loading was
  11.267064146697521 seconds;
- warm cache admission: 1.1462332271039486 seconds; preparation and publication
  were explicitly `not_run_cache_hit`;
- eval hydration: 0.18719061091542244 seconds;
- entry to first successful optimizer step: 35.769406 seconds, with the
  first-step phase itself taking 9.412104953080416 seconds;
- post-warm-up steps 3-5: 9.149637639522552, 8.743157681077719, and
  8.782313872128725 seconds. Their exact sum is
  26.675109192728996 seconds, mean 8.891703064242998 and median
  8.782313872128725 seconds; the run is steady-state eligible at 3/3;
- evaluation execution: 1.4611424170434475 seconds; checkpoint publication:
  0.9521081708371639 seconds.

Across the 24 measured rank-step rows, input build was 0.942868236 to
1.31310224 seconds (median 1.0225790355, mean 1.053240180125), all recorded
input wait values were zero, and per-step build skew was 0.370234004,
0.108427266, and 0.196260004 seconds. Global observed process-lifetime
high-water values were 9,154,424,832 RSS bytes, 11,554,700,288 CUDA allocated
bytes, 15,407,775,744 CUDA reserved bytes, zero read bytes, and 335,872 write
bytes. These stay well below every frozen ceiling.

Read-only cache replay measured train utilization as
357,559/384,000 = 0.9311432291666667 and eval utilization as
90,054/96,000 = 0.9380625. Every pack contains eight atomic image segments;
the unused capacity is headroom rather than materialized padding. This is a
compatibility baseline, not a binpacking or provider-promotion result.

An independent executed-receipt probe strictly parsed every JSON row,
recomputed the steady duration with zero delta, confirmed all counters and
policy/cache/config identities, read both safetensor payloads, and scanned for
named secret/environment markers. The adapter payload is 72,111,984 bytes with
588 readable keys; the special-token payload is 8,224,864 bytes with one
`shared_embed_delta` key. It returned task 1.8 `PASS` with no discrepancy;
arbitrary sensitive-string detection remains outside that bounded marker scan.

### Final Wave 0 gate

Task 1.9 closed only after the executed W0-GPU receipt was joined with all of
the following final-tree checks:

- targeted artifact/resource/runtime/pipeline tests: `127 passed`;
- isolated timeout-guarded real two-rank peer-failure convergence: `1 passed`;
- strict config and train-entry tests: `60 passed`;
- strict all-OpenSpec validation: `19 passed, 0 failed`;
- targeted Ruff and `git diff --check`: passed;
- no task-created repository temp/output/cache residue, no checkpoint staging
  directory, and all eight GPUs released after the run.

The separate standards-compliance auditor and intent/contract auditor both
returned `PASS` with no P0, P1, or P2. The former independently parsed and
recomputed run rows, phases, hashes, counters, resource scopes, checkpoint
readback, and provenance. The latter independently verified the user-owned
ordering/no-upgrade/default-policy boundaries, temporary-root-only execution,
and the absence of a promotion or optimality claim. The prior rank-local
convergence P1 remains closed. These verdicts accept Wave 0 only; every Wave 1
and later behavior remains unimplemented and must pass its own gate.

## Wave 1 fixed point — 2026-08-09

This section supersedes the preceding Wave 0 statement only for Wave 1. Tasks
2.1-2.6 are implemented; task 2.7 remains open until the final standards
auditor joins this durable packet with the independent intent verdict.

### Immutable v3 identity and authenticated admission

`src/training/pack_cache.py` now resolves one immutable target at
`<cache-root>/coordexp-swift-pack-cache-v3/<64-hex-fingerprint>`. `Rebuild`
means publishing to a previously absent v3 fingerprint. A valid target is a
byte-preserving hit; an occupied invalid target is a collision. Publication
uses a private sibling stage, re-resolves the complete determinant registry
after staged payload validation, and installs with
`renameat2(RENAME_NOREPLACE)`. The normal path never repairs, replaces,
deletes, retains, or garbage-collects a published cache.

Every public reader and writer now requires the caller-selected cache root and
rejects alternate well-shaped roots or symlinked path components. Each required
chunk is opened once with no-follow directory/file descriptors, required to be
a regular file, bounded to 4 GiB, read into one descriptor-stable byte snapshot,
SHA-256 checked, and restricted-unpickled from those exact bytes. Replacement,
short/grown reads, metadata drift, non-regular entries, and symlinks fail closed.

The determinant registry binds semantic identities plus exact source digests
for an independently enumerated 31-owner surface, including raw-data geometry,
renderer/parser, image resolution/loading, encoding, packing, supervision,
MRoPE/FA2/forward payloads, the production micro-step constructor/schema, and
the serializer. It also binds referenced image bytes, the complete realized
vocabulary membership through bounded canonical digests, serialized precision
and FA2-proof fields, and a recursive model-front-end envelope. Only exact
root-level conventional model weights or content-index-declared shards are
excluded; nested same-basename assets remain hashed. Total discovered files,
weight-index declarations, hashed files/bytes, and unclassified file size are
bounded.

### Model-free startup boundary

Training now requires paired, validated launcher `RANK`/`WORLD_SIZE`, creates a
temporary CPU/Gloo convergence plane and one launcher-rank-zero run owner, and
keeps `cache_admission` active while resolving processor/token identities,
realized vocabulary groups, train/eval fingerprints, the current rank's train
chunks, and every eval payload. The temporary group closes before Accelerator
construction. A second bounded convergence boundary attests exact Accelerator
rank/world identity before model, adapter, optimizer, or GPU materialization.
Missing, retired, unpublished, corrupt, colliding, rank-local, and ownership-
handshake failures converge to bounded typed diagnostics and finalize the one
shared failure artifact rather than orphaning it. Retained rank-selective eval
hydration remains explicitly deferred to Wave 4.

### Final-tree executed receipts

- authenticated-snapshot replacement/symlink/size/short-read/metadata-drift
  negatives: `5 passed`;
- complete Wave 1 cache, determinant, production-constructor, preflight,
  assembly, rebuild, and convergence subset: `267 passed`;
- expanded `tests/training tests/config tests/artifacts tests/runtime` gate:
  `513 passed`;
- the production constructor was executed and pickle-round-tripped for all four
  BF16/FP16 by first-micro-step/every-forward combinations;
- real two-process CPU/Gloo tests covered a rank-one required-chunk failure,
  successful temporary-Gloo teardown into the controlled Accelerator surface,
  and a rank-one-only Accelerator identity mismatch; the focused two-failure
  command returned `2 passed`;
- concurrent same-fingerprint writers produced exactly one no-replace
  publication and a byte-identical loser hit in temporary roots;
- targeted Ruff, strict active-change validation, strict all-OpenSpec validation
  (`19 passed, 0 failed`), and `git diff --check` passed.

The shared production cache remained 61 GiB with 37 top-level publications and
mtime `2026-08-04 10:31:12.901756461 +0000`. No v3 namespace, stage, or backup
was created there, and no task-created stage/backup residue was found in the
repository. No GPU, dependency upgrade, shared-cache write, Git stage, commit,
or push occurred. Source-order next-fit packing and fixed sorted intra-image row
order are unchanged.

### Audit and claim boundary

The final intent/contract auditor returned `PASS` with no P0, P1, or P2 and
confirmed immutable-v3, ordering, no-upgrade, pre-model admission, Wave 4 eval
ownership, and authorization boundaries. Standards audits first exposed and
then verified repairs for the owner-oracle/geometry gap, nested weight
classification, selected-root binding, inventory bounds, constructor execution,
and chunk checksum/decode race. The last standards pass confirmed the code and
tests technically satisfy tasks 2.1-2.6 and held task 2.7 only until this receipt
packet and task state were persisted; its post-packet verdict is recorded below.

These receipts prove cache identity/admission and control-plane semantics in
temporary-cache CPU/direct and two-rank Gloo scope. They do not prove GPU/NCCL
transition behavior, training numerical correctness, or a performance win. A
failure during Accelerator construction itself cannot converge through the
post-Accelerator group; rank-zero best-effort finalization plus launcher peer
supervision remains the bounded fallback. The 4 GiB chunk snapshot bound was not
exercised against a production v3 cache. Wave 4 still owns retained selective
eval hydration, and Wave 2 owns real-model all-layer FA2 parity/proof.

### Final Wave 1 gate

After this packet was persisted, the standards-compliance auditor re-read the
task state and receipts and returned `PASS` with no P0, P1, or P2. It reconciled
the `267 passed` focused subset, `513 passed` expanded gate, strict validation,
Ruff/diff/residue receipts, shared-cache non-mutation, and the independent
intent `PASS`. Task 2.7 is therefore closed. This accepts Wave 1 only and does
not authorize or validate Wave 2 execution.

## Wave 2 technical hold — 2026-08-09

Tasks 3.1-3.3 have implementation and CPU-test evidence, but Wave 2 is not
accepted and tasks 3.4-3.7 remain open. The final authorized real-model probe
used v2 plan SHA-256
`e5c2b1eb0c7ff99b7a66de8dc172f5af0331959d9b5f12a07e61096dd79b4197`
and produced terminal receipt SHA-256
`fbd0556f597aab3facae4af1ef6bc7ebb24c2f447cf172df1c24f4d127ac2820`.
The byte-identical artifact and claim-boundary record are under `receipts/`.

The terminal `qwen.parity.clean_failed` context records that semantic atoms,
the exact denominator projection, supervised logits, total loss, and all four
mandatory BF16-derived per-term scalars passed their frozen gates, while the
complete trainable-gradient comparison returned false. This consumes the
single replacement authorization. It is not sufficient evidence that packed
training gradients are intrinsically wrong: the generic failure path discarded
the already-computed per-parameter comparison, arm, all-layer proof,
negative-control, timing, GPU-memory, and measurement payloads.

Independent contract, model-diagnosis, and Opus 5/max reviews classify the
artifact as both a valid terminal failure of the frozen gradient gate and an
incomplete diagnostic artifact. They identify two leading infrastructure risks:

- gradient tolerance is selected from parameter storage dtype even though PEFT
  DoRA and the shared special-token delta are FP32-stored but their gradients
  originate in distinct BF16-autocast/FA2 forwards;
- aligned `None` gradients are currently counted as a parity failure instead of
  a separate trainable-coverage failure.

Neither diagnosis is recoverable as the actual cause because the bounded
gradient failures were not persisted. Frozen v2 tolerances MUST NOT be widened
or retrospectively reinterpreted. Non-GPU remediation owns evidence-preserving
failed receipts, fail-closed receipt publication, parity-versus-coverage
diagnostics, and CPU synthetic evidence. Any later real-model execution requires
a new user-owned stop-rule/version decision and a fresh pre-result contract.

Wave 3 and later waves remain on hold under the sequential correctness gates.
No accepted Wave 2 parity, FA2 execution proof, gradient equivalence,
performance, or promotion claim exists.

### Wave 2 non-GPU failure-artifact remediation closure

The evidence-preservation remediation is complete at the following fixed source
identities:

- `src/qwen/parity.py`:
  `e3be883d65f604a383780d031b0d483f18548d9e26cdbff8f68f2aa58977b490`;
- `scripts/probes/coordexp_swift/wave2_packed_parity.py`:
  `5102f4b7601b7dfbd74f31c7c2a272e36eeff7e437d6678b2238a9578b04a9c5`;
- `tests/qwen/test_packed_parity.py`:
  `30fd9bc3174d012d2f2e3fec84c7222c0e6979325b9501747e1f3999383823f8`.

Rich failed receipts now bind each reached stage to one exact completed-phase
prefix, exact root-field inventory, and stage-appropriate arm inventory. Their
completed comparison, proof, negative-control, timing, resource, and measurement
subtrees are deeply validated. Gradient parity and trainable coverage are
separate results, while the overall frozen v2 gate still requires both. A
terminal receipt publication failure is surfaced through an absent-target-only
publication-failure sidecar rather than being swallowed.

Both passed receipts and post-plan rich failed receipts require the same
authenticated plan during validation. The receipt plan hash and plan-owned
config, repository, dependency, model-weight, and complete source-owner
identities must match exactly; the loaded-model transition remains governed by
the existing component-attestation contract. Genuine pre-plan rich failures and
the two immutable historical minimal v2 failures remain readable without
inventing evidence they never contained.

The bounded production-shaped CPU fixture retained and revalidated all 589
gradient-comparison rows while keeping each arm inventory capped at 64; the
published receipt was 711,473 bytes, below the 8 MiB limit. More than 4,096
comparison rows and an oversized pre-arm trainable inventory fail closed. Final
independent receipts were `97 passed` for the focused parity file, `261 passed,
1 skipped` for `tests/qwen tests/losses`, `416 passed` for the proportional
artifact/runtime/training matrix, and `56 passed` for the focused rich-receipt,
capacity, provenance, publication, sidecar, collision, and mutation selection.
Targeted Ruff, format, `py_compile`, strict OpenSpec validation, and
`git diff --check` all passed. The final independent contract audit found no
P0, P1, or P2 in this remediation.

This closure repairs future evidence fidelity only. It does not alter the
immutable failed v2 GPU receipt, change any frozen tolerance, restore the lost
v2 per-parameter observations, authorize a retry, complete tasks 3.4-3.7, or
release Wave 3. A new versioned gradient/cadence contract remains a user-owned
decision.

## Wave 2 v3 conditional reopening decision — 2026-08-10

The user selected the conditional versioned reopening and then simplified its
numerical rule: keep Qwen attention/linear execution under production BF16
autocast, retain the production graph-connected `ConvertOutputsToFp32`
logits/loss seam for high-precision computation, and detach only serialized
comparison observations. Gradients return through that cast into the BF16
upstream graph. The accepted v3
contract does not add cosine similarity, relative-L2 gates, adaptive near-zero
floors, or a result-dependent tolerance. All 589 FP32-stored but
BF16-compute-derived DoRA/special-token gradients use the same elementwise
FP32-computed `rtol=5e-3`, `atol=5e-3` gate as the other cross-forward BF16
observations.

V3 retains the exact authenticated v2 model/config/sample/supervision/boundary
workload, while the terminal v2 receipt remains immutable and cannot be
retroactively accepted. One identical packed repeat supplies only a fixed
`max_abs<=2.5e-3` measurability check; both packed arms must independently pass
against the separate reference. The separate reference must reproduce the
production two-micro-step streaming backward cadence rather than summing two
differentiable graphs. Coverage is exact and separate from parity, and the
boundary-only negative must be detected by supervised logits or total loss.
The probe process requires `FLASH_ATTENTION_DETERMINISTIC=1` before plan
preparation and runtime/model setup, and binds it in the plan and receipt; this
is not a production-default or backend/dependency change.
Each packed arm clears gradients once immediately before its forward. The
one-shot boundary is made durable by one absent-target attempt-start marker
published immediately before real-model GPU setup; a post-marker failure
consumes the attempt and a pre-marker failure requires a fresh launch audit.

This decision authorizes planning and reversible implementation/CPU validation,
not an immediate GPU launch. Exactly one new real-model GPU attempt may run only
after the updated OpenSpec, v3 implementation, fresh immutable launch plan, and
independent standards and intent/contract audits are P0/P1-free. The existing
eight-GPU-hour ceiling applies. Any attempt that reaches real-model GPU setup is
consumed by its terminal result; there is no automatic retry, sample switch,
tolerance edit, dependency upgrade, or v4 authorization. Wave 3 remains held
until a v3 receipt passes every mandatory gate.

The finalized planning packet passed both independent read-only audits after
closing two contract gaps: every packed arm now clears stale gradients exactly
once before forward, and one immutable absent-target attempt-start marker makes
the pre-GPU one-shot boundary durable. Both auditors reported no remaining
P0/P1, and `openspec validate --all --strict` reported `19 passed, 0 failed`.
Task 3.4 is therefore complete. This planning PASS authorizes reversible code
and CPU-test work only; it does not authorize a GPU launch.

A subsequent owner trace clarified that v3 plan preparation intentionally stays
model-free, so it cannot truthfully bind PEFT-installed concrete parameter names.
The plan therefore binds the strict structural declaration
`196/196/196/1=589`; after CPU model/adapter/delta installation and before GPU
setup, the live exact names/shapes/dtypes are authenticated from the installed
model and existing adapter/delta receipts, then bound into the attempt marker
and terminal receipt. This avoids an extra model load during plan preparation
without weakening per-arm exact coverage. Task 3.4 was reopened for a focused
text-only re-audit of this clarification.

Both independent auditors returned PASS with no P0/P1 for the refined split.
They confirmed that fixed counts plus strict suffix owners are an independent
expectation rather than self-authentication, while the marker-bound live rows
provide exact runtime coverage before GPU work. Full strict OpenSpec validation
remained `19 passed, 0 failed`; task 3.4 is closed again on this final wording.

At the time of the immutable v3 execution, the later Wave 5 strict
synchronous-provider default changed the then-current resolved config
fingerprint to
`0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1`
without changing the authored YAML bytes. The executed Wave 2 v3 plan retains
that exact historical identity and its projection-v1 attestation:
remove exactly `training.forward_input_provider_mode: synchronous`, yielding
the immutable parent-v2 fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`.
Extra removed paths, non-synchronous mode, wrong projection digest, or any
other config drift fail closed. Whole-plan revalidation remains exact, and the
live resolved mode is reasserted before marker/GPU work and retained in the
plan, marker, and terminal receipt. This closes the later-default launch-audit
P1 without regenerating or reinterpreting the immutable parent plan and without
authorizing GPU execution.

## Wave 2 v3 immutable result and scoped release — 2026-08-10

The sole audited v3 launch completed and consumed its attempt marker. Durable
artifacts are `receipts/wave2-v3-plan.json`,
`receipts/wave2-v3-attempt-marker.json`, and
`receipts/wave2-v3-terminal-receipt.json`, with file SHA-256 values recorded in
the measurement plan and the accompanying postmortem receipt. The terminal
status is intentionally unchanged: `failed/qwen.parity.clean_failed`.

Executed evidence is internally consistent and deeply validated. The complete
supervised forward is byte-identical across packed and streaming-separate arms,
all objective and denominator fields pass, the two packed gradient inventories
are bit-identical, the corrupted-boundary negative is decision-bearing, and the
all-layer proof observes the accepted deterministic varlen route in all 28 text
layers. The cross-shape gradient comparison fails only its originally frozen
numeric gate; all 589 rows are present and finite.

Independent contract, model-diagnosis, and Opus/max postmortems agree that this
is a valid terminal measurement rather than evidence of a packing-forward
semantic defect. The one-packed-backward versus two-streaming-backwards contrast
changes BF16 reduction shape; 196 LoRA-A tensors are degenerate zero-signal rows
at cold start, and the remaining differences are concentrated on reduction-
heavy LoRA-B/DoRA/special-token surfaces. The artifact does not establish exact
Jacobian equivalence, and no threshold is retroactively widened.

The user selected the narrow claim disposition: the demonstrated
forward/logit, loss/denominator, segment-isolation, and all-layer FA2 semantics
release Wave 2; cross-shape gradient equality remains a preserved rejected
diagnostic and does not block Wave 3. There is no retry, v4, sample switch, or
receipt reinterpretation. A separate future-artifact fix retains completed GPU
idle preflight evidence in rich failed receipts; it cannot and will not mutate
this historical terminal receipt.

## Current Wave 2/3 config compatibility readers — 2026-08-10

Later strict packing and exact-resume defaults changed the live resolved config
fingerprint to
`da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`.
Current Wave 2 and Wave 3 config compatibility readers use their respective
projection-v2 schemas and remove the same exact ordered path/value inventory before requiring legacy
fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`:

- `training.forward_input_provider_mode=synchronous`;
- `packing.policy=source_order_next_fit`, `packing.window_size=null`,
  `packing.lookahead=null`, `packing.seed=0`, `packing.worker_count=1`,
  `packing.fragment_item_budget=1024`,
  `packing.fragment_byte_budget=4194304`,
  `packing.cursor_byte_budget=65536`, and
  `packing.max_packs_per_fragment=null`;
- the complete `resume={checkpoint_dir:null,mode:disabled}` object.

Any missing, extra, reordered, or changed entry and any current/projected digest
drift fail closed. The immutable executed Wave 2 plan and failed receipt remain
projection-v1 historical evidence; they are not rewritten. The current reader
admits the old failed receipt without `execution.requested_device` and
`execution.gpu_idle_preflight` only when all of these fixed identities match:
plan hash `03834e309357dace9d7b51a6c51186b14b09bea8953b3c3d12e82609cff1a49d`,
plan payload SHA-256
`6f27e22b3e8a2c8aff626e9062afc0d664cee04e77341d0d4fdf7723c08a3483`,
receipt payload SHA-256
`8bdea86554c64b656b6a60325c2b5d69a12d9e58b2a7cbfdac01e0a9f9aabe13`,
terminal `failed/qwen.parity.clean_failed`, and stage `comparisons`. New rich
receipts still require both execution fields. This is an exact-payload reader
exception, not a schema relaxation or result reinterpretation.

Current implementation/test identities for this compatibility surface are:

- `src/qwen/parity.py`:
  `e87fae9514379b60b9cdf5b39421bf65d9ff38b30453f74e4dfd62c0d2573e2d`;
- `scripts/probes/coordexp_swift/wave2_packed_parity.py`:
  `96854d55cd7f905e0d450c62e5f5b9c87a2464c4a4b44742deaf035c2095260f`;
- `scripts/probes/coordexp_swift/wave3_zero_weight_gpu.py`:
  `ec89961157e0e8230eb1d836d757492989cea3b25f2dffe2b8e6901265340830`;
- `tests/qwen/test_packed_parity.py`:
  `6df44168924ce8053f0f4903d6239e2afc44290b076680e9c9291a53fbe25a63`;
- `tests/losses/test_wave3_zero_weight_probe_contract.py`:
  `208fe5f99b026771396bd383752d04b070e0095ba8869edb524f68482bfda172`.

Wave 3 artifact schemas are now v3 even though its config compatibility
projection remains v2. The active plan binds the complete frozen base-model
weight index/shard identity and aggregate
`e128f5f42f1a042702efc1eed5a787da36a4d586a17b538671260996056284aa`;
the worker rehashes it before model load, and marker/runtime/receipt evidence
cross-binds it. The immutable r2 v2 chain is admitted only by the fixed-root,
fixed-raw-hash historical reader as `historical_non_executable`. All active v2
entry attempts fail before CUDA, child creation, model/cache access, or artifact
publication. The unexecuted r3-v2 plan remains immutable prelaunch evidence and
must not be overwritten.

The fresh-root r3-v3 replacement has now run exactly once. Its plan, marker,
and terminal receipt file SHA-256 values are respectively
`f4936e01729722ce297f9d9a09bd791967fcb741ada827d32b1321deaa86288e`,
`27b2b957d6f96b9a454d3394773fbc65672b673824e14b45c0f5373847c9b297`,
and `6283baa2b72c0e730bb671179980423564188b9d910778cf29ba66042298958c`.
It terminated `failed/wave3.accelerator` after marker publication and before
any model forward/backward, so it is an incomplete infrastructure artifact and
not a zero-weight result. The exact source cause was Accelerate 1.10.1's
indexless one-process `torch.device("cuda")` representation versus the probe's
indexed `cuda:0` equality check. Current source now performs CPU-only
Accelerate/distributed/launcher admission before the marker, pins exact
`ACCELERATE_TORCH_DEVICE`, and strictly validates the constructed Accelerator
after the marker. The fixed source/test identities above passed 98 contract
tests, 19 accelerator-focused tests, loss/trainer adjacency, Ruff, format,
compile, strict OpenSpec, and independent P0/P1/P2-free source audit. This is
source-only closure: the replacement authorization was exhausted and no
r4/retry was allowed at that 2026-08-10 fixed point, when tasks 4.4/4.5 remained
open. The later 2026-08-11 disposition closed both as
`no_performance_promotion` without another launch.

## Wave 8 pinned runtime and reference points — 2026-08-10

Wave 8 keeps the installed Transformers, FlashAttention, Torch, CUDA runtime,
Accelerate, PEFT, tokenizers, and `flash_attention_2` route unchanged. Pinned
runtime-baseline schema 3 has digest
`cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`.
In addition to the previously recorded `libtorch_cuda`, `libc10_cuda`, cuDNN,
and driver identities, admission now binds distribution
`nvidia-cuda-runtime-cu12==12.8.90` (`RECORD` SHA-256
`bef69015da09064656ac31d35fc73d6436712380c602ff38fb5fc16c30512bdf`,
11,369 bytes) to the `libcudart.so.12` object actually mapped into the process.
The admitted object is the distribution-relative
`nvidia/cuda_runtime/lib/libcudart.so.12`, 728,800 bytes, SHA-256
`c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920`.
The receipt persists its live absolute origin; admission requires that origin
to be the same file as the selected distribution object and fails closed if the
distribution, loaded object, origin relation, size, or content is unavailable
or drifts.

The current authorized minimal CUDA initialization attestation is retained at
`outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json`
with internal receipt SHA-256
`829d86ec8d977f30d37e8e979acba6527b3640180825e551785078a4876b1043`.
The receipt file SHA-256 is
`d88c9c4ded7c698786c54721f8fbbffdad1467e467f845f72dbdb8cfe9eb75ac`.
It records the actually mapped `libtorch_cuda`, `libc10_cuda`, `libcudart`,
`libcuda`, `libcudnn`, and `libcudnn_graph` objects and admits all six against
the pinned content, size, origin, and ELF build identities. This is a native
runtime identity receipt only: it loaded no model, wrote no cache, and carries
no training-quality or throughput claim.
The immutable r1 receipt remains historical pre-owner-expansion evidence and is
not the current Wave 8 admission fixed point.

Current provenance owner identities are `src/artifacts/provenance.py` SHA-256
`a0a9072b343f168e8efa0a110075d595be09c28f75408a596e4bc6f75350abbb`
and `src/training/pipeline.py` SHA-256
`ff1d5bb7eef352ae9c191a1deaf5c6f0beb31456735aa2fa73e8108e6189985a`.
Their focused tests are `tests/artifacts/test_provenance.py` SHA-256
`0ce18fafaa6f4cdc8379dba13cda964849709011f5edc1eea9b43524ced5d918`
and `tests/artifacts/test_wave8_native_runtime_attestation.py` SHA-256
`fe8e112a3dc6c99c0300c1ef2c22393c2312c36c9453b294779a2674e063b384`.

The following coordinates are immutable reference points for later work, not
runtime admission owners or upgrade candidates in this change. Line numbers
refer to the exact pinned files and are paired with content hashes so future
line drift cannot be mistaken for the recorded implementation.

| Reference identity | Content-bound file | Exact symbols and call sites | Disposition |
|---|---|---|---|
| ms-swift commit `f2797138dba0e224cfff735cd89a528a08d8732a` | `swift/dataset/packing.py`, SHA-256 `f873186b6841170804f76ef64320b223838179e7b63ec24222858ff08ee85548` | `calculate_matched_group` L16-L26; `PackingDataset.create_packed_idx` L89-L102 with 1,000-row chunks; `IterablePackingDataset._put_data_in_queue` L156-L163 and `__iter__` L182-L203 with `packing_interval` | Reference-only; `swift` is not imported by the CoordExp training route and its drift remains non-admission provenance. |
| Transformers `4.57.1` | `transformers/modeling_flash_attention_utils.py`, SHA-256 `293fe81c6bd38aac8a3bc6ae086ff21b9695b349e97c9650b1fe61e42a36d710` | `lazy_import_flash_attention` L127; `prepare_fa_kwargs_from_position_ids` L316-L357; `_process_flash_attention_kwargs` L453-L526; `_flash_attention_forward` L529-L665, including explicit-varlen dispatch L601-L655 | Imported runtime reference already covered by fail-closed source identity; no backend change. |
| Transformers `4.57.1` | `transformers/integrations/flash_attention.py`, SHA-256 `850aa63f9473391188c357454cfb6a89339394d26f6024ccbe246f00f6344144` | `flash_attention_forward` L14-L84 and `_flash_attention_forward` call L66-L82 | Selected `flash_attention_2` adapter reference; no promotion or benchmark arm. |
| Transformers `4.57.1` Qwen3-VL | `transformers/models/qwen3_vl/modeling_qwen3_vl.py`, SHA-256 `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` | vision dispatch `Qwen3VLVisionAttention.forward` L185-L223 with explicit `cu_seq_lens` at L216-L217; text dispatch `Qwen3VLTextAttention.forward` L416-L457 through `ALL_ATTENTION_FUNCTIONS` L440-L452; decoder call site L502-L510 | Qwen call-site reference for the repository-owned text-layer proof; vision calls remain separately classified. |

## Wave 7 r5 immutable consumed failure — 2026-08-11

The one-shot r5 sequence is no longer prospective. Its immutable artifact chain
is:

| Artifact | File SHA-256 | Internal canonical payload SHA-256 |
|---|---|---|
| plan | `ea3cef1d96412cd3f425e9c7f0c0cd562ceafa2fe59c39c225755898dbb66a1f` | `73c502d69ec8b999a9e451620702b54eb1d573bf775e18d32edde08d216f53c4` |
| sequence marker | `58aa7425dc6c019a23e790ba072495f0fb787b805eabaf922f079c35eddb19b5` | `a620caa0e64a12be980120e310a914571c78ee0889d6da0760aae208f3c82224` |
| terminal receipt | `d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1` | `f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656` |

The uninterrupted phase exited `rc=1` after 20.094 seconds. The controlled
parent, pre-child gate, exact-resume child, and final comparator remained
`not_started`. Bounded cleanup and recovery passed. Marker publication consumed
the attempt and the terminal receipt authorizes no retry.

CPU-only attestation reproduces the root cause: launching the absolute
`src/train.py` path without adding the repository root to `PYTHONPATH` raises
`ModuleNotFoundError`; module invocation succeeds. This confines r5 to a
launcher/entrypoint environment failure before model or cache access and before
any training phase. It does not establish model behavior, cache behavior,
training numerics, interruption behavior, or exact-resume correctness, and it
does not close tasks 8.2, 8.7, 8.8, or 8.9.

The successor source fix is in progress only. No r6 execution is authorized.
Before any successor marker can be published, the user must explicitly
authorize a new attempt with fresh absent roots, runtime attestation,
deterministic preflight, request, immutable plan, marker, and private cache. The
new packet must retain the same at-most-once no-retry rule, cost ceiling,
shared-load restrictions, cleanup/recovery requirements, and diagnostic claim
scope. Until such an attempt passes the Wave 7 gate, tasks 9.3 and 9.5 remain
open. Wave 3-5 stay closed without performance promotion, their production
defaults remain unchanged, and Wave 6 remains pending future design with
`source_order_next_fit` as the production default.

## Wave 7 r6 one-shot successor authorization — 2026-08-11

Later on 2026-08-11, the user explicitly authorized one fresh r6 one-shot successor.
This later authorization supersedes only the prospective no-authority
statement in the immutable r5 section above; it does not alter or reinterpret
the r5 failure. r6 is not an r5 retry. It is exactly one prospective successor
under a fresh r6 run/cache/runtime/preflight/request/plan namespace, with
absent-only publication roots, a fresh private cache, and its own at-most-once
marker.

The r6 authority uses amendment schema
`coordexp-swift-wave7-r6-amendment-v3`, request schema
`coordexp-swift-wave7-exact-resume-sequence-request-v5`, plan schema
`coordexp-swift-wave7-exact-resume-sequence-plan-v5`, marker schema
`coordexp-swift-wave7-exact-resume-sequence-marker-v5`, and terminal schema
`coordexp-swift-wave7-exact-resume-sequence-receipt-v5`. The request retains the
existing `legacy_r4_failure` binding and adds exactly one
`predecessor_sequence_failure` binding to the immutable r5 terminal. That r5
binding is copied unchanged from request to plan and is historical,
non-executable evidence; it cannot satisfy a current terminal, request, plan,
marker, or launch authorization.

Each r6 phase runs at most once with a timeout of at most 600 seconds
(`<=600s/phase`). The complete sequence is bounded to at most 2400 seconds wall
time (`<=2400s wall`) and at most 14400 GPU-device-seconds
(`<=14400 GPU-device-seconds`). Shared-GPU observations remain non-promotional.
Failure at any gate publishes terminal evidence and stops: no later phase may
start, there is no retry or root switch, and there is no automatic r7.

This one successor does not reopen the Wave 3-5 performance campaign or promote
Wave 6. Their no-promotion/default-retention dispositions remain intact, Wave 6
matched training remains pending, and `source_order_next_fit` remains the
production default. Tasks 9.3 and 9.5 remain open until r6 passes the Wave 7
gate.

## Wave 7 r6 immutable consumed preflight failure — 2026-08-11

The r6 authorization is consumed by a deterministic-preflight failure after
`KeyboardInterrupt`. No model load, training arm, or sequence-level
request/plan/marker/terminal was reached. Its immutable precursors completed
before preflight. The passed config-bundle receipt has file/payload SHA-256
`82214495b1e75e44155303a28e456d6269942fde926959ce846d5c03fdb6aa89` /
`6f4a16b524701dac91cf03a1ac802e590766f1a4ab0881023a400a339809dd3c`
and semantic-projection SHA-256
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
The canonical cache root is
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-11-wave7-r6`, and its
passed preparation receipt has file/payload SHA-256
`52df7e0b3bcca23fcfd5bde69f59f7a97030bf455d704a2009e19a6d22a74c3b` /
`bd9a8e4b937f000547d7f7de676bd642773c03edbedd7c5b0fdfee3c0bf9f718`.

| Cache split | Fingerprint | Manifest SHA-256 | Payload |
|---|---|---|---|
| train | `8307cf2dd9cac344b8a50bb783a831d683c3686ae73678425d6245e9df9eae87` | `775c0c87f6c6e63b11f17c4c737c7649cadb2e0b20494b02d4694274d2ad3567` | 32 micro-steps in one chunk, SHA-256 `59a38ddb036ae644e15bfb25db022dc7b28ca82216679262bf6af44824ba8956` |
| eval | `881fe84188657eeb705dd1bf2b754f11b9f69d61eb62a83954fc867a9d1744f8` | `dce7449e1267a9217166b6297249856d94f0def88fd9cb7fd7c3422f09a15634` | 8 micro-steps in one chunk, SHA-256 `047135041298ed869a078d78f22a3cf54d949963950ffe3dedc54c120e910ca3` |

Both splits used the production `fork_process_pool` / 16-worker payload path;
production payload loading passed and the cache tree remained stable. This is
bound by two repeated seven-file snapshots with identical tree SHA-256
`943fdbc325e143dbb8748c97f905f95fecd11836d6f6039b728df8b816a3de85`.
It is completed cache construction/admission evidence only and makes no model
or training claim. The passed runtime-admission receipt has file/payload SHA-256
`51822d203651799cf44279c85b2de64d50a2c2a5ba2ad3e8513ab1f2a22f0469` /
`3b06e0ce6eefa31fb414b6d8c9eb50c8a2bdbd2f0ca448da71fbdbdb929fd0d2`.

| Immutable r6 preflight artifact | File SHA-256 | Internal canonical payload SHA-256 |
|---|---|---|
| plan | `2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` | `9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f` |
| attempt-start marker | `f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` | `967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e` |
| failed terminal receipt | `eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` | `4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab` |

Publishing the attempt-start marker consumed r6. The immutable failed terminal
contains `mismatches=["KeyboardInterrupt"]`, `launch_count=0`, no rank-receipt
bindings, and no comparison. The leaf directory independently contains exactly
eight passed `launch-a` receipts, ranks 0-7, all reporting workload aggregate
SHA-256
`41f57c8c5ff14d25b1c7c20711f53becec5853fadcfe68dd0c116ec8fd1d9422`;
there is no `launch-b` receipt. This leaf evidence is narrower than a terminal
pass. The terminal cannot be mutated to absorb it, and the absent second launch
means the two-launch comparison never reached a fixed point. R6 therefore
supplies only partial plumbing evidence, not a deterministic-preflight pass,
model behavior, training evidence, or exact-resume evidence; it does not widen
the completed cache-precursor claim above.

The post-failure repairs supersede the failed implementation for any future
authorization and are content-bound as follows:

| Surface | SHA-256 |
|---|---|
| `scripts/probes/coordexp_swift/wave7_determinism_preflight.py` | `52d6e2a3b709ede73f5794eefdbfd6c57997714af29f5329e936b8953492346c` |
| `tests/training/test_wave7_determinism_preflight.py` | `8af9a58d1f425c13bd5e25ffeaca3ab60865b9dcf650d0469abb846be48279a8` |
| `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py` | `cadcbf5ec413b4eea7314684181b4a3ef514144212f9814e3f58d2b999ddfaea` |
| `tests/training/test_wave7_exact_resume_sequence.py` | `ae0f1615410cce187722e1473fed93908033e94b1b6ed6dc98caf4c6df41c236` |

An independent read-only audit returned PASS for that repair set. The repair
hashes and audit are future-authorization engineering evidence only; they do
not rewrite or retry r6, establish a Wave 7 pass, admit Wave 8, or authorize an
automatic r7. Wave 3-5 retain their no-promotion disposition. Wave 6 matched
training remains pending and `source_order_next_fit` remains the production
default.

## Wave 7 r7 one-shot successor authorization — 2026-08-12

The user explicitly authorized exactly one fresh Wave 7 `r7` one-shot
successor. It is not a retry or repair of immutable r6. Its absent-only sequence
root is `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-r7`, its
fresh private-cache root is
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-r7`, and its
private runtime, preflight, amendment, request, plan, marker, and terminal
namespaces are respectively `runtime/`, `determinism-preflight/`,
`amendment-v4.json`, `request-v6.json`, `sequence-plan-v6.json`,
`sequence-marker.json`, and `sequence-receipt.json` below the sequence root.
No fallback to an earlier root is permitted.

The authorized schemas are:

| Artifact | Schema |
|---|---|
| amendment | `coordexp-swift-wave7-r7-amendment-v4` |
| request | `coordexp-swift-wave7-exact-resume-sequence-request-v6` |
| plan | `coordexp-swift-wave7-exact-resume-sequence-plan-v6` |
| marker | `coordexp-swift-wave7-exact-resume-sequence-marker-v6` |
| terminal receipt | `coordexp-swift-wave7-exact-resume-sequence-receipt-v6` |

Request and plan MUST copy these immutable bindings without projection or
reinterpretation:

| Binding | Artifact path | File SHA-256 | Internal canonical payload SHA-256 |
|---|---|---|---|
| `legacy_r4_failure` | `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-10-r4/exact-resume-comparison-receipt.json` | `1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b` | `13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a` |
| `predecessor_sequence_failure` | `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r5/sequence-receipt.json` | `d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1` | `f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656` |
| `predecessor_preflight_failure.plan` | `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/determinism-preflight/plan.json` | `2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` | `9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f` |
| `predecessor_preflight_failure.attempt_marker` | `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/determinism-preflight/attempt-start-marker.json` | `f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` | `967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e` |
| `predecessor_preflight_failure.terminal_receipt` | `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-11-r6/determinism-preflight/terminal-receipt.json` | `eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` | `4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab` |

Every r7 phase is at most once and bounded to `<=600s`; the full sequence is
bounded to `<=2400s wall` and `<=14400 GPU-device-seconds`. Admission uses the
same shared eight-GPU envelope: exactly 81920 MiB total, at most 49152 MiB
pre-existing allocation, and at least 32768 MiB headroom on each device.
Shared-load timing, utilization, memory, and throughput remain non-promotional.
Any gate failure publishes a terminal and stops all later r7 phases; no retry,
root switch, or automatic Wave 7 `r8` successor is authorized.

The user also explicitly directed continuation Wave 7 -> Wave 8 -> Wave 9 on
2026-08-12, and task 10.3 already authorizes final cache/convergence work after
the gates. These later waves do not require a new user authorization merely to
proceed. Wave 8 remains conditional on the accepted r7 Wave 7 gate and its own
frozen packet and budget. Wave 9 remains conditional on the accepted Wave 8
gate and its own sealed packet, frozen identities, and budget. Neither may
bypass its gate or reuse the r7 packet or budget.

## Wave 7 r7 immutable consumed cache-admission failure — 2026-08-12

The r7 config-bundle precursor passed and is immutable:

| Artifact | File SHA-256 | Internal payload SHA-256 |
|---|---|---|
| `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-r7/config-bundle-receipt.json` | `f07e4bc9f4e272e6ed831c714b55f8c1c7a6ee721a86c691e00e5cd3eee34ac6` | `a72c97856ce2689077c91a19652aca3e6fd6b914c8dbce15bfac9008da46ebdb` |

Its semantic-projection SHA-256 is
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
The uninterrupted, interrupted-parent, and resume-child config SHA-256 values
are respectively
`88ccc01f69deab5242405593032c31ff8374c455f08247da20322e3709be2fc4`,
`f9441f5a9ccb27635289a49739b4449c486770b970736549ea9b4dfee4fdf997`,
and
`8bec80b9a5eaefcd9212968b3f9dd8d38eb0601308791a686a2ff622d9dcb9f8`.

The initial production preparation invocation stopped before receipt
publication because its authorized private-cache parent did not exist. After
creating exactly
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-r7`, the
production command was invoked without the strict determinism environment. It
atomically published only `preparation-receipt.json` with file SHA-256
`9077a0458bece11b630ab90c68a9811dfca2941171b0575717b3dba670ae817d`
and internal receipt SHA-256
`e7d7d5c4f2d6cf35e823c80faed4c9391f0368449f6be27560db9f628f8bcc2b`.
The receipt records schema `coordexp-swift-pack-cache-preparation-receipt-v1`,
`terminal_status="failed"`,
`failure.error_type="RuntimeContractError"`,
`failure.error_code="runtime.determinism_environment_conflict"`, and
`result=null`. There are no cache payload files below the private root.

This immutable failed admission consumed r7 under the at-most-once rule. No
runtime directory, deterministic-preflight directory, `request-v6.json`,
`sequence-plan-v6.json`, sequence marker, sequence terminal, model load, CUDA,
training arm, comparison, or post-run gate exists. Consequently the attempt
stops with no retry, root switch, or automatic Wave 7 `r8` successor and makes
no cache, model, training, exact-resume, performance, or Wave 7 pass claim. The
user-authorized Wave 7 -> Wave 8 -> Wave 9 order remains blocked at Wave 7;
later waves keep their frozen-packet and budget gates and cannot proceed through
this failed chain.

The post-failure r7 producer/controller implementation is frozen only as
future engineering evidence. It cannot reopen r7 or satisfy any execution gate:

| Surface | SHA-256 | CPU receipt |
|---|---|---|
| `scripts/probes/coordexp_swift/wave7_exact_resume_config_bundle.py` | `6b41c4a9a56dd77fea9715dcb03e9fe58a3f905300a09459f966eb03da846c54` | config/request lane: `80 passed` |
| `tests/training/test_wave7_exact_resume_config_bundle.py` | `b805cf4c701c47e060e29afbc9af034c47e1c98b34b6869ca7ca25dd653c8752` | same lane |
| `scripts/probes/coordexp_swift/wave7_exact_resume_request.py` | `e0c32fe68f001adcf0c3c34f1add666e6e5bd919de0c4d9e0d695d54fe5f6b2b` | config/request lane: `80 passed` |
| `tests/training/test_wave7_exact_resume_request.py` | `32bd619244d79442903aed4014719afae66daa1b5a934c9b1cb3ef26e7373f0f` | same lane |
| `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py` | `3896d643c12c110ad8ed1557bb189720fa36f8e33527bed2357b8f057fe4cddc` | owned: `169 passed` |
| `tests/training/test_wave7_exact_resume_sequence.py` | `92170b4cd8d2895363733325dbadbf1ed8d36dab8e4ff4a6796382176eb8babd` | same lane |

The request and sequence bind only the bounded r7 authorization section SHA-256
`6b94dafab01c0845e32a807a1d2b8dc5d31e677d5ae81e448a3110a7dd05a491`
while rehashing the complete live measurement plan at SHA-256
`6b088bc3ba7befcca26963b08ef493ce67a14dbfa190158b4ec270a410332730`.
This keeps the authorization immutable while permitting the additive consumed-
failure record. The sequence lane also passed `86` adjacent Wave 7 post-run and
Wave 8 matrix tests. Ruff, format, compile, and diff checks were clean. No
runtime artifact was authored by these post-failure verification lanes.

## Wave 4 task 5.7 no-promotion closure — 2026-08-11

Wave 4 is closed as
`no_performance_promotion/current_default_retained`. The matched
startup/resource experiment was dropped, no performance marker was consumed,
and no RSS, I/O, startup, timing, resource, or efficiency result is inferred
from the omitted experiment. Production rank-selective eval hydration and its
tests-only full-hydration oracle remain the current correctness-tested owners.

The abandoned runnable controller/test cluster was untracked and was removed
only after its exact pre-delete SHA-256 identities matched the cleanup packet:

| Removed surface | Pre-delete SHA-256 |
|---|---|
| `scripts/probes/coordexp_swift/wave4_selective_eval_hydration.py` | `55a7eb114195a70172752b907d848859613dd674d47d104bd4310a2701fccad1` |
| `scripts/probes/coordexp_swift/wave4_eval_full_model_reference.py` | `e888cd02e9d4e18d06894d388c0a351f978b0b7e353df627bb9f647e6cd2ffe8` |
| `scripts/probes/coordexp_swift/wave4_eval_startup_join.py` | `fd42b74130f241221e727f7939a6f350d74d7ef9b4698e5667cb479c7c4214fd` |
| `tests/training/test_wave4_selective_eval_probe.py` | `fa6f1148a62c77476f64133acd186f8db43664f3edfd6255b397a89ccf9bf922` |
| `tests/training/test_wave4_eval_full_model_reference.py` | `d9b7f5a0cc113b700b44d886c7ef5087345d9e7947c7a8494feeb0bfea7976b7` |
| `tests/training/test_wave4_eval_startup_join.py` | `4aad6c6b93e5a18b55d98596e20b84ea9d16eb796741a4fc2f855c466a9e59a3` |

Before deletion, the outside-cluster Python reference scan for all three module
names plus their promotion/join symbols was empty. After deletion, exact-path
existence, Wave 4 basename, and controller-symbol residue scans were empty.
The intent audit then found nine ignored bytecode copies of the deleted
controllers/tests; those exact `.pyc` files were removed, and the final source,
bytecode, symbol, and `find_spec` checks were empty.
`load_rank_eval_micro_steps_from_cache` remains owned by
`src/training/pack_cache.py`, called by `src/training/pipeline.py`, and covered
by the retained pack-cache, eval, assembly, and phase-convergence tests. Wave 5
historical/controller surfaces, immutable output receipts, private cache roots,
production source, and all other tests were outside this cleanup and untouched.

Executed closure gates:

- focused CPU correctness suite with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`:
  `200 passed` across `tests/training/test_pack_cache.py`,
  `tests/eval/test_forward_eval.py`,
  `tests/training/test_pipeline_assembly.py`, and
  `tests/training/test_pipeline_phase_convergence.py`;
- the initial unbounded shared-host lane reached `198 passed, 2 failed` because
  the two spawned Gloo cases timed out during TCPStore rendezvous before a
  CoordExp phase body; the same two cases passed `2/2` with bounded thread
  pools before the complete `200/200` rerun. A separate direct-`ms` Python
  isolation with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and cacheprovider disabled
  passed the same two cases `2/2` in 20.06 seconds, classifying the prior
  combined failures as transient invocation/shared-host isolation unrelated to
  the six-file deletion;
- `openspec validate harden-optimize-coordexp-swift-training-infrastructure --strict`:
  valid;
- targeted Ruff over the preserved pack-cache/eval/pipeline owners and four
  focused test files: clean; targeted `python -m py_compile` over the same
  surfaces: clean;
- whitespace/diff and final residue checks: clean.

The bounded closure review accepts the recorded successful full-suite receipt
plus the independently passing isolated Gloo cases; it does not require repeated
shared-host reruns of the same non-production cleanup. Only the six
content-bound untracked cleanup targets and the owning OpenSpec records changed,
with no output, cache, production-source, GPU/model, staging, or commit action.
The separate intent/contract reading is `PASS`: the runnable promotion path is
absent, current correctness owners remain, the current default is unchanged,
and the closure makes no performance claim.

## Wave 7–9 CPU-only prelaunch owners — 2026-08-11

Three fail-closed owners were completed after the immutable r6 failure. They
reduce post-launch ambiguity but do not change any Wave 7, Wave 8, or Wave 9
task status and do not authorize an automatic r7.

| Surface | SHA-256 | Focused receipt |
|---|---|---|
| `scripts/probes/coordexp_swift/wave7_exact_resume_postrun.py` | `72f34ae9906adc9915a0f7ee355228c685663c6324e7f25d8e6d3a24d4f2eb04` | `73 passed` |
| `tests/training/test_wave7_exact_resume_postrun.py` | `4b423104b7e61ad14c38b3e55921f30b67c835c243fade521d08f5bb3b8db4c4` | v6 fixture rotation; post-run + Wave 8: `86 passed` |
| `scripts/probes/coordexp_swift/wave8_compatibility_matrix.py` | `12e5381a447f75376e2ef946664d0eb24b047d021faf7fbd2d0979ec6d1ad893` | `13 passed` |
| `tests/training/test_wave8_compatibility_matrix.py` | `ded6ed8a690e196efcc59c6a6275e0fc0f5956e1b73b71ce7f3bd6c82fa00d65` | same focused file |
| `scripts/probes/coordexp_swift/wave9_transition_packet.py` | `032c05facf6391b70162ce649529c0fb082fd2caa0748a6463c28bb10c229b30` | `30 passed` |
| `tests/training/test_wave9_transition_packet.py` | `850bd219048660e5d75c9c685f088179f87753015ac5bc784e683e1408ff11f8` | same focused file |

The Wave 7 post-run owner authenticates the successful sequence/comparison,
re-admits the final inference payload before and after a genuinely fresh
`src.infer` subprocess, binds the derived config and exact artifact set, and
requires bounded process/GPU cleanup. It cannot complete task 8.2 without a
future successful exact-resume run and real fresh-process inference.

The Wave 8 matrix is a CPU-only plan/aggregate/validate owner over the five
required cells: cache admission, packed forward/backward plus all-layer FA2,
protected loss, exact resume, and pinned native runtime. A current passed Wave
7 sequence, comparison, and post-run receipt are mandatory; the failed r5/r6
artifacts cannot aggregate to `passed`. No Wave 8 matrix artifact exists, so
tasks 9.3 and 9.5 remain open.

The Wave 9 transition packet owner is a pure validator with `draft` and
`sealed` lifecycles. Drafts are never executable. A sealed projection requires
green Wave 7 and Wave 8 bindings, a selected production config, frozen
source/runtime/cache/model identities, retained `source_order_next_fit` and
`synchronous` defaults, bounded multi-worker cache preparation, explicit
CPU/I/O/time/storage/GPU budgets, immutable rollback, absent targets, and
one-shot no-retry launch semantics. No packet was sealed and task 10.2 remains
open.

All three focused files were rerun concurrently in the default `ms` Python
environment with plugin autoload and pytest cache disabled and exited zero.
Their implementation lanes also reported clean Ruff, format, compile, and
whitespace checks. No model, GPU, cache, output artifact, task checkbox, Git
stage, commit, or push action was performed.

## Wave 7–9 shortest semantic path result — 2026-08-12

The authorized `2026-08-12-core-4` execution completed the shortest
decision-bearing path. It does not reopen Waves 3–5 performance promotion,
does not resolve Wave 6, and makes shared-load timing and resource observations
descriptive only.

Wave 7 exact resume passed on eight ranks. The uninterrupted, interrupted
parent, and resumed child used the same config semantics, cache payload, model
weights, world size, and strict runtime policy. Through the interruption point,
next-pack cursors, accumulation and optimizer positions, Python/NumPy/Torch CPU
and CUDA RNG, optimizer and scheduler state, trainable state, learning rate,
losses, parameter updates, checkpoint selection, and authoritative integer
metric statistics matched. The final step-5 state remained inside the frozen
BF16-derived tolerance, and the final comparison reported no mismatches.

Checkpoint publication produced four committed checkpoints totaling
8,051,779,001 bytes: 7,729,725,929 bytes of exact training state and
322,053,072 bytes of inference payload. Total publication time was
38.69695 seconds. Both step-3 and step-5 checkpoints remain present; no
automatic pruning occurred. A genuinely fresh `src.infer` process loaded the
step-5 base model, DoRA adapter, and special-token embedding delta and generated
and scored one row successfully. The stricter post-run attester itself did not
pass because its GPU-child observation/output-identity projection rejected the
otherwise successful consumer run; therefore this record claims real
loadability and one-row generation, not a passed post-run attestation or Wave 8
matrix receipt.

The Wave 8 compatibility reruns reached the following bounded result:

- the fresh core-4 cache preparation completed model-free and was admitted;
- fresh packed execution reproduced the already documented immutable Wave 2
  boundary: 138 supervised logits, total/per-term loss, denominators, semantic
  atoms, packed-repeat gradients, boundary isolation, and all 28 Qwen text-layer
  deterministic-varlen FA2 captures passed, while three packed-versus-streamed
  BF16 gradient tensors exceeded the cross-shape diagnostic band. The native
  terminal remains `failed`; no threshold was changed and no retry was run.
  Under the existing post-result disposition, that cross-shape comparison is
  diagnostic-only rather than a release gate, so only the explicitly accepted
  forward/loss/FA2 scope is carried into task 9.3;
- the fresh protected-loss probe passed all raw-logit, zero-weight,
  nonzero-control, total-loss, and 589-gradient-tensor comparisons;
- the exact-resume compatibility cell is the passed core-4 Wave 7 comparison
  above.

The same core-4 execution satisfies the production-shaped eight-rank path: it
ran preflight, model load, five optimizer steps, step-3 sharded evaluation,
inference checkpoint publication, controlled interruption, same-world-size
continuation, and final comparison. Cache preparation took 12.996096 seconds,
loaded no model, did not initialize CUDA, and peaked at 1,282,338,816 bytes RSS.
Uninterrupted steady steps were 9.64–10.40 seconds with zero recorded input wait
and roughly 10.55–10.81 GiB peak allocated GPU memory per rank; resumed steps
were 9.46–10.14 seconds with the same integer metrics and accepted loss/state
tolerances. These measurements are descriptive under shared GPU use and do not
promote a performance claim.

Decision-bearing artifacts are retained under:

- `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-4/`;
- `outputs/probes/coordexp_swift/wave3_zero_weight/2026-08-12-wave8-core4-v4/`;
- `outputs/probes/coordexp_swift/wave8_compatibility/2026-08-12-core-4/packed-parity-r3/`.

Tasks 8.2, 8.7, 9.3, and 10.5 are complete on this bounded evidence. The formal
Wave 7/Wave 8 gate ceremonies, matrix aggregation, transition packet,
production-cache materialization, final reference comparison, broad cleanup,
ledger, documentation, audits, and archive workflow remain open and are not
implicitly satisfied by this result.
