# Production optimization acceptance — 2026-09-12

Status: eight-rank training, fresh exact resume, old/new HF consumption and
direct HF evaluation verified. The user has accepted the one-grid BF16
composition contract, but the final checkpoint exceeds it in a later generated
coordinate. Full acceptance requires resolution of that observed larger drift
and a current qualified vLLM consumer witness. See [tasks](tasks.md)
for the current completion boundary and [review](review.md) for the independent
proposal review. No full training or model-quality claim is made.

[Current verification](/data/CoordExp/outputs/infra_base/optimization-20260912/final-current-verification.json)
confirms unchanged training-witness source hashes and 33 valid local Markdown
links. Affected Ruff checks, whitespace checks and strict OpenSpec validation
pass. Tasks 5.4 and the vLLM portion of 5.5 remain open; this change is not
archived or represented as fully accepted.

## Fixed workload and code

- Baseline: `70e576f9606c48d322b6c67df9baac78c22fc3f6`, executed from the
  isolated `/data/CoordExp/.worktrees/codex-production-opt-baseline-20260912`.
- Final training candidate: current `coordexp-infras` worktree. The
  [fresh launch packet](/data/CoordExp/outputs/infra_base/optimization-20260912/fixed-ddp-launch-packet-02.json)
  binds all 125 Python source files, parent/child configs and environment.
- Runtime: eight A100 80GB PCIe; Python 3.12.11, Torch 2.9.1,
  Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1, FA2 2.8.3,
  vLLM 0.14.1. No package upgrades.
- Qwen3-VL 2B natural-adjacent, BF16/FA2, DoRA r16 plus special-token
  embeddings, production loss/optimizer/schedule, EBS48, six accumulation
  micro-steps per rank, four applied updates.
- Frozen COCO source prefixes: 1,024 training and 128 evaluation examples;
  101 train packs, 13 eval packs. Four updates consume 192 global pack
  presentations with the existing repeated-stream semantics. Evaluation uses
  disjoint shards across eight ranks.
- [Launch packet](/data/CoordExp/outputs/infra_base/optimization-20260912/launch-packet.json)
  records actual counts/configs/environment; all raw evidence is under
  `/data/CoordExp/outputs/infra_base/optimization-20260912`.

## Completed checks

| Check | Evidence and result |
| --- | --- |
| Independent plan review | Two fresh Astra reviewers; two launch-plan blockers corrected; strict OpenSpec validation passed. |
| Submission bound | Old source attempts a ninth unfinished task for a limit of eight; corrected caller tests pass. Default real entry uses 16 processes and a 32-task window. |
| Preparation parity | [Receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/preparation-parity.json): every decoded field/tensor/alias in 101 train and 13 eval packs matches. Only process-local PyTorch storage keys differ in pickle bytes. Both cache-hit verification entries pass. |
| Baseline eight-rank lifecycle | [Summary](/data/CoordExp/outputs/infra_base/optimization-20260912/baseline-training-summary.json): four finite applied updates, eval at 2/4, inference and training-state publications at 2/4, final status completed; exit 0. |
| Initial bounded-preparation candidate | [Summary](/data/CoordExp/outputs/infra_base/optimization-20260912/candidate-training-summary.json): same completed lifecycle; [all semantic logging rows exactly equal](/data/CoordExp/outputs/infra_base/optimization-20260912/baseline-candidate-training-comparison.json). This predates the DDP correction. |
| Final strict-DDP eight-rank lifecycle | [Parent summary](/data/CoordExp/outputs/infra_base/optimization-20260912/candidate-fixed-ddp-02-training-summary.json): four finite applied updates, eval/save at 2/4, final publication and exit 0. [Frozen accounting](/data/CoordExp/outputs/infra_base/optimization-20260912/baseline-final-candidate-accounting.json) preserves membership, denominators, weights, LR and counters; floating outcome changes from the new reduction order are recorded separately. |
| Fresh eight-rank exact resume | [Child summary](/data/CoordExp/outputs/infra_base/optimization-20260912/candidate-fixed-ddp-resume-02-training-summary.json): restores step 2, applies 3/4, evaluates and publishes, exit 0. [Exact comparison](/data/CoordExp/outputs/infra_base/optimization-20260912/fixed-ddp-exact-resume-comparison.json): eight rank pairs, 16 admission calls, 18,864 tensor pairs / 481,538,280 elements, zero differences across model, optimizer, scheduler, RNG and cursor state. [Step accounting and bucket evidence](/data/CoordExp/outputs/infra_base/optimization-20260912/fixed-ddp-resume-accounting.json) also match exactly, including gradient norms. |
| Supported inference payload tests | 66 passed; old components without root manifest, independent authentication, tamper rejection, DoRA reload and embedding-delta identity. [Receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/payload/focused-payload-tests.receipt.json). |
| Loss/checkpoint/resume CPU tests | 232 passed before the real-entry reader correction; [log](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training/pytest.log). Correction-specific verification follows separately. |
| Cache/packing tests | 269 passed plus the corrected frozen-byte assertion; [receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/packing/receipt.json). The complete corrected payload test file has four passing cases. |
| Strict-DDP regression | Caller/policy and affected runtime/pipeline checks: 266 passed. Reporting plus actual two-rank Gloo/shared-delta regression: 48 passed, independently replayed by the lead. [Source and RED/GREEN receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training/strict-ddp-candidate.json). |
| Old/initial-new HF consumers | Four fixed requests, BF16, batch 2, max 64 tokens. Old payload baseline/candidate predictions, scores and 132 token-trace rows are byte-identical; initial candidate new payload outputs also match. [Parity](/data/CoordExp/outputs/infra_base/optimization-20260912/hf-inference-parity.json). |
| Final candidate HF consumer | [Receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/final-candidate-hf-consumption.json): final strict-DDP checkpoint, four successful decodes, zero score failures, 132 trace rows, exit 0. This checkpoint differs numerically from the initial candidate and is not claimed to preserve its exact generated tokens. |
| HF direct evaluation | Baseline-old, candidate-old, initial-new and final-new scored artifacts consumed with exit 0. [Final evaluation receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/final-candidate-direct-evaluation.json) authenticates metrics, lineage and COCO conversion artifacts. |

The legacy compatibility specimen is a preserved pre-change producer payload,
not a historical scientific experiment. [Its identity](/data/CoordExp/outputs/infra_base/optimization-20260912/pre-change-payload.identity.json)
records exact copied component bytes. Real HF consumed these independent
components without a root publication manifest. No old training-state
migration was introduced.

All four inference rows are diagnostic: generation/scoring completes, but the
few-step checkpoint's outputs fail the object parser and produce no eligible
detections. The evaluator retains `benchmark_eligible: false`. This verifies
the consumer path and exact output preservation, not detection quality.

## Reproduced corrections and pending gates

1. Initial strict-replay preparation rejected missing environment variables.
   The launch packet now explicitly exports `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   and `FLASH_ATTENTION_DETERMINISTIC=1` before Python; source validation remains.
2. The first 64-example eval prefix produced seven packs. It was increased to
   128 before either training arm; both arms share the final frozen prefix.
3. A frozen payload golden predated naming migration `70e576f96`. Reconstructing
   the old planner exactly recovered its old bytes; only algorithm identity
   and derived plan/fragment hashes differ. [Recovery evidence](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training/golden-identity.json)
   justifies re-pinning the oracle; all nonmetadata values and byte assertions
   remain protected.
4. The fresh step-2 continuation failed before restoration because the reader
   admitted only the parent's latest completed publication. The accepted
   design requires independently validating selected step 2 and current parent
   progress at step 4. The reader correction passes 155 focused tests and the
   original public admission reproducer without changing parent artifacts.
   The subsequent eight-rank continuation completed, but exact comparison
   found a further backward/update divergence: step-3 forward/loss is equal,
   while its pre-clip gradient norm differs by approximately 1.97e-9. Final
   parameters/Adam moments differ; scheduler, options, counters and RNG/cursor
   state match in the diagnosed rank. Native fixed-DDP buckets now resolve the
   accepted fresh-parent/continuation scenario: final state and step accounting
   compare exactly on all eight ranks. No equality tolerance was relaxed.
   The earlier failing evidence remains available:
   [first reader failure](/data/CoordExp/outputs/infra_base/optimization-20260912/logs/candidate-resume.log),
   [reader fix](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training/resume-publication-green.json),
   [state mismatch](/data/CoordExp/outputs/infra_base/optimization-20260912/exact-resume-comparison.json),
   [state diagnosis](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/training/resume-state-diagnosis-v2.json).
5. The first inference subset used absolute image references, which the source
   format correctly rejected. References were made relative to the subset,
   resolving to the same original images; all four rows passed the real loader
   before successful consumers. Input identity retains the failed attempt.
6. vLLM materialization requires an explicit absolute
   `COORDEXP_EXECUTION_MODEL_CACHE_ROOT`; the initial missing-environment failure
   is preserved. Qualification now uses the task's execution-model cache.
   The subsequent composition qualification was correctly rejected: the fixed
   row's dynamic and merged HF executions differ in two coordinate tokens,
   with maximum full/selected logit differences 1.875/0.78125. All 196 merged
   targets and selected-row identities agree. A real same-input layer probe
   reconstructs the published weights exactly on CPU and GPU but still shows
   BF16 operator-order differences; changing the merge device cannot fix that
   observed discrepancy. [Diagnosis](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/vllm-composition/diagnosis-summary.json).
   That historical strict qualification failure remains preserved. The user
   subsequently accepted one-grid coordinate error. The revised bounded
   composition contract has passed one independent numerical-boundary review;
   implementation and fresh qualification use the final strict-DDP checkpoint.
   [User decision and reviewed plan](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/vllm-bounded-composition/contract-decision.json).
   Implementation passes 164 affected CPU tests and 31 lead-replayed bounded
   cases. Fresh final-checkpoint qualification correctly rejects its observed
   free-generation result: positions 5/7 differ by one grid unit, while position
   12 changes `coord_733` to `coord_858` (125 units). Length, non-coordinate
   tokens, EOS and structural weight checks match. [Current rejection](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/vllm-bounded-composition/final-grid1-rejection.json).
   The user has been asked whether the merged model should instead have an
   independent inference contract with explicit generation-drift diagnostics;
   no broader allowance has been applied pending that answer.
7. The first fixed-DDP parent failed in Gloo control-group creation, before
   model loading or any update. DDP flags had not yet been consumed; model-free
   cache identity access also explains the preceding dtype warnings. The
   [failure receipt](/data/CoordExp/outputs/infra_base/optimization-20260912/fixed-ddp-network-failure.json)
   retains the host-interface timeout. The same source/workload parent and
   child passed with `GLOO_SOCKET_IFNAME=lo` for this single-host witness. The
   underlying transient network cause is not established.

## Performance interpretation and roadmap

One descriptive warm-cache training pair took 197.059 seconds for baseline and
165.229 seconds for the initial bounded-preparation candidate, before the DDP
replay correction, from training entry to terminal publication.
These observations are not repeated or noise-qualified; candidate preparation
also overlapped baseline startup. **No speedup claim is accepted.** The
demonstrated optimization is the worker-proportional submission bound with
unchanged semantics. Raw/encoded data remain materialized.

The final fixed-DDP parent took 249.791 seconds and its step-2 continuation
took 155.035 seconds. Those runs verify exact restart correctness; different
executed step counts and unpaired shared-system conditions preclude treating
them as a performance comparison. The initial pair does not measure the final
DDP policy's overhead or speedup.

[Phase/resource observations](/data/CoordExp/outputs/infra_base/optimization-20260912/performance-observations.json)
record GPU allocated high water of 10,759,374,848 bytes and reserved high water
of 22,598,909,952 bytes in both initial arms. Observed CPU RSS high water is
10,581,606,400 / 10,000,969,728 bytes respectively. These are existing logging
observations across ranks, not attribution to a particular data structure or
an isolated preprocessing memory measurement.

| Roadmap area | Observed trigger and disposition |
| --- | --- |
| Lazy cache hydration | Training rank hydration takes 1.73 / 1.89 seconds; retained-pack RSS is not isolated. No demonstrated dominant cost; defer. |
| Loss/logits kernels | Steady-state observations do not isolate logits or CE cost. Preserve the objective and defer until direct attribution identifies a bottleneck. |
| Optimizer acceleration | Optimizer assembly is measured, but optimizer update cost is not separately attributed. No justified optimizer change. |
| Data/template management | No new data mixture or source consumer was requested. Preserve existing owners and semantic cache admission. |
| vLLM throughput | Composition admission fails before engine execution. Resolve the explicit inference contract before throughput work; no concurrency performance claim. |
| DoRA precision | Same-input evidence explains BF16 dynamic/merged rounding differences. It does not establish a useful or compatible precision replacement; defaults remain unchanged. |
| Rollout to learning | No production RL objective or consumer exists within this scope. Defer implementation; retain the proposed exact-token and policy-version boundary. |
| Distributed capacity | This workload fits replicated training on eight A100 80GB cards. No capacity failure justifies FSDP2 or a new topology. |
| Async checkpointing | Candidate saves take 10.58 and 10.02 seconds, approximately 12.5% of its 165.23-second lifecycle. Save stalls are not dominant here; retain synchronous durable publication. |
| General modularity | Concrete faults are contained in the cache, publication reader, and DDP/session owners. No competing owner requires a new registry or runtime framework. |

No new RL objective, FSDP2 topology, DoRA precision default, data mixture, or
asynchronous checkpoint path has been introduced. These dispositions apply to
the measured smoke, not to an unmeasured full-scale workload.
