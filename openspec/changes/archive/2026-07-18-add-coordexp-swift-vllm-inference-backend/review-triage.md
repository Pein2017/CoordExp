# Review Triage

## Scope

Round 1 reviewed the complete initial OpenSpec artifact set in four independent
read-only lanes: contract/spec, vLLM/upstream, module architecture, and
roadmap/acceptance. The reviewed worktree was
`/data/CoordExp/.worktrees/CoordExp-swift` on branch `coordexp-swift`.

## Accepted Findings

| Severity | Finding | Decision | Resolution |
| --- | --- | --- | --- |
| P1 | Backend trace corruption conflicted with parser-level salvage and empty scored rows. | fix | The scoring and pipeline deltas now make backend likelihood corruption terminal while retaining malformed-object salvage and row-local prediction exclusion. |
| P1 | vLLM 0.14.1 was called qualified before any executed receipt existed. | probe | 0.14.1 is now only the initial candidate. Qualification requires a matching passed receipt binding probe, model, fixture, dependency, installed-source, process, prompt, likelihood, replay, CUDA, and cleanup identities. |
| P1 | Dynamic-HF versus materialized-HF parity lacked numeric thresholds. | fix | Selected embedding rows are exact after target-dtype casting; FP32 full logits use `rtol=1e-4`, `atol=5e-3`; selected-token logits use `rtol=1e-4`, `atol=2e-3`; tied storage and generated ids remain exact. |
| P1 | The eight-GPU gate could activate fewer than eight ranks. | fix | Acceptance now requires eight visible tokens, eight active nonempty ranks, at least eight decode blocks, eight session receipts, strict merge, and evaluator consumption. |
| P1 | Stable failure-only requirement names were reused for executable behavior. | fix | The old reservation/HF-only requirements are removed and replaced by explicitly named executable offline-vLLM and HF/vLLM shard requirements. |
| P1 | Process topology ambiguously mixed offline API ownership and engine child processes. | fix | The worker invokes the offline API; an explicit qualified engine process mode owns any children. Uniprocess mode is preferred when its real probe passes. |
| P1 | Semantic image handoff lacked byte identity and vLLM could not satisfy the HF grid artifact wording honestly. | fix | Requests bind media SHA-256 and dimensions. Backends rehash before projection. HF records executed grid; vLLM records actual returned ids, placeholder ranges/count, processor/no-resize policy, and exact executed media identity. |
| P1 | Base-only direct loading had no immutable controller/worker handoff. | fix | Every vLLM run now resolves an execution-model receipt; base-only receipts hash all required model/config/tokenizer/processor assets and are revalidated before engine construction. |
| P1 | Prompt text and prompt IDs had no single execution authority. | fix | Executable `chat_text` is authoritative. Requests separately carry unexpanded input ids and expected expanded executed ids. vLLM parity must use `RequestOutput.prompt_token_ids`. |
| P1 | HF-only attention and runtime-patch fields would become accepted vLLM no-ops. | fix | Shared model config is backend-neutral; HF-only fields move to a strict `backend.hf` block and vLLM uses a mutually exclusive `backend.vllm` block. |
| P1 | Materialization identity omitted processor/chat-template assets. | fix | Fingerprints and cache-hit validation now bind every processor/template asset copied into or loaded from the execution snapshot. |
| P1 | Raw replay did not distinguish one-placeholder input ids from visual-token-expanded executed ids. | fix | Generation and replay submit unexpanded ids plus media, then require returned expanded ids to match the expected executed prefix exactly. |
| P2 | Controller had no bounded stalled-worker case. | fix | Tasks and runtime contract now require bounded waits, complete owned-process-tree termination, terminal diagnostics, and no canonical publication. |
| P2 | Static source studies had no reproducible handles. | fix | The source study now records exact package versions, installed source paths, symbols/roles, and SHA-256 identities; the executed receipt rebinds them. |
| P2 | Deleting old HF factories lacked a named replacement seam and fresh real parity. | fix | The semantic session opener remains injectable and Wave 1 now requires fresh real single-row and heterogeneous-batch HF artifact parity before deletion. |
| P2 | vLLM cleanup depended on a private upstream surface. | fix | The design names a version-pinned lifecycle adapter and requires explicit engine-core shutdown plus process-tree evidence; garbage collection is not the contract. |

## Rejected Findings

| Claimed severity | Finding | Decision | Reason |
| --- | --- | --- | --- |
| P1 | Add another user approval gate before the Wave 0 executable probe. | wrong | The current user request explicitly authorizes implementation of this plan. Wave 0 intentionally ends with post-probe independent review before Wave 1 begins. |
| P2 | Split raw replay out of vLLM version qualification because raw tracing is disabled by default. | wrong | Raw likelihood is an explicit first-class feature of this change. Per-run activation remains optional, but a backend version is accepted only after both its default policy channel and optional raw channel are proven. |

## Round 1 Status

All accepted P0/P1 contract findings have corresponding artifact revisions.
Strict OpenSpec validation passes after the revisions. The real Wave 0 receipt
now passes for vLLM 0.14.1 and records five preserved diagnostic attempts that
led to the qualified memory, stop-text, and worker-lifecycle behavior. A second
independent review round will inspect the revised artifacts and executed
receipt before Wave 1.

## Round 2 Triage

Four fresh read-only lanes reviewed the revised contracts and canonical Wave 0
receipt. The roadmap lane approved with P2 refinements. The contract,
upstream, and architecture lanes correctly held Wave 0 on qualification gaps.

| Severity | Finding | Decision | Resolution |
| --- | --- | --- | --- |
| P1 | Runtime qualification incorrectly required every composed execution model to equal the base-only probe fingerprint. | fix | Runtime/model-family qualification is now separate from exact per-run execution-model identity. A derivative must bind the qualified source base and a passed execution-model composition-fidelity receipt; dynamic-HF/materialized-HF behavior remains diagnostic rather than an identity claim. |
| P1 | The receipt omitted live tokenizer, processor, chat-template, multimodal, engine-core, and distributed-cleanup source identities. | fix and probe | Snapshot identity now exhaustively hashes every regular model file. The regenerated canonical receipt records all 19 snapshot files and 24 static/runtime-discovered source owners. |
| P1 | The probe did not bind the submitted in-memory image bytes or nonempty placeholder ranges. | probe | Generation and replay now independently reopen one byte payload, hash source and RGB pixels, derive one exact placeholder range from returned prompt ids, and require equal media identities. |
| P1 | One-rank vLLM could bypass the only qualified process-exit cleanup boundary. | fix | Every vLLM run, including one active rank, must use a fresh worker. Parent-observed worker exit, no descendants, and GPU-memory return precede canonical publication. |
| P2 | Old HF path deletion preceded real artifact parity. | fix | Wave 1 retains the old path as a temporary oracle, runs exact single-row and heterogeneous-batch parity, and deletes residue only afterward. |
| P2 | Repeatability had no exact acceptance definition. | fix | Wave 3 now requires two fresh-worker runs with exact token, stop, parser, row, and non-timing artifact equality plus approved likelihood tolerances. |
| P2 | Engine arguments did not distinguish invariants from run-varying values. | fix | The receipt and design now identify invariants; concurrency, model length, memory sizing, and automatic KV sizing require executed value coverage before production use. |
| P2 | Wave 5 could satisfy failure injection with only a subset of listed cases. | fix | The gate now requires an executed terminal receipt for every named failure class. |
| P2 | Design text still described upstream probes as unexecuted. | fix | Open Questions now reflects the completed probe and the pending corrected-receipt review gate. |

## Round 2 Status

The corrected real receipt and strict OpenSpec validation now pass. Wave 0
remains held only until a fresh independent review confirms no unresolved
P0/P1. Wave 1 implementation has not started.

## Round 3 Triage

The third review round produced three approvals and one accepted P1. The P1
observed that 24 named source owners represented only 16 unique files and did
not cover lazily loaded `UniProcExecutor`, `GPUModelRunner`, model loading, and
sampling modules.

The probe now captures a post-generation/replay manifest of every loaded source
module under vLLM, Transformers, PEFT, and qwen-vl-utils, while asserting seven
required execution paths explicitly. The regenerated receipt contains 853
unique source files, including the uniprocess executor, GPU model runner,
default model loader, Qwen3-VL owner, multimodal processor, sampler, and top-k/
top-p sampler. Attempt 7 is preserved as superseded evidence.

Two independent final revalidation lanes recomputed the 853-file manifest,
verified every required owner and snapshot identity, matched the current probe
hash, and approved with no unresolved P0/P1. Wave 0 is converged. The
branch-durability P2 is handled by the scoped Wave 0 commit before Wave 1.

## Wave 1 Implementation Triage

Wave 1 moved the existing HF implementation behind the semantic backend-session
contract and then used the old path only as an artifact-parity oracle. Four
review/fix rounds covered runtime correctness and executable evidence. The
protected temporary oracle files were excluded from approval because their
deletion requires a separate manual gate.

| Severity | Finding | Decision | Resolution |
| --- | --- | --- | --- |
| P1 | Raw-model likelihood could be partially present or malformed across writer and merge boundaries. | fix | Enabled runs now require one finite, non-positive raw value for every non-pad generated token; disabled runs forbid it. Strict merge rejects missing, positive, non-finite, or status-inconsistent evidence. |
| P1 | Executed media identity could describe source bytes instead of transformed RGB pixels. | fix and probe | HF and shared Qwen image materialization now hash canonical transformed RGB8 pixels. A real asymmetric hflip receipt proves the executed hash equals mirrored pixels and differs from the original. |
| P1 | The initial parity policy allowed unknown descendants under broad additive roots. | fix | The executable verifier now enumerates exact object keys, scalar leaves, repeated rows, and fixed sequence lengths. The final mutation matrix rejected all 332 injected schema drifts across 19 roots. |
| P1 | Empty parity rows did not prove policy-owned prediction scoring or evaluator compatibility. | probe | A real raw-enabled DoRA-plus-delta run produced 14 scored predictions. Every stored score equals the policy channel, all 14 differ from raw counterfactual scores, and the unchanged evaluator consumed the artifacts successfully. |
| P1 | The raw-likelihood receipt named adapter and embedding-delta paths without binding their bytes. | fix and probe | The rerun receipt hashes adapter config/tensor and delta metadata/tensor before and after execution, records sizes plus component/combined fingerprints, and chains those identities into the policy-score verifier. Altered payload hashes fail closed. |
| P1 | Equally incomplete shards could merge because identity equality did not require semantic evidence to be nonempty. | fix | Shard ingestion now requires nonempty model, processor, tokenizer, session, likelihood, frontend, generation, template, dataset, and scalar identities; nullable composition fields must be explicit. HF may use a null execution identity, while vLLM requires a nonempty execution-model object. |
| P2 | The raw-likelihood receipt recorded authored config identity but not one reconstructable post-override configuration. | fix | The receipt now embeds the complete effective config, a recomputable fingerprint, and argv. The real rerun used a nondefault `max_new_tokens: 160` and reproduced the fingerprint exactly. |

## Wave 1 Status

The final evidence lane and strict-merge lane both returned GO with no unresolved
P0/P1. The complete targeted inference, evaluator, and Qwen slice passed with
286 tests; strict OpenSpec validation, receipt/catalog hash checks, executable
receipt replay, and `git diff --check` passed. The temporary HF oracle and its
legacy test were subsequently removed, residue checks passed, and Wave 2 began
only after that protected deletion gate was completed.

## Wave 2 Implementation Triage

Two independent read-only lanes reviewed execution-model materialization,
dynamic-HF ownership, DoRA merge ownership, selected-token folding, immutable
snapshot validation, path-independent identity, and the real step-4887
composition-fidelity receipt. Both returned GO with no P0, P1, or P2 finding.

The reviews confirmed that HF remains a first-class dynamic backend: it loads
the live DoRA adapter and special-token embedding delta without passing through
materialization. Only vLLM resolves the immutable execution snapshot. PEFT owns
the safe DoRA merge, while CoordExp owns the single FP32 selected-token fold and
the tied embedding/lm-head checks.

The current materialization and composition-fidelity receipts pass all exact
identity checks. The deliberately separate BF16 behavior diagnostics retain
their observed numeric and greedy differences without being misrepresented as
exact state failures. Targeted adapter, selected-token, execution-model,
composition, and config coverage passed with 153 tests. Strict OpenSpec
validation, legacy parity-symbol residue checks, and `git diff --check` passed.

## Wave 2 Status

Wave 2 is converged. No unresolved P0/P1 remains, and task 3.6 is closed. Wave
3 may proceed while preserving both supported execution paths: dynamic HF and
materialized vLLM.

## Waves 3-6 Pre-Final Hardening

Implementation and real-run evidence were reviewed during the vLLM tracer,
dual-likelihood, distributed, and acceptance waves. The following findings
were accepted and fixed before the final convergence review.

| Severity | Finding | Decision | Resolution |
| --- | --- | --- | --- |
| P1 | A completed worker could leave a pending sibling alive after another rank failed. | fix | Controller cleanup now terminates the complete owned worker set and removes private runtime caches before terminal publication. |
| P1 | vLLM artifacts copied the frontend's expected image grid as if the backend had observed it. | fix | vLLM now records returned prompt ids, placeholder ranges, media identity, and a null observed grid; HF continues to record its actual tensor grid. |
| P1 | Runtime qualification covered installed package sources but not every CoordExp application owner. | fix and receipt | A separate exact 23-path application-source receipt is mandatory and is rehashed before vLLM session creation. |
| P1 | Global GPU-memory equality was treated as a cleanup gate on shared GPUs. | narrow | Owned worker/process exit and private-cache removal are publication gates. Parent-observed global GPU memory remains diagnostic because unrelated jobs may change it. |
| P1 | Benchmark eligibility could be claimed by tiny smokes. | fix | Canonical benchmark eligibility now requires a non-smoke scope and at least 200 rows; one/two-rank fixtures remain explicitly non-benchmark. |
| P1 | Likelihood failures lacked request, token, and channel-local diagnostics. | fix | Alignment failures now identify request id, token position/id, semantic channel, and observed value while remaining terminal. |
| P2 | Concurrency qualification was tied to one derivative checkpoint instead of the qualified source family. | fix | Concurrency receipts bind the qualified base family, dtype, and `max_num_seqs`; each derivative still requires its own execution-model identity. |
| P2 | Failure evidence did not table every required class with publication and cleanup outcomes. | fix and execute | The v2 failure matrix binds all eight classes to executed tests and requires terminal diagnostics, complete cleanup, and no canonical publication. |

Fresh current-code runs then passed for one active rank, two active ranks, and
eight active ranks. The eight-rank vLLM val200 run used 50 nonempty decode
blocks, completed all 200 rows with zero parser/drop/truncation/score failures,
and was consumed by the unchanged evaluator. The matched eight-rank dynamic-HF
run preserved the same dataset, prompt, generation, and scoring contracts.
Absolute mAP and mRecall deltas were `0.0011065226287719776` and
`0.0003795564820910924`, both below the approved `0.005` gate. These results
are bound in the distributed and val200 receipts; final review convergence is
still pending.

## Exact-Source Revalidation

Pre-final hardening moved runtime qualification authority into
`src/inference/qualification_receipts/`, replaced checkout-absolute source
identity with repo- or package-relative identity, made strict FP32 composition
proof durable across a clean materialization cache, and hardened controller
cleanup for interrupts and descendants surviving an exited worker leader.

After the final source edit, all four real vLLM 0.14.1 qualification probes
(BF16/FP32, concurrency four/raw replay one) were regenerated successfully.
Fresh current-source one-rank and two-rank FP32 dual-likelihood runs completed
through strict merge and the unchanged evaluator, with exact two-row order,
zero parser/score/drop/truncation failures, available raw likelihood, and all
workers exited. The complete shared HF/vLLM, execution-model, evaluator, DoRA,
and selected-token suite passed with 431 tests. Strict OpenSpec validation,
delta inspection, config/Markdown hygiene, failure-matrix verification, and
`git diff --check` also passed before final independent review.

The acceptance/artifact lane then found one archive blocker: the executable
failure-matrix test still loaded its receipt from the active change directory.
The receipt now lives under `tests/inference/receipts/`, and both the test and
task record use that durable path. The lane's missing-stable-spec observation
is resolved by the required delta-spec sync before archival.

## Final Convergence Review

Four independent final lanes reviewed backend semantics, distributed cleanup,
receipt integrity, and OpenSpec/archive readiness. Their accepted P1 findings
were fixed before a focused two-lane re-review:

- worker cleanup now attempts every owned rank and aggregates failures instead
  of aborting after the first termination error;
- controller launch, wait, and interruption failures publish terminal-only
  diagnostics even when cleanup itself fails;
- startup and CUDA-OOM injections cover the vLLM controller-owned shard path,
  while the durable failure matrix states honestly that fresh-worker cleanup is
  composite evidence from the process-tree cases;
- acceptance receipts name every hashed artifact relative to its run root, and
  all recorded paths and SHA-256 values were revalidated;
- the failure-matrix receipt records the exact 94-test command that produced its
  result; and
- backend sessions remain independent of evaluator/artifact orchestration so a
  later GRPO or post-training rollout owner can reuse the same HF/vLLM execution
  boundary without introducing another engine wrapper in this change.

Both focused re-review lanes returned `APPROVE` with no unresolved P0/P1. The
complete inference/evaluator/adapter/special-token suite passed with 455 tests.
Fresh current-source one-rank and two-rank FP32 vLLM dual-likelihood runs also
completed through strict merge and the unchanged evaluator:

- one rank: `outputs/coordexp_swift/infer/smoke/qwen3-vl-2b-step4887-vllm-fp32-parity-smoke-20260718T152112Z`;
- two ranks: `outputs/coordexp_swift/infer/smoke/qwen3-vl-2b-step4887-vllm-fp32-parity-smoke-20260718T151658Z`.

The accepted val200 evidence remains bounded to its recorded source identities:
dynamic HF mAP/mRecall are `0.4253271187757268` and `0.5087078247640996`;
vLLM FP32 mAP/mRecall are `0.4264336414044988` and
`0.5090873812461907`. Absolute deltas are `0.0011065226287719776` and
`0.0003795564820910924`, below the `0.005` gate. The accepted delta specs were
synced into stable authority and the change was archived on 2026-07-18.
