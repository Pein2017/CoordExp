## Context

The comparison against ms-swift `0673cf75dca7d0b9b608b4a76632fb508ead5076`
identified bounded preparation, explicit execution phases, and resource-aware
validation as useful patterns. CoordExp already owns stricter geometry,
packing, loss, DoRA-plus-embedding, cache, and resume semantics. Upstream code
is a comparator, not a dependency or semantic authority. Baseline infras is
`70e576f9606c48d322b6c67df9baac78c22fc3f6`.

The production report is [the supporting inventory](../../../research/investigations/ms-swift-upstream-comparison-2026-09-12/production-training-report.md).
This change supersedes its suggested runtime matrix where the user narrowed
acceptance to eight GPUs and a few steps. No research-probes work is included.

### User-accepted decisions and execution authority

The user accepted grilling items 1–3 and 6–8, with these explicit refinements:

| Decision | Frozen scope |
| --- | --- |
| Outcome | Optimize the existing production lifecycle; fixed-workload end-to-end time and GPU cost are the performance priorities. |
| Semantics | Retain examples/order, template/geometry/tokens, pack and optimizer-step membership, losses/denominators, effective batch, optimizer/schedule and precision defaults. |
| Compatibility | Supported pre-change checkpoint components must load and infer; old training-state resume is unnecessary. New checkpoints keep strict same-world-size exact resume. |
| Runtime acceptance | Eight GPUs, a few optimizer steps, forced save/eval and fresh consumers; no full training or exhaustive model/topology matrix. |
| Resources | No GPU-hour or run-time budget cap. This does not expand the finite workload into full epochs. |
| Expansion | Plan broader capabilities with concrete triggers; avoid redundant compatibility and speculative frameworks. |
| Performance | Require repeatable, comparable evidence for claimed speedups; accept correctness and maintenance gains separately. |
| Authority | One independent reviewer round after proposal; then implement directly without another approval request. |

## Goals / Non-Goals

**Goals:** bound outstanding preprocessing work; retire actual eight-rank
lifecycle failures; establish supported old-payload inference and new-state
resume; preserve semantic oracles; record resource and timing evidence; leave
concrete, conditional follow-on choices for all surveyed production domains.

**Non-goals:** full training, research objectives or research-probes edits,
multi-node validation, blanket package upgrades, all historical checkpoint
formats, an asynchronous RL framework, new scientific data mixtures, an
unconditional FSDP2/TP implementation, or a new generic registry/runtime layer.

## Decisions

### 1. Bound work in the current preprocessing owner

Change the existing process-pool loop in `src/training/cache_workflow.py` to
keep at most `min(example_count, 2 * worker_count)` submitted unfinished
futures, replenish after completion, and restore results by source index using
the current validation owner. Retain the existing fork initialization,
worker-count/seed behavior and exception propagation. Keep the limit local;
no user configuration or scheduling abstraction is needed.

The current loop submits one future per raw example. Bounding it removes this
unbounded queue; it does not remove the materialized raw and encoded tuples.
Do not claim streaming or whole-process constant memory. A gated executor test
through the preparation caller must fail on the baseline's over-submission,
then pass for the candidate. Out-of-order results and failure publication are
decision-bearing cases. Reuse the current payload/cache tests for semantics.

Alternatives: keeping eager submission leaves the demonstrated resource-bound
problem; a streaming cache rewrite changes ownership and startup semantics
without a measured need. Both are excluded from this first implementation.

`cache_workflow.py` is not itself a registered semantic determinant owner.
Keep the existing determinant registry: scheduling-only edits should preserve
cache compatibility, while edits to actual content producers still invalidate
their caches. Demonstrate encoded/packed identity equality explicitly.

### 2. Preserve module ownership while fixing measured faults

Use the current config resolution, preparation/cache, packed forward/loss,
adapter, session/checkpoint, inference qualification and artifact consumers.
Only add a module when a concrete owner conflict cannot be resolved locally.
The production inventory is not a mandate to refactor every module. A real
failure exposed by the accepted vertical path may be fixed at its shared
owner with a reproducible regression, without reopening user authorization.

### 3. Freeze a production-shaped eight-rank smoke

Add `configs/smoke/eight_gpu_qwen3_vl_2b_coco.yaml`, inheriting production.
Retain Qwen3-VL 2B natural-adjacent, BF16, FA2, packed length 12000, DoRA r16,
embedding deltas, segment-balanced objective, AdamW groups and effective batch
48. Eight ranks therefore consume six micro-steps per rank per update.
Use four planned optimizer updates, evaluation/checkpoint cadence `[2, 4]`,
and per-forward FA2 evidence. Set both uninterrupted parents (baseline and
candidate) to `resume.mode: exact_same_world_size` with `checkpoint_dir: null`
so they publish training state; retain strict CUDA replay. Inspect successful
applied updates, not just planned step numbers. Keep the existing
one-through-four-rank interface.

Before GPU execution, prepare a fixed COCO source prefix (initial proposal:
1024 training and 64 evaluation examples). Inspect actual pack counts and
freeze the prefix and resulting schedule in the launch receipt. Four updates
consume 192 global pack presentations; repeated presentations retain the
existing planner semantics and must be reported. Require at least eight eval
packs to exercise the existing sharded evaluation route. Prefix adjustment
before freezing is allowed; no adjustment between baseline/candidate runs.

The real preparation entry uses 16 materialization workers by default; its
CLI has no materialization-worker selector. Retain that value for paired
preparation measurements, yielding a 32-future candidate window. Record the
actual materialization metadata. Keep `packing.worker_count: 1`: that separate
planner field is part of the cache identity. Materialization worker count is
nonsemantic; the controlled regression can select four through its existing
internal argument. No new CLI or configuration option is needed.

Prepare the cache with the real `src.prepare_train_cache` entry, verify hits
with `--require-all-hit`, then launch `torch.distributed.run` with eight
processes and `-m src.train --config ...`. Save exact commands/configs, code
identity, dependency versions, rank count, pack counts, logs, exit status,
successful optimizer counts, evaluation receipts, peak RSS/GPU memory and
checkpoint payloads under a run-specific artifact root. Launch directly in
the shared GPU environment; react to actual contention failures.

### 4. Separate old inference payloads from new exact resume

Before implementation changes, preserve a checkpoint generated by the
baseline source from the same bounded smoke, with adapter and embedding bytes
hashed. If no supported prior checkpoint is available, explicitly label this
as a pre-change producer fixture, not an old scientific experiment. Obtain
baseline HF outputs from those immutable bytes before candidate consumption.
Do not regenerate those bytes with the candidate writer.

Keep supported base-only, adapter-only, embedding-delta-only and combined
component configuration routes. Combined historical components may lack
`inference_payload_manifest.json`; per-component checks remain mandatory.
If a root manifest exists, reject byte mismatches. DoRA magnitude vectors,
embedding token-ID/base identity, tied delta semantics and supported stored
dtypes remain protected. Inference must not read optimizer/RNG/cursor state.
Use existing tests for the format matrix and a real combined HF inference
reload for the preserved pre-change bytes. Remove compatibility only where
the implementation actually needs to change old training-state handling.

For new exact resume, run an uninterrupted four-update candidate that saves
step 2. Launch a fresh eight-process run from that step-2 checkpoint with the
same four-update schedule and admitted identity, using
`resume.mode: exact_same_world_size` and the parent's step-2 checkpoint path.
Require the parent's published training-state files before continuation.
Compare final trainables,
optimizer, scheduler, RNG and cursor state plus steps 3–4 loss/accounting to
the uninterrupted candidate. A parent configured for only two total steps
would create a terminal checkpoint and is not a valid substitute. Keep all
existing fail-closed admission tests. Do not relax comparisons after seeing a
failure; investigate differences against the existing deterministic contract.

The first real continuation exposed a latest-publication-only restriction in
the reader. The accepted step-2 continuation requires separating two checks:
authenticate the selected nonterminal checkpoint against its own completed
publication event, and authenticate the parent's current progress against its
latest completed publication. Selecting step 2 must neither roll back parent
metadata nor bypass selected-payload, terminal-cursor, or event-identity checks.
This correction implements the originally frozen continuation scenario.

The next real continuation restored optimizer, scheduler, RNG and cursor
identically, but differed at gradient reduction and subsequent updates. Native
DDP rebuilds its buckets after initial execution; a fresh process can therefore
reduce gradients in a different floating-point order from its warmed parent.
Strict replay now uses native `find_unused_parameters=True` with
`static_graph=False` to retain the initial bucket layout. The reduction policy
is part of strict runtime and exact-checkpoint identity; an earlier checkpoint
without it is not a compatible training-state parent. Legacy mode retains its
existing DDP configuration, and inference payload compatibility is unchanged.
Compact rank-zero native bucket observations accompany existing step rows,
without extra collectives. The CPU test covers accumulation, checkpointing and
shared embedding deltas with no selected input tokens; the fresh eight-rank
parent/continuation comparison remains the required GPU acceptance.

### 5. Cross real inference and evaluation consumers

Use the current inference config fields to consume the new final checkpoint
and the immutable pre-change combined components through dynamic HF. Freeze
a small shared COCO inference set, deterministic decode and token limits.
Compare baseline/candidate HF token outputs and scoring from the same old
bytes; exact token and payload identity is expected for this scheduling-only
change. Use existing numeric tolerances where a component oracle already
defines them and record them before candidate execution.

For composed vLLM, use `python -m src.qualify_vllm run` and `admit` with the
actual model/config identity, then the real `src.infer` entry. The user accepted
the observed one-grid coordinate error on 2026-09-12: “差一格无所谓,这个误差可以接受.”
The merged BF16 artifact therefore has an explicit bounded composition
contract: prompt IDs, sequence length and every non-coordinate token (including
EOS/structural tokens) must match dynamic HF exactly; differing coordinate
tokens must map through the canonical coordinate vocabulary and differ by at
most one grid unit. There is no allowance for larger coordinate, lexical,
structural or length drift. Exact tied rows, merged target weights, dtype and
component/model identity checks remain mandatory. Report changed positions
and values as well as unchanged strict greedy/logit comparison diagnostics;
do not rewrite failed diagnostic booleans as exact parity. The existing finite
numeric evidence and threshold values remain authentic diagnostics, rather
than dynamic-HF probability-equivalence gates for this BF16 execution contract.

Version and bind the revised composition acceptance in current qualification
receipts. Validate the coordinate mapping against the admitted tokenizer;
recompute bounded token acceptance from explicit token IDs and mapping at the
consumer boundary. A forged success flag, invalid coordinate mapping, missing
policy evidence or drift outside the bound must reject. Keep this in the
current composition and qualification owners, without a new configuration
profile, registry or historical receipt migration.

Runtime and concurrency qualification execute the bound merged snapshot and
retain their existing completion, request identity, finite policy-logprob and
cleanup checks. They do not establish HF-versus-vLLM output/logit parity.
Forced replay stays exact within vLLM: request, prompt, generated continuation,
stop evidence and raw-logprob alignment must match. Do not apply the coordinate
tolerance to replay or imply that merged-policy likelihoods equal dynamic HF.
Dense exports are derived artifacts; old admission receipts are not
grandfathered. Run one independent review of this changed numerical boundary
before implementation, then complete the already authorized qualification,
inference, evaluator and archive work without another approval request.
Use the current direct evaluator on the scored artifact
family (`--artifact-dir` and `--out-dir`) and check lineage and finalization.
This is consumer plumbing evidence, not a model-quality assessment.

### 6. Separate boundedness, correctness and speedup claims

The required result is semantic preservation plus bounded preprocessing and
the eight-rank vertical witness. Record baseline and candidate preparation
and end-to-end lifecycle timings with identical workload and runtime shape.
Label cold caches and warm hits separately, and identify shared-GPU or OS
cache disturbances. A single lifecycle pair is descriptive only.

If claiming a speedup, use at least three paired, alternating baseline and
candidate observations for the affected path, require every pair to improve
and median improvement to exceed the larger within-arm observed range, with
semantic/resource checks passing. Otherwise report measured values and an
inconclusive speedup result. Do not launch repeated full lifecycles merely to
claim speed for a preprocessing resource-bound fix. No quality or convergence
claim follows from four updates.

### 7. Evidence-triggered roadmap

These items are disposition decisions, not unchecked mandatory code tasks.
Record whether each trigger occurred in the smoke; absent evidence means
deferred. A new objective or topology requires a separately frozen contract.

| Area | Adopt when | First coherent change | Protected boundary |
| --- | --- | --- | --- |
| Lazy cache hydration | Retained packed payloads materially dominate RSS/startup | Load admitted local chunks lazily in the existing cache/session path | Order, semantic admission, no repeated whole-cache scans |
| Loss/logits kernels | Timings identify logits/CE as the dominant useful target | One qualified implementation behind the current loss owner | Segment weights, masks, denominators and gradients |
| Optimizer acceleration | Optimizer step is a measured bottleneck | Existing AdamW option plus parity check | Groups, dtype, update and resume state |
| Data/template management | A real new source combination is requested | Deterministic preparation manifest using existing template owners | No implicit scientific mixture/order change |
| vLLM throughput | Qualified request workload shows queue/engine bottleneck | Measure supported concurrency before changing engine topology | Current TP1/DP1, exact request/token lineage, admission |
| DoRA precision | Measured memory/cast overhead warrants a precision experiment | Explicit opt-in precision contract and merge/reload parity | Never silently alter current default or magnitude semantics |
| Rollout to learning | A real RL consumer and objective exist | Exact token trajectory, masks, policy version and weight-handoff boundary | No decode/re-encode substitution; invalidate stale weights/caches |
| Distributed capacity | Replicated model state blocks a required workload | FSDP2 loading, optimizer and distributed save/resume vertical slice | New topology has its own evidence; no config-only support claim |
| Async checkpointing | Measured save stall dominates lifecycle cost | Bounded writer in existing artifact owner | Completion and latest aliases only after durable success |
| General modularity | A concrete change requires competing semantic owners | Extract a focused ordinary module with one owner | No generic registry, event bus or orchestration framework |

## Risks / Trade-offs

- Bounded futures may change completion timing: canonical indexing and payload
  equality close semantic risk; a controlled gated test proves the bound.
- Four updates are expensive with production EBS48 but expose actual collectives,
  save/eval and resume seams. They do not establish long-run behavior.
- Existing entry/consumer defects may prevent first-pass smoke completion:
  preserve the first failing evidence and fix only the demonstrated owner.
- No usable historical experiment checkpoint is assumed available: the
  immutable pre-change producer fixture limits the historical inference claim.
- vLLM qualification may fail in the current runtime: that is an incomplete
  required gate, not permission to weaken qualification.
- Resource/timing observations are noisy on shared hardware: report scope and
  refrain from unsupported performance claims.

## Migration Plan

1. Validate/freeze these artifacts; perform one independent Astra review round
   covering semantic/resource invariants and lifecycle/compatibility acceptance.
   Resolve material findings together and retain a concise review disposition.
2. Capture the baseline producer fixture and production-shaped launch evidence
   before source edits, or use an isolated baseline checkout with the exact
   recorded code identity. Preserve existing untracked reports and user work.
3. Implement the bounded scheduler and smoke configuration, prove the bound
   and existing semantic contracts, then execute the candidate vertical path.
4. Correct reproduced failures, consume old/new payloads, verify resume and
   qualified inference/evaluation, and write the acceptance receipt. Update
   current operator documentation and task state to the actual evidence.
5. Rollback is source/config reversion; immutable baseline artifacts stay
   available. Cache reuse remains governed by existing semantic admission,
   never by blindly copying cache directories. No old state migration needed.

## Open Questions

No user decision blocks implementation. Actual prefix/pack counts, runtime
failures, timings and triggered roadmap items are discoverable execution facts
to record in the acceptance receipt, not reasons for a new authorization gate.
