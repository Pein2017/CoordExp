## Context

CoordExp already completed a broad internal runtime refactor program, but the active training path still shows several concrete optimization and operability gaps.

The most important observation from the current codebase review is that DDP is not the primary bottleneck by itself.
The active bottlenecks sit above it:

- duplicated coordination and reduction ownership across the Stage-2 trainer layers,
- serial or fully local Python work in Channel-B before synchronization points,
- repeated data preparation and encoding in the packing path,
- incomplete executed-runtime provenance,
- and heuristic observability contracts.

One important scoping detail:

- the live Stage-2 path still uses trainer-owned dynamic post-rollout packing,
- while the shared dataset layer still pays repeated sample-fetch and encode costs upstream.

So the data-path work in this change should be treated as a shared dataset/sample-fetch optimization program, not as a replacement for the current Stage-2 post-rollout packing contract.

That matters because it changes the right optimization order.
Switching distributed frameworks too early would risk a lot of churn while leaving the existing sources of skew and overhead intact.

The intended end-to-end flow we need to protect is:

`typed config + dataset authority + geometry authority`
-> `dataset preparation + encoding + packing`
-> `trainer/runtime orchestration + rollout coordination`
-> `teacher-forcing / eval / logging`
-> `metrics + artifacts + restartability`

Constraints and invariants:

- Keep config-first engineering; avoid new ad-hoc CLI flags by default.
- Preserve geometry invariants and object ordering contracts.
- Preserve Qwen3-VL chat-template compatibility.
- Do not modify upstream HF internals.
- Preserve Stage-2 learning semantics unless a later explicit spec delta says otherwise.
- Keep single-dataset training and packing-first efficiency assumptions as the default posture.

## Goals / Non-Goals

**Goals**

- Improve DDP scaling efficiency by reducing duplicated coordination and expensive rank-local work before barriers.
- Make distributed step behavior explicit and testable.
- Improve tokens/sec and startup/runtime efficiency in the data path without weakening dataset or prompt contracts.
- Make training artifacts describe the executed runtime and checkpoint intent clearly enough for operators to reason about replay and resume.
- Make observability more explicit, cheaper under DDP, and easier to interpret.

**Non-Goals**

- No broad Stage-2 semantics redesign.
- No immediate migration to FSDP, DeepSpeed, or a different launcher model.
- No new public CLI workflow for training.
- No speculative multi-dataset or fusion-first redesign.
- No silent metric-schema or artifact-schema drift.

## Decisions

### Decision 1: Optimize coordination ownership before changing distributed backends

The first optimization target is duplicated coordination logic, not the DDP backend itself.

We will treat the following as one cohesive ownership problem:

- phase barriers,
- rank-symmetric failure propagation,
- per-step readiness checks,
- and metric reduction semantics.

This work should converge on a shared `DistributedStepCoordinator` or equivalent explicit seam that is used by both:

- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_rollout_aligned.py`

Why:

- current coordination is split across trainer-local reducers, helper modules, and executor-local barrier wiring,
- and that split is directly connected to skew sensitivity and maintainability cost.

Alternative considered:

- move directly to a new distributed runtime stack.
  Rejected because it would not fix duplicated ownership or Python-side skew on its own.

### Decision 2: Replace metric-name heuristics with an explicit metric contract

Stage-2 metric reduction semantics should no longer be inferred from string prefixes or suffixes.

Instead, the optimization program should define an explicit metric contract with fields such as:

- reduction mode,
- local aggregation mode,
- DDP aggregation mode,
- snapshot policy,
- and health-signal behavior.

The important requirement is that local pending-log aggregation and global DDP reduction consume the same contract.

Why:

- today the same metric can pass through multiple heuristic reducers,
- and adding a new metric currently risks silent semantic drift.

Alternative considered:

- keep string conventions and tighten tests only.
  Rejected because the problem is duplicated ownership, not just missing regression coverage.

### Decision 3: Split sample preparation from token encoding in the dataset hot path

The data layer should make it possible to reuse work safely.

We should introduce an explicit internal distinction between:

- preparing a record:
  - path resolution,
  - preprocessing,
  - ordering,
  - rendering payload construction,
  - metadata shaping
- encoding a prepared record:
  - chat template application,
  - tokenization,
  - final example shaping

Static packing, length precompute, and encoded-cache-backed paths should be able to reuse the appropriate layer instead of recomputing the whole pipeline.

This decision does **not** redefine trainer-owned dynamic post-rollout packing in:

- `stage2-ab-training`
- `rollout-matching-sft`

Why:

- current packing and plan-building paths repeatedly enter expensive `dataset[index]` flows,
- and the current method boundaries make reuse difficult and concurrency riskier.

Alternative considered:

- add more cache layers around the current dataset path.
  Rejected because caching around a side-effectful boundary would increase correctness risk.

### Decision 4: Make precompute and caching explicit about mutability and memory bounds

Threaded precompute should only be used when the dataset path is side-effect free for the relevant operations.
If a dataset uses mutable template state or stochastic preprocessors, the system should either:

- use an immutable helper path for precompute,
- or fall back to serial/process-isolated work.

Similarly, encoded-cache shard residency should be bounded explicitly rather than growing implicitly with persistent workers.

Why:

- thread safety and RSS growth are both operational risks that currently look like incidental behavior rather than explicit policy.

Alternative considered:

- trust the current thread-pool and worker model because existing tests pass.
  Rejected because the identified risks are workload-dependent and insufficiently encoded as contract today.

### Decision 5: Distinguish authored config from executed runtime

The training stack should preserve both:

- the authored typed config,
- and the executed runtime after launcher/bootstrap mutation.

The optimization program should add explicit executed-runtime artifacts such as:

- `effective_runtime.json`
- `pipeline_manifest.json`
- stronger dataset provenance sidecars

This also implies making checkpoint intent explicit:

- `artifact_only` for model-selection artifacts,
- `restartable` for checkpoints that must include all future-affecting state.
- `artifact_only` remains the compatibility-preserving default until a restartable artifact contract is explicitly implemented.
- `restartable` must be opt-in and paired with hard-fail resume preflight when required artifacts are missing or incompatible.

Why:

- today operators can inspect useful artifacts, but the replay/resume contract is still under-specified.

Alternative considered:

- keep current artifact files and document the caveats.
  Rejected because operator ambiguity is already a real maintenance cost.

### Decision 6: Train-time eval and observability health should be explicit about rank ownership

This change should optimize train-time eval transport and diagnostic health visibility without broadening the offline evaluator contract or reopening the entire file-logging surface.

We should prefer:

- rank-0-owned artifact writing,
- scalar-first or shard-first DDP aggregation for train-time eval,
- and explicit health metrics when best-effort diagnostics disable themselves.

Why:

- these are low-risk optimizations that can reduce confusion and DDP overhead without changing core learning behavior.

Alternative considered:

- defer all observability changes until the larger runtime surfaces are cleaned up.
  Rejected because metric and logging ownership are already part of the coordination problem.

### Decision 7: Defer broader multi-server scale-out to a follow-on change

The runtime already contains multi-server concepts, but the launcher and preflight still hard-limit the stable path to one server.

This optimization program should not remove that restriction inside this change.
It should first:

- stabilize coordinator ownership,
- stabilize metric and artifact contracts,
- and make launcher/runtime topology parity explicit.

Then, in a follow-on change, multi-server parity can be enabled deliberately.

Why:

- scale-out should come after correctness and operability boundaries are explicit.

Alternative considered:

- enable multi-server dispatch immediately because the runtime already has partial support.
  Rejected because the current control plane is not yet explicit enough.
