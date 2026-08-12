## Context

CoordExp-Swift owns packed multimodal training semantics that are stricter than
the generic contracts exposed by its external libraries. The accepted path
already prepares deterministic source-order packs, preserves segment isolation,
maps supervision and four-row MRoPE per segment, and uses explicit
FlashAttention varlen boundaries. The completed
`streamline-coordexp-swift-base-infrastructure` change also established bounded
input preparation, rank-selective cache publication, sharded evaluation, phase
timings, and exact distributed metrics. Its deltas were synced to stable specs
and the change was archived on 2026-08-06, so it is now the stable authority for
this follow-on.

The 2026-08-06 audit found three correctness or claim-strength gaps:

1. the cache fingerprint does not cover every realized tokenizer/vocabulary and
   transitive cached-payload determinant;
2. the current FA2 runtime proof can accept one matching varlen call instead of
   attesting every executed Qwen text layer;
3. a zero-weight protected token-type loss still constructs a full-vocabulary
   gradient graph solely to retain a raw diagnostic.

It also found efficiency opportunities whose benefit is not yet established on
the production eight-rank workload: earlier cache admission, direct eval-shard
hydration, input-preparation overlap, stronger bin utilization, and exact
resume. These have different semantic and operational risk, so they are split
into independently reversible waves.

The external references are evidence and implementation comparators, not new
owners of CoordExp semantics:

- ms-swift 4.2.2 exposes offline chunked binpacking and a bounded iterable
  packing interval. Those are useful comparator shapes, but their reordering,
  multiprocessing, error handling, and generic loss/token contracts cannot be
  imported as the CoordExp production contract.
- Transformers 4.57.1 exposes varlen FlashAttention utilities, lazy backend
  selection, and padding-free boundary plumbing. Those utilities are useful
  backend references, but selecting a backend does not prove that every Qwen
  text layer executed it with CoordExp's exact boundaries.
- Accelerate can supply low-level distributed state helpers, but the exact
  checkpoint contents, compatibility keys, cursor meaning, and artifact lineage
  remain owned by this repository.

## Goals / Non-Goals

### Goals

- Close the three audited correctness and claim-strength gaps before unrelated
  throughput tuning.
- Measure end-to-end time-to-first-step and steady-state optimizer-step wall
  clock, including CPU preparation, cache admission, rank skew, I/O, and GPU
  idle time.
- Make every candidate policy explicit, fingerprinted, reproducible, and
  reversible while retaining compatibility defaults until promotion.
- Add exact same-world-size resume with fail-closed admission and append-only
  lineage.
- Preserve evidence sufficient to distinguish executed code and binary state
  from a package-name or Git-commit approximation.
- Permit each accepted wave to land independently; a failed optimization must
  not hold correctness work hostage.

### Non-Goals

- Changing renderer, dataset, coordinate, supervision, loss-normalization,
  segment-isolation, visual-replacement, MRoPE, or DDP semantics.
- Treating dynamic packing, reordering, or changed co-presentation as a free
  infrastructure substitution.
- Training-time KV cache, cached GPU visual states, or cached hidden states from
  a trainable tower.
- Automatic dependency upgrades, FA3/FA4 promotion, or production-cache
  deletion.
- Exact resume across a different world size, topology, or trainable parameter
  set in the first supported contract.
- Bitwise equality across independent CUDA launches.

## Decisions

### Decision 1: Use a follow-on change with explicit baseline dependency

This change will not revise the completed
`streamline-coordexp-swift-base-infrastructure` plan in place. Its accepted
M2/M4/M6 behavior has been synced to stable specs and archived at
`openspec/changes/archive/2026-08-06-streamline-coordexp-swift-base-infrastructure/`.
Implementation commits for this change reference that baseline and carry their
own receipts.

Alternative considered: append new tasks to the completed change. Rejected
because it would reopen a 29/29 acceptance boundary, mix prior evidence with new
claims, and leave two meanings of “complete.”

### Decision 2: Run ten dependency-ordered, independently reversible waves

The work is split as follows:

| Wave | Scope | Required exit | User stop |
|---|---|---|---|
| 0 | authority, frozen baseline, provenance and measurement contract | stable old specs, clean comparison matrix, no shared-cache writes | no |
| 1 | complete cache identity and pre-model admission | mutation tests and fail-before-model receipts | no; production materialization is pre-authorized for Wave 9 |
| 2 | all-layer FA2 proof and real parity oracle | real-model forward/backward parity plus negative control | no |
| 3 | zero-weight protected-loss optimization | correctness implementation retained; performance experiment dropped with no efficiency claim | yes; closed without promotion on 2026-08-11 |
| 4 | eval shard-on-load and startup/resource receipts | current correctness-tested hydration retained; startup/resource experiment dropped | no-promotion selected; cleanup gate 5.7 remains |
| 5 | three-arm input-provider comparison | synchronous remains the production default; performance matrix dropped | yes; closed without promotion on 2026-08-11 |
| 6 | offline/windowed/online packing comparators | CPU research retained; matched training and policy decision remain pending | yes for this production path; future design remains open |
| 7 | exact training-state resume | interrupted-versus-uninterrupted same-world-size proof | no; bounded contract is delegated to lead |
| 8 | freeze upstream/backend baseline | provenance and no-upgrade compatibility guard | no; upgrades are out of scope |
| 9 | production convergence | one authorized cache build and full pipeline acceptance | no within the frozen cost envelope |

Every active decision-bearing wave has correctness, operational, and independent
audit gates and stops on its first failed mandatory gate. An explicitly dropped
performance experiment may instead close only through a user-owned
no-promotion/default-retention disposition plus its remaining cleanup and
correctness audit; an omitted measurement is never treated as a favorable
operational result. A performance candidate that does not beat the frozen
reference is removed or left experimental and does not change defaults.

The user clarified on 2026-08-11 that the r5 correctness/plumbing sequence could
run concurrently with pre-existing eight-GPU work.  That sequence therefore
binds a stable pre-marker driver-process baseline and requires every post-phase
GPU row to be a subset of it, while preserving exact owned-process cleanup and
never signalling the baseline jobs.  The live-state amendment admits at most
49152 MiB of pre-existing allocation on each 81920 MiB GPU, retaining at least
32768 MiB for the owned smoke.  This is an execution-availability
decision, not a performance-oracle change: shared-load timing and resource
observations are non-promotional, and every efficiency decision retains its
matched or otherwise-idle evidence gate.

That one-shot r5 authorization is now consumed. Its immutable plan, sequence
marker, and terminal receipt are bound by file/payload SHA-256 pairs
`ea3cef1d96412cd3f425e9c7f0c0cd562ceafa2fe59c39c225755898dbb66a1f` /
`73c502d69ec8b999a9e451620702b54eb1d573bf775e18d32edde08d216f53c4`,
`58aa7425dc6c019a23e790ba072495f0fb787b805eabaf922f079c35eddb19b5` /
`a620caa0e64a12be980120e310a914571c78ee0889d6da0760aae208f3c82224`,
and
`d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1` /
`f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656`.
The uninterrupted phase exited `rc=1` after 20.094 seconds; every later phase
is `not_started`, cleanup and recovery passed, and no retry is authorized.
CPU-only reproduction attests an entrypoint-environment defect: invoking the
absolute `src/train.py` path without repository-root `PYTHONPATH` raises
`ModuleNotFoundError`, while module invocation succeeds. R5 is therefore a
diagnostic-only launcher/plumbing failure. It makes no model, cache, training,
or exact-resume claim.

The successor source fix is in progress but has no execution authority. There
is no authorized r6. Any successor requires explicit new user authorization and
fresh absent roots, runtime attestation, deterministic preflight, request,
immutable plan, marker, and private cache, while retaining the same at-most-once
no-retry rule, cost ceiling, shared-load restrictions, and diagnostic claim
boundary. This does not reopen the Wave 3-5 no-promotion dispositions or Wave 6:
their defaults remain retained, and `source_order_next_fit` remains the
production packing policy while Wave 6 matched training stays pending.

#### Wave 7 r6 one-shot successor authorization — 2026-08-11

Later on 2026-08-11, the user explicitly authorized one fresh r6 one-shot successor.
This later authorization supersedes only the prospective no-authority
statement immediately above; it does not alter or reinterpret the immutable r5
failure. r6 is not an r5 retry. It is exactly one prospective successor under a
fresh r6 run/cache/runtime/preflight/request/plan namespace, with absent-only
publication roots, a fresh private cache, and its own at-most-once marker.

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
production default.

#### Wave 7 r6 immutable consumed preflight failure — 2026-08-11

The authorized r6 attempt is consumed and failed in deterministic preflight
after `KeyboardInterrupt`; it did not reach model loading, training, or any
sequence-level request, plan, marker, or terminal. Its immutable precursors
completed before preflight. The passed config-bundle receipt has file/payload
SHA-256
`82214495b1e75e44155303a28e456d6269942fde926959ce846d5c03fdb6aa89` /
`6f4a16b524701dac91cf03a1ac802e590766f1a4ab0881023a400a339809dd3c`
and semantic-projection SHA-256
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
The canonical private-cache root is
`outputs/probes/coordexp_swift/private_v3_cache/2026-08-11-wave7-r6`.
Its passed preparation receipt has file/payload SHA-256
`52df7e0b3bcca23fcfd5bde69f59f7a97030bf455d704a2009e19a6d22a74c3b` /
`bd9a8e4b937f000547d7f7de676bd642773c03edbedd7c5b0fdfee3c0bf9f718`.
The `fork_process_pool` / 16-worker production payload path produced train
fingerprint
`8307cf2dd9cac344b8a50bb783a831d683c3686ae73678425d6245e9df9eae87`,
manifest SHA-256
`775c0c87f6c6e63b11f17c4c737c7649cadb2e0b20494b02d4694274d2ad3567`,
and one 32-micro-step chunk with SHA-256
`59a38ddb036ae644e15bfb25db022dc7b28ca82216679262bf6af44824ba8956`;
the eval fingerprint is
`881fe84188657eeb705dd1bf2b754f11b9f69d61eb62a83954fc867a9d1744f8`,
with manifest SHA-256
`dce7449e1267a9217166b6297249856d94f0def88fd9cb7fd7c3422f09a15634`
and one 8-micro-step chunk with SHA-256
`047135041298ed869a078d78f22a3cf54d949963950ffe3dedc54c120e910ca3`.
Production payload loading passed and the cache tree remained stable. This is
bound by two repeated seven-file snapshots with identical tree SHA-256
`943fdbc325e143dbb8748c97f905f95fecd11836d6f6039b728df8b816a3de85`.
It is cache construction/admission evidence without a model or training claim.
The passed runtime-admission receipt has file/payload SHA-256
`51822d203651799cf44279c85b2de64d50a2c2a5ba2ad3e8513ab1f2a22f0469` /
`3b06e0ce6eefa31fb414b6d8c9eb50c8a2bdbd2f0ca448da71fbdbdb929fd0d2`.

The immutable determinism-preflight plan, attempt-start marker, and failed
terminal receipt have file/payload SHA-256 pairs
`2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` /
`9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f`,
`f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` /
`967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e`,
and
`eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` /
`4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab`.
Marker publication consumed r6. The failed terminal records
`mismatches=["KeyboardInterrupt"]`, `launch_count=0`, no bound rank receipts,
and no completed comparison. Direct inspection of the immutable leaf files
nevertheless finds exactly eight passed `launch-a` rank receipts, ranks 0-7,
all with workload aggregate SHA-256
`41f57c8c5ff14d25b1c7c20711f53becec5853fadcfe68dd0c116ec8fd1d9422`;
no `launch-b` rank receipt exists. This is narrow supplemental plumbing evidence
only. Because the consumed terminal cannot incorporate those leaves or reach a
two-launch fixed point, it is not a preflight pass and cannot be repaired or
reinterpreted in place.

Post-failure code repair is content-bound by SHA-256
`52d6e2a3b709ede73f5794eefdbfd6c57997714af29f5329e936b8953492346c`
for `wave7_determinism_preflight.py`,
`8af9a58d1f425c13bd5e25ffeaca3ab60865b9dcf650d0469abb846be48279a8`
for its focused test,
`cadcbf5ec413b4eea7314684181b4a3ef514144212f9814e3f58d2b999ddfaea`
for `wave7_exact_resume_sequence.py`, and
`ae0f1615410cce187722e1473fed93908033e94b1b6ed6dc98caf4c6df41c236`
for its focused test. An independent read-only audit returned PASS for this
repair set, but only as engineering evidence for a possible future explicit
authorization. It does not alter the immutable r6 result, establish a Wave 7
pass, admit Wave 8, or authorize an automatic r7. Wave 3-5 retain their
no-promotion dispositions; Wave 6 matched training remains pending and
`source_order_next_fit` remains the production default.

#### Wave 7 r7 one-shot successor authorization — 2026-08-12

On 2026-08-12 the user explicitly authorized exactly one fresh Wave 7 `r7`
one-shot successor. This later authority supersedes only the prospective
no-`r7` statement above. It does not alter, repair, or retry immutable `r6`:
`r7` is a new successor with an absent-only sequence root
`outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-r7`, private-cache
root `outputs/probes/coordexp_swift/private_v3_cache/2026-08-12-wave7-r7`, and
private `runtime/`, `determinism-preflight/`, `amendment-v4.json`,
`request-v6.json`, `sequence-plan-v6.json`, `sequence-marker.json`, and
`sequence-receipt.json` namespaces beneath the sequence root. None may fall
back to or mutate an earlier root.

The r7 packet uses amendment schema
`coordexp-swift-wave7-r7-amendment-v4` and request, plan, marker, and terminal
schemas `coordexp-swift-wave7-exact-resume-sequence-request-v6`,
`coordexp-swift-wave7-exact-resume-sequence-plan-v6`,
`coordexp-swift-wave7-exact-resume-sequence-marker-v6`, and
`coordexp-swift-wave7-exact-resume-sequence-receipt-v6`. Its request and plan
MUST copy unchanged these immutable predecessor bindings:

- `legacy_r4_failure`: r4 comparison receipt file/payload SHA-256
  `1d56e8139500ac09e62f57b2d2ce074401bc7596a64bee3b3915ff5036575e5b` /
  `13a39f43f1e504f48a2e70a8037d5ea7ccfd919003ffa1f89417e1b592973f4a`;
- `predecessor_sequence_failure`: r5 terminal file/payload SHA-256
  `d5244fb6d96b24b1cfac4438628ae141496f7357c286dff5454c19c9f39731b1` /
  `f560cbd7bfeac8e25f031021d9d0053c3df0d69961d52f1b854754ce4f4e7656`;
- `predecessor_preflight_failure`: the complete r6 preflight failure chain,
  consisting of plan file/payload SHA-256
  `2d5d5108c07e02ff964f5b3e6e4e42ce84e58bd0c17a8310dd7e2f3e6147e6ec` /
  `9b5c820c5f2c366c994a15732b99498b4072d1d552d9a7048c6afb65e0532a8f`,
  attempt-marker file/payload SHA-256
  `f19a894ac3113aecc3a126dab7e3327389518c9e12e34d7806ad401adf239ce9` /
  `967fc87daabc19dd82d9d5a0f2567a2700da71bf9cb3773f904da63488585d7e`,
  and terminal file/payload SHA-256
  `eaf20c86a3fa95d7fa6533faaf7fb0e9b8fe938aa6f62d2ba725c3c604a64784` /
  `4b400e094c494099b611e85d78b2156ce53b0fe372e12de6ff780c493b5770ab`.

Each r7 phase is at most once and `<=600s`; the complete sequence is
`<=2400s wall` and `<=14400 GPU-device-seconds`. The shared eight-GPU envelope
remains exactly 81920 MiB total, at most 49152 MiB pre-existing allocation, and
at least 32768 MiB headroom per device. Shared-load timing, utilization, memory,
and throughput remain non-promotional. Any gate failure publishes terminal
evidence and stops all later r7 phases, with no retry, root switch, or automatic
Wave 7 `r8` successor.

The user's 2026-08-12 instruction already authorizes the gated order Wave 7 ->
Wave 8 -> Wave 9, and task 10.3 already authorizes final cache/convergence work
after its gates. This amendment adds no new user-authorization requirement for
Wave 8 or Wave 9. Wave 8 may proceed only after the accepted r7 Wave 7 gate and
under its own frozen packet and budget; Wave 9 may proceed only after the
accepted Wave 8 gate and under its own sealed packet, frozen identities, and
budget. Neither later wave may bypass its gate or inherit r7's packet or budget.

#### Wave 7 r7 immutable consumed cache-admission failure — 2026-08-12

R7 is consumed and failed before runtime admission. The passed config-bundle
receipt has file/payload SHA-256
`f07e4bc9f4e272e6ed831c714b55f8c1c7a6ee721a86c691e00e5cd3eee34ac6` /
`a72c97856ce2689077c91a19652aca3e6fd6b914c8dbce15bfac9008da46ebdb`
and semantic-projection SHA-256
`f1047d46a93ebb4bb4b97b25aec74ce89c3bce43a62b238eb40bf86e30da0a5e`.
Its uninterrupted, interrupted-parent, and resume-child config file SHA-256
values are respectively
`88ccc01f69deab5242405593032c31ff8374c455f08247da20322e3709be2fc4`,
`f9441f5a9ccb27635289a49739b4449c486770b970736549ea9b4dfee4fdf997`,
and
`8bec80b9a5eaefcd9212968b3f9dd8d38eb0601308791a686a2ff622d9dcb9f8`.

The first preparation invocation stopped before receipt publication because the
fresh private-cache parent was absent. After that exact authorized root was
created, the production preparation command was invoked without the required
strict determinism environment and atomically published the immutable failed
`preparation-receipt.json`. Its file/internal receipt SHA-256 pair is
`9077a0458bece11b630ab90c68a9811dfca2941171b0575717b3dba670ae817d` /
`e7d7d5c4f2d6cf35e823c80faed4c9391f0368449f6be27560db9f628f8bcc2b`.
It records `terminal_status="failed"`,
`failure.error_code="runtime.determinism_environment_conflict"`, and
`result=null`. No cache payload file was created.

Under the r7 at-most-once authority, this failed admission consumes r7 and
stops the sequence. There is no retry, root switch, or automatic Wave 7 `r8`
successor. No runtime, deterministic-preflight, request, plan, marker, model,
CUDA, training, comparison, or post-run gate was reached, so r7 supplies only a
cache-preparation admission failure and no cache, model, training, exact-resume,
performance, or Wave 7 pass claim. The already-authorized Wave 7 -> Wave 8 ->
Wave 9 route remains gated and is blocked at Wave 7; Wave 8 and Wave 9 cannot
start unless a successful preceding Wave 7 gate exists under valid authority.

This shared execution availability also covers fresh, versioned Wave 4, Wave 5,
and Wave 6 correctness/plumbing successors. It does not revive a consumed root
or reuse a historical marker: every successor has its own absent root and
one-shot marker, authenticates the exact shared baseline, and preserves complete
owned-process cleanup. Numerical and artifact evidence may survive shared load;
startup, provider, packing, throughput, timing, memory, and efficiency promotion
remain governed by the original matched/otherwise-idle oracle.

The later 2026-08-11 production-path decision supersedes those optional Wave
3-6 successor launches before any fresh marker was consumed. The active order
is Wave 7 -> Wave 8 -> Wave 9. Waves 3 through 5 are closed without performance
promotion; Wave 4 task 5.7 cleanup and verification are complete. They retain
current production semantics/defaults.
Wave 6 remains a bounded
CPU research result and pending future design, keeps `source_order_next_fit` as
the production default, and is explicitly nonblocking for this convergence.
An omitted experiment is recorded as omitted; it is never relabeled as a pass.

### Decision 3: Replace ad hoc fingerprint fields with a declared determinant registry

Cache identity will be assembled from a versioned determinant registry rather
than a growing hand-maintained tuple. Each determinant has a name, owner,
content identity, reason, and schema version. The registry covers:

- dataset and image content identities used by preparation;
- renderer, parser, raw-data geometry, ordering, packing, supervision, MRoPE, and serialization
  owners, with both their semantic identity and the content digest of every
  declared owner source file; an independent hard-coded expected-owner matrix
  prevents the production registry from serving as its own completeness oracle;
- the production `SupervisedMicroStep` constructor and schema owners, including
  every config value serialized into a cached micro-step such as model logits
  dtype and FA2 proof flags;
- one recursive, sorted envelope of local model-front-end assets. Every regular
  file is content-bound except exact model-weight payloads named by a
  content-bound weight index or a small declared set of conventional standalone
  weight basenames at the model-root top level only; a nested same-basename file
  remains part of the front-end envelope unless an index declares it as a shard.
  The exclusion policy itself is fingerprinted. Total discovered regular files,
  weight-index declarations, hashed front-end files, and hashed bytes are each
  bounded. Symlinks, unreadable subtrees, unclassified large files, and other
  ambiguous entries fail closed instead of being silently skipped;
- the full resolved token-vocabulary groups, bound by canonical per-group
  membership digests and counts rather than unbounded member lists;
- policy configuration and algorithm versions.

The manifest stores the determinant list and aggregate fingerprint. A startup
preflight resolves the same identities with model loading disabled, validates
the train/eval publications, the current rank's required training payloads, and
every evaluation payload under the compatibility eager-validation path, and
only then admits expensive setup. Retained rank-selective evaluation hydration
is a Wave 4 optimization and is not claimed by Wave 1. Cache v3 resolves to
`<cache-root>/coordexp-swift-pack-cache-v3/<fingerprint>`. Publication builds in
a unique same-filesystem staging directory and atomically installs only when
that canonical final target is absent. The publisher validates the version
namespace and 64-hex fingerprint path before staging, re-resolves every
determinant after staged-payload validation and immediately before the
no-replace install, and rejects observed drift without exposing the target. A
valid existing target is a read-only cache hit; an existing incomplete,
corrupt, or conflicting target is an immutable collision and fails closed
without automatic repair. In this change, `rebuild` means only resolving the
current v3 identity and publishing to a previously absent v3
namespace/fingerprint. v1, v2, and existing v3 directories are never mutated,
replaced, or deleted.

The canonical path rule applies to public admission and hydration as well as
publication. Every public reader and writer receives the caller-selected cache
root explicitly and validates the supplied target against that root; it cannot
infer a new root from an alternate well-shaped cache directory. Symlinked cache
path components fail before manifest consumption. Internal validation of a
private staging directory is a separate non-public operation; copying otherwise
valid bytes outside the selected v3 namespace/fingerprint does not create an
admissible cache.

Chunk authentication and decoding share one bounded stable snapshot. The loader
opens each path component without following symlinks, requires a regular file,
reads one size-bounded byte snapshot while checking descriptor stability,
compares that snapshot's SHA-256 with the manifest, and only then runs the
restricted unpickler over those exact bytes. Path replacement between checksum
and decode therefore cannot substitute unauthenticated payload state.

The determinant registry is explicit rather than pretending static import-graph
hashing proves transitive runtime ownership. Mutation tests are the acceptance
mechanism: changing every named class of determinant must either change the
fingerprint or fail because the new owner is undeclared.

Alternatives considered:

- hash the whole repository. Rejected because unrelated edits would force
  rebuilds and would still not identify external assets or executed binaries.
- hash only package versions and selected source files. Rejected because
  same-path tokenizer edits and realized vocabulary-group changes can remain
  invisible.

### Decision 4: Treat numerical parity as the oracle and backend tracing as attestation

The FA proof will derive the expected Qwen text-layer identities from the loaded
model topology. It will record one backend event per executed layer, including
layer identity and the exact `cu_seqlens`/maximum-length contract. Vision and
other attention calls use separate event classes. Acceptance requires an exact
set/count match, not merely at least one matching event.

The proof is paired with a versioned real-model packed-versus-separate harness
over the same supervised segments. Wave 2 v3 binds the exact authenticated v2
workload but does not reinterpret the immutable v2 result. It executes two
identical packed arms and one separate reference under restored model/RNG state.
Each packed arm clears gradients exactly once immediately before its forward and
performs one backward without another clear in between. The separate reference
reproduces the production micro-step loop: clear gradients once, then perform each of its two
forward/loss/backward operations immediately and in order without retaining a
summed differentiable graph. A detached sum is permitted only for reporting.
A deliberately corrupted boundary negative changes only the FA2 boundaries and
must be detected in supervised logits or total loss; gradients are not an
acceptable sole negative-control signal.

Qwen attention and linear execution remains under production BF16 autocast.
The production Accelerator-prepared model then applies its graph-connected
`ConvertOutputsToFp32` seam before `LossContext`, so logits and high-precision
loss arithmetic are FP32 while autograd returns through the cast into the BF16
upstream graph. V3 preserves that production route; it does not introduce a
second recast. Receipt/comparison snapshots of logits, losses, per-term scalars,
and gradients are detached and kept or upcast in FP32. The mandatory per-term
`raw_loss`, `weighted_loss`, `segment_mean_numerator`, and
`token_weighted_diagnostic` values and all 589 expected DoRA/special-token
gradients are BF16-compute-derived, even where parameter or gradient storage is
FP32. Each packed arm must independently agree with the separate reference by
elementwise `torch.allclose(rtol=5e-3, atol=5e-3)` after FP32 conversion. The
stricter same-forward FP32 protected-diagnostic band is not used for these
cross-forward comparisons. The immutable v2 receipt remains readable historical
evidence; v3 does not recompute, emit, or gate on a parallel legacy
storage-dtype comparator.

The same-packed repeat is a measurability precondition, not a second candidate
or a source of adaptive tolerance. It must have the identical model state,
input, train mode, and RNG state as the primary packed arm, complete the exact
same 589-name trainable inventory, and satisfy global FP32 maximum absolute
gradient difference `<=2.5e-3`. Missing, extra, `None`, non-finite, or mismatched
name/shape/storage/gradient-dtype rows, or a larger repeat difference, produces
terminal `unmeasurable`. Both packed repetitions must still pass the independent
`5e-3/5e-3` comparison against the streaming separate arm; the implementation
may not choose the better repetition or widen a band after results.
The diagnostic process MUST start with `FLASH_ATTENTION_DETERMINISTIC=1` before
plan preparation, dependency/model import, Accelerator construction, or model
setup, and the immutable plan and receipt MUST bind that value. This is a
probe-only repeat-measurability control; it does not change or promote the
production training default or dependency/backend baseline.

The model-free authenticated v3 plan binds the structural inventory declaration
and strict owner/suffix rules: 196 LoRA-A, 196 LoRA-B, 196 DoRA magnitude, and
one shared special-token delta parameter (589 total). After CPU model, adapter,
and delta installation but before GPU setup, the runtime derives the exact
concrete names, shapes, storage dtypes, and expected gradient dtypes from the
installed model and authoritative adapter/delta receipts. That concrete
inventory is bound into the immutable attempt-start marker and terminal receipt,
then revalidated for every arm. Every expected gradient must be present and
finite in all three clean arms, with no extra parameter; exact-zero individual
tensors are allowed, while the complete surface must contain a nonzero aggregate
signal. Coverage and numerical parity are separate mandatory gates.

The one-packed/two-separate contrast uses a declared denominator projection,
not whole-record equality. Term inventory, `term_name`, denominator scope,
eligible/skipped segment counts, selected-atom counts, term weights,
normalizer/formula version, and the resulting planned-step denominator must be
identical. `context_count` is structural evidence and must be exactly `1` for
the packed arm and `2` for the shared separate arm; that one field is excluded
from cross-arm semantic equality. The authenticated projection is evaluated
before Accelerator construction, model loading, or either comparison forward.

The v3 plan additionally binds the exact parent v2 plan/model/sample and arm
identities while retaining the exact current resolved runtime-config identity.
The live compatibility reader uses
`coordexp-swift-wave2-config-compatibility-projection-v2`: it authenticates
current fingerprint
`da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`,
removes exactly the later strict defaults for the synchronous provider, the
source-order next-fit policy and its eight planner fields, and the complete
disabled-resume object, and MUST reproduce immutable parent-v2 fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`.
Missing, extra, or changed path/value rows fail closed; no identity field or
generic plan comparison is projected. The live resolved provider mode is
reasserted immediately before the attempt marker and GPU setup and is retained
in the plan, marker, and terminal receipt. Plan and receipt schemas are versioned as
`coordexp-swift-wave2-packed-parity-plan-v3` and
`coordexp-swift-wave2-packed-parity-receipt-v3`. Passed, failed, and
`unmeasurable` receipts preserve all completed evidence through one bounded
phase-state machine and absent-target atomic publication. The parent-v2 plan
bytes are retained durably at
`receipts/wave2-v2-parent-plan.json`; a temporary-directory copy is not an
authority or launch dependency. Exactly one audited
v3 real-model launch is allowed within the existing eight-GPU-hour Wave limit;
immediately before any real-model GPU setup, the command atomically publishes
one immutable absent-target attempt-start marker bound to the plan hash,
command/source/dependency identities, receipt target, and exact installed
589-row trainable inventory. A pre-existing marker rejects another invocation.
Marker publication consumes the attempt regardless
of later terminal outcome; a failure before publication does not claim model
execution and requires a fresh launch audit rather than a blind retry. The
terminal receipt references the marker. No automatic retry, workload
substitution, threshold change, or v4 is authorized by this decision.

The executed immutable v3 plan and failed receipt predate projection v2 and
remain bound to their historical v1 projection and fingerprint
`0f8fda29362a46e67cecccdd5fee7d7539fafc7d91b52f49b4d4b036556224a1`.
The current reader admits that receipt only when the known plan hash, complete
plan payload, receipt payload, `failed/qwen.parity.clean_failed` status, and
`comparisons` stage all match exactly. That historical payload alone may omit
the later-required `execution.requested_device` and
`execution.gpu_idle_preflight`; every newly produced rich receipt still
requires both fields. This is a read-only compatibility exception, not a plan
rewrite, schema relaxation, or result reinterpretation.

#### Post-result release disposition — 2026-08-10

The one v3 execution remains an immutable valid failure of its predeclared
cross-shape gradient comparison. It is not rescored, retried, or converted to a
passing receipt. The execution nevertheless established stronger forward
evidence than the release question required: all 138 supervised full-vocabulary
logit rows were byte-identical between packed and streaming-separate execution;
total loss, every mandatory loss-term scalar, denominator semantics, and
semantic-atom identity agreed; the same-packed repeat was bit-identical over all
589 gradients; the boundary-only negative was detected by logits and loss; and
the 28/28 text-layer FA2 proof passed.

The cross-shape backward comparison changed reduction shape and accumulation
grouping between one packed backward and two streaming backwards. Its 589-row
coverage receipt remains mandatory diagnostic evidence, but its numerical
equality result is no longer a Wave 2 release gate. In particular, 196 cold-start
LoRA-A gradients were identically zero because LoRA-B was zero, while the 393
signal-bearing rows showed deterministic BF16 reduction-order differences even
though supervised logits were byte-identical. Treating the inherited
`5e-3/5e-3` elementwise band as a semantic backward oracle would conflate
cross-shape numerical reproducibility with segment isolation.

The user therefore selected the narrow release claim: accept the demonstrated
packed forward, loss/denominator, boundary-isolation, and all-layer FA2
semantics; retain the packed-versus-streaming gradient comparison as a rejected
diagnostic; and release Wave 3 after the remaining artifact-fidelity finding is
fixed and independently audited. This decision does not assert exact Jacobian
equivalence, does not promote a wider gradient tolerance, and does not authorize
another Wave 2 GPU attempt or a v4 experiment.

Transformers' backend-selection and varlen utilities may be reused behind a
small adapter only when they preserve the explicit CoordExp boundary contract.
The repository-owned proof stays above that adapter so an upstream dispatch
change is observable.

### Decision 5: Compute zero-weight diagnostics outside autograd

When a protected auxiliary term has effective weight zero, its raw diagnostic
is computed in FP32 under `torch.no_grad()` or from detached inputs using the
same numerical formula. The differentiable reduction is not constructed and
the diagnostic is never added to total loss. Nonzero weights continue to use
the accepted differentiable path unchanged.

Acceptance compares raw diagnostics to the current reference, and total loss
plus every trainable gradient to a base-only reference. A production-shaped
memory/time receipt is required because removal of a graph is only a proposed
efficiency gain until the observed critical path confirms it.

The strict FP32 raw-diagnostic comparison reuses one frozen FP32 logits tensor,
targets, semantic-atom inventory, denominators, and weights for both paths; only
graph attachment versus detachment/no-grad may differ. A second model forward is
not an admissible strict-diagnostic reference because it would reintroduce BF16
forward variation.

The frozen Wave 3 plan retains current resolved config fingerprint
`da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`
and explicit `synchronous` provider mode. Its versioned v2 compatibility
projection removes the same exact 11 later-default rows as Wave 2 v2 and must
reproduce legacy frozen fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`;
the current identity, exact removed path/value inventory, projected identity,
and provider are independently validated and the live values are reasserted
immediately before the attempt marker and GPU setup.

Wave 3 plan, marker, terminal receipt, and publication-failure sidecar writes
reuse the repository strict JSON absent-target/no-replace publisher. The file
and containing directory are fsynced and the final bytes are strictly reloaded
and hash-validated. Post-link recovery requires both an own-link callback and
exact reloaded bytes; a collision, even with byte-identical content, never
establishes publication ownership, and a pre-link failure exposes no partial
final file.

The active Wave 3 plan, marker, and terminal receipt use their versioned v3
schemas. The plan binds the complete base-model weight identity: resolved model
root, safetensors index, every declared shard path, size and SHA-256, and the
aggregate SHA-256. The worker freshly recomputes and exact-compares that identity
before `load_model=True`; the marker, runtime evidence, receipt, and receipt plan
binding preserve the same identity. Run, controller, and worker entry points
strictly validate and bind the v3 plan before CUDA parsing, process creation, or
artifact publication. The immutable r2 v2 plan/marker/failed receipt remain
readable only through a fixed-path, fixed-raw-hash historical reader that returns
`historical_non_executable`; no v2 artifact is accepted by an execution entry
point. The unexecuted r3-v2 prelaunch plan is likewise preserved and rejected,
not overwritten or upgraded in place.

The sole fresh-root r3-v3 replacement then published its immutable marker and
terminated during one-rank Accelerator admission with
`failed/wave3.accelerator`, before any model forward or backward. This is an
infrastructure failure, not zero-weight numerical evidence, and it consumes the
replacement authorization. Source-only remediation now performs a CPU-only
pre-marker admission of fresh Accelerate/distributed/launcher state, pins the
exact selected `ACCELERATE_TORCH_DEVICE=cuda:<index>`, and after the marker
requires `DistributedType.NO`, rank/local-rank `0/0`, world size `1`, the exact
indexed CUDA device/current device, BF16 native AMP, accumulation `1`, and no
scaler. That remediation is CPU-tested and independently audited but is not a
runtime result; at that historical fixed point no r4, retry, or replacement was
authorized, and Wave 3 remained without its required GPU comparison.

On 2026-08-11 the user's instruction to continue the remaining smoke runs,
together with the explicit authorization to share the current eight GPUs,
superseded that historical launch hold for exactly one fresh Wave 3 v4
successor. The v4 successor MUST use a new absent root and marker, MUST retain
the immutable r2/r3 evidence as non-executable history, and MUST bind the live
config, source, cache, runtime, and full model-weight identities. Its selected
GPU admission and terminal evidence use the shared-process contract in this
change: exact 81920 MiB total memory, at most 49152 MiB pre-existing use, at
least 32768 MiB headroom, stable pre-existing driver PIDs, and mandatory
session-wide cleanup plus two post-run subset samples after every spawned-worker
outcome. This is a one-shot correctness/plumbing authorization, not a retry of
the consumed v3 root and not performance or efficiency promotion evidence.

Before that fresh v4 marker was published, the user subsequently dropped Wave
3-6 performance experiments and selected Wave 7 -> Wave 8 -> Wave 9 as the
production path. The unconsumed Wave 3 v4 launch authorization is therefore
retired. Its tested controller may remain as non-executed engineering evidence,
but it MUST NOT be prepared or run under this change and supplies no GPU,
memory, timing, or efficiency result.

Alternative considered: skip the diagnostic at weight zero. Rejected because
it would break observability and existing artifact expectations.

### Decision 6: Separate cache admission, rank hydration, and model setup phases

The converged launch path will have explicit phases, introduced at their owning
wave:

1. resolve strict configuration and executed provenance;
2. resolve cache determinants and expected fingerprints;
3. in Wave 1, validate both publication manifests, hydrate and validate the
   current rank's required training payloads, and fully validate every eval
   payload without retaining decoded eval state;
4. load model/adapters and prepare distributed optimizer state;
5. in Wave 4, replace the later retained full-eval hydration with retained
   rank-selective eval hydration after exact ordinal and metric parity is
   proved;
6. execute the first optimizer step and the accepted post-warm-up steps;
7. execute each scheduled evaluation under one separately aggregated receipt;
8. publish the terminal checkpoint.

Before Accelerator exists, the launcher-provided global rank and world size are
strictly validated and used only by a bounded CPU control plane. That control
plane lets the eventual rank zero select the one shared run directory, own the
RunWriter, converge cache-admission status, and finalize an admission failure.
After admission, Accelerator MUST resolve the same rank and world size before
any model work. The `cache_admission` artifact phase remains active until every
live rank converges Accelerator validation and identity through a bounded
post-Accelerator status boundary keyed by the already attested launcher
rank/world size. A rank-local mismatch therefore cannot let another rank enter
model work or a later collective. This is one continuous logical rank-zero
owner, not a second training backend or a second run tree.

The steady-state duration is the sum of the already all-rank-max-reduced
optimizer-step durations for the complete accepted post-warm-up sample. It is
not a continuous outer interval that also contains callback, evaluation, or
checkpoint work. Each scheduled evaluation retains an event-level duration and
per-rank measurement row; its phase summary aggregates only those events.
Resource counters observed after evaluation remain explicitly labeled as
process-lifetime high-water values, not evaluation-only incremental peaks.

Once the rank-zero artifact owner exists, ordinary pre-trainer phase failures
converge through the same bounded fixed-frame rank-report collective used by
the runtime gates. Every rank must report the same phase boundary before it is
accepted. This covers caught rank-local Python exceptions when all ranks can
reach the boundary; killed ranks and mismatched/blocking collectives remain
outside this receipt contract and are handled by launcher supervision.

Evaluation publication retains one canonical ordinal index. Each rank derives a
disjoint ordinal slice without changing the accepted modulo assignment. The
loader hydrates only indexed steps assigned to that rank; a coarse chunk may
still need to be read when every rank references that chunk, so chunk skips and
I/O reduction are measured outcomes rather than assumed benefits. Global
evaluation refuses partial coverage. Full hydration remains a test reference
until parity is proved, not a hidden fallback; a selective implementation that
does not improve retained RSS, deserialization work, or end-to-end startup is
not promoted on the strength of its name alone.

### Decision 7: Keep the synchronous provider as reference and compare three arms

Wave 5 compares:

- legacy fused device-direct preparation, reached when no provider owns the
  build/transfer lifecycle;
- the current synchronous provider, which already separates CPU construction
  from device transfer and remains the compatibility reference;
- the bounded overlapped provider with one finite lookahead.

All arms consume the same precomputed pack plan and cache identity. Receipts
compare pack IDs, tensors, loss components, optimizer steps, checkpoints, and
evaluation inputs. The benchmark uses the production eight-rank shape, paired
arm ordering, matched warm-up exclusions, and at least three independent paired
observations unless a predeclared failure or resource stop fires.

Before executing a candidate, the baseline receipt freezes the practical noise
band and promotion threshold. Promotion requires semantic equivalence, no
resource-bound violation, and a paired end-to-end wall-clock gain greater than
that frozen noise band. The lead owns the final default disposition and retains
synchronous when the evidence is inconclusive or the practical gain is absent.

Alternatives considered:

- promote on single-rank or per-batch preparation latency. Rejected because it
  does not measure distributed skew or time hidden behind GPU execution.
- unbounded producer queues. Rejected because they move latency into memory and
  complicate failure and resume semantics.

### Decision 8: Compare packing policies without silently changing research meaning

Three policy families share one repository-owned `PackPlan` receipt:

- `source_order_next_fit`: current compatibility default;
- `window_binpack`: deterministic fixed-window/chunk binpacking inspired by the
  ms-swift offline comparator;
- `online_window_binpack`: deterministic finite-lookahead streaming comparator
  inspired by ms-swift's iterable interval.

One encoded image example is the atomic planner item. The resolved sorted policy
owns the deterministic object/row order inside that image, and no packing policy
may reorder, split, or rewrite those rows. A planner may reorder whole image
examples across packs or batches and may change cross-image co-presentation.
That freedom is explicit user-approved scope, not evidence that any particular
policy is beneficial.

The receipt contains input ordinals, intra-image order identity, segment
lengths, pack membership, within-pack image order, utilization,
dropped/rejected examples, tie-breaker, seed, algorithm version, and cursor
state. The planner is evaluated first against already encoded lengths without
training. A changed-order candidate proceeds to matched training only after it
proves fixed intra-image order, deterministic each-once coverage, replayability,
and bounded resource use. The lead may promote it only after the frozen
semantic, quality, and end-to-end wall-clock gates pass.

Worker count is provenance-only if exact `PackPlan` equality is demonstrated
across counts. Otherwise it is a semantic fingerprint determinant. Online
packing has a finite lookahead and host-memory bound and must serialize its
pending window/cursor for exact resume; otherwise exact-resume configuration is
rejected.

Alternatives considered:

- import ms-swift's dataset classes directly. Rejected because generic template
  packing does not own CoordExp's renderer, visual replacement, loss mapping,
  cache publication, or exact resume contract.
- global epoch-wide binpacking. Rejected as the first production comparator due
  to order drift, startup latency, and unbounded plan materialization.

### Decision 9: Add a repository-owned, same-world-size exact-resume schema

Inference-minimal checkpoints and exact training checkpoints remain distinct.
Exact resume is opt-in and stores a versioned `training_state/` child alongside
the scheduled inference-loadable checkpoint. The exact state manifest includes
every trainable model surface (including adapters and selected embeddings),
optimizer, scheduler, applicable scaler, optimizer and accumulation position,
per-rank RNG, data/pack cursor, configuration and policy identities, world size,
trainable parameter signature, dependency compatibility fields, and checksums.
Frozen base-model weights are bound by immutable content identity and are not
copied into every exact checkpoint.

Publication uses a temporary checkpoint directory and a committed manifest only
after every rank-owned state is durable. Restore validates all compatibility
keys before mutating model or optimizer state. The first supported contract is
same world size and checkpoints at optimizer-step boundaries. Mid-accumulation
save is rejected unless pending gradients and the exact micro-step state are
later implemented and separately proved.

Accelerate state helpers may serialize low-level state, but the manifest and
compatibility admission remain repository-owned. Resume creates a new additive
run segment linked to its parent; it never appends ambiguously into completed
logs or overwrites an inference checkpoint.

The first contract performs no automatic pruning of committed exact-state
checkpoints. Storage is controlled by opt-in enablement and checkpoint cadence;
operator retention or pruning is a future explicit policy so an accepted resume
ancestor cannot disappear silently.

### Decision 10: Record executed source and binary provenance before optimization claims

The run receipt records Git commit and dirty state, a non-secret digest of
relevant local changes, and resolved versions/identities for Transformers,
FlashAttention, Torch, Accelerate, PEFT, and tokenizers. FlashAttention records
the imported module/binary identity and selected implementation; Transformers
records the imported source root and relevant owner identities. An inability to
resolve identity is explicit and blocks decision-grade backend comparisons.
The selected ms-swift checkout is recorded separately as a reference-only
comparison identity: the current training route does not import `swift`, so its
drift is visible provenance but is not a fail-closed runtime admission key.

Only compact digests and selected metadata enter normal run artifacts. Full
patches and environment dumps are optional external evidence and are never
copied automatically, avoiding credentials and oversized artifacts.

### Decision 11: Freeze dependency and attention-backend versions in this change

The installed Transformers, FlashAttention, Torch, CUDA, Accelerate, PEFT,
tokenizers, and FA2 route remain fixed throughout this change. Wave 8 verifies
that strict config and provenance prevent an incidental upgrade or backend
switch and records upstream comparison points for a future change. It does not
install or execute an upgraded stack. Any future candidate requires a separate
approved OpenSpec with isolated environment/cache identities and the complete
compatibility and end-to-end performance matrix.

The current admission fixed point is pinned runtime-baseline schema 3, digest
`cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`.
Its minimal CUDA initialization attestation is the r2 receipt at
`outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json`
with internal receipt SHA-256
`829d86ec8d977f30d37e8e979acba6527b3640180825e551785078a4876b1043`.
That receipt binds the actually mapped CUDA/Torch/cuDNN objects only; it loaded
no model, wrote no cache, and supplies no training-quality or throughput claim.
The earlier r1 receipt is historical pre-owner-expansion evidence, not a
current admission alternative.

## Risks / Trade-offs

- **Complete identities increase cache churn.** Mitigation: settle all owner
  changes before one authorized production materialization; use tiny temporary
  fixtures for mutation tests. Do not bulk-copy legacy roots into v3 and do not
  add automatic retention, catalog, lease, deduplication, or GC machinery in
  this change.
- **Preflight may duplicate lightweight processor work.** Mitigation: keep
  preflight model-free, measure it separately, and reuse resolved immutable
  identities in later setup.
- **All-layer tracing can perturb compiled execution.** Mitigation: use a
  bounded proof mode for admission/smoke, retain the numerical oracle, and keep
  tracing disabled in ordinary steady-state steps after proof succeeds.
- **Detached diagnostics can still be compute-heavy.** Mitigation: measure
  diagnostic-only cost; later frequency reduction would be a separate artifact
  contract, not part of this change.
- **Binpacking can improve utilization while changing optimization dynamics.**
  Mitigation: separate planner/utilization evidence from model-quality evidence
  while preserving fixed sorted object/row order inside every atomic image.
- **Overlap can increase host memory and nondeterministic failure surfaces.**
  Mitigation: one finite lookahead, explicit cancellation/error propagation,
  resource bounds, and synchronous reference fallback selected explicitly.
- **Exact checkpoints consume storage and synchronization time.** Mitigation:
  keep exact resume opt-in, save trainable rather than frozen base weights,
  support optimizer-boundary saves first, and benchmark publication separately.
- **Dirty-state provenance can leak content.** Mitigation: store only path-class
  metadata and digests in normal artifacts; never environment secrets or patch
  bodies.
- **Many waves can create partial combinations.** Mitigation: each accepted wave
  has a strict config/version identity and the final matrix tests all promoted
  defaults together.

## Migration Plan

1. Preserve the accepted Waves 0-2 identities and immutable historical
   evidence. Do not reinterpret or rerun their consumed attempts.
2. Keep the Wave 3 correctness implementation and Wave 5 synchronous default
   under their explicit no-promotion closures; launch no new performance arms.
3. Keep the Wave 6 CPU packet as pending future research, retain
   `source_order_next_fit`, and do not run matched training in this change.
4. Complete the delegated opt-in, same-world-size, optimizer-boundary Wave 7
   exact-resume proof, including post-run checkpoint/inference/retention
   evidence, under its one-shot shared-GPU contract.
5. With Wave 7 and retained production-default compatibility owners frozen,
   run the Wave 8
   pinned cache/parity/all-layer/protected-loss/resume compatibility matrix
   without changing dependencies or backend.
6. Before Wave 9 task 10.1, complete only Wave 4 task 5.7
   cleanup/verification without starting its dropped startup/resource
   experiment.
7. Produce the Wave 9 transition packet, then atomically materialize the one
   required cache under a previously absent v3 namespace/fingerprint and run
   the full eight-rank train/eval/checkpoint/resume convergence within the
   frozen cost envelope.
8. Update only accepted operator docs/defaults, run the complete gate and two
   final audits, and present the bounded evidence ledger before any separate
   archive action.
9. Rollback is wave-local: restore the prior strict-config default, keep
   artifacts and immutable cache versions for evidence, and remove only
   task-created temporary roots. Never delete old production caches as part of
   rollback.

## Open Questions

None. The user resolved ownership, ordering, exact-resume, dependency, cache,
and final-launch decisions on 2026-08-06. Implementation returns to the user
only if a wave would exceed the protected intra-image semantics, bounded resume
contract, fixed dependency baseline, or frozen cost envelope.
