## Context

The repeated execution contract is narrower than a “research probe.” Four
committed HF scripts independently materialize one multimodal request through
`HFBackendSession._materialize_native_inputs`, append literal persisted token
IDs, call a raw model forward, and extract chosen-token evidence. The two pilot
callers exercise different shapes:

- `run_continuation_locality_boundary_scoring.py` observes two alternative
  one-token continuations at a fixed row boundary;
- `run_exact_prefix_owner_compositionality.py` additionally scores a complete
  multi-token candidate row and reduces its token evidence by experiment-owned
  row phases.

The backend already owns request materialization, loaded HF objects, runtime
lifecycle, and raw likelihood semantics. It also contains
`teacher_forced_chosen_token_logprobs`, but that function requires the private
model and native prompt mapping, so it is not a usable caller seam. The current
research forward path also derives Qwen M-RoPE positions explicitly over the
full sequence; a public wrapper cannot assume that merely extending an
attention mask preserves this behavior.

An independent Opus 5 audit rejected the draft artifact-verification
capability. Canonical inference manifests do not provide a common `run_id`,
completed `terminal_status`, or per-artifact digest map, while continuation
locality receipts inline records and bind external inputs rather than their own
bytes. A projector would therefore invent identity rather than verify it. The
selected reducers also do not have a demonstrated byte-binding failure that a
new handle would prevent.

The same audit executed the explicit research and rollout-calibration suites:
711 tests were collected, 670 passed, and 41 failed. The failures separate into
38 mutable `unit.md` hash drifts and 3 processed-data provenance drifts. They
are pre-existing evidence-governance issues, not justification for expanding
this runtime change.

## Goals / Non-Goals

**Goals:**

- Put single-request HF materialization, literal token-history extension,
  Qwen position construction, raw chosen-token likelihood, vocabulary rank,
  session ownership, and cleanup behind one HF-owned interface.
- Preserve exact executed prompt IDs, caller-supplied continuation IDs, token
  order, and SHA-256 without decode/re-tokenize cycles.
- Support both one-token boundary evidence and multi-token teacher-forced
  evidence with the same operation.
- Let the two pilot callers reproduce their existing token hashes, FP32
  log-probabilities, ranks, and derived receipt fields exactly.
- Expose observed dtype and attention facts in the existing backend receipt so
  research callers do not inspect the raw model merely to report them.

**Non-Goals:**

- A generic `Probe`, `Case`, `Condition`, `Intervention`, `Observation`, runner,
  plugin registry, DAG, or declarative research schema.
- Artifact-family normalization, verified-run handles, new receipt status or
  identity vocabulary, or offline reducer infrastructure.
- Backend-neutral exact-history support or a vLLM implementation.
- Full logits, hidden states, feature hooks, suffix generation, forced release,
  stopping criteria, parser evidence, matching, owner ledgers, row phases, or
  statistical reduction.
- Choosing prompts, prefixes, candidates, controls, cohorts, owner policies,
  estimands, uncertainty, transfer claims, preservation risks, or stop rules.
- Migrating `run_complete_candidate_row_scoring.py`,
  `run_paired_terminal_forced_opener_release.py`, completed artifacts, or all
  private accesses in the two pilot scripts. Vision-parity and experiment-local
  release machinery remain out of scope unless separately justified.
- Fixing or re-pinning the 41 known research/provenance test failures.

## Decisions

### 1. Publish an HF-session evidence seam, not a probe abstraction

The behavioral interface is:

```python
@dataclass(frozen=True)
class HFExactHistory:
    request_id: str
    conditioning_token_ids: tuple[int, ...]
    conditioning_token_ids_sha256: str
    # Session binding and native multimodal state are private.

@dataclass(frozen=True)
class HFChosenTokenEvidence:
    token_id: int
    raw_model_logprob: float
    candidate_vocab_rank: int

class HFBackendSession:
    @property
    def special_token_ids(self) -> Mapping[str, int]: ...

    def prepare_exact_history(self, request: DecodeRequest) -> HFExactHistory: ...

    def extend_exact_history(
        self,
        history: HFExactHistory,
        token_ids: Sequence[int],
    ) -> HFExactHistory: ...

    def teacher_forced_evidence(
        self,
        history: HFExactHistory,
        continuation_token_ids: Sequence[int],
    ) -> tuple[HFChosenTokenEvidence, ...]: ...
```

`special_token_ids` exposes only backend-owned decode tokens such as `im_end`
and `pad`; dense-enumeration row-entry and coordinate IDs remain caller-owned.
There is no text-encoding method: both selected consumers already own exact
token IDs, and adding a text path would weaken the no-retokenization contract.
There is no public suffix-decode method because row stopping and release are
scientific intervention semantics in the current consumers.

The dataclasses expose audit-safe scalar/tuple evidence only. Callers cannot
construct a usable history by copying those fields: the session retains an
unforgeable internal binding and the private multimodal context. Reuse with a
different or closed session fails through the existing `RuntimeContractError`
family before a model call. No second error taxonomy is introduced.

**Alternatives considered:**

- Keep a utility in `scripts/research/`: this preserves experiment locality but
  leaves native model/input lifecycle knowledge duplicated across four real
  consumers.
- Add the methods to backend-neutral `BackendSession`: rejected because vLLM
  has no equivalent full-vocabulary, exact-history forward contract.
- Build a composable runner around session/condition/intervention/observation:
  rejected because it would flatten scientific controls and stopping semantics
  that are intentionally different in the existing scripts.

### 2. Histories append IDs only; attention and positions materialize at use

`prepare_exact_history` reuses the existing one-request processor validation
and records the executed prompt IDs as the initial conditioning sequence. The
session keeps the native image inputs in private storage keyed by the handle.

`extend_exact_history` validates integer vocabulary IDs, concatenates them in
order, computes the canonical token-ID SHA-256, and returns a new immutable
handle. It does not mutate its parent and does not claim to update attention or
position tensors.

`teacher_forced_evidence` constructs the full conditioning-plus-continuation
sequence at the point of use, derives the current Qwen M-RoPE position IDs over
that full sequence, and performs one raw forward under inference mode. This
preserves the semantics of the existing `_forward_logits` path. The existing
free teacher-forced helper may be reused only if tests prove identical position
and likelihood behavior; mechanically wrapping it is not itself acceptance.

For each continuation position, the operation returns FP32 raw-model
log-probability and rank `1 + count(logit > selected_logit)`. Ties therefore
share a rank. It never returns the full logits. A boundary comparison is two
one-token calls on the same immutable history; a candidate row is one
multi-token call. The first implementation does not add cross-call caching or
batching; measured need can justify that separately.

### 3. Keep scientific reductions in the two callers

The boundary caller continues to choose the row-entry and terminal token IDs
and compute their margin. The compositionality caller continues to define row
tokens, row phases, phase sums/means, candidate metadata, owner interpretation,
release interventions, matching, and its claim boundary. Migration replaces
only how raw token evidence is obtained.

Real caller shapes are intentionally plain:

```python
history = session.prepare_exact_history(request)
history = session.extend_exact_history(history, item["prefix_token_ids"])
opener = session.teacher_forced_evidence(history, (OBJECT_REF_START,))[0]
terminal = session.teacher_forced_evidence(
    history, (session.special_token_ids["im_end"],)
)[0]
# Caller computes opener.raw_model_logprob - terminal.raw_model_logprob.
```

```python
history = session.prepare_exact_history(request)
history = session.extend_exact_history(history, case["prefix_token_ids"])
tokens = tuple(case["target"]["row_token_ids"])
evidence = session.teacher_forced_evidence(history, tokens)
# Caller retains row-phase grouping, sums, means, metadata, and interpretation.
```

### 4. Add observed runtime facts to the existing receipt, without new identity

The HF receipt's `effective_settings` gains:

- `observed_model_dtype`: the current parameter-dtype names and element counts,
  or `null` when the loaded runtime cannot establish them;
- `observed_attn_implementation`: the value observed on the loaded model, or
  `null` when unavailable.

These names distinguish runtime observation from `BackendLaunch.model_dtype`
and configured backend options. The implementation must not relabel a declared
setting as observed. The pilot smoke requires non-null values and parity with
the existing script-local observations. No new run fingerprint or duplicate
receipt identity is added.

### 5. Use two committed consumers and exact parity as the promotion gate

The first real smoke uses the frozen Source config and existing immutable
baselines:

- locality case
  `same_image_near_continue-image-10040-depth-1-16497c21af80`, whose prefix
  SHA-256 is
  `16497c21af806b3b5d9386238614d807d7059f8cb7781d028e621d4e706021fa`;
- owner case
  `owner-case-image-225458-depth-1-owner-225458-78480-prefix-eb93c607ad0f`,
  whose prefix and candidate-row SHA-256 values are respectively
  `eb93c607ad0f2b32957eb6ab070031ae1231d6210ea5973ddfa41e3f8b89c0e2`
  and
  `216b5c1561c30877ba4f64c744281bdc0794e0a3e57a40d131c7ae3f3f397509`.

Both consumer shapes are checked for Source and transition step 36 with:

- Source config
  `configs/coordexp_infras/infer/research/qwen3_vl_2b_row_local_long_promotion_64_max3084_b4_v2/row-local-long-source-64-max3084-b4-hf.yaml`;
- transition config
  `configs/coordexp_infras/infer/research/qwen3_vl_2b_row_local_long_promotion_64_max3084_b4_v2/row-local-long-transition-lr1e5-step-36-64-hf.yaml`;
- manifest
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality/panel-v1/manifest.json`;
- source JSONL
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/candidate-pool-v1/candidate-pool-2432.coord.jsonl`;
  and
- FP32 runtime mode.

New outputs go to a temporary run root and never overwrite the historical
receipts.

On the same installed runtime and checkpoint, acceptance requires exact prompt,
prefix, and candidate token sequences/hashes; exact per-token FP32
log-probabilities and vocabulary ranks; and exact caller-derived boundary and
row-phase values. If exact equality cannot be reproduced, implementation stops
before deleting the private path. A tolerance or changed estimand requires an
explicitly revised change, not an ad hoc test relaxation.

The two other committed HF consumers remain untouched. They are evidence that
the seam may have more reuse, not migration scope.

### 6. Treat the known failing suites as routed baseline evidence

Before code work, rerun the explicit suite in `ms`, inventory exact node IDs,
and classify them against the two known classes. The 38 narrative-hash failures
and 3 processed-data provenance failures are recorded as pre-existing and
routed to their current owners. This change does not repair, skip, re-pin, or
add them to default collection. A third failure class is a stop condition
because it would invalidate the reviewed baseline.

Only new interface tests under `tests/inference/` are hard gates for this
change. Structural `openspec validate --strict` is necessary planning
validation but is not runtime evidence.

## Risks / Trade-offs

- **The two consumers may not share exact position semantics.** → Prove both
  against their old private path before deletion; retract the public seam if
  either needs full logits, custom hooks, native stopping, or a different
  history representation.
- **Two one-token forwards are slower than one boundary-logit forward.** → Do
  not add caching or a multi-candidate API before measuring the representative
  smoke. Optimize inside the session later only if it preserves evidence.
- **An opaque session registry adds lifecycle state.** → Keep histories
  single-session and immutable, clear private state on `close`, and test
  cross-session/closed-session failures before model execution.
- **Observed runtime fields may be unavailable on another transformers model.**
  → Record `null` rather than substituting configured values; current Qwen
  pilot requires concrete observations.
- **Stable spec surface may be premature.** → The change is accepted only after
  two real consumers pass exact parity. Failure retracts the delta rather than
  leaving a one-consumer abstraction.
- **Pilot scripts retain other private accesses.** → Delete only scoring and
  runtime-observation accesses superseded by this seam. Vision parity and
  release-generation machinery stay visible and become separate evidence for
  any future seam.

## Migration Plan

1. Wait for explicit user approval; planning artifacts alone authorize no code
   or GPU execution.
2. Reproduce and record the two-class 711-test baseline. Stop on a third class.
3. Add failing `tests/inference/` contract tests for preparation, immutable
   append, lifecycle binding, at-use position construction, token evidence,
   ranks, and observed receipt fields.
4. Implement the HF-only records and methods while leaving existing callers on
   their private path.
5. Migrate the named locality case and prove exact parity in a temporary output
   root; then migrate the named owner case and prove exact multi-token parity.
6. Remove only the now-superseded private scoring/runtime-observation accesses
   in those two scripts. Keep all experiment semantics and historical outputs.
7. Run the first independent gate over the interface plus both consumers.
8. Run focused/default tests, the two real smokes, strict OpenSpec validation,
   residue inspection, and a final independent audit. Stop for user review
   before archive or any further consumer migration.

Rollback is deletion of the additive session surface and restoration of the
two caller-local accesses before archive. No persisted artifact or stable
configuration migration is involved.

## Open Questions

- Will at-use Qwen M-RoPE materialization reproduce both existing private
  forward paths bit-for-bit on the installed runtime?
- Does the measured duplicate boundary-forward cost justify a narrow
  multi-candidate observation later, or is the simpler API sufficient?
- After the two pilots, do the untouched complete-candidate and paired-release
  scripts actually fit this evidence-only seam without exposing new semantics?
  They are evidence to inspect later, not scope for this change.
