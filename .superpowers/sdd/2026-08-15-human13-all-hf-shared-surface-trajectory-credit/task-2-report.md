# Task 2 report: live no-cache HF sampler and replay

## Status and scope

Task 2.1--2.5 are implemented and the OpenSpec task is checked.  The final
bounded live rereview is recorded against the durable no-update witness below;
all later OpenSpec tasks remain unchecked.  The implementation owns:

- `scripts/research/human13_hf_shared_surface_live.py`;
- `tests/research/test_human13_hf_shared_surface_live.py`.
- the BF16/cache normalization seam in `scripts/research/human13_live_model.py`
  and its matching live-model tests.

The exact seam consumes an existing `Human13LiveAssembly`, its BF16/FA2
plan/components, and the image-1584 processor skeleton.  Earlier Task-2
focused evidence was CPU/injected only; the production-shaped no-update
witness below is recorded separately and still adds no vLLM, prefix cache,
retry, fallback, update, or publication path.

## Delivered boundary

- One `HFSharedSurfaceSession` owns the existing prepared model object in eval
  mode and admits only the frozen image-1584 K16 plan, seed groups
  `35001..35016`, RP 1.0 before temperature 0.4, top-p 1.0, no top-k, cap 512,
  and `<|im_end|>`-only stopping.
- Sampling uses full prompt-plus-generated histories and literal
  `use_cache=False` forwards.  Stopped rows leave the active batch without
  reordering survivors.  Task-1 values bind request identity, causal histories,
  chosen raw logits/log-probabilities, active shapes, and connected per-request
  RNG-state transitions.
- Replay reconstructs every recorded sampler step with the same active request
  membership and causal history length, selects only each chosen causal
  position, runs the same HF repetition-penalty/temperature/top-p processors,
  and crosses Task-1 parity admission before retaining any live
  chosen-log-probability graph.  Non-reentrant activation checkpointing keeps
  the later single backward bounded; full-sequence vocabulary logits are not
  retained.
- Surface checks fail terminally on model/parameter/checkpoint/adapter/delta,
  train/eval, dtype/backend, prompt/image/tokenizer/processor, cache, causal
  shape, non-finite, and history/padding drift.  Failure has no retry or
  fallback.
- `SharedSurfaceResourceReceipt` binds the model/parameter/tokenizer/processor
  identities, processor order, sampling/replay/no-cache counts, prepared FP32
  logits, graph count, cleanup reason, double-close state, and retained live
  resource count.  Close clears graphs and session references.

The surface remains BF16 in parameter/autocast identity.  The prepared model's
observable logits are required to be FP32 because Accelerate wraps BF16 native
AMP outputs through its FP32 conversion boundary.

## TDD evidence

Initial RED before the production module existed:

```text
conda run -n ms pytest -q tests/research/test_human13_hf_shared_surface_live.py
Pytest: 0 passed, 18 failed
ModuleNotFoundError: No module named 'scripts.research.human13_hf_shared_surface_live'
```

Additional RED cycles preceded their implementation corrections:

- adapter, embedding-delta, and tokenizer-hash mutations produced three
  false admissions;
- checkpoint-path mutation produced one false admission;
- close receipt lacked the zero-retained-live-resource witness;
- an Accelerate-shaped BF16-parameter/FP32-output fake was rejected as if its
  logits had to remain BF16.

Final focused GREEN:

```text
conda run -n ms pytest -q tests/research/test_human13_hf_shared_surface_live.py
Pytest: 23 passed
```

Final adjacent GREEN:

```text
conda run -n ms pytest -q \
  tests/research/test_human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface_live.py \
  tests/research/test_human13_live_model.py
Pytest: 93 passed
```

## Static and review gates

```text
conda run -n ms ruff check <Task-1, Task-2, adjacent live-model files>
[]

conda run -n ms python -m compileall -q <same files>
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

git diff --check
(exit 0; no output)
```

At that historical checkpoint Serena diagnostics were unavailable in the active
tool surface and were not retried.  The later exact-worktree Serena/Pyright
diagnostics are recorded in the final gate below.

The independent review initially returned HOLD on one P1: Accelerate native
BF16 AMP keeps parameters/compute BF16 but converts prepared-model outputs to
FP32.  A local CPU reproduction confirmed
`parameter_dtype=torch.bfloat16`, `output_dtype=torch.float32`, and
`native_amp=True`.  The single correction bundle changed the fake and live
admission/receipt accordingly.  The same reviewer then returned
`P1_CLOSED -- PASS`; no other P0/P1 was found.

## Claim boundary and concern

The focused and adjacent sections above are injected CPU plumbing evidence.
The production-shaped no-update witness below establishes only real Qwen
BF16/FA2 sampler/replay parity and resource headroom; it does not establish
model quality, optimizer/backward behavior, checkpoint ownership, audit
behavior, or GPU readiness for the full vertical.

Commit: `3b814aba4cbd1badde04623ed18ec95b51042694`
(`research: add live HF shared-surface session`).  The ignored report itself is
kept as local execution evidence and is not part of the commit.

## Fix round 1: Task-2 rereview correction bundle

The bounded rereview in `task-2-rereview.md` found four systemic gaps.  At that
historical checkpoint OpenSpec 2.5 remained unchecked; the correction changed only the
Task-2 live module/test plus the exact live-model builder provenance seam and
its matching tests.

### RED and correction

The first focused regressions failed against the original implementation:

```text
conda run -n ms pytest -q tests/research/test_human13_hf_shared_surface_live.py \
  -k 'resource_receipt_is_sealed or wrong_language_surface'
Pytest: 0 passed, 2 failed
```

The failures proved that zero-work close incorrectly emitted `completed` and a
wrong language/vision trainable surface opened successfully.  The correction:

- seals `Human13LiveAssembly` only at the existing live builder after exact
  validation/loaded adapter and selected-delta hashes are checked, and
  re-admits that provenance at Task-2 open;
- matches actual trainable parameter names to the receipt's exact language
  DoRA group, requires the selected delta and vision/aligner surface frozen,
  binds the exact Source A1 plan/validation hashes, and checks prepared runtime
  model/optimizer/scheduler aliases;
- makes `SharedSurfaceResourceReceipt` a canonical admitted terminal value with
  content hash and strict `from_dict`, exact ordered seed/sample/replay coverage,
  forward/no-cache counts, zero retained graphs/session references, borrowed
  external ownership, cleanup outcome, and no caller-release claim;
- permits `completed` only after all four frozen groups were sampled and
  replayed in order; incomplete, missing-replay, duplicate/wrong-order, forged,
  copied/replaced, and tampered receipts fail closed; and
- terminalizes in `finally`, attempts `zero_grad` and `free_memory`
  independently, records cleanup subfailures, clears every session reference
  and graph, preserves an existing primary exception, and never reruns cleanup
  on double-close.

### Final evidence

```text
conda run -n ms pytest -q \
  tests/research/test_human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface_live.py \
  tests/research/test_human13_live_model.py
Pytest: 114 passed

conda run -n ms ruff check <Task-1, Task-2, adjacent live-model files>
[]

conda run -n ms python -m compileall -q <same files>
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

git diff --check
(exit 0; no output)
```

The standalone `pyright` executable is absent from the `ms` environment.
Serena's Python language server was activated for the exact worktree and its
Pyright diagnostics returned `{}` (zero error-severity diagnostics) for all
four touched source/test files.

All tests used injected CPU causal fakes and value-only plans/receipts.  No real
model load, CUDA/GPU action, network access, launch, output publication, vLLM,
prefix cache, retry, fallback, backward, or optimizer update occurred in that
historical correction gate; Task 2.5 was still unchecked at that point.

## Fix round 2: parameter provenance and nested lifecycle lineage

The second bounded rereview left T2-R1/T2-R2 open.  The user explicitly
authorized another correction round covering those fixable issues.  OpenSpec
2.5 and all later tasks remain unchanged.

### RED

Fresh adversarial tests were added before production changes.  They proved six
failures: post-builder mutation of the trainable adapter and frozen selected
delta was admitted; predecessor A1 was admitted instead of an owning vertical
plan; no owning plan builder existed; completed lifecycle serialization had no
nested Task-1 groups; and a canonical outer rehash could fabricate completed
K16 evidence.

```text
conda run -n ms pytest -q <round-2 focused selectors>
Pytest: 0 passed, 6 failed
```

### Correction

- Added the value-only all-HF vertical Source plan with unit
  `2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical`, arm
  `C-One-Image`, BF16/FA2 language DoRA, frozen selected delta, AdamW
  `3e-6`, scheduler horizon one, and pre/post milestones `(0,1)`.  Existing
  predecessor plan builders and consumers remain valid and unchanged.
- Builder-issued `Human13LiveAssembly` now carries canonical parameter-state
  provenance for every named prepared-model parameter: name, shape, dtype,
  `requires_grad`, tensor version, and exact CPU-byte SHA-256.  The exact named
  selected-token delta has an additional frozen/content binding.  The builder
  seals that state; downstream admission recomputes it before constructing the
  HF shared-surface identity, so post-builder value or trainability mutation is
  rejected rather than snapshotted as a new Source.
- `SharedSurfaceResourceReceipt` now embeds the admitted Task-1
  `SampledHFGroup` and `GradientReplayGroup` values.  Admission/reload checks
  exact shared identity, group index, seed order, sampled-to-replay lineage,
  nested admission seals, and derives group hashes plus completed sample/replay
  forward counts from those nested values.  Arbitrary hashes, zero-forward
  completed evidence, nested tamper, derived-hash rehash, copy/replace, and
  wrong order fail; a valid four-group terminal receipt serializes and reloads.
- Borrowed external assembly ownership, finally-safe cleanup, primary-error
  preservation, one terminal cleanup attempt, and double-close behavior remain
  unchanged.

### Verification and boundary

```text
conda run -n ms pytest -q \
  tests/research/test_human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface_live.py \
  tests/research/test_human13_live_model.py
Pytest: 122 passed

conda run -n ms ruff check <six Task-1/Task-2/live-model paths>
[]

conda run -n ms python -m compileall -q <same six paths>
(exit 0; no output)

Serena/Pyright error-severity diagnostics
{} for all four changed source/test files

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

git diff --check
(exit 0; no output)
```

This remains injected CPU/value-contract evidence.  It does not prove a real
Qwen load, CUDA RNG, BF16/FA2 forward/backward, GPU memory headroom, live
numerical parity, optimizer update, or model quality.  No model, GPU, network,
launch, output, cache, vLLM, retry, fallback, backward, or update action ran.
Task 2.5 was still unchecked until the fresh bounded rereview below was clean.

## Fix round 3: production-shaped no-update parity witness

The original padded full-history replay was not viable on the real Qwen
BF16/FA2 surface: it produced large survivor-logit drift, and retaining the
full-vocabulary grad output exhausted GPU 0.  A bounded TDD correction therefore
made replay reproduce every recorded sampler step, pass position-selective
`logits_to_keep`, and use non-reentrant activation checkpointing.  The focused
live suite now has 50 passing tests, including delayed-stop shape parity and
checkpoint invocation regressions.

On 2026-08-16, the real public assembly boundary loaded the frozen image-1584
Source model/adapter/selected-token delta, encoded the real prompt skeleton,
and ran all four K16 groups on GPU 0.  Every group passed Task-1 parity with
`max_abs_error=0.0` and `mean_abs_error=0.0`.  The terminal resource receipt
was:

```text
sample_forward_count=463
replay_forward_count=463
total_forward_count=926
no_cache_forward_count=926
cleanup_state=closed
cleanup_call_count=1
retained_graph_count=0
session_held_reference_count=0
```

The run performed no backward, optimizer step, private checkpoint write,
HF-fp32 audit load, network action, or output-root write.  The durable raw
receipt is recorded at
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-16-human13-all-hf-shared-surface-trajectory-credit-vertical/no-update-k16-parity-witness.md`.
The configured output root was not written by this run; a pre-existing stale
`run-reservation.json` for PID 377949 remains separately documented there.
Together with the final focused/static/reviewer gates, this closes Task 2.5 and
the real Task-2 no-update parity/resource witness.  Task 4.6 and Task 5 remain
open.

## Final bounded rereview and durable evidence

The follow-up review checked the durable witness above rather than relying on
the earlier report-only line.  Current evidence is:

```text
focused live suite: 50 passed
adjacent HF/shared/live-model suite: 124 passed
Pyright --level error: 0 errors, 0 warnings, 0 informations
Ruff, compileall, strict OpenSpec, and git diff --check: clean
```

The reviewer found no remaining Task-2 P0/P1.  Two non-decision-bearing
hardening notes remain recorded for a later schema revision: failed receipts do
not yet encode per-step partial-attempt counters, and the terminal value receipt
retains the nested immutable lineage even after the session clears its own
group lists.  Neither changes the completed K16 receipt or its exact 463/463
counts.
