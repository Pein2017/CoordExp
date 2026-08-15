# Task 2 report: live no-cache HF sampler and replay

## Status and scope

Task 2.1--2.4 are implemented.  Task 2.5 and all later OpenSpec tasks remain
unchecked.  The implementation creates only:

- `scripts/research/human13_hf_shared_surface_live.py`;
- `tests/research/test_human13_hf_shared_surface_live.py`.

No `human13_live_model.py` change was needed: the exact seam consumes an
existing `Human13LiveAssembly`, its BF16/FA2 plan/components, and the
image-1584 processor skeleton.  No real model was loaded, no GPU/network/output
action ran, and no vLLM, prefix cache, retry, fallback, update, or publication
path was added.

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
- Replay performs one padded teacher-forced group forward with gradients
  enabled, vectorizes every `(row, causal_position)` gather, runs the same HF
  repetition-penalty/temperature/top-p processors, and crosses Task-1 parity
  admission before retaining any live chosen-log-probability graph.
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

Serena diagnostics were unavailable in the active tool surface and were not
retried.  This is one reason the broader Task 2.5 gate remains unchecked.

The independent review initially returned HOLD on one P1: Accelerate native
BF16 AMP keeps parameters/compute BF16 but converts prepared-model outputs to
FP32.  A local CPU reproduction confirmed
`parameter_dtype=torch.bfloat16`, `output_dtype=torch.float32`, and
`native_amp=True`.  The single correction bundle changed the fake and live
admission/receipt accordingly.  The same reviewer then returned
`P1_CLOSED -- PASS`; no other P0/P1 was found.

## Claim boundary and concern

This is injected CPU plumbing evidence only.  It does not establish a real
Qwen BF16/FA2 forward, CUDA RNG behavior, FA2 padded replay compatibility,
memory headroom, numerical parity, model quality, optimizer/backward behavior,
or GPU readiness.  Those remain owned by later reviewed tasks; no live action
was taken here.

Commit: `3b814aba4cbd1badde04623ed18ec95b51042694`
(`research: add live HF shared-surface session`).  The ignored report itself is
kept as local execution evidence and is not part of the commit.

## Fix round 1: Task-2 rereview correction bundle

The bounded rereview in `task-2-rereview.md` found four systemic gaps.  This
single correction bundle keeps OpenSpec 2.5 unchecked and changes only the
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
prefix cache, retry, fallback, backward, or optimizer update occurred.  This is
still plumbing evidence only; Task 2.5 remains unchecked pending rereview.

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
Task 2.5 remains unchecked until the fresh bounded rereview is clean.
