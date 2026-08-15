# Task 1 report: Human-13 all-HF shared-surface contracts

## Status

DONE

## Scope and commit

- Commit: `383db17cfaa2caa71132646b8435648b7e663eec`
  (`research: add all-HF shared-surface contracts`).
- Implemented only the CPU/value-contract foundation in
  `scripts/research/human13_hf_shared_surface.py`, its focused tests, and the
  Task 1.1--1.4 checkboxes.  Task 1.5 remains unchecked for the independent
  review owner.
- No torch, transformers, accelerate, vLLM, model, or GPU import/action was
  added.  The pre-existing untracked
  `memories/notes/2026-08-14-scalable-k-trajectory-successor-direction.md`
  was not touched.

## Delivered contract

- Frozen image-1584 K16 plan: four ordered groups of seeds
  `35001..35004`, `35005..35008`, `35009..35012`, and `35013..35016`.
- Immutable BF16/FA2/eval/no-cache identity and immutable policy binding:
  RP 1.0, RP-before-temperature, prompt-inclusive history, temperature 0.4,
  top-p 1.0, no top-k, cap 512, and `<|im_end|>` only.
- One `_admit(...)` choke point creates consumer-usable sampled groups, parity
  receipts, and replay groups.  Sampling admission binds the expected live
  surface; replay admission separately binds the replay identity to the sampled
  identity.
- Exact `(request_id, token_index, history_sha256, chosen_token_id)` replay
  lineage, finite processed log probabilities, canonical strict-JSON content
  hashes, and frozen `max <= 0.02`, `mean <= 0.002` parity admission.
- Pure sign-aware RP-before-temperature scalar logic plus a zero-action K16
  resource estimate/dry-run receipt: 2,048 batched image/prompt forwards,
  four replay forwards, one backward, 8,192 maximum generated tokens, required
  GPU role labels, and two distinct output roots.

## TDD evidence

### RED

The required first command was run before the production module existed:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
```

It failed at collection as expected.  The captured pytest output was:

```text
ModuleNotFoundError: No module named 'scripts.research.human13_hf_shared_surface'
```

Two subsequent focused RED cycles also preceded their implementation changes:

- resource-estimate test collection failed with `cannot import name
  'HFSharedSurfaceResourceEstimate'`;
- replay-surface test failed with `TypeError: admit_gradient_replay() got an
  unexpected keyword argument 'replay_identity'`.

### GREEN and static gates

Final commands and observed outputs:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 24 passed

conda run -n ms ruff check scripts/research/human13_hf_shared_surface.py tests/research/test_human13_hf_shared_surface.py
[]

conda run -n ms python -m compileall -q scripts/research/human13_hf_shared_surface.py
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid
```

Additional completion checks:

```text
git diff --check
(exit 0; no output)

rg import-absence check for torch|transformers|accelerate|vllm
(exit 0; no matches)

git diff --cached --check
(exit 0; no output)
```

## Coverage and concerns

- The failure-mode matrix covers model-object identity, parameter digest,
  adapter/delta, dtype, attention backend, eval mode, tokenizer/prompt/image,
  seed coverage/order, history/token lineage, processor order, no-cache,
  finite values, replay-surface identity, and max/mean parity thresholds.
- Canonical round-trip, forged content hash, and malformed nested-value cases
  are covered.
- No stop concern.  The contract deliberately records static resource bounds;
  it does not reserve GPUs, create output roots, or execute a live HF session.

## Fix round 1: independent-review correction bundle

### Status and commit

DONE — `e1263285db60a062711a707e047d9aad0f8f6c3d`
(`research: harden all-HF shared-surface contracts`).  This correction leaves
Task 1.5 unchecked; it does not claim the independent re-review is complete.

Changed committed paths:

- `scripts/research/human13_hf_shared_surface.py`
- `tests/research/test_human13_hf_shared_surface.py`

The active OpenSpec task file was explicitly included in the staging command
but had no correction-round content change.  The unrelated untracked memory
note was not touched.

### TDD evidence

Before changing the implementation, the expanded focused suite was written and
run with:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
```

The previous implementation failed at collection because it did not export the
new Task-2-facing value evidence (`HFActiveBatchStep`), producing the expected
missing-feature RED (`Pytest: No tests collected`, exit 127 through the Conda
wrapper).  The first implementation run then exposed the intended exact-mean
parity boundary failure; decimal-preserving error construction was added before
the final GREEN run.

Final commands and outputs:

```text
git diff --check
(exit 0; no output)

conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 33 passed

conda run -n ms ruff check scripts/research/human13_hf_shared_surface.py tests/research/test_human13_hf_shared_surface.py
[]

conda run -n ms python -m compileall -q scripts/research/human13_hf_shared_surface.py
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

import-absence check for torch|transformers|accelerate|vllm
(exit 0; no matches)

git diff --cached --check
(exit 0; no output)
```

### Corrected contracts

- Replaced the public `_admitted` boolean with an identity-bound, hash-checked
  sealing registry.  Constructors have no admission argument; direct values and
  every `dataclasses.replace` copy fail before serialization/consumption.
- Requests now require observed non-empty request IDs, non-negative chosen token
  IDs, 1--512 tokens, canonical prompt-inclusive causal history, and exact
  request/token identity.  Token 0 binds to the prompt history and later token
  histories hash the exact prior generated IDs.
- Sampled groups now bind active-batch history/shape and RNG transitions;
  requests bind stop reason and raw chosen logit; replay binds processor order
  and causal chosen-token gathering.  Parity receipts retain the complete
  absolute-error distribution as well as max and mean.
- Added the sealed canonical `HFSharedSurfaceCloseReceipt`, sealed/reloadable
  resource and dry-run receipts, forged-hash/nonzero-action/root-mismatch tests,
  and the active Task-1 failure-mode matrix with executable counterexamples for
  surface, lineage, processor, parity, denominator, optimizer delta, audit,
  and rollback owners.

### Remaining concern

No implementation stop concern.  The new objective/optimizer/audit/rollback
entries are deliberately pure handoff guards and matrix owners, not execution;
their live behavior remains owned by later OpenSpec tasks.  A localized
independent re-review is still required by unchecked Task 1.5.

## Fix round 2: scoped re-review correction bundle

### Status and commit

DONE — `d3fadf25919e6f17482768167883d65c43740b14`
(`research: tighten all-HF shared-surface evidence`).  Task 1.5 remains
unchecked: this is a repair response, not a claim that the re-review itself is
complete.

Committed paths:

- `scripts/research/human13_hf_shared_surface.py`
- `tests/research/test_human13_hf_shared_surface.py`

The OpenSpec task file was explicitly included in the stage command but had no
content change.  The unrelated untracked memory note was preserved.

### TDD evidence

The new adversarial tests were added before implementation and run with:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 31 passed, 5 failed
```

The RED failures included the review's false-admission conditions: reversed or
duplicated active steps did not raise, a three-dimensional batch shape was
admitted, shifted replay/gather positions were admitted, and the proposed
replay-bound close API was unavailable.  The exact-boundary fixture was then
corrected to publish three active steps for three-token requests before the
implementation green run.

Final commands and outputs:

```text
git diff --check
(exit 0; no output)

conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 37 passed

conda run -n ms ruff check scripts/research/human13_hf_shared_surface.py tests/research/test_human13_hf_shared_surface.py
[]

conda run -n ms python -m compileall -q scripts/research/human13_hf_shared_surface.py
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

import-absence check for torch|transformers|accelerate|vllm
(exit 0; no matches)

git diff --cached --check
(exit 0; no output)
```

### Corrected evidence contracts

- Active-batch steps now require exactly two integer dimensions, token indexes
  in exact `0..max-1` order, one nonduplicated step per index, exact request
  membership/order and causal-history order derived from sampled requests, and
  connected RNG transitions between adjacent steps.
- Replay causal-gather positions must agree with both sampled and replayed
  tokens, preventing a coordinated caller-side offset from being sealed.
- Close admission now receives an admitted `GradientReplayGroup`, derives both
  identity and replay hash from it, and requires exact
  `HFSharedSurfaceIdentity` construction; policy/duck identity substitutions
  fail before publication.
- Added explicit `0`/`1` token and isolated max-parity (`0.02` accepted,
  `0.020001` rejected while mean remains within limit) boundaries.
- The failure-mode matrix now names the actual later owner symbols and uses a
  CPU-only explicit owner-binding adapter to freeze their minimal invalid input
  contracts without importing or executing Torch/model/runtime paths.

### Remaining concern

No implementation stop concern.  The owner-binding adapter is intentionally a
pure Task-1 handoff seam; it does not execute later-wave objective, projected
apply, audit, checkpoint, or transaction runtime behavior.  Those executions
remain out of scope and Task 1.5 still requires independent re-review.

## Fix round 3: close lineage and source-resolved owner matrix

### Status and commit

DONE — `e6694dfdcc85aa9aade9702445cc463746c0cecf`
(`research: seal all-HF close lineage`).  Task 1.5 remains unchecked.

Committed paths:

- `scripts/research/human13_hf_shared_surface.py`
- `tests/research/test_human13_hf_shared_surface.py`

The OpenSpec task path was explicitly staged but did not change.  The unrelated
untracked memory note was not touched.

### TDD evidence

The nested close replay, owner-resolution, and malformed active-request tests
were added before their implementation.  The first focused RED command was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 38 passed, 2 failed
```

The failures were the expected missing `HFSharedSurfaceOwnerBinding.resolve()`
and missing `replay_group` close payload.  The malformed active-request fixture
was narrowed to a one-member shape so it specifically reaches the former
unhashable-element path rather than an earlier length guard.

Final commands and outputs:

```text
git diff --check
(exit 0; no output)

conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
Pytest: 40 passed

conda run -n ms ruff check scripts/research/human13_hf_shared_surface.py tests/research/test_human13_hf_shared_surface.py
[]

conda run -n ms python -m compileall -q scripts/research/human13_hf_shared_surface.py
(exit 0; no output)

openspec validate add-human13-all-hf-shared-surface-trajectory-credit-vertical --strict
Change 'add-human13-all-hf-shared-surface-trajectory-credit-vertical' is valid

import-absence check for torch|transformers|accelerate|vllm
(exit 0; no matches)

git diff --cached --check
(exit 0; no output)
```

### Corrected contracts

- Close payloads now embed the complete admitted `GradientReplayGroup` canonical
  value.  Reload reconstructs that nested replay, derives/compares its exact
  identity and content hash, and rejects a canonically rehashed A-identity /
  B-replay substitution before sealing or `to_dict()`.
- `admit_shared_surface_close` still accepts only an admitted replay group and
  derives both close identity and replay hash from it.  Direct close construction
  requires exact identity and replay types.
- The failure-mode owner bindings now parse the named repository source with
  `ast`, resolve the real function/class-method seam, require its actual guard
  fragments, and expose a value-only callable that accepts only the frozen real
  invalid input.  Missing symbols and resolved non-guard symbols fail tests.
  The audit seam is now `PrivateCheckpointRef.__post_init__`; the rollback seam
  names `TrainingStateTransaction.reject` with its real `snapshot` input.
- `HFActiveBatchStep` validates request-id/history element types before set or
  other hashable-only operations, so `active_request_ids=[[]]` raises
  `SharedSurfaceContractError` consistently.

### Remaining concern

No implementation stop concern.  Owner resolution deliberately reads/parses
source instead of importing/starting later Torch/model/runtime owners, keeping
Task 1 CPU/value-only.  Its purpose is to fail on missing or non-guard seams;
Task 3/4 remain responsible for executing those later behaviors.  Independent
Task 1.5 re-review is still required.

## Fix round 4: actual source-guard evaluation

### Status and scope

Final repair bundle for T1-R3.  Task 1.5 remains unchecked.  Owned paths are
the CPU-only contract module, its focused tests, and this report; the unrelated
untracked memory note remains outside the bundle.

### RED evidence

The owner-resolution/rejection tests were changed before production code.  The
first focused command failed during collection because the prior module did not
provide the required distinct result/error API:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py
ImportError: cannot import name 'HFSourceOwnerNonRejectionError'
1 error during collection
```

After the first GREEN implementation, a second RED mutation test proved that
an incomplete objective binding was still being classified like the exact
rollback `snapshot=None` field failure:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_hf_shared_surface.py -k missing_value_field
Pytest: 0 passed, 1 failed
Failed: DID NOT RAISE HFSourceOwnerNonRejectionError
```

### Corrected owner contract

- All four rows resolve their repository module/path, exact symbol, positional
  versus keyword-only parameter signature, owner source fragments, and one
  unique rejecting `if` guard before rejection is evaluated.
- `_validate_backward(spec, receipt)` evaluates the actual
  `receipt.trajectory_denominator != 13 * 16` predicate with a zero-valued
  source-level receipt field.
- `apply_projected_delta(...)` uses the exact current
  `torch.isfinite(applied_flat).all()` / `the applied delta is not finite`
  source guard.  Its import-free evaluator maps a finite/non-finite numeric
  sequence through that actual AST expression; no Torch module is imported.
- `PrivateCheckpointRef.__post_init__(self)` evaluates the actual
  `self.private is not True` predicate over `private=False`.
- `TrainingStateTransaction.reject(self, snapshot)` resolves its first
  `_require_active(snapshot)` delegation and that helper's exact
  `snapshot.transaction_id != self._active_transaction_id` guard.  The frozen
  `snapshot=None` input produces typed `required_field_unavailable` provenance;
  a matching snapshot produces non-rejection.
- Successful rejection returns `HFSourceOwnerRejection` with owner/guard
  symbol, signature, source, and rejection kind.  Missing symbols, wrong
  signatures, or missing guard fragments raise
  `HFSourceOwnerResolutionError`; resolved non-rejecting or incompletely bound
  values raise `HFSourceOwnerNonRejectionError`.  Neither error can satisfy an
  expected owner-rejection assertion.

### Verification evidence

The focused GREEN suite reached `45 passed`.  Fresh final pytest, Ruff,
compileall, strict OpenSpec, import-absence, and staged diff checks are recorded
in the final task handoff for the committed bundle.

### Remaining concern

No implementation stop concern within Task 1.  This is source/value-level
handoff evidence only; it does not claim that later-wave Torch owners have been
executed.  Task 1.5 remains the independent review gate and is deliberately
unchecked.
