# Task 3 report: all-HF objective and transaction adapter

## Status

IMPLEMENTED — correction bundle and local gates are green; fresh bounded
rereview is pending against the committed target.

## Scope boundary

- New adapter/test only: `human13_all_hf_vertical.py` and its focused test.
- CPU/injected tensors and a toy `torch.nn.Parameter`/fresh AdamW transaction only.
- No HF model load or forward, CUDA/GPU, network, checkpoint, output publication,
  accepted state, retry, fallback, or legacy objective/projection change.
- The pre-existing untracked memory note is preserved.

## TDD RED

The first focused command ran after the production-shaped K16 test was written
and before the production module existed:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: No tests collected
ModuleNotFoundError: No module named 'scripts.research.human13_all_hf_vertical'
exit 127 through the Conda/RTK wrapper
```

The test fixture already routes through admitted Task-1 sampled/replay values,
an admitted K16 trajectory ledger, an admitted compiler ledger/compact-logit
receipt, the existing fresh AdamW proposal/projection/apply owners, and the
existing complete training-state transaction.  The required behavior is one
global denominator 16, all three components, one proposal/apply, detached
receipts, one private update, and exact explicit rollback.

## Owner seam review

- No missing `human13_rp_crossover_runtime.py` transaction seam has been
  proven, so that legacy runtime and its tests remain unchanged.
- Task 2 deliberately retains live replay tensors inside its session owner.
  Task 3 therefore consumes injected CPU live tensors keyed by the admitted
  `GradientReplayGroup` hashes and serializes only detached values/hashes.
- `capture_exact_adamw_proposal` owns its own begin/step/reject transaction.
  The Task-3 adapter can keep a caller transaction around backward and private
  projected apply while using one inner capture transaction over the exact same
  parameter/optimizer surface; no optimizer or projection math is copied.

## Initial GREEN and bounded review

The initial implementation passed 13 focused tests and 244 tests across the
focused plus adjacent shared-surface/live-model/trajectory/compiler/
preservation/runtime/transaction suites. Ruff, compileall, Serena diagnostics,
and strict OpenSpec validation were clean.

The independent bounded review returned HOLD with three P1 findings and no P0:

1. post-prepare `model.train()` or an equal-valued live module parameter-registry
   swap was not revalidated;
2. detached-value equality did not prove replay/compiler autograd ownership, so
   equal-valued foreign leaves could enter the objective;
3. mutually consistent trajectory/compiler source identity was not explicitly
   joined to the exact HF shared-surface checkpoint payload identity.

These were accepted as vertical-owner contract gaps. They did not prove a
missing legacy runtime transaction seam, so the legacy runtime remains
unchanged.

## Fix round 1 — TDD correction bundle

Before production correction, five new adversarial cases were added (two
parameterized model-surface cases plus foreign replay, foreign compiler, and
foreign Source cases). The focused RED was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 13 passed, 5 failed
```

All failures were at the intended missing checks and no proposal receipt was
expected. The single correction bundle now:

- seals the exact `(name, parameter identity, tensor version)` model-surface
  fingerprint, requires eval mode/current registry and Source state before
  backward, and revalidates the post-capture versioned surface immediately
  before projected apply;
- traverses autograd leaves for every replay and present compiler tensor and for
  each corresponding component numerator, requiring nonempty roots exclusively
  in the exact bound trainable parameters; canonical absent compiler sites keep
  the admitted zero-component semantics;
- requires trajectory Source identity to equal the HF identity checkpoint
  payload, carries that digest in the sealed proposal receipt, and includes the
  surface/source join in the objective binding consumed by the existing AdamW
  proposal/projection receipts.

A sixth focused test injects model-mode drift during projection and proves the
second surface check runs after exact AdamW capture and before projected apply,
with the outer transaction restoring Source.

## Fix round 1 gates

```text
focused: 19 passed
focused + adjacent Human-13 suites: 250 passed
Ruff: clean
compileall: clean
Serena diagnostics (production and test): {}
strict OpenSpec: valid
```

All execution remained CPU-only with injected toy tensors. There was no real HF
load/forward/backward/optimizer action, CUDA/GPU use, network access, checkpoint
write, output publication, accepted state, adaptive retry, or fallback path.
OpenSpec Task 3.6 remains unchecked pending the fresh committed-target rereview.

## Fix round 2 — full registered model state

The fresh rereview of commit `d6b7fdf` confirmed that all three round-1 P1s
were closed, then returned HOLD with one additional P1: the runtime fingerprint
and rollback boundary covered the optimizer-bound trainable subset, but not a
registered frozen base parameter. A frozen value could therefore change after
preparation, pass proposal admission, and remain changed after rollback.

This was accepted as a repairable full-model contract gap, with no change to
the research objective, projection semantics, or legacy runtime seam. Before
production correction, the toy surface gained one registered frozen parameter
and the adversarial focused test mutated it after preparation. The exact RED
was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 19 passed, 1 failed
Failed: DID NOT RAISE AllHFVerticalError
```

The correction keeps AdamW and the supplied `TrainingStateTransaction` bound
only to the trainable proposal parameters. Separately, the vertical owner now:

- captures the exact complete `model.named_parameters()` registry at prepare,
  including every parameter's name, object identity, shape, dtype,
  `requires_grad`, tensor version, and content hash;
- carries a detached full-model Source hash in the objective/proposal receipt;
- revalidates the full registry and content before backward and again after
  exact AdamW capture immediately before projected apply;
- begins the existing trainable transaction before the final live-state
  revalidation, then on rejection uses that transaction for optimizer,
  trainable, counter, and RNG restoration and restores all still-identical
  registered model parameters from the private Source snapshot;
- records equal full-model Source/restored hashes in the sealed rollback
  receipt. A registry identity change remains an explicit fail-closed restore
  error rather than silently rebinding optimizer ownership.

The existing trainable Source-drift test was also strengthened to require a
rollback receipt and exact Source parameter/transaction digest restoration.

## Fix round 2 gates

```text
focused: 20 passed
focused + adjacent Human-13 suites: 251 passed
Ruff: clean
compileall: clean
Serena diagnostics (production and test): {}
strict OpenSpec: valid
diff check: clean
```

No legacy runtime file changed. Execution remained CPU-only with injected toy
tensors and no real HF load/forward/backward/optimizer action, GPU/CUDA,
network, checkpoint/output action, accepted checkpoint, retry, or fallback.
OpenSpec Task 3.6 remains unchecked pending another committed-target rereview.

## Fix round 3 — post-probe surface certification

A subsequent bounded review found one additional P1 at the external realized
margin callback seam: the last full-surface check ran before
`realized_margin_probe`, while that callback runs inside the existing projected
apply owner. A successful callback could switch the model to train mode or
mutate frozen/trainable parameters after the preservation owner measured its
physical delta, and the vertical could then seal a proposal receipt. Rollback
restored parameter values but did not restore `model.training`.

The finding was accepted as a vertical receipt/rollback contract gap. Before
production correction, three parameterized adversarial cases were added for a
probe that returns valid margins after calling `model.train()`, mutating a
frozen parameter, or mutating a trainable parameter. The exact RED was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 20 passed, 3 failed
Failed: DID NOT RAISE AllHFVerticalError
```

The correction now:

- requires the complete Source module tree to be in eval mode and includes its
  module-mode entries in the full-model Source content address;
- immediately before sealing a private proposal receipt, rechecks eval mode,
  the exact full parameter registry/metadata, the expected single version
  increment for each trainable projected apply, unchanged frozen versions and
  values, and the existing apply owner's pre-probe trainable post-state hash;
- rejects any post-probe drift before a proposal receipt exists, then uses the
  existing transaction plus the private full-model snapshot to restore all
  values and Source eval mode;
- adds explicit Source/applied and Source/restored eval-mode evidence to the
  sealed proposal and rollback receipts. The original post-probe validation
  error remains the primary `AllHFVerticalError`; any restore failure is
  reported explicitly as rollback failure with the primary error retained in
  its message.

The successful probe path remains unchanged and now asserts eval-mode evidence
in both proposal and rollback receipts.

## Fix round 3 gates

```text
focused: 23 passed
focused + adjacent Human-13 suites: 254 passed
Ruff: clean
compileall: clean
Serena diagnostics (production and test): {}
strict OpenSpec: valid
diff check: clean
```

No legacy runtime or owner objective/projection math changed. Execution remained
CPU/injected only, with no real HF model action, GPU/CUDA, network,
checkpoint/output action, accepted checkpoint, retry, or fallback. OpenSpec
Task 3.6 remains unchecked pending fresh committed-target rereview.

## Fix round 4 — callback ownership and version-exact rollback

Two independent follow-up findings were combined into this authorized repair:

1. a valid realized-margin callback could replace an AdamW param-group entry
   with an equal-valued foreign parameter after the existing ownership check;
   the proposal could seal, then rollback could become non-terminal with the
   transaction inactive, optimizer still foreign, and a live gradient;
2. value/eval restoration used `copy_`, so tensor version counters advanced
   while the rollback receipt claimed exact full Source restoration but carried
   no version evidence.

Before production correction, focused tests were added for callback optimizer
substitution, exact normal-rollback versions, and exact callback-failure
versions. The RED was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 22 passed, 3 failed
```

The callback is now wrapped by the vertical owner. It snapshots the exact
AdamW group keys/options and original parameter references plus the existing
transaction's named parameters, optimizer/scheduler/counter/runtime owners,
CUDA-capture setting, active transaction identity, update count, runtime
counters, defaults, and empty state. After the callback it requires all of
those owners unchanged. Any callback mutation is restored before a typed
`ProjectedApplyError` is raised, allowing the existing projected-apply revert
and outer transaction rejection to complete. The same ownership check runs
again after apply with the expected one update and before proposal sealing.

`_reject` now restores source optimizer/transaction ownership before calling
the existing transaction rejection. Public rollback is exception-safe and
terminal: if ordinary rejection fails, a narrow fallback restores optimizer,
scheduler/runtime counters, update count, RNG, full model values/mode/versions,
and clears gradients, then the owner enters `rolled_back` and raises typed
`AllHFVerticalError`. A second rollback is deterministic. The fallback is
tested with an injected transaction-reject failure.

Full Source and applied/restored version tuples are now sealed in proposal and
rollback receipts. On the pinned `ms` runtime (`torch 2.9.1`), the narrow
version restore helper uses the supported
`torch._C._autograd._unsafe_set_version_counter` entrypoint via guarded
attribute lookup, verifies availability, applies it only after Source values
are restored, and rechecks every version. If the entrypoint is unavailable or
does not restore exactly, rollback fails explicitly rather than claiming exact
Source restoration.

## Fix round 4 gates

```text
focused: 26 passed
focused + adjacent Human-13 suites: 257 passed
Ruff: clean
compileall: clean
Serena diagnostics (production and test): {}
strict OpenSpec: valid
diff check: clean
```

No objective/projection math or legacy runtime file changed. Execution remained
CPU/injected only with no real HF model action, GPU/CUDA execution, network,
checkpoint/output action, accepted checkpoint, retry, or fallback objective.
OpenSpec Task 3.6 remains unchecked pending fresh committed-target rereview.

## Fix round 5 — recoverable pre-transaction preflight

The rereview of round 4 confirmed its callback ownership, exact version, and
terminal public rollback repairs, then found the same ownership invariant could
fail after preparation but before `TrainingStateTransaction.begin()`. The
preflight correctly rejected a substituted optimizer parameter reference,
nonempty optimizer state, or nonzero counter, but `snapshot` remained `None`;
the error path therefore released the private Source evidence without restoring
the mutated ownership state or producing a rollback receipt.

Three parameterized focused cases were added before production correction. The
exact RED was:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 26 passed, 3 failed
Failure: rollback receipt was None
```

Preparation now retains a complete private Source recovery snapshot alongside
the existing full-model snapshot: optimizer state dict, parameter groups,
defaults, scheduler state, update count, transaction owner references, runtime
counters, CPU/CUDA RNG, model values/mode, and tensor versions. If preflight
fails before an active transaction exists, the vertical:

1. restores every prepare-time Source component and verifies the complete
   transaction digest, optimizer ownership/defaults/empty state, gradients,
   model content/mode, and versions;
2. begins a real transaction on that restored Source;
3. rejects that recovery transaction through the ordinary `_reject` path,
   producing the same sealed rollback receipt contract;
4. only then releases the one-shot live evidence and raises the original
   preflight error with the rollback receipt attached.

If either recovery or receipt construction fails, a second complete Source
restore is attempted, the owner still terminalizes to `rolled_back`, and the
primary plus recovery/fallback failures remain explicit in the typed error.

The fixed-point review also identified a second case in this same boundary:
model parameter registry substitution and `requires_grad` mutation were
detected, but value-only restoration could not reconnect the exact Source
objects/trainability and therefore produced no receipt. Two additional tests
were written before that repair:

```text
conda run -n ms python -m pytest -q tests/research/test_human13_all_hf_vertical.py
Pytest: 29 passed, 2 failed
Failure: rollback receipt was None
```

The prepare-time snapshot now also retains every Source module's registered
parameter and child-module entries plus each parameter's exact `requires_grad`
flag. Full-model restoration reconnects those exact registered objects and
module structure, restores trainability before values, clears gradients, and
then certifies eval mode, content, and exact tensor versions. Registry swaps and
trainability flips now follow the ordinary sealed rollback-receipt path.

## Fix round 5 gates

```text
focused: 31 passed
focused + adjacent Human-13 suites: 262 passed
Ruff: clean
compileall: clean
Serena diagnostics (production and test): {}
strict OpenSpec: valid
diff check: clean
```

No legacy runtime or objective/projection math changed. Execution remained
CPU/injected only with no real HF model action, GPU/CUDA execution, network,
checkpoint/output action, accepted checkpoint, retry, or fallback objective.
OpenSpec Task 3.6 remains unchecked pending fresh committed-target rereview.
