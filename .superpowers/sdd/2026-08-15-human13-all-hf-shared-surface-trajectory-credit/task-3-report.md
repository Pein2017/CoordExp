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
