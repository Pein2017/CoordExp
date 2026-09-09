# Local execution checks

Read the section implicated by the task. Repository paths below are relative to
the **selected checkout**. The canonical research base is
`/data/CoordExp/.worktrees/research-probes`; a production checkout may expose
different APIs. Missing research helpers are not a reason to import another
worktree. Stable specs define supported behavior; a research direction owns its
scientific objective. Resolve conflicts before translating between them.

## Precision and numerical acceptance

Distinguish parameter storage, autocast/forward arithmetic, sensitive reductions,
optimizer state, saved payload and replay precision. Upcasting a BF16 output
cannot recover precision already lost in the forward. For sensitive probability
math, use the supported scoring owner (`src/losses/token_scores.py`) rather than
adding another softmax/log implementation. Preserve gradients and explicit caller
reductions; do not silently detach, transfer to CPU or change all model weights
to FP32.

For a tolerance-bound solve/intervention, distinguish solver feasibility from
the precision of materialized execution. Allocate slack before a cast/replay;
do not relax the frozen acceptance threshold to admit a numerically failing
candidate. A past QP failure arose when the FP64 solve consumed the whole FP32
replay tolerance. The reusable lesson is this separation, not a universal epsilon.

Identical parameters or logits do not establish identical gradients across
different reduction/packing shapes. Inspect the claimed equivalence and its
accepted tolerance; preserve a previously dispositioned diagnostic as such.
Relevant owners: `src/qwen/forward.py`, `src/losses/`, `src/runtime/`, and the
direction's solver/materialization code. Check the exact runtime before assuming
dynamic adapters, merged weights or another engine implement the same arithmetic.

## Objective, denominator and distributed reduction

Identify the loss-bearing unit and population: token, action, object, original
example or pack. Write numerator, mask and denominator for the actual objective,
including EOS and empty-support behavior when relevant. Keep raw loss and
weighted contribution distinct. Token mean, mean of per-example token means,
and mean of pack means generally differ when lengths differ.

Trace the actual backward scale through local contribution, DDP averaging,
accumulation and optimizer step. Do not add a world-size or accumulation factor
just because a formula elsewhere uses one. Uneven local work needs the global
denominator of the declared objective, not an unweighted mean of rank means.
For a normalization-equivalence claim, compare gradients on unequal-length
examples and uneven partitions, not just equal-sized batches or displayed loss.

Packed-training owner:
`openspec/specs/coordexp-swift-supervision-losses/spec.md`, `src/losses/`,
`src/runtime/train_runtime.py`, `tests/losses/`, `tests/runtime/`.
Research examples: `probes/dora_owner_learning/README.md` and
`probes/dora_owner_learning/tests/test_native_learning.py` keep Source CE/RLOO
and coordinate/full-action credit distinct. Their constants are profile-owned.

## Causal positions, padding and packing

For an action starting after a prefix of length P, the first action target is
scored from the preceding causal row; do not use same-position logits or shift
twice. Use the caller's explicit target-to-logit map. With compact selected
logits, returned row numbers are not original sequence positions. Packed
segments must not predict across sample boundaries. An attention mask, a loss
mask and segment-local position metadata solve different problems.

Unequal lengths expose padding errors. Check literal token histories, per-request
budgets, terminal requests and result association. A pad token already inside a
stored history is not necessarily disposable batch padding. Do not retokenize
stored actions or infer terminal status merely by stripping token values.

Owners: the Qwen manual's `Local Execution Boundaries`,
`src/qwen/native.py`, `src/qwen/generation.py`, `src/qwen/positions.py`,
`openspec/specs/coordexp-swift-packing-forward/spec.md`.
Counterexamples: `tests/qwen/test_native.py`, `tests/qwen/test_generation.py`,
`tests/qwen/test_positions.py`, `tests/qwen/test_forward.py`.

## Autograd, tensor meaning and capture

Check dimension **meaning**, not just dimension size: batch versus image/patch,
physical versus selected token row, sample-local versus packed position, and
pre-injection versus post-injection activation. The Qwen visual tensors are not
ordinary batch-major text metadata.

Equal integer values can still differ in autograd usability when constructed
under inference mode. Hook outputs may alias tensors later modified in place by
DeepStack; snapshot at the required boundary, before mutation. Preserve gradient
flow when the consumer differentiates, and clean hooks on failure. These are
already executable counterexamples in `tests/qwen/test_native.py` and
`tests/qwen/test_inspection.py`; use them rather than adding a generic shape gate.
The public `CaptureHiddenRows` and `CaptureInputs` helpers deliberately detach
snapshots. A snapshot can enter a new replay graph, but cannot differentiate
through the original captured computation. Choose a graph-connected local hook
when that derivative is the question; do not assume every clone helper retains it.

## Chat templates and real encoded inputs

Distinguish generation from an assistant opener, continuation of a partial
assistant action, and teacher-forced replay of literal IDs. Inspect the actual
processor/template output for image placeholders, assistant boundaries, special
tokens and EOS. Do not assume rendered strings concatenated and retokenized
produce the same IDs as the stored history. Preserve image order, grids,
resize policy and coordinate interpretation together.

Owners: `src/templates/`, `src/qwen/encoding.py`, the Qwen manual's
`Processor And Grid Rules`, and
`openspec/specs/coordexp-swift-data-template-encoding/spec.md`.
One real encoded sample with relevant image/template boundaries is more useful
than repeated synthetic strings. For an actual launch, validate effective
runtime settings at the real child/worker entry, not only in the parent shell.
