# Lead ruling 02: matched live-adapter source reference

2026-09-22. Authorize the bounded qualification correction below. Scientific
fitting remains held until the complete qualification passes; this is not
source-equivalence acceptance or a relaxation of the numerical gate.

This amends only the runtime-reference boundary of [unit.md](unit.md) and the
no-runtime-comparison-change sentence in
[feedback 01](lead-qualification-feedback-01.md). Source checkpoint, data,
architecture, objective, optimizer/LR, selected surfaces, 2e-4 parity tolerance,
original clock and resource envelope remain unchanged. The user-authorized
task includes training-infrastructure adaptation; no new user permission is
needed for this correction.

## Verified evidence and correction to the diagnosis

The lead independently verified SHA256
`5f7f98acde14b23712e054ca60accf7afd5c13e40652394d540f90db2307394f`
for source-path-conflict-v1/manifest.json and all ten bound files. Corrected
parity-v4 still reports maxima 1.96875, 0.75 and 3.1689453125, so equivalence
has not passed. The reported 430.167983 allocated GPU-seconds and zero scientific
updates remain the conflict snapshot, not an updated live cost total.

The worker's merged-source explanation is NOT supported by the executed path.
`parity._load_source` calls `attach_dora_adapter`, whose actual operation is
Transformers `PeftAdapterMixin.load_adapter`. parity-v4's source receipt records
that API, 196 adapter layers and `merged_adapters=[]`. The separate function
`merge_dora_adapter_for_execution` contains `autocast_adapter_dtype=False` and
`merge_and_unload`; it is not the source loader reached by this parity entry.
Keep the original conflicting report immutable and correct its interpretation
in the new manifest. Do not call its drift "merged-source drift".

The distinct Mixin-injection versus training PeftModel-wrapper construction
paths are a concrete candidate explanation. The latter explicitly promotes
adapters by default; record actual tensor dtypes/values before attributing the
residual discrepancy to that policy. No claim that it explains all drift is
accepted yet.

## Authorized reference and decisive comparison

Build an independent reference from the pristine base, the exact mature source
adapter payload and both original selected-token deltas. Keep the mature adapter
live using the same explicit PEFT wrapper/promotion policy as the training
runtime. The reference must contain ONLY the original mature target set. It
must not be copied from an expanded model, include newly added adapter targets
or an address module, or load a qualification/trained checkpoint.

Bind base/source payload hashes and original target membership. Verify copied
mature A/B/magnitude tensors and effective input/output rows against the source
with explicit dtype conversion recorded; never renormalize or repair mature
tensors to make parity pass. Reference and expanded model must have matching
effective mature values, actual base/adapter/embedding dtypes, eval/dropout and
attention settings, runtime patches, processor tensors, serialized prompts,
causal target alignment, and comparison shapes. Freezing the reference for
evaluation must not silently recast its tensors or merge it.

Compare this reference against the independently constructed, corrected
expanded training model at ZERO updates with codebook OFF. Verify new B tensors
are zero, new magnitudes use the runtime norm, and mature copies are unchanged.
Retain the original full-response/full-vocabulary max-absolute-logit gate of
2e-4 on ALL three admitted qualification cases. Short greedy equality is an
additional check, never a substitute. A failure remains HOLD and returns to the
lead with localized evidence rather than another reference change.

Version the source contract/launch before this comparison; preserve earlier
launches and failures. Where saved tensors permit, record old Mixin-reference
versus new live-reference drift separately. A bounded bridge on the same three
qualification inputs is authorized if tensors were not retained; do not add
cases or expand this into merge/export compatibility research. Label the bridge
as a loader/runtime comparison, with no equality claim or tolerance relaxation.

## Training, evaluation and reuse of existing evidence

If the new comparison passes, its conclusion is that target expansion preserves
the source UNDER THE DECLARED LIVE TRAINING RUNTIME. It does not establish
equivalence to historical Mixin, merged, dense or vLLM evaluations.

All scientific source baselines and trained natural evaluations in this package
must use that same explicit live-adapter runtime and selected-delta composition,
with the intended difference in address enablement/learned weights declared.
Do not mix old-route source scores with new-route trained scores. Evaluate any
required source baseline under the new bound route, retaining old values only
as separately labeled diagnostics.

Refresh source parity and corrected-checkpoint fresh-process reload, full-row
logit comparison and native generation under this route. Corrected single/
two-rank update, loss and optimizer evidence may be reused IF its training
assembly, dtype policy and exact executed producer remain unchanged. A change
to shared setup, promotion or optimizer behavior requires requalification of
the affected path; do not blanket-rerun unaffected evidence. Finish the planned
single/distributed comparison on the corrected states if still outstanding.

After all unit.md qualification conditions pass, continue the bounded fitting
automatically. Report the matched-reference milestone with source bindings,
per-case errors, preservation checks, reload evidence, cost and job state. No
scientific fitting, full-dataset training, self-acceptance or successor is
released by this ruling alone. Send reports directly to the existing lead task.
