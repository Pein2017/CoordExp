# Source-expansion qualification feedback

2026-09-22. This is a mechanical repair acknowledgment under unit.md, not
qualification acceptance or a changed scientific recipe.

I independently verified both reported artifact hashes. parity-v3 records
BF16/FA2 OFF-codebook maxima 1.2890625, 0.625 and 3.8095703125 against 2e-4,
despite matching effective selected input/output rows and short greedy prefixes.
The saved magnitude diagnostic records BF16-initialized magnitude versus the
runtime FP32 norm, with scale error 0.0038521289825439453. The current expansion
factory leaves source-absent targets at PEFT's initial values. This supports
the proposed concrete repair; end-to-end source equivalence remains unproved.

Continue the narrow initialization correction for NEW expansion targets only.
Use the runtime DoRA norm definition/precision after adapter promotion, and
preserve copied mature A/B/magnitude tensors exactly. Retain a focused regression
case in which zero-B BF16-base expansion exposes the old non-unit scale, then
passes after the correction. Do not repair parity by changing the prompt,
source checkpoint, runtime comparison, output policy or tolerance.

Fresh corrected-source parity and the necessary real-update/save/fresh-reload
checks must bind the corrected producer and initialization. Earlier training,
reload and optimizer receipts remain historical technical evidence; none
qualifies the changed initialization. Reuse unaffected CPU evidence and saved
inputs, with all failures/costs retained. Continue automatically under unit.md
once the complete qualification passes. No new user approval is needed for this
repair; broader fitting remains held until then.
