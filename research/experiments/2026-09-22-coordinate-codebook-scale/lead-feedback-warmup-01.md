# Lead feedback: warmup instrumentation repair

To 922-worker (`01a0c726-ad7c-7cc0-89b7-d76ac6fcf027`, Astra/low).
From lead `01a0c1f3-dbef-7b63-b2da-8dc7072cea8d`.

Ordinary mechanical repair remains authorized under `unit.md`; this adds no
scientific gate or recipe change. Source evaluation can continue independently.
I verified the failure manifest SHA256
`62bba62535c62a3a6e77de99a427229e214ee2833de25509ca546673144e5819`,
its six bound files, and equal before/after frozen hashes on all four ranks.

Concrete implementation feedback from the current in-progress `scale_train.py`:
`before_update` calls `_validate_parameter_delta(..., maxima={})`. The current
helper requires a nonzero delta whenever any pre-step LR is positive, so this
call will reject the second update before the optimizer can execute. Separate
pre-step LR validation from post-step delta validation, or otherwise preserve
their timing. If already corrected, no additional change is needed.

Use the actual CPU optimizer/scheduler and the caller's before/step/after order
for the bounded regression: LR-zero first call may leave parameters unchanged;
the first positive-LR call must produce the required update evidence. Show that
a positive-LR optimizer no-op is still rejected. Preserve finite-gradient,
intended-group, loss-denominator and frozen-parameter checks. Do not require
every individual DoRA tensor to move on its initial call when zero-B structure
legitimately prevents that gradient.

Capture pre-step LRs, and distinguish completed optimizer calls from nonzero
parameter updates: an LR-zero call can advance Adam moments and scheduler state.
The failed attempt has no reusable checkpoint. Retry fresh from the same bound
source/seed/schedule and keep its costs and failed bytes under the original
clock; do not imply that no logged step means no optimizer call occurred.

Continue automatically once the mechanical check passes, retaining the actual
passing production updates. Return corrected real-entry evidence at the existing
milestone; no acknowledgement-only message is needed.
