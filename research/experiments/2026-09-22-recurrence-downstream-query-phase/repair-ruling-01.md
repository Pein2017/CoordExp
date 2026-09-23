# Root repair ruling: consumer routing and composed interventions

Attempt001 is technical-invalid before any text layer: the global SDPA registry
also dispatches vision attention, which has no layer_idx. Its source snapshot,
manifest and receipt remain immutable. It used1 model/1 vision invocation,
7.162s package time and no completed cell. PID4019255 is terminal.

Root takes over the producer and authorizes one corrected attempt002, at most
3 model/3 vision calls,10minutes on GPU4. Total package budget becomes4 model/
4 vision invocations including the retained failed attempt. Scientific cells,
primary prediction, source selection, tolerances and stop rule are unchanged.
The user has renewed autonomous research; no further approval is needed.

The bundled correction filters exact text module IDs before reading layer
metadata, verifies actual Q as well as post-GQA K/V/mask at the SDPA consumer,
and checks layer0's composed output AFTER its intentional head substitution.
The latter would otherwise reject a valid held-state condition. Selected
pre-Q/K captures now clone the small slices to avoid retaining whole sequence
storage, and prior cell GPU state is released before the next cell.

The previous CPU check compared SDPA on identical producer-patched inputs and
did not exercise the production consumer checker. The revised positive case
constructs the expected patch by explicit paired coordinates and calls the
installed sdpa_attention_forward with the actual F.sdpa observer/checker. Wrong
query and reversed-sine mutations must reach that observer and be rejected.
This qualifies the same consumer boundary used by production. The new native
model call additionally tests real vision passthrough; no extra GPU smoke.

Root preserves original producer snapshots, checks CPU qualification, executes
the three cells, and independently accepts raw results. No worker retry or peer
surface is authorized. Any further failure returns to root with its evidence;
technical invalidity is not a scientific null.
