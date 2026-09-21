# Untied coordinate gradient-path accounting

## Question and authority

At the mature tied2444 and untied+axis2444 checkpoints, how do input and output
paths contribute to the selected-token gradient on the same trusted supervision,
and do endpoint radial components differ from interior coordinates?

This is a user-authorized, forward/backward-only diagnostic. Root owns scientific
decisions and acceptance; the existing 916-worker owns its bounded execution.
No optimizer update, training, new checkpoint, source-data edit or successor.

## Predecessor and incremental evidence

Saved effective weights show coord999/0 remain the two largest output-row norms
after untied training, while the new input rows do not have those endpoint ranks.
This rejects tying as necessary for those output norm peaks, without explaining
bursts. The present unit measures current gradient paths rather than inferring
them from weight norms. It cannot recover gradients during early training.

## Shared anchors and complete population

Inherit exact checkpoint, paired embedding, tokenizer, prompt, media and source
hashes from [unit A](../2026-09-18-untied-highconfidence18-natural/unit.md).
Models are old mature tied step2444 and new untied+axis step2444, not R16.
Population is the fixed 18 scenes / 570 trusted positive annotations: historical
Human13 plus the current five-image refinement. Do not substitute live COCO rows.
Freeze one canonical geo_sorted_xy teacher per scene, including normal EOS;
use identical token targets, supervision masks and global normalization in both
models. These are trusted positive lists, not exhaustive negatives or held-out
generalization evidence. Keep all 18 denominators and source identities.

Process all 18 once per declared gradient condition, or freeze chunks covering
all 18 before aggregation. Cached tokenization is allowed after hash binding.
Use fixed seed17, source order and batch/chunk schedule; record exact execution.
OOM recovery may split microbatches only if the same global reduction is verified.
Never remove unfavorable scenes or average unequal chunk means unweighted.

## Gradient surfaces and objectives

Keep every parameter value fixed. Enable gradients only on the 1,004 selected
coordinate/wrapper rows; language DoRA and other parameters remain unchanged.
Preserve activation autograd through the entire input-to-loss path: freezing
weights must not detach embeddings, hidden states or input-path computation.
No full-vocabulary untied training is introduced.

For old tied2444, measure native shared-delta gradient g_shared. In a diagnostic
clone, use equal-valued but independently differentiable input/output deltas;
frozen base storage may remain shared. Measure g_in and g_out, and verify the
actual backward identity g_shared = g_in + g_out. Include selected wrappers,
base-plus-delta effective weights and both wrapper forward paths in the check.
For new untied2444, use the actual separately saved deltas without retying them.

Measure CE and raw_axis_validity_hinge gradients separately. CE is segment-balanced
token mean, then mean over the same globally eligible 18 segments, with normal
EOS supervised. Axis loss is the admitted coordinate-only FP32 expected-xy hinge,
margin1/999, mean axes, mean complete boxes per segment, then the same global
segment mean, retaining zero-box segments. Freeze complete/incomplete box counts.
For new2444 report CE, raw axis, weighted axis0.01 and combined CE+0.01*axis.
The old checkpoint's axis response is an analytical diagnostic, not its historical
training objective; report its historical CE gradient separately.

## Mechanical gate and numerical execution

Reuse unit A's admitted untied native-HF load and complete paired payload identity.
This diagnostic may run in parallel after that load gate; it does not delay A.
Freeze precision, attention, autocast, padding, teacher inputs and global reduction
before measurements; use FP32 accumulation for losses, gradients and reducers.
Require fresh equal-weight tied/split forward parity before interpreting gradients.
Freeze numerical tolerances in the execution receipt before target aggregation.
Check finite-difference directional sensitivity of each selected delta path on a
fixed supervised gate slice, plus actual backward sum equality for the tied clone.
Include a fixed endpoint-row direction and an interior direction; preserve exact
perturbation/restoration receipts. These are temporary numerical evaluations,
never optimizer steps. Failure blocks gradient interpretation, not natural A.

## Saved measurements and acceptance

Retain per-scene and aggregate gradients for each declared objective/path, effective
selected weights, target/mask fingerprints, losses, counts and gate residuals.
Report gradient norms, input/output norm ratio and cosine, explicitly handling
zero norms. For every coordinate row report radial component g dot w / ||w||;
negative of that scalar is the first-order norm change under plain gradient descent.
Compare coord0 and coord999 individually against the complete interior1..998
distribution; include wrapper summaries separately. Magnitude dominance is not
conflict. Negative cosine describes current vector disagreement, not historical
damage, an Adam update or a causal burst mechanism. Do not infer Adam behavior
without optimizer evidence; this unit does not require or simulate its moments.
Per-axis/role summaries are optional only if available from the frozen positions
without extra model passes; otherwise leave them unmeasured.

Publish raw gradient tensors and a deterministic CPU reducer reproducing all
tables, objective combinations, denominators and equality residuals. Root checks
saved-output reconstruction and the fresh forward/backward gate before acceptance.
Interpret alongside A's natural outputs; gradients alone cannot establish burst
incidence, explain a behavioral difference or isolate untie from added axis loss.

## Cost and stop

Hard ceilings: 6 allocated GPU-hours, 300 forward/backward evaluations and 4GiB
retained tensors, including gates/retries. Count forwards and backwards separately,
complete 18-scene passes, chunk sizes, GPU allocation time and actual artifact bytes.
Stop on a failed gate, exhausted ceiling or completed fixed measurements; preserve
partial denominators and classify technical-invalid versus completed evidence.
No automatic tolerance relaxation, additional scenes, stages, objectives, layers,
seeds, optimizer experiments or expanded checkpoint sweep. End owned jobs and return
one candidate result with exact artifacts, cost and limits for lead acceptance.
