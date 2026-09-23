# Recurrence exit: history and current-row position crossing

Date: 2026-09-22. Lead: current research task; execution: native subagent
`trace_dynamics` (existing gpt-5.6-sol / xhigh). The user explicitly authorized
autonomous research judgments and use of all GPUs after discussing this contrast.
This is a new GPU probe, not a change to the closed CPU readback's authority.

## Active question

From mature untied+axis step-2444's val:7511 native exit, does exchanging the
current row prefix's MRoPE positions between rows88 and89 transport the x2
decision between bins38 and999 when crossed with both native complete histories,
under exact source identity, causal-mask preservation and native replay parity?

The [predecessor](../2026-09-22-recurrence-transition-readback/results.md) found
opposing role dynamics and a fixed-pair margin oscillation inconsistent with a
literal stationary two-positive-decay readout. It did not identify the cause.
Earlier history perturbations establish sensitivity, not a selective covered-set
update. This probe tests counterfactual transport of the same local decision,
not whether some perturbation changes any logit. The strongest alternative is
history-dependent computation, including interaction with positional retrieval.

The [user's visual adjudication](../2026-09-22-recurrence-transition-readback/supporting/user-visual-adjudication.md)
identifies the repeated water-region person as hallucination and the first long
beach rectangle as a bad multi-owner prediction. Later isolated kite and some
person predictions are valid. Thus the primary outcome here is numerical exit;
neither owner exclusion nor valid localization recovery is being estimated.

## Frozen source and intervention

Select only `val:7511` from the predecessor's bound
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/dynamics/selection.json`.
Its native batch is `refined-01`, target index2. Bind the selection and its
raw, trace, runtime receipt and image by their stored hashes. Preserve the
source checkpoint composition, effective selected input/output rows and untied
deltas, FP32/SDPA arithmetic, image processing, prompt, batch companions and
physical padding. Checkpoint adapter:
`/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444/adapter`.
Load current maintained code, not preserved historical executable snapshots.

Indices are zero-based. Let H_E and H_L be the complete literal histories
before row88 and row89 respectively. Let S be the current row's literal prefix
from its opener through the consumed y1 token that predicts x2. Verify S is
identical in both sources before launch. P_E and P_L are S's native three-axis
MRoPE position arrays. The four cells are H_E/P_E, H_E/P_L, H_L/P_E, H_L/P_L.

Within each H, keep tokens, history positions, image, batch companions, physical
token order and the actual causal attention relation fixed. Replace positions
only for the entire S in the target sample, shifting all text MRoPE axes equally
and preserving S's internal relative positions. Crossed cells intentionally
introduce a gap or positional overlap against the history. Record that surgical
condition; it is not a naturally generated prefix. Do not change causal order to
match the artificial positions. Use direct forward; generation may overwrite
manual positions. A four-row position representation is allowed only to preserve
the native physical causal relation, with native three/four-row parity qualified.

Capture the position/mask values consumed by the model, not merely requested
kwargs. Causal prefix computation implies earlier history states are unaffected
when only S positions change; verify the needed boundary through actual inputs
and installed forward behavior. Do not claim history contents and positional
retrieval are independent underlying mechanisms.

## Readout, acceptance and decision

Save full vocabulary logits at the correctly aligned x2 predictor, the signed
margin z[38]-z[999], actual vocabulary argmax and top candidates for all four
cells. Keep serialized token identity separate from the coordinate bin number.
Native diagonal replay must reproduce saved top-two values within absolute
2e-4 and exact native argmax. Preserve a no-op replay and a mutation check that
detects wrong causal slot or edits outside S. Qualify any mask representation
change before interpreting crossed cells. Parity failure is technically invalid
for the affected contrast, not a scientific null.

The advance positional-transport prediction is H_E/P_L choosing bin999 and
H_L/P_E choosing bin38, alongside correct native diagonals. Both directions
must transport the same argmax; an unrelated competing token does not count.
Each crossed cell must have vocabulary top1-minus-top2 gap greater than 4e-4
(twice the per-logit parity tolerance) for a decisive transport label; otherwise
mark that direction numerically unresolved. This is a predeclared numerical
guard, not a certified error bound or statistical confidence interval. Fresh
full-vocabulary no-op comparisons qualify any representation change.
Report each direction separately as well. The history-following pattern is both
E-history cells choosing38 and both L-history cells choosing999. Other patterns
and continuous interaction are valid outcomes, not reasons to search more doses.
Report the factorial margin interaction without pretending to estimate sampling
uncertainty from four deterministic cells. Use replay discrepancy to bound
numerical interpretation, not to tune success thresholds after seeing results.

Recompute the whole S under its assigned positions. This changes S's contextual
states as well as its queries/keys, so the estimand is not isolated final-query
readout. A history factor similarly transports a whole contextual package.
Transport supports local sufficiency of the current-row positional intervention
for this fixed decision. It does not establish a position-only clock, positional
sufficiency for the entire natural burst, physical recovery, or a general
mechanism across images. History-following or interactions instead weaken the
proposed simple positional account and determine the lead's next question.

## Execution and stop rule

One GPU initially; all GPUs are user-authorized but this small experiment does
not need a parallel model fleet. Initial cap: 12 top-level model forwards and
20 minutes of model execution; count and report qualification as well as main
cells. One owner and live invocation; no waiting for stress workloads to idle.
Concrete OOM, identity or alignment failures return immediately to the lead.
The lead may authorize a bounded repair under the user's current grant; retain
failed attempts and use a fresh output path after source changes.

No training, free generation, other checkpoint/image, layer scan or automatic
follow-up in this package. Stop after one stable four-cell candidate or a
decision-bearing technical failure. Root independently verifies artifacts and
declares acceptance before choosing any further package.

Maintained producer: `probes/training_set_completion/recurrence_position_history.py`.
Artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-position-history/`.
Bind/capture producer and used maintained dependencies, source inputs and runtime
identity before model execution. Preserve logs, failed attempts, all cell inputs,
logits, resource counts and terminal status. Root owns this unit, state, result,
catalog/index and acceptance; the execution child owns its module and artifacts.
