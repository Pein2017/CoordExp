# Lead ruling: gradient admission failure and authorized continuation

Date: 2026-09-18 UTC. Decision owner: root task01a0a3d5-dc45-7693-8467-4801aa7190df.
Execution owner: existing916-worker task01a0a81a-9e32-7db1-bd07-86fa601f4276.
Cwd: /data/CoordExp/.worktrees/research-probes.

## Evidence independently inspected

The frozen gradient job ended technical-invalid after7 forwards and2 backwards,
with0/18 population scenes. Shared-path CE all-selected directional derivative:
autograd0.0007213164935819805; finite difference0.0006258487701416016;
absolute residual0.0000954677234403789 exceeds bound0.00005606582467909903.
Original/restored delta SHA256 agrees:
57830aee3f15fe2af38e7ac332f2fd9d0efcff7266bdb41d4f7cb957a4f1c719.
Producer SHA256:
3df5796f45663edc7f4537b0eb64901a72184041d89583889f934fcb6ddcac5c.

Root inspected job-closure.json, runtime/receipt.json, finite-difference.json,
and producer terms()/finite-difference call sites. A concrete independent flaw
exists in probes/training_set_completion/untied_gradient.py: the fallback
`if not axis.requires_grad: axis=logits.sum()*0` overwrites a real nonempty axis
loss during the finite-difference torch.no_grad() calls. Saved endpoint/interior
axis plus/minus values are zero; their small derivative discrepancies then fit
inside atol. Those apparent passes do not validate the axis derivative.
This flaw does not explain the separate CE discrepancy. Neither failure
establishes a defect in checkpoint training or the model's true gradient.

## Decision and bounded correction

1. Preserve this attempt as technical-invalid, not a scientific null. Do not
   interpret its gradient norms, endpoint effects or tied/untied comparisons.
2. Fix only the concrete axis-value/autograd-context bug. Preserve the original
   producer bytes/hash and failure artifacts before editing. Use one meaningful
   CPU RED/GREEN check through the actual loss-term path: a nonzero hinge has
   equal numerical value with grad enabled and disabled, and the real zero-box
   case keeps its zero and permits zero-gradient backward. Test must fail under
   the original fallback. Do not change the objective, scene mask or denominator.
3. No gradient model rerun, epsilon/tolerance relaxation, precision or direction
   sweep is released by this ruling. The lane remains unanswered after the CPU
   correction. A later numerically justified gate successor needs a separate
   root release. No additional review chain is needed for this correction.

## Continue the existing package now

Resume from the existing natural runtime-handoff.json, session64789 and its six
producers, PIDs3016639-3016644; root confirmed those exact commands are live.
Reuse completed qualification/groups. Never launch duplicate producers.
Finish units A and B within the original protocols and remaining shared budget.
All8 GPUs remain available; place admitted readout work on free lanes when its
frozen source-order event selection can be completed, rather than assigning a
new gradient experiment. Do not select first-finishing images in place of the
frozen source order. Preserve exact replay companions and source bindings.

Feedback unit D remains CPU admission only, model budget ZERO. No training,
optimizer updates, new checkpoint stages, new cohorts or scientific arms.

Worker owns its existing probe code, unit results/state and output roots. Root
owns frontier/catalog and acceptance. Preserve unrelated dirty changes. Return
an integrated candidate with A/B evidence, C technical-invalid plus the bounded
CPU correction receipt, D admission/HOLD, exact counters/hashes and ended-job
witnesses. Distinguish incomplete work from accepted results. Reuse current
native implementers if still useful; no extra delegation is required.

Use event-driven completion rather than polling or restarting. If a runtime
handoff is unavoidable, persist exact live owners, exit witnesses and remaining
work; do not present an idle worker as completion. Continue until the registered
package is ready for root acceptance or a material new ruling is needed.
Append PACKAGE_CANDIDATE, PACKAGE_BLOCKED or NEEDS_LEAD to the existing
coordination/worker-events.jsonl as appropriate. Root will rearm its monitor.

Evidence root:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-gradient-path-accounting/
Natural runtime handoff:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/runtime-handoff.json
