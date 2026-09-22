# Independent researcher: owner recovery with continuous geometric deduplication

You own this bounded research branch in an independent Codex task. The user and
the originating lead remain in a different task discussing research direction.
Do not depend on that lead staying active or resuming you after every phase.
Follow the applicable AGENTS.md. Direct user exchanges are Chinese; technical
records and worker briefs default to concise English.

Suggested researcher model: gpt-6-astra / medium. This is a routing suggestion,
not a claim about the model actually running or measured superiority.

## Objective and user rulings

Determine whether a continuously applied training-time deduplication regularizer
can retain or improve Rweak's recovery of distinct annotated owners while
reducing repeated predictions and annotation-relative false positives.

The user explicitly clarified:

> 是否能找到更多的 owners(而不是duplication)!
> IOU达到 50%其实也已经算是 OK 了; 优化已经`找到`的 object 的 IOU 并不是 top priority;
> 只要 IOU>0.95,可以无脑`惩罚` ... 不管是不是same-desc.

These rulings supersede the previous experiments' IoU60/80 nondecrease gates
for NEW work. Preserve historical results and their original acceptance rules.

- Primary metric: category-compatible, one-to-one GT-owner matches at IoU >=0.50.
  Count each durable owner ID once. Report TP, FP, FN, precision, recall, and
  paired recovered/lost/net owner IDs relative to the same baseline.
- Duplicate eligibility: within one image, prediction-to-prediction IoU >0.95,
  regardless of description/category and without requiring GT attribution.
  Equality at0.95 does not qualify. This is a user-approved research predicate,
  not proof that such pairs can never be distinct physical objects.
- Deduplication is intended to remain active during optimization, not merely
  remove boxes after generation. Postprocessing may be a cheap diagnostic but
  is not acceptance evidence for a training regularizer.
- IoU60/80, invalid/drop counts, length and caps are diagnostics, not automatic
  vetoes on meaningful owner recovery. Do not silently invent a new FP tolerance
  or all-metrics-must-improve gate. Show the trade-off when it is not dominant.
- Research usefulness over production ceremony: no generic framework, repeated
  hash audits, automatic review committee, model benchmark or broad sweep.

## Workspace and minimum reading path

Work in /data/CoordExp/.worktrees/coco-gt-correction-portfolio
(observed HEAD73b8b3cc2; recheck live state). The checkout contains uncommitted
prior research scripts/tests/records and a modified experiments index. Preserve
them; a fresh Git worktree alone would omit these untracked implementations.
The originating lead will not edit your owned experiment/code surfaces in
parallel. Declare any shared-file ownership before delegating writes.

Read only what is needed, in this order (paths below are relative to cwd):
1. Applicable AGENTS.md and this handoff's explicit user rulings.
2. research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-owner-focus-ablation/unit.md
   and results.md: existing R/M/Rweak recipe and evidence, NOT new acceptance gates.
3. research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-07-coco-weak-correction-dose/results.md
   and dose.py: saved-dose comparison and unused confirmation input.
4. scripts/research/train_coco_gt_correction.py, coco_gt_correction_bank.py,
   reduce_coco_owner_focus.py, eval_coco_owner_focus.py, and their nearest tests
   as needed. Trace the real objective/consumer before selecting a loss.

Artifact base: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration
- Existing recipe/outputs: 2026-09-07-coco-owner-focus-ablation/focus-v1/
- Promising checkpoint: that directory's Rweak/checkpoint-000064/
- Existing512 panel: that directory's inputs-v1/holdout512.jsonl
- Available but NOT evaluated confirmation input:
  2026-09-07-coco-weak-correction-dose/dose-v1/confirmation-input/confirmation512.jsonl
  and manifest.json. It contains512 images/3886 owners; do not use its outcomes
  to tune a regularizer and still label it fresh confirmation.
- Proposed new output root: 2026-09-08-coco-owner-recovery-dedup/.
  Check for existing work before creating an invocation; never overwrite a run.

## Established evidence, not hypotheses

On the existing512 images/3759 owners, the current matcher gives:

| Arm | TP50 | FP50 | Precision | Recall |
|---|---:|---:|---:|---:|
| Source |2225|3037|42.28%|59.19%|
| R64 |2233|4916|31.24%|59.40%|
| M64 |2258|4796|32.01%|60.07%|
| Rweak16 |2240|3410|39.65%|59.59%|
| Rweak32 |2250|3244|40.95%|59.86%|
| Rweak64 |2310|3369|40.68%|61.45%|

Rweak64 recovers188 Source-missed owners and loses103, net+85. These TPs already
exclude repeated matches to the same owner. FP here means valid predictions
unmatched to the available annotations, including duplicates; invalid/parser-
dropped outputs are separate. This is not a COCO AP table or a hallucination count.
The512 panel has now informed selection; it is not a new independent test.

Rweak is NOT a new optimizer: it uses R's complete correction teacher and mask,
scales correction loss per image by sum(M_mask)/sum(R_mask), keeps R's denominator
and canonical anchor, and trains the existing language DoRA surface. The earlier
paired recipe used original Source step2444, train256,64 global updates, effective
image batch32 and seed20260908; inspect the unit for exact settings. Token-weight
mass is matched, not gradient norms. Existing16/32/64 are one training trajectory.
Old duplicate counts used a narrower GT-attributed same-category predicate; do
not reuse them as counts under the user's new cross-description geometric rule.

## First independently owned package

Proceed without asking the originating lead for discoverable facts:
1. Recount the NEW duplicate predicate on saved Source/Rweak64 outputs, with
   per-image concentration and a clearly defined redundant-box count (do not
   confuse pair count with number of removable predictions). Report raw TP/FP;
   if you inspect deduplicated outputs, keep that table explicitly separate and
   rematch owners rather than assuming TP is unchanged. Use CPU, not new decode.
2. Select the smallest viable differentiable training penalty and state where
   its model-generated repetition signal comes from. Explain how gradients
   reach trainable parameters, how often the signal refreshes, and normalization.
   Do not call GT-only teacher supervision or a nondifferentiable NMS operation
   a model deduplication regularizer. Retain the user's predicate exactly.
3. Write one concise new unit with the hypothesis, strongest alternative,
   primary contrast and bounded execution proposal. Existing recipe risk checks
   need not be repeated unless the new loss changes their boundary.
4. Implement and CPU-check the minimal probe-level loss/measurement surface if
   it can be done within the above semantics. Verify cross-description overlap,
   threshold boundary, no-repeat behavior, meaningful gradient signal, and
   consumer TP/FP accounting through the smallest decision-bearing checks.

Important contrast: training from original Source with/without dedup tests
prevention; continuing from Rweak64 tests repair. Do not compare a longer-trained
dedup branch with the old64 checkpoint and attribute the difference solely to
dedup. Propose one matched contrast, not both routes or a full R/M/W portfolio.

This handoff does not invent a GPU-hour budget or transfer last night's exhausted
one-to-two-round authorization. Before GPU smoke/training, present the concrete
loss, starting checkpoint, matched comparator, update count, GPU/wall-time ceiling
and stopping rule to the user IN YOUR OWN TASK for approval. Continue useful CPU
analysis/preparation meanwhile; do not wait for the originating lead to schedule
your implementation. Do not launch new training, inference or a coefficient sweep
until that bounded resource/experiment proposal is authorized.

## Optional team and durable execution

You may delegate bounded independent subwork when it saves total effort: at most
two simultaneous direct native children, no grandchildren. You own integration
and corrections. Set fork_turns:none, model and effort explicitly. Use Astra/low
or Sol/medium for exact analysis/execution; Sol/high for a demonstrated difficult
implementation boundary. No mandatory scout or reviewer. All share your scope
and eventual resource allowance; delegation does not expand either.

When GPU work is later authorized, own its durable launcher, terminal artifacts,
wait/recovery and result reduction yourself. Reconcile existing processes first.
Follow wake-me-up's actual contract and accept only an armed receipt as evidence
of later wakeup. Never assume native child completion starts an idle root turn.
Do not repair runtime services or build cross-task messaging infrastructure.
Run Python through conda run -n ms. Shared GPU use is normal; act only on actual
conflicts. Keep all failed/capped images in the declared denominators.

## Deliverable, communication and stop

Own this experiment directory and the smallest relevant script/test additions.
Do not rewrite old result files, shared skills, AGENTS.md, credentials or memory.
No commit/push/archive/cleanup, new dependency, model family or dataset acquisition.

Return a compact research update: what was observed; what remains hypothetical;
TP/FP and unique-owner gains/losses; the recommended single next contrast and its
resource request; exact artifact paths; blocking questions if any. Finish the
CPU/design package once this is sufficient to authorize or reject the GPU round.
After a later authorized round, complete its analysis and stop at that round's
rule, not at a convenient intermediate checkpoint or an unrequested new search.

Send only decision-bearing changes/completion to the originating discussion task
if an actual cross-task tool is available. Otherwise persist results here and
tell the user in your own task; missing outbound messaging must not prevent you
from completing an authorized package. Never claim a message or wake succeeded
without a receipt. Do not depend on copied conversation history beyond this brief.
