> SUPERSEDED by the user-requested closeout. Do not resume proposal enumeration. See ../closeout-v1/final-receipt.json when published.

# Current COCO22 teacher review continuation

User priority: finish current round. Owner-binding ideas are deferred; no mechanism experiment or extra reviewer calibration is a prerequisite. Preserve dirty work. No new teacher refresh/ordering/CE factor.

Accepted boundaries:
- discovery-v2: all44 rows independently cold-replayed by lead; see discovery-final-lead-acceptance.json. Not teacher truth.
- gt-review-admission-v1/lead-admission.json: all169 new rawGT covered,142 verified,22 held,1 false,4 duplicate. Original records remain unchanged; old11/227 untouched.
- annotations-v1:22 rows and79 prior unlabeled preserved, no new extras yet.
- 29 runtime/evaluator/annotation tests passed in /tmp/coco22-lead-runtime-eval-annotations.log. Research knowledge check passed before latest state-only refresh.
- reviewer-calibration-v1/user-ruling-v2.json supersedes prior C09 rejection. A bbox may validly identify one person while containing another; minor clipping acceptable. C07 was constructed, not a model rollout. Luna-max fixed windshield uncertainty and incidental-overlap cases but still over-rejected C09; bounded comparison closed in lead-max-followup.json, no more comparison work needed now.

Pending complete-package owners (native agents, same active invocation must be reconciled on resume):
- /root/visual_a: visual-review-v1/a-proposals, images196090/210584/200288/457861, proposal review plus direct missing-owner candidates; earlier GT54 package frozen.
- /root/visual_b: visual-review-v1/b-proposals, images19413/510122/116096/438671; earlier GT59 frozen.
- /root/review_luna: visual-review-v1/luna-proposals, images296894/124185, first-pass only; lead acceptance required.
- /root/review_sol: visual-review-v1/sol-proposals/image-335722, one-image proposal review.
- /root/cohort_teacher: actual22 CPU cross-consumer draft at teacher-integration-draft-v1 using current142 newGT plusold227; NOT final teacher or GPU admission. Owns fixes only coco22_data.py.

Next:
1. Inspect candidate proposal packages and exact images for accepted new owners / disputed decisions. Ground-truth proximity and model self-report alone do not establish acceptance. Existing held GT correction is not automatically a distinct new unlabeled owner.
2. Publish exact lead owner-admission receipt and actual annotations-v2 (coco22_annotations.build_version); retain originals/prior79 and version binding. Unknown physical/class cases remain explicit; no fabricated COCO classes.
3. Build final data-v1 from accepted GT sidecar plus final annotations. Lead verify old227 exact first, new11 membership, owner identities, native parser, prompt/media/mask/coordinate/EOS alignment.
4. Use runtime candidate-implementation-v3/predeclared-qualification and readback amendment. Run actual teacher-bound qualification before8GPU main256 updates. Do not use integration draft for GPU. Resolve exact CLI from current code.
5. Main source S latest sample-equal+256, freshoptimizer, fixedsampleequalCE+0.01geometry, global22 fullreplay. Source2444 control only after completed-dose genuine failure per unit.md. No hour kill. Long GPU command namedtmux/receipts and fresh durable monitor.

No main22 training launched as of this checkpoint.
