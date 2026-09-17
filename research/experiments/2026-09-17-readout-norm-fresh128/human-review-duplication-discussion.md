# Human review and duplication discussion

Status: lead-synthesized discussion and accepted CPU recount, 2026-09-17.
No new model execution, training, annotation edits, or approved successor protocol.
The completed fresh128 experiment and its frozen metrics remain unchanged.

## Human evidence and scope

The user reviewed many images in `user-comparison-v1/fresh/png` and reported
that many nonduplicate annotation-unmatched predictions denote real objects;
remaining cases include localization/extent differences, weak visual evidence,
and ambiguous instance granularity. This is meaningful human evidence for
changing the research priority, without a measured population prevalence or
blanket admission of all unmatched rows. No exhaustive review is required.

Seven highlighted image IDs: 99184,84241,114820,313465,495443,483867,542582.
Three Astra xhigh advisors independently handled census, mechanism/prior evidence,
and owner/evaluation semantics. The lead reran the census, independently recounted
literal repeats, checked original annotation metadata, and viewed only the
unpainted model-input images495443/542582 for the decisive ambiguities.

## Evidence changing the interpretation

The visualization's `dup-cand` is the number of same-category prediction PAIRS
with pixel IoU>=.30, including matched/matched pairs. Purple is an overlap hint,
not an adjudication of repeated physical identity. The accepted study's strict
counter instead counts each later valid row once for any earlier box with
bin-space IoU>.95, irrespective of category. These are distinct proxies.

| Fresh128 measure | Original | Normalized |
|---|---:|---:|
| Exact valid repeated rows beyond first occurrence |586|37|
| Images with those exact repeats |6|4|
| Exact invalid repeated rows |446|0|
| Accepted strict near-repeat rows |604|40|
| Images with strict near-repeats |8|4|

Exact valid repeats by category, O/N: person327/1, book201/1,
carrot43/33, bird15/0, apple0/2. Person is emitted on63 images in each arm,
but its strict repetition is confined to one image. Two images99184/356238
contribute491/604 original strict rows. Thus row-weighted category concentration
cannot establish category-specific failure prevalence. Bird is a current
counterexample to restricting the problem to person/produce/books; historical
bottle/knife loops provide further counterexamples.

The census replays64 raw-file SHA bindings and all256 image-arm saved counters.
Same-category/pixel sensitivity is small for the strict count (O601 versus
accepted604; N40 in either), but neither predicate establishes physical identity.

### The selected examples

- **99184:** original emits342 person rows and reaches the cap. The exact same
  person box repeats261 consecutive times from one-based row82. Normalized
  emits26 person rows plus a ball and reaches EOS. This is a clear recurrence
  witness, separate from the overlay's55,314 pair hints.
- **495443:** normalized adds7 orange rows, with maximum pair IoU about.498;
  neither arm has exact or >.95 repeats. The unpainted input visibly contains
  a cluster of fruit at the upper edge, and raw GT lists only9 banana annotations.
  New discoveries, extent errors and repeated physical owners may coexist;
  the entire orange group must not be preclassified as a negative burst.
  Its y1=0 is also compatible with real frame clipping, not sufficient endpoint
  pathology evidence.
- **542582:** neither arm has exact/strict repeats; overlay pairs decrease3→2
  while class-aware matches decrease14→12. Raw COCO additionally contains
  traffic-light crowd annotation901000542582; the processed frozen bank has29
  ordinary entries versus30 raw annotations including that crowd. Raw crowd
  context is not represented by ordinary overlay FP labels. This finding does
  not by itself verify every traffic-light prediction.
- **313465:** class-aware matches increase6→8 despite pair hints increasing4→6.
  Overlapping carrot pieces and part/group conventions need owner judgments.
- **84241/483867:** examples of annotation/category/extent interpretation, with
  no exact/strict repeats. A real object can have a reference-IoU miss.
- **114820:** the tie is an uncertain visibility obligation in the supplied
  view, not a verified absent object. Judge at actual unpainted model-input
  resolution; preserve raw annotation and isolate uncertain future obligations.

## Mechanism: what is supported and what is not

Working hypothesis: recurrent history and coordinate decisions interact,
allowing a spatially plausible owner hypothesis to win repeatedly; output-row
norms affect which trajectory is entered or escaped. Annotation granularity,
density, similarity, small size and occlusion can increase ambiguity but are
confounded. The present evidence does not identify annotation inconsistency
as the training-origin cause of hundreds of repeated emissions.

Do not assume that the model lacks all coverage memory. The
[July22 bottle counterfactual](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md)
found spatially specific suppression from a previously covered owner, while
the first differing coordinate still preferred the duplicate even when the
complete distinct-owner row had a higher average score. This is a bounded
counterexample: an available history signal need not control greedy selection.

Avoid repeating an already answered training question. The
[Sep08 matched dedup experiment](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-coco-owner-recovery-dedup/results.md)
reduced repeats by207 but had G19/L14, net+5 with interval[-11,+27]; distinct
owner recovery was not established. Reduced overlap can also mean box jitter,
invalid output, or premature stopping, rather than recovered owners.

External analogy, not CoordExp mechanism proof: the
[DITTO study](https://papers.nips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf)
observed context-dependent self-reinforcement of repeated text under greedy
generation despite scarce such repetitions in its corpus. The
[COCO paper](https://vision.ics.uci.edu/papers/LinMBBGHPRZD_ECCV_2014/LinMBBGHPRZD_ECCV_2014.pdf)
describes explicit annotation/verification procedures and crowd grouping;
grouped or ignored instances are not automatically careless annotation.

## Lead recommendation: a bounded inference discriminator, not training yet

Decision question: at a naturally reached confirmed duplicate, can a change
of one complete object row restore subsequent free enumeration of distinct
owners without sacrificing prior credible owners?

Proposed small panel: a few unambiguous recurrence cases, plus dense distinct-owner
overlap and healthy controls. Reuse candidates already present in saved rollouts;
do not make teacher-bank completion or broad new review a prerequisite.

Compare original continuation, a valid alternative expression of the same owner,
and one already-supported distinct-owner row. Use the same checkpoint, image,
native prefix, category/token length where feasible, and total attempted-token
budget; intervene for one row, then use original free greedy. The forced owner
itself receives no recovery credit. Score subsequent owner G/L, including
credible unlabeled owners, plus duplicate identity, invalidity and EOS/cap.
Review only changed owner clusters and suppression conflicts, within a fixed
small budget; uncertain gains are not accepted and uncertain losses remain visible.

This can distinguish useful complete-row route changes from mere shortening
or superficial overlap reduction. It does not alone separate owner-set memory
from spatial scan advancement; perturbation magnitude and route differences
remain explicit alternatives. Reproducing the July22 local-versus-row score
comparison at these current recurrence states would test its transfer, not
constitute discovery of a new mechanism.

An advisor instead favored applying the existing norm policy through the first
divergent complete row, then withdrawing it. That is a cheaper timing contrast
and remains a useful alternative, but primarily tests compression of the norm
policy rather than owner-specific duplication. No withdrawal or branch experiment
has been launched or frozen by this discussion. Neither successful forcing nor
a pulse would establish trainability or population generalization.

Post-hoc NMS cannot release past generation budget. Global penalties on repeated
`person` or coordinate tokens can suppress legitimate neighboring instances.
The intended target is repeated physical identity, not repeated category words,
coordinate bins, red overlays, or overlapping boxes generally.

## Reproducible evidence

- [Census and definitions](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/duplication-discussion/census.json)
- [CPU recount script](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/duplication-discussion/census.py)
- [Original annotation context](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/duplication-discussion/annotation-context.json)
- [Existing review axes](../../../docs/eval/UNMATCHED_REVIEW.md): physical identity,
  visibility, class and geometry remain separate; no blanket GT deletion.
