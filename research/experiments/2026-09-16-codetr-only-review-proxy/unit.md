# Co-DETR-only review proxy: registration and retained-output diagnostic

Current accepted result: [results.md](results.md). The pre-user-correction replay is preserved as [retained-replay.md](retained-replay.md).

## Authority and bounded question

On 2026-09-16 the user requested project-wide TIDE-aligned unmatched analysis,
a Co-DETR-first proxy without a default Qwen-VL judge, conservative learning
admission, and lead/subagent review of unresolved cases.

From the frozen 2026-09-10 context-detector observations, what changes when
the semantic-VLM filter is removed at the SAME detector score>=.50 and
candidate/detector IoU>=.75 operating point? Decision: whether this detector-only
rule can already justify automatic teacher admission, or should only nominate
review candidates. Strongest alternative: removing the VLM improves recall
while exposing detector-supported geometry/category errors.

## Source and estimand

Source root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator`.
Use all 64 retained candidate/image identities; compare existing combined
decisions to detector-only decisions computed WITHOUT semantic observations or
reference labels. Only then score against BOTH original and lead-adjudicated
provisional visual references. These images are already unblinded historical
evidence; this is retrospective diagnosis, not new held-out validation.

Report supported clean/defective/gray counts, clean retention, support coverage,
and incremental cases. Preserve reference uncertainty and the original reference.
Neither detector agreement nor a provisional agent reference is certified GT.
Do not infer population prevalence from this image-balanced selected panel.

## Cost, stop and products

One CPU replay of existing64 observations; no model invocation, GPU launch,
threshold search, fresh visual review, new labels, teacher write or SFT run.
Stop after publishing immutable input bindings, per-case replay and a summary;
register the shared review terminology and a concrete conservative proxy design.
The 22-image training unit stays closed, with its original teacher and metrics.

Further multi-view inference and image-disjoint calibration are a proposed
follow-on, not completed or validated by this replay. Existing Co-DETR+VLM
scripts and their frozen scientific meaning are preserved.

## Review semantics

The project-wide interpretation owner is
[UNMATCHED_REVIEW.md](../../../docs/eval/UNMATCHED_REVIEW.md).
TIDE reference-relative error labels, detector support, physical judgments and
training admission are separate fields. The current choice excludes default
VLM judging; unresolved cases remain for bounded lead/subagent visual review.
