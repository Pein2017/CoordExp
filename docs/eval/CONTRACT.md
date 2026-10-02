# Evaluation meaning and evidence

Generation, parsing, scoring, assignment and interpretation are separate stages.
An evaluator must consume a consistently bound artifact family, not combine
predictions, scores or labels from different producers. Retain source, model,
configuration, dataset, row order and score provenance at the relevant boundary.
Exact artifact filenames and schemas are owned by `src/inference/`, `src/eval/`
and the local stable specs.

Name the unit and denominator before reporting a metric. Token likelihood,
conditional row realization, complete greedy trajectories, annotation matching
and distinct physical-owner coverage answer different questions. A box overlap
or category match is not by itself physical truth, and an unmatched prediction
is not automatically a hallucination on incompletely annotated data.

Keep newly recovered owners and lost incumbents separate. Keep duplicate,
unsupported, malformed, invalid-geometry and stopping debt visible rather than
folding them into a favorable aggregate. Rows with no predictions still belong
to the evaluated population. Do not pool incompatible cohorts or label versions.

Geometry conversion must be explicit on each side of a match. The parser owns
prediction normalization; the evaluator must not silently reinterpret units or
reparse text to manufacture a different prediction set. Trace/scored-row binding
and empty-output behavior deserve negative tests.

A successful import, synthetic smoke, clean process shutdown or schema-valid
receipt does not establish a model-quality improvement. Technical invalidity
means the scientific question was not answered, not that the hypothesis failed.
Visualizations are inspection aids, not a replacement for the declared metric.
