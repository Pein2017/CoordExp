# Research Decisions

Decision nodes are the mutable belief layer of the CoordExp research graph.
They summarize what the current evidence changes about the next experiment.
They are non-normative: they do not define runtime behavior, stable interfaces,
or final architecture.

## Active Decisions

- [Prefer x-then-y ordering for sorted detection targets](prefer-x-then-y-object-ordering.md)
- [Use visual designation as a causal teacher](use-visual-designation-as-a-causal-teacher.md)
- [Separate selection, transcription, commit, and stop](separate-selection-transcription-commit-and-stop.md)
- [Require target-specific causal consumption](require-target-specific-causal-consumption.md)
- [Protect native capability during specialization](protect-native-capability-during-specialization.md)
- [Let architecture emerge from hypothesis gates](let-architecture-emerge-from-hypothesis-gates.md)

## Lifecycle

- `active`: governs the current research route.
- `revisit`: contradicted or materially narrowed; awaiting a replacement.
- `retired`: no longer governs work, with the replacing evidence recorded.

The frontmatter `evidence` paths must resolve inside the repository. Relations
between decisions use stable decision IDs. Run
`python scripts/research/check_research_graph.py` to check the graph.
