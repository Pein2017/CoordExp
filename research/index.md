# Research

CoordExp research knowledge is organized as an OKF-style Markdown bundle.

Use this tree for research development, idea lifecycle records, probe units,
investigations, mechanism notes, experiment interpretation, negative results,
source-linked interpretation, and provenance handles. Raw worktree intake
remains under `docs/history/worktree-union/` and is linked from `research/`,
not copied into it. Current coding, architecture, infrastructure, and operator
behavior remain in `docs/`. Stable compatibility-sensitive contracts remain in
`openspec/`.

## Entry Points

- [Decisions](decisions/) - current research belief updates, constraints, and route gates linked to their evidence
- [Ideas](ideas/) - synthesized research directions that may become CoordExp capabilities
- [Investigations](investigations/) - bounded analysis, ablation, diagnosis, checkpoint surgery, and post-analysis
- [Mechanisms](mechanisms/) - reusable semantic explanations and failure models
- [Archive](archive/) - raw, superseded, deprecated, or non-main-reading-path material

## OKF Conventions

- `index.md` files are routers and contain no frontmatter.
- Non-router Markdown files use YAML frontmatter with `type: decision`, `type: idea`, `type: investigation`, or `type: mechanism`.
- `type` denotes the top-level semantic bucket: decision, idea, investigation,
  or mechanism. Document role is carried by filename, headings, tags, and links.
- Markdown links carry graph edges; surrounding prose explains the relationship.
- Decision nodes summarize a current research choice without becoming a product
  contract. Each decision names its evidence, belief update, and next
  discriminator. Evidence may revise or retire a decision later.
- Artifact files stay outside `research/`; research docs store handles and interpretation.
- Individual probes belong under the owning idea or investigation, usually
  `research/ideas/<topic>/experiments/<unit-id>/unit.md` or
  `research/investigations/<topic>/experiments/<unit-id>/unit.md`.
  Mark them with `role: research-unit`, `authority: non_normative_research`,
  and `promotion_status: not_promoted` unless a later promotion review says
  otherwise.
  Promote only durable metric/artifact/config/runtime contracts into
  `openspec/`; keep exploratory hypotheses and measured outcomes here.

## Layout Contract

```text
research/
  index.md
  decisions/<slug>.md         # current evidence-backed research choices
  ideas/<topic>/              # synthesized idea reading path
  ideas/<topic>/experiments/<unit>/unit.md
  investigations/<topic>/     # bounded cross-cutting analysis
  investigations/<topic>/experiments/<unit>/unit.md
  mechanisms/<slug>/          # reusable explanation, only when repeated
  archive/                    # non-main-reading-path material
```

Research units are not OpenSpec changes. They may recommend an OpenSpec
promotion when a result hardens into a stable compatibility-sensitive contract.

## Active Frontier

| Question | Current belief | Decision |
|---|---|---|
| Should sorted detection targets use four or eight coordinates, and which top-left ordering? | Four and redundant eight-coordinate supervision were operationally near-equivalent on the tested val-200 slice. The stronger actionable signal is to sort by top-left x then y. | [Prefer x-then-y ordering for sorted detection targets](decisions/prefer-x-then-y-object-ordering.md) |
| Can Qwen3-VL consume an object-specific visual control signal? | Yes, under bounded painted/post-scatter interventions; this is a privileged causal handle, not a final interface. | [Use visual designation as a causal teacher](decisions/use-visual-designation-as-a-causal-teacher.md) |
| Does pure-CE serialization learn an order-free object ledger? | No evidence yet. It learns a strong order-conditioned, mostly coordinate-level transition. | [Separate selection, transcription, commit, and stop](decisions/separate-selection-transcription-commit-and-stop.md) |
| Is a decodable proposal representation sufficient? | No. The tested bridge produced non-specific continuation and unsafe rollout behavior. | [Require target-specific causal consumption](decisions/require-target-specific-causal-consumption.md) |
| May a row-specialized checkpoint replace the detector baseline? | No. Cursor-row specialization can preserve designation while collapsing enumeration. | [Protect native capability during specialization](decisions/protect-native-capability-during-specialization.md) |
| Should slots, a ledger, or a final architecture be built now? | Not from current evidence. Promote only the smallest mechanism that passes its discriminator. | [Let architecture emerge from hypothesis gates](decisions/let-architecture-emerge-from-hypothesis-gates.md) |
