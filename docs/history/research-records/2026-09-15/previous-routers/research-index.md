# Research

CoordExp research knowledge is organized as an Open Knowledge Format
(`OKF`)-style Markdown bundle.

Use this tree for research development, idea lifecycle records, probe units,
investigations, mechanism notes, experiment interpretation, negative results,
source-linked interpretation, and provenance handles. Raw worktree intake
remains under `docs/history/worktree-union/` and is linked from `research/`,
not copied into it. Current coding, architecture, infrastructure, and operator
behavior remain in `docs/`. Stable compatibility-sensitive contracts remain in
`openspec/`.

## Entry Points

- [Qwen3-VL Dense Enumeration Current Compass](investigations/qwen3-vl-dense-enumeration/compass.md) - current question-oriented synthesis for finite-panel programmability, owner coverage, semantic compression, and causal row/layer evidence
- [Qwen3-VL Dense Enumeration Detailed Overview](investigations/qwen3-vl-dense-enumeration/overview.md) - historical hypothesis and evidence atlas
- [Dense Enumeration Experiment Router](investigations/qwen3-vl-dense-enumeration/experiments/index.md) - selected protocols/results and bounded technical records
- [CoordExp Historical Experiment Knowledge Handoff](investigations/coordexp-experiment-knowledge-handoff/) - audited historical results, negative evidence, execution lessons, and provenance ledgers; never a current-runtime authority
- [Decisions](decisions/) - current research belief updates, constraints, and route gates linked to their evidence
- [Ideas](ideas/) - synthesized research directions that may become CoordExp capabilities
- [Investigations](investigations/) - bounded analysis, ablation, diagnosis, checkpoint surgery, and post-analysis
- [Mechanisms](mechanisms/) - reusable semantic explanations and failure models
- [Archive](archive/) - raw, superseded, deprecated, or non-main-reading-path material

## Open Knowledge Format (`OKF`) Conventions

- `index.md` files are routers and contain no frontmatter.
- Non-router Markdown files use YAML Ain't Markup Language (`YAML`) frontmatter
  with `type: decision`, `type: idea`, `type: investigation`, or
  `type: mechanism`.
- `type` denotes the top-level semantic bucket: decision, idea, investigation,
  or mechanism. Document role is carried by filename, headings, tags, and links.
- Markdown links carry graph edges; surrounding prose explains the relationship.
- Every abbreviation, arm code, hypothesis code, metric symbol, and coined name
  must be expanded and defined at first use or in a local terminology registry.
  A machine path or legacy run name does not count as a definition.
  Expansion alone is insufficient when the expanded term still hides its
  operational behavior; prefer a plain-language mechanism or objective name
  and omit the abbreviation when that is clearer.
- Decision nodes summarize a current research choice without becoming a product
  contract. Each decision names its evidence, belief update, and next
  discriminator. Evidence may revise or retire a decision later.
- Artifact files stay outside `research/`; research docs store handles and interpretation.
- Research execution artifacts use the logical root
  `outputs/research/<investigation>/<unit-id>/<run-id>/`, where `unit-id` means
  the immutable research-unit identifier, `run-id` means one immutable
  execution identifier, and `investigation` means the stable owning
  investigation identifier. The resolved absolute
  root, manifest, receipt, checkpoint, config, and evidence scope must be
  recorded by the owning unit. A path alone is not execution evidence.
- Individual probes belong under the owning idea or investigation, usually
  `research/ideas/<topic>/experiments/<unit-id>/unit.md` or
  `research/investigations/<topic>/experiments/<unit-id>/unit.md`.
  Mark them with `role: research-unit`, `authority: non_normative_research`,
  and `architecture_promotion_status: not_promoted` unless a later
  architecture-promotion review says otherwise.
  Promote only durable metric/artifact/config/runtime contracts into
  `openspec/`; keep exploratory hypotheses and measured outcomes here.

## Evidence Flow

```text
executed artifacts
  -> research unit (protocol, handles, observations, bounded verdict)
  -> investigation synthesis (competing explanations and belief state)
  -> research decision (current route choice)
  -> next discriminator
```

A mechanism note requires repeated support across independent units and a
successful novel prediction. An OpenSpec change owns only a reusable
implementation or compatibility contract; it never owns the hypothesis or the
scientific verdict.

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

outputs/research/
  <investigation>/<unit-id>/<run-id>/  # external executed evidence
```

Here `<slug>` is a stable human-readable node identifier, `<topic>` is a stable
idea or investigation identifier, `<unit>` and `<unit-id>` are immutable
research-unit identifiers, `<investigation>` is the owning investigation
identifier, and `<run-id>` is one immutable execution identifier.

Research units are not OpenSpec changes. They may recommend an OpenSpec
promotion when a result hardens into a stable compatibility-sensitive contract.

## Active Frontier

The current frontier is maintained in the [Qwen3-VL research compass](investigations/qwen3-vl-dense-enumeration/compass.md)
with historical context in its [overview](investigations/qwen3-vl-dense-enumeration/overview.md).
Operational benchmarks and dated snapshots are routed through the
[archive](archive/index.md).
