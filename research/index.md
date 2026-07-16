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

- [Qwen3-VL Dense Enumeration Weekly Research Report, 2026-07-13 through 2026-07-16](investigations/qwen3-vl-dense-enumeration/2026-07-13-to-2026-07-16-weekly-research-report.md) - integrated executed evidence, mechanism synthesis, implementation retrospective, and GPT-Pro handoff
- [Qwen3-VL Autoregressive Detection Research Compass](investigations/qwen3-vl-dense-enumeration/compass.md) - program-level north star, current belief register, discriminator queue, and paper-thesis boundary
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

| Question | Current belief | Decision |
|---|---|---|
| Can Qwen3 Vision-Language (`Qwen3-VL`) consume an object-specific visual control signal? | Yes, under bounded painted/post-scatter interventions; this is a privileged causal handle, not a final interface. | [Use visual designation as a causal teacher](decisions/use-visual-designation-as-a-causal-teacher.md) |
| Does pure cross-entropy (`CE`) serialization learn an order-free object ledger? | No stable order-free ledger is established. At one exact state, a coherent phrase-geometry row controls the successor, but this may still be a textual geometry-sorted serialization transition rather than visual commit. | [Separate selection, transcription, commit, and stop](decisions/separate-selection-transcription-commit-and-stop.md) |
| Is a decodable proposal representation sufficient? | No. The tested bridge produced non-specific continuation and unsafe rollout behavior. | [Require target-specific causal consumption](decisions/require-target-specific-causal-consumption.md) |
| May a row-specialized checkpoint replace the detector baseline? | No. Cursor-row specialization can preserve designation while collapsing enumeration. | [Protect native capability during specialization](decisions/protect-native-capability-during-specialization.md) |
| Should slots, a ledger, or a final architecture be built now? | Not from current evidence. Promote only the smallest mechanism that passes its discriminator. | [Let architecture emerge from hypothesis gates](decisions/let-architecture-emerge-from-hypothesis-gates.md) |
| What does repeated full-image bagging establish? | It can expose valid competing object modes at one identical prefix, but mode availability is trajectory-state dependent and bagging is not a safe final enumeration policy. | [Use bagging as an object-support probe](decisions/use-bagging-as-an-object-support-probe.md) |
| Where should the dense-enumeration program look next? | Privileged hard routing can compile tight instance-owned geometry, and one decoder-block-`23` pre-`x1` state switches a same-description spatial owner where the block-`13` control does not. The effect is almost entirely a first-coordinate basin switch. This unit is closed; `x1` causal mediation is the smallest later discriminator, while training and architecture remain unauthorized. | [Dense-enumeration research compass](investigations/qwen3-vl-dense-enumeration/compass.md) |
