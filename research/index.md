# Research

CoordExp research knowledge is organized as an OKF-style Markdown bundle.

Use this tree for research development, idea lifecycle records, investigations,
mechanism notes, experiment interpretation, negative results, source-linked
interpretation, and provenance handles. Raw worktree intake remains under
`docs/history/worktree-union/` and is linked from `research/`, not copied into
it. Current coding, architecture, infrastructure, and operator behavior remain
in `docs/`. Stable compatibility-sensitive contracts remain in `openspec/`.

## Entry Points

- [Ideas](ideas/) - new research directions that may become CoordExp capabilities
- [Investigations](investigations/) - bounded analysis, ablation, diagnosis, checkpoint surgery, and post-analysis
- [Mechanisms](mechanisms/) - reusable semantic explanations and failure models
- [Archive](archive/) - raw, superseded, deprecated, or non-main-reading-path material

## OKF Conventions

- `index.md` files are routers and contain no frontmatter.
- Non-router Markdown files use YAML frontmatter with `type: idea`, `type: investigation`, or `type: mechanism`.
- `type` denotes the top-level semantic bucket: idea, investigation, or mechanism. Document role is carried by filename, headings, tags, and links.
- Markdown links carry graph edges; surrounding prose explains the relationship.
- Artifact files stay outside `research/`; research docs store handles and interpretation.
