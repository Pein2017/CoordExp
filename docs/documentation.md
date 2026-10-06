# Documentation ownership and retention

This is the shared documentation policy; checkout READMEs add local routes and
recovery context, not copies of this policy. Agents are the primary readers.
Keep material that changes a decision or prevents a non-obvious mistake.
Source search discovers implementation; it does not confer user authority,
accepted evidence or scientific ownership.

## Choose the content owner

`docs/` keeps durable contracts, policy, architecture boundaries and design
reasons. Runtime behavior remains checkout-local: use its source, typed configs,
CLI help, tests and exact-checkout CodeGraph. A useful implementation explanation
may stay beside its code; do not maintain a parallel call graph, option inventory
or support matrix that merely falls behind it. Stable compatibility requirements
belong to the local OpenSpec. A sibling's newer implementation is not local truth.

Main owns shared checkout, storage and Agent-integration policy. Infrastructure
owns version-local implementation boundaries. Canonical research-probes owns
scientific questions, protocols, results and interpretation. Its
`research/conventions.md` owns research organization: tentative ideas, negative
results, counterexamples and unresolved disagreement can be worth retaining.
Do not apply docs' stability test to research evidence.

Prefer one maintained owner with explicit links. Absorb conclusion-changing
conditions, controls, numbers and limitations before removing a duplicate.
Temporary coordination and superseded execution status need not remain separate
files. Handoff remains a supported way to start or transfer a conversation; a
handoff prompt is not a second knowledge ledger or a memory-store replacement.

## Names and reading paths

Use English lowercase kebab-case for ordinary document names and directories.
Keep consumer/ecosystem names such as `README.md`, `AGENTS.md`, `SKILL.md` and
OpenSpec's artifact names. Keep intentionally user-facing language exceptions.
Use stable topics or roles, not `final`, `revised` or `v2` packaging. Dates belong
to genuinely chronological records, use `YYYY-MM-DD`, and retain event time;
a dated parent usually makes another date in its child filename redundant.

Let directories carry topic and files carry role where that helps retrieval.
Prefer `overview` for orientation, `design` for rationale, `results` for an
executed comparison and `findings` for synthesis; these are suggestions, not a
required file set. `README.md` may combine routing and necessary explanation;
keep `index.md` too only when it serves a distinct collection. Do not add
frontmatter, thin folders or templates without a real consumer.

## Retire meaningfully

Absorb useful semantics into the formal owner; retain a source only for a named
independent evidence need; otherwise delete it. Do not create archive/reference
inboxes, redirect tombstones, duplicate source snapshots or new SHA inventories.
Git recovery applies only to bytes actually committed. Inspect untracked unique
information before disposal; rerunning or rewriting it is not the original evidence.

A frozen source that remains evidence keeps its original identity and bytes.
It may instead be retired after dependency review and necessary distillation;
frozen does not mean permanent. Preserve OpenSpec archives as framework history,
and do not remove code/data/raw artifacts while tidying a mixed document tree.
Historical paths may remain historical; prioritize live Agent/skill/consumer and
key evidence links over reconstructing every old environment.

## Existing closed documentation recovery

The finite [retirement seal](../manifests/documentation/retirement-20261003.json)
binds the original Git commit and docs tree. Removed files are recoverable from
that exact snapshot, including useful material that has not been scientifically
reinterpreted. Sealing is not a claim that every old conclusion is accepted or
fully distilled. Preserve original provenance and distinguish invalid runs from
scientific null results. Never re-execute recovered historical code by default.

Each checkout's `docs/README.md` links its own existing seal; it covers that
retirement's declared old docs, not arbitrary research, progress or untracked
files. Ordinary new cleanup does not extend or reproduce these packages.
Use `python -B -m scripts.check_documentation --show-original <old-doc-path>` in
the owning checkout to read an original, or `git show <commit>:<old-doc-path>`
for known Git-backed content. Its optional `--verify-recovery` is for a concrete
recovery/integrity need, not a recurring all-history check. Neither operation restores a tree or authorizes execution.
Git ancestry preserves the snapshot in ordinary descendants; it is not an
independent off-machine backup. Keep the sealed commit reachable when explicitly
rewriting history or exporting a repository. Do not append new history to this
closed retirement package.

The current `docs/README.md` is the only live inventory. The checker validates
its linked assets, local docs links and absence of runtime prose dependencies.
A new lasting asset requires an intentional README update, not a change to the
closed historical seal or a second catalog. Boilerplate generation and
indiscriminate archival growth are not default workflows.
