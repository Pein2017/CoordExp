# Agent-oriented research knowledge convention

This is the repository owner for research **placement, reading and maintenance**. It does not authorize experiments, change scientific acceptance rules, or supersede the current user's instructions. Runtime/implementation contracts remain in `docs/` and `openspec/`; the research-flow Skill routes here rather than maintaining a second layout policy.

## 1. Optimize for catch-up and retained knowledge

All primary readers are agents. Optimize useful information recovered per reading path, not minimum line count or a human-facing dashboard. Preserve decisive observations, denominators, counterexamples, abandoned hypotheses, technical failures, untested ideas and reopening conditions. Semantic compression is encouraged only when the original evidence remains recoverable.

Offer two paths: **fast** = current context plus the current state/result; **deep** = research story plus selected question pages and their sources. Neither path requires reading every historical experiment. The full catalog and source manifest support targeted expansion. Do not bury a caveat needed to interpret a headline merely to meet a length target.

## 2. Organize by research program and question

```text
research/
  index.md
  CONVENTIONS.md
  <program>/
    index.md
    current.md
    story.md
    vocabulary.md
    questions/<question>.md
    ideas.md
    experiments.jsonl
    experiments/<unit-id>/
      unit.md          # new unit's outline/protocol; freeze at execution
      state.json       # only live lifecycle/evidence/disposition routing
      results.md       # accepted facts and bounded interpretation
  ideas/               # retained independent idea owners
  decisions/           # existing cross-cutting research decisions
  mechanisms/          # only genuinely supported reusable explanations
  archive/             # retained older curated collections; no new raw dump

docs/history/research-records/<capture>/  # immutable source provenance
probes/<direction>/                     # maintained experiment code
outputs/research/<program>/<unit-id>/<run-id>/  # durable execution evidence
```

A program may cover many experiments and hypotheses. A question page is not a claim that a mechanism is proven. Add directories only for a real distinction; do not recreate a generic `investigations`, `progress`, or `misc` dumping ground. Existing independent ideas/decisions need not be bulk-migrated merely to match a new taxonomy.

The sole legacy `research/investigations` alias exists for demonstrated old-path consumers. It is not a category, a current-state entry or an authorized writing surface. See the [archive guide](../docs/history/research-records/2026-09-15/README.md).

## 3. One owner per kind of statement

| Statement | Owner | Other surfaces do |
|---|---|---|
| Current lifecycle, latest user boundary, result/protocol pointers | The unit's `state.json` | Link or give an explicitly contextual summary, not a separately maintained status table |
| Frozen question, contrast, population, conditions, cost and stop rule | `unit.md` or the state's exact preserved protocol reference | Link; never silently rewrite it after seeing results |
| Accepted metrics, owner identities, measurement rule and bounded verdict | Accepted result plus its named immutable output receipt | Summarize only decision-bearing facts with attribution, not copy the full ledger |
| Current interpretation and strongest alternatives | Owning question page | Link the interpretation and distinguish it from observation |
| Why research choices changed | `story.md` | Preserve causal/decision transitions, not every command or round |
| Metadata retrieval | `experiments.jsonl` | IDs, titles, topic tags, protocol/result/state paths; no live counters or inferred runtime state |
| Historical bytes and old assertions | Manifest-bound `docs/history/` sources | Treat dated `current`, `running` and grants as historical data |
| Temporary transport | A bounded handoff | Integrate its useful delta into owners, then archive it |

A result can be accepted while the stage is incomplete and the task is paused. Keep these axes separate. A stored permission describes an earlier grant; fresh execution still needs current user authority and runtime re-observation.

## 4. New-unit lifecycle without a growing notebook

During design, `unit.md` is a proportionate outline. Freeze its decision-bearing content when execution begins. A small exploratory probe does not require a production-sized preregistration. Add a separate explicit amendment only for a semantic change; ordinary mechanical repair does not require a new ceremony. Never change a frozen denominator, criterion or stop rule retroactively.

`state.json` uses `schema_version: 1`, `unit_id`, `lifecycle`, `evidence`, `disposition`, `state_as_of`, `protocol`, `result`, `state_source`, `boundary`, `not_authorized`, and `next_action`. Paths are repository-relative. `result` may be null before accepted output; provenance must explain the actual state. Lifecycle is one of `planned`, `ready`, `running`, `blocked`, `paused`, `closed`, `superseded`; evidence is one of `none`, `partial`, `unreviewed`, `accepted`, `invalid`. Disposition states the bounded scientific outcome in plain language or a defined label, independently of lifecycle.

Use a run/evaluation receipt for changing counters and attempt identities. Write an accepted result when there is an actual interpretable outcome. For accepted-result corrections, preserve the prior version and record the changed evaluator/semantics; do not overwrite hash-bound records. Long iteration histories belong in provenance, not appended indefinitely to `unit.md` or `current.md`.

Historical units are exempt from the new schema. Their old status fields are frozen source labels, not current state. A migrated continuing unit may point directly to the byte-preserved original protocol, rather than manufacture a retroactive new preregistration. Historical catalog entries without a state are not automatically complete, failed, or resumable.

## 5. Preserve ideas and avoid repeating roads

Before proposing a new experiment, read the relevant question page, search the catalog and read the nearest predecessor result/protocol. Record a short predecessor note: what was tested, what it answered, what remains unresolved, what will differ and what observation changes the decision. A new date or name is not a new discriminator.

Distinguish a bounded scientific negative, a technically invalid attempt, an unexecuted plan, a missing-support gate, a superseded interpretation and an unexplored idea. Do not convert “this recipe failed” into “this family cannot work.” Do not convert an old proposal into a standing launch queue.

Each substantive question page should retain the evidence chain, strongest remaining alternative, closest counterexample/failed shortcut and reopening condition. `ideas.md` holds important dormant alternatives with source links. Preserve source-local names for retrieval but define their meaning once in the shared vocabulary; every materially different checkpoint, metric or arm still needs an unambiguous local identity.

## 6. Update only the surfaces whose meaning changed

Close from evidence outward: accepted receipt/result → `state.json` → question page when belief changes → story when the research trajectory changes → current context when the frontier/user boundary changes. Update catalog metadata when paths or units change. Do not rewrite all layers after every optimizer step or repeat full background in each result.

The current page is replaceable context, not append-only history. The story explains transitions, not every experimental episode. Short attributed overlap is allowed when required for safe catch-up; redundant full tables and independently edited volatile counters are not.

Create a handoff only for a real transfer, not for every conversation or closeout. A live entry must resolve to an owning current/state/result/question/decision surface, not to a transcript, reviewer prompt or consumed handoff. Do not create or update durable project memory unless the user explicitly requests it.

## 7. Code, artifacts and relocation

New maintained code belongs in ordinary `probes/<direction>/` packages using existing public owners; runtime outputs belong at the declared output root. The research tree holds interpretation and compact state/index metadata, not model payloads, logs, cache files or another execution framework. Archive retired experiment source without pretending it is a maintained library.

Before moving a source, search actual callers and evidence bindings, inspect fresh Git/content, preserve original bytes and logical path in a manifest, and distinguish readable-source recovery from executable replay. A compatibility alias may be retained only for a demonstrated dependency, with an explicit limitation and a check. Do not add recursive alias forests or rewrite old hashes to hide a broken binding. `Path(__file__).resolve()` and parent-depth assumptions require special attention.

Do not flatten old results into one leaderboard. Retain checkpoint/parameter surface, natural versus forced/teacher-forced conditioning, data-use population, owner/category/geometry policy, denominator, decode policy, technical validity and claim scope. A documentation migration must not manufacture a scientific promotion.

## 8. Verification and closeout

Run from the verified research-probes checkout:

```sh
python -B scripts/research/check_research_knowledge.py check
python -B -m unittest discover -s tests/research -p 'test_research_knowledge.py'
conda run -n ms python -B scripts/research/check_research_graph.py
git diff --check
```

The knowledge check verifies live local links, catalog/state references, source preservation and the legacy JSON-reader boundary. Its historical-link report distinguishes source-time gaps from new live-link failures. It does not validate external artifacts, scientific correctness, all Markdown syntax, or a model run. No new GPU run is required merely to reorganize documentation.

At closeout report actual file/route changes, source preservation, tests and known gaps, Project/path/branch, affected sessions/jobs, and retained unrelated work. Commit, push, publication, memory changes and experiment resumption are separate actions, not implied by maintaining this tree.
