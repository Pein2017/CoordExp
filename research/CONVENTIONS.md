# Agent-oriented research knowledge convention

This file owns research placement, reading and maintenance. It grants no experiment, architectural or publication authority. The current user's instructions outrank historical plans; current implementation and compatibility contracts stay with their existing `docs/` and `openspec/` owners. The research-flow Skill routes here, rather than maintaining another layout policy.

## Flat root, meaningful questions

The whole repository serves this research topic. Do not reintroduce a topic/program wrapper or classify knowledge by its changing lifecycle. The active shape is:

```text
research/
  index.md                     # current frontier, user boundary, reading map
  CONVENTIONS.md               # this placement/maintenance contract
  story.md                     # how evidence changed the research direction
  glossary.md                  # shared definitions and ambiguous historical aliases
  alternatives.md              # selective unresolved ideas and reopening conditions
  questions/<question>.md      # evidence, interpretation, counterexamples, decisions
  literature/
    index.md                   # public papers mapped to local research questions
    papers/<paper-id>-<slug>.md # source-bounded critical reading notes
    sources/                   # retained intake material with attributed provenance
  experiments/
    catalog.jsonl              # complete searchable metadata, including cold records
    <unit-id>/
      unit.md                  # proportionate outline, frozen once execution starts
      state.json               # current lifecycle and result/protocol pointers
      results.md               # accepted evidence and bounded interpretation
```

No `ideas/`, `decisions/`, `mechanisms/`, `investigations/`, `archive/`, `reports/`, `handoffs/`, or `progress/` buckets inside the active research root. Temporary migration staging must be emptied before closeout. A new question earns a page through a distinct scientific decision and evidence chain, not through a quota or a speculative future need. Reconsider a program layer only if an actually independent second topic exists.

Maintained experimental code lives in `probes/<direction>/`. Existing `outputs/research/...` roots and immutable study/run identifiers remain unchanged: flattening the knowledge tree is not permission to rename model artifacts. Retired protocols, evidence records, code, consultations and old schemas live in manifest-bound `docs/history/`, outside the maintained reading path. Reusable public literature and its source material live in `literature/`; publication age or completed reading is not a reason to archive them.

## Read for a task, not for a file count

Fast catch-up is `index.md` → current `state.json` → accepted result. Deep catch-up adds `story.md` → relevant question pages → decisive original sources. Search `experiments/catalog.jsonl` and the source manifests for expansion. Maximize useful information recovered, not minimum line count. Keep denominators, conditions and limiting counterexamples beside the claims they limit. Do not require every old experiment to be read on every new task.

When reviewing our own research, interpreting a mechanism, choosing a next discriminator, borrowing a method/metric, or assessing novelty, consult [the public-literature map](literature/index.md) when external evidence could change that judgment. Start with the local question and nearest accepted result/counterexample, then select the relevant paper notes and original sections. Compare the actual models, tasks, populations, conditioning, interventions, controls and outcomes; state whether the external result corroborates, challenges, or leaves our interpretation unresolved. Search beyond the retained collection when a decision-bearing gap or novelty claim requires it. A missing entry is not evidence that no prior work exists.

This is targeted retrieval, not a mandatory whole-library review: reuse already checked sources within their scope, and do not require literature work for routine runtime/status updates. Reading a paper neither authorizes an experiment nor changes an active frozen contract. Keep a proposed extension separate until the current unit's evidence and boundary are reconciled.

## One owner per statement

| Statement | Owner |
|---|---|
| Current task and reading route | `index.md`, as an attributed synthesis of the exact unit state/result |
| Lifecycle, latest user boundary, protocol/result pointers | The unit's `state.json`; no independently maintained current-status tables |
| Frozen question, population, contrast, cost, criteria and stop rule | `unit.md` or the state's exact preserved protocol reference |
| Accepted numbers, identities, evaluation semantics and bounded verdict | Accepted `results.md` plus named immutable execution/evaluation receipts |
| Current scientific interpretation, alternatives and route choice | The relevant `questions/*.md` |
| External paper identity, tested conditions, evidence, interpretation limits and reusable methods | `literature/papers/<paper-id>-<slug>.md` |
| Paper-to-question retrieval and retained intake provenance | `literature/index.md`, linking paper notes and `literature/sources/` |
| Why the direction changed | `story.md`, not a chronological run log |
| An important untested or unresolved idea | A concise entry in `alternatives.md`, linked to its question and predecessor |
| Record identity and retrieval | `experiments/catalog.jsonl`; no live metrics or inferred process state |
| Old bytes, old grants and retired interpretations | Frozen `docs/history/` sources |

Short attributed overlap is useful for catch-up; duplicated ledgers, full repeated backgrounds and separately edited volatile counters are not. A directory name, source code search result or tool success wrapper is not evidence of an accepted scientific claim.

## Public literature and critical synthesis

Keep one note per paper, named by a stable identifier and short slug, and link it from every relevant question route rather than duplicating it across topic folders. An unread recommendation needs only an index entry; create a note when there is a substantive reusable claim, method or critique. Record the primary URL/version and reading scope (intake only, abstract, selected sections, or full text), so a secondary summary or a partial reading cannot masquerade as a verified full-paper account. Preserve original intake bytes and their provenance separately from edited synthesis. Canonical source links and section/table pointers are sufficient; do not copy full copyrighted papers into notes.

A paper note covers five things proportionately: source and reading scope; the question and tested conditions; observations versus the authors' interpretation and our assessment; limits and strongest remaining alternatives; and local question/predecessor links with a reason to reread. Mark our derivations and proposed tests explicitly. Decodability, intervention sensitivity, selective causal use, natural behavior and trainability are different evidence claims. Check privilege from annotations, forced histories or external selectors and preserve counterexamples and negative/invalid distinctions when transferring a result.

The relevant `questions/` page owns cross-paper synthesis and comparison with local evidence. Update it when reading changes our belief or next discriminator; update `story.md` only when the research direction changes. Keep experiment counts, acceptance and live status with their existing owners, not in a second literature ledger. The index is a question-oriented reading map, not a publication-date queue or an instruction to reproduce every paper. An important untested alternative may enter `alternatives.md` with a concrete reopening condition.

Maintain useful notes in place as papers are revised, recording which version supports a claim and any decision-changing revision. Only superseded local material that leaves the maintained reading path belongs in `docs/history/`; retain useful old papers and justified criticisms in the active literature collection.

## Units and evidence lifecycle

During design, `unit.md` is a proportionate executable outline. Freeze decision-bearing content at execution. A scientific-semantic change needs an explicit amendment or new unit; an ordinary mechanical repair does not require another protocol ceremony. Never rewrite the original denominator, conditioning, intervention, criterion or stop rule after observing outcomes. A migrated paused unit may reference its original frozen protocol instead of creating a retroactive one.

`state.json` uses `schema_version: 1`, `unit_id`, `lifecycle`, `evidence`, `disposition`, `state_as_of`, `protocol`, `result`, `state_source`, `boundary`, `not_authorized` and `next_action`. Paths are repository-relative; `result` may be null before acceptance. Lifecycle is `planned|ready|running|blocked|paused|closed|superseded`; evidence is `none|partial|unreviewed|accepted|invalid`; disposition states the bounded scientific outcome independently. Accepted evidence, an incomplete stage and a paused task can coexist. Historical units keep their old schema as provenance, never as present launch authority.

Keep live/paused or actually continuing units in `research/experiments/`. At a substantive closeout, distill the useful result and safely retire the source record to `docs/history/`; keep its stable ID and exact paths in the catalog. Do not move whole directories merely because another week passed. Reopening creates an explicit new current contract with a predecessor link; it does not rewrite frozen evidence. Historical phase syntheses and operational studies are labeled as such rather than counted as newly executed scientific experiments.

## Preserve sparks without preserving obsolete furniture

Before calling an idea new, inspect its question page, catalog predecessors, actual result and remaining scope. Record what was tried, what was answered, what remains open, what changes now and which observation changes the decision. A new name/date is not a new discriminator.

A tested idea is absorbed into the question/story and catalog, not kept as a parallel active document. Retain unresolved alternatives when evidence is missing, an attempt was technically invalid, a materially new condition changes its value, or it remains a consequential counterfactual. Distinguish these cases explicitly. A failed recipe is not a ban on a whole family; a later related experiment is not automatically the matched control an older idea lacked. When the answer is still unknown, keep the source pointer rather than manufacturing closure.

`alternatives.md` is selective, not an exhaustive brain-dump or a second evidence atlas. Its question links own full scientific context. Keep any special architectural idea as an optional hypothesis with a decisive comparator, never as the assumed implementation. Retire dated governance, old agent prompts and old resource grants without carrying them into current scientific constraints.

## Maintain only what changed

Close from evidence outward: accepted receipt/result → unit state → question when belief changes → story when direction changes → index when frontier/user boundary changes. Update catalog paths/records when necessary. A single optimizer step does not require updates to every layer. Create a handoff only for a real transfer; integrate its useful delta into owners, then retire the transport. Do not create/update durable Project Memory without explicit user authorization.

Keep observations, supported inference, hypotheses and untested proposals distinct. A technically invalid contrast leaves that contrast unanswered, not scientifically negative. Teacher-forced scores, exact conditional patches and mechanics smoke support their own surfaces, not automatic native-greedy transfer. Unknown/unmatched is not negative, and neutral reward need not imply zero gradient. Preserve checkpoint, parameter surface, token/geometry definitions, decode policy, population and evaluator meaning; do not flatten incompatible histories into a leaderboard.

## Preservation and validation

Before any move/removal, inspect fresh Git, current bytes, references and live consumers; preserve original bytes and logical coordinates. Use exact-source hashes, an explicit source-to-archive mapping and a bounded check. Never alter sealed receipts or source hashes to hide an identity break. Readable source recovery and runnable original-context replay are separate claims. Do not keep old-path aliases permanently: migrate demonstrated consumers, verify their decision-bearing behavior, and remove the alias. Historical links resolve through the source-aware read-only resolver, not by reviving the removed taxonomy.

From the verified research-probes checkout:

```sh
python -B scripts/research/check_research_knowledge.py check
python -B -m unittest discover -s tests/research -p 'test_research_knowledge.py'
git diff --check
```

The single knowledge checker covers the live layout/links, catalog/state, original-source preservation and known retirement records. Consumer changes also need their focused CPU tests and an actual data-read equivalence check. Historical missing sources must be explicitly reported or bound to verified Git recovery, never ignored merely to pass. The obsolete decision-graph checker is retired, not left as a misleading zero-node success.

Checks do not certify all scientific interpretations, external artifacts, every Markdown construct or model execution. No GPU run is needed for document restructuring. Report exact scope, source preservation, validation/gaps, Project/path/branch, remaining work and active jobs. Git publication, memory writes and research resumption remain separate authorizations.
