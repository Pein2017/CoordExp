# Comprehensive Recap And Memory Bootstrap Prompt

Use `$project-memory` in the repository
`/data/CoordExp/.worktrees/research-probes`.

Your job is not merely to summarize recent turns. Traverse the complete
historical main-thread session, write a comprehensive research retrospective,
and then use that retrospective to reconstruct useful project memory.

## Source

Session identifier:

```text
019f4a19-d81c-75a2-84b0-2c20379e686e
```

Local transcript:

```text
/data/CoordExp/.codex/sessions/2026/07/10/rollout-2026-07-10T03-37-15-019f4a19-d81c-75a2-84b0-2c20379e686e.jsonl
```

The transcript is approximately 831 megabytes and 128,000 lines. Process it as
a streaming historical reconstruction. Never load or print the complete file
at once.

## Read First

```text
memories/README.md
memories/config.yaml
memories/current.md
memories/template.md
handoff/customized-memory.md
research/index.md
research/investigations/qwen3-vl-dense-enumeration/compass.md
research/investigations/qwen3-vl-dense-enumeration/overview.md
```

Use the existing `research/` tree as a formal evidence map and cross-check. It
does not replace reading the session, because the conversation contains
reasoning, alternatives, user steering, and discarded ideas that may not have
been promoted into formal research documents.

## Phase One: Traverse The Whole Session

Read from the first turn through the final meaningful turn in bounded
chronological spans. Keep a lightweight private coverage ledger while working
so no large interval is silently skipped.

Prioritize:

- every user message and correction;
- substantive assistant reasoning and proposals;
- approvals, reversals, stopping criteria, and changes of direction;
- experiment launches, outcomes, negative results, and interpretation changes;
- named checkpoints, datasets, artifacts, documents, and source paths;
- mathematical and statistical arguments;
- implementation and workflow lessons that affected research quality.

Read compact event summaries and tool-call identities during the first pass.
Inspect raw tool output only when a retained conclusion, metric, failure, or
artifact identity requires verification. Ignore repeated status polling and
large outputs that add no meaning.

When later turns correct earlier ones, preserve the evolution and make the
current interpretation clear. Do not flatten the history into a false story in
which the final view was obvious from the beginning.

## Phase Two: Write A Comprehensive Recap

Create:

```text
memories/notes/019f4a19-d81c-75a2-84b0-2c20379e686e-comprehensive-recap.md
```

This is the primary deliverable. It should be detailed enough that a strong
researcher who never saw the transcript can understand the long session's
trajectory and why the project reached its present state.

Organize it naturally, but ensure it covers:

- the original goal and how the problem definition evolved;
- the major chronological research phases;
- the strongest observations and their exact evidence boundaries;
- competing mechanisms and how belief in them changed;
- mathematical or statistical models that shaped decisions;
- experiment designs, controls, findings, negative results, and unresolved
  confounders;
- proposed treatments, training attempts, why some failed, and what remained
  promising;
- data quality, annotation ambiguity, object-versus-geometry distinctions, and
  evaluation lessons;
- important code, infrastructure, worktree, and research-workflow lessons;
- durable user preferences, constraints, and research philosophy;
- rejected, deferred, or superseded paths and why they changed;
- the final open questions, stopping criteria, and best continuation options;
- a curated map of the most important files, artifacts, checkpoints, and
  session handles.

Distinguish clearly between executed evidence, interpretation, speculation, and
historical reconstruction. Link to formal research documents instead of copying
their full contents.

## Phase Three: Build Usable Memory Notes

After the comprehensive recap exists, extract additional natural-language notes
under `memories/notes/` where separate chapters would materially improve later
recall. Let the session's real transitions determine their boundaries. Do not
impose a fixed taxonomy or create files merely to satisfy a target count.

Preserve high-value rejected or abandoned reasoning when it prevents repeated
work. Merge or delete duplicate, stale, or misleading memory. The note set
should complement the comprehensive recap rather than fragment it into a second
database.

## Phase Four: Reconstruct The Terminal State

Replace `memories/current.md` with a concise account of the state reached at
the end of the session:

- the active goal;
- the current understanding and strongest alternatives;
- what is established, tentative, rejected, or still ambiguous;
- active or pending experiments and implementation work;
- important constraints and user decisions;
- the best next actions;
- the minimum reading path for a new main-thread agent.

This file is a continuation brief, not another historical recap.

## Phase Five: Self-Audit

Before finishing:

1. Confirm that the beginning, middle, and end of the transcript were all
   traversed.
2. Check the recap against the existing research compass, overview, experiment
   units, and results it cites.
3. Remove unsupported precision and clearly label uncertain reconstruction.
4. Ensure `current.md` reflects the terminal state rather than an interesting
   earlier phase.
5. Ensure no raw transcript, large tool output, credential, secret, cache, or
   generated index was copied into `memories/`.

You have full create, read, update, and delete access inside `memories/`. Keep
all changes there. Do not modify `research/`, `docs/`, OpenSpec, source code,
other skills, or `AGENTS.md`. Do not stage or commit. Preserve unrelated
working-tree changes.

## Completion Report

Report:

- the chronological source range actually covered;
- the recap and memory files created, rewritten, merged, or deleted;
- the major historical phases recovered;
- the most important belief changes;
- what remains uncertain or intentionally omitted;
- whether the recap and `current.md` together allow another agent to continue
  without reopening the 831-megabyte transcript.
