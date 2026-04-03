# Memory Operations

Use the smallest possible file for every write.

## Default Read Flow

1. Read `.self-improving/memory.md`.
2. If the task obviously matches a domain, read the matching file in `domains/`.
3. If the task is specific to one project or workflow, read the matching file in `projects/`.
4. Skip unrelated files.

## Default Write Flow

1. Classify the lesson:
   - correction
   - confirmed preference
   - domain rule
   - project override
   - reflection candidate
2. Write it to exactly one primary file.
3. Update `index.md` if counts changed materially.

## File Meanings

- `memory.md`: durable global preferences or rules
- `corrections.md`: explicit corrections and repeat history
- `reflections.md`: post-task lessons not yet proven durable
- `domains/*.md`: lessons that apply only within a domain like code or writing
- `projects/*.md`: overrides for one project or workflow
- `archive/*.md`: inactive or superseded items

## Suggested User-Facing Queries

- "What have you learned?" -> summarize recent corrections and durable rules
- "Show my patterns" -> show `memory.md`
- "Show project patterns" -> show the relevant file in `projects/`
- "Memory stats" -> summarize counts from `index.md`
- "Forget X" -> remove or archive the matching entry, then confirm

## Maintenance

Keep maintenance conservative:

- merge obvious duplicates
- archive superseded items rather than deleting them
- prefer append-only updates
- do not rewrite large files without a concrete reason
- if `.self-improving/` is git-tracked, avoid churny edits that create low-value commit noise

Do not introduce autonomous cron-style maintenance. In Codex, run maintenance only when the user asks or when the skill is explicitly being exercised.
