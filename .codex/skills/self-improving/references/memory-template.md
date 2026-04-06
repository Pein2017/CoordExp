# Memory Templates

The default local memory root for this skill is:

```text
.self-improving/
```

If your workspace uses another local path, substitute it consistently everywhere.

## `memory.md`

```md
# Self-Improving Memory

## Confirmed Preferences

## Active Patterns

## Recent Candidates
```

## `corrections.md`

```md
# Corrections Log

## YYYY-MM-DD
- Context:
- Correction:
- Scope: global | domain/<name> | project/<name>
- Status: candidate | promoted
```

## `reflections.md`

```md
# Reflections Log

## YYYY-MM-DD
- Task:
- Reflection:
- Lesson:
- Status: candidate | promoted | archived
```

## `index.md`

```md
# Self-Improving Index

## HOT
- memory.md: 0 entries

## WARM
- projects/: 0 files
- domains/: 0 files

## COLD
- archive/: 0 files
```
