---
name: rtk-token-saver
description: "Use when CoordExp shell output is likely to be noisy: broad search, docs reads, git diff/status/log, tests, logs, file discovery, or repo orientation."
---

# RTK Token Saver

Use `rtk ...` as the first shell path when compact output is more useful than verbatim stdout.

## Good Fits

```bash
rtk git status --short --branch
rtk git diff --stat
rtk git diff --name-only
rtk grep "TargetSymbol" src
rtk read docs/IMPLEMENTATION_MAP.md
rtk conda run -n ms python -m pytest tests/test_example.py
```

Preserve project wrappers under RTK; in this repo, tests should usually be:

```bash
rtk conda run -n ms python -m pytest <target>
```

## Skip RTK

Use raw commands when:

- the user asks for exact output;
- stdout is machine-readable or parsed downstream;
- quoting/heredocs/shell state are delicate;
- the command is a narrow exact line read such as `sed -n`;
- the command is already tiny;
- RTK has no useful rewrite.

Probe support with:

```bash
rtk rewrite "git status"
rtk gain --project
```

## Pairing

- RTK: noisy shell, docs, logs, tests, git summaries.
- Serena: Python symbols, references, call graph, precise code edits.
- Raw shell: exact text, config snippets, JSON, narrow line reads.
