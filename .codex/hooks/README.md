# Conservative RTK hook

`rtk-pretooluse.py` owns rewrite safety for Codex and the single Pi adapter,
`../../.pi/extensions/coordexp-rtk.ts`. Qualified runtime: RTK **0.48.0** at
`/root/.local/bin/rtk` (must be on the caller's `PATH`); the Pi lookup uses
`/root/miniconda3/envs/ms/bin/python -B`, without a shell. The lookup never
executes the requested command itself.

## Automatic whitelist and raw exclusions

Automatic support is deliberately narrower than the former broad hook:

- Direct `git status`, `git diff`, `git log`.
- Direct `rg`, `grep`, and `pytest` for human-facing output. Pytest options
  are limited to `-q`/`-v` repetitions, `-s`, `-x`, `-h`, `-V`, `--help`,
  `--version`, and `-k`/`-m` with a following value. Other options stay raw.
- Simple unquoted `&&`, `||`, `;` chains only when every segment is eligible.
- Exact three-argv `bash/sh/dash/zsh -c/-lc COMMAND` wrappers only when the
  inner command is eligible and the original wrapper contains no `$` or backtick.

Other commands, including **ruff, npm, make, Python, uv and conda runners**, stay
raw. Previous automatic capabilities are **not** all preserved. Manual
`rtk ...` remains available and is never wrapped again.

The hook leaves the original bytes untouched for collection (`--collect-only`,
`--co` and conservative abbreviations), known machine/count/file-list/report
flags, interpreter-sensitive invocations (including absolute Python paths),
leading assignments or environment/runner prefixes, explicit executable paths,
pipes/redirections, multiline commands, comments, backslashes, unsupported shell
syntax, and wrappers with trailing `$0`/positional argv. Shell wrappers with `$`
or backticks remain raw to avoid moving expansion between shells. Direct search
expansions can receive a textual prefix; pytest expansions remain raw.

Known scientific artifact directories (`outputs`, `research`, `results`, `data`,
`datasets`, `checkpoints`, etc.) and suffixes (`.json[l]`, `.csv`, `.tsv`, `.npy`,
`.npz`, `.parquet`, `.safetensors`, `.pt`, `.log`) also stay raw. These checks are
conservative heuristics, not a complete detector of scientific or exact-output
intent. **Use a raw bypass whenever exact output or research meaning matters.**

## Raw bypass and Pi loading

Set either `RTK_DISABLED=1` or `RTK_HOOK_DISABLE=1` in the agent environment.
For one call, use `RTK_DISABLED=1 COMMAND` or `RTK_HOOK_DISABLE=1 COMMAND` (leading
assignments already exclude automatic rewriting). Bypass does not remove an
explicit, manually written `rtk` prefix.

In the main checkout, trusted-project discovery auto-loads the one adapter from
`.pi/extensions/coordexp-rtk.ts`; restart or `/reload` to discover it. Do not add a
duplicate settings entry or second adapter. Project trust and main-owned resource
enablement still apply; this work does not change `.pi/settings.json` or profiles.
The adapter handles only model/nested `bash` tool calls, mutates
`event.input.command`, and leaves timeout and other fields unchanged. User `!`/`!!`
commands have no `user_bash` interception. Neither owner grants permission or
returns `permissionDecision: allow`; existing permission handlers still apply.

Missing RTK/interpreter/owner, failed or invalid lookup, cancellation, timeout,
and oversized commands/output fall back to the original command. Commands are
limited to 32 KiB; the adapter bounds combined lookup stdout/stderr to 128 KiB
and lookup time to 3 seconds (Python's RTK lookup has a 2-second timeout).

## Qualification and token limits

Run only the focused, CPU-only fixtures from the checkout root:

```bash
PATH=/root/.local/bin:$PATH /root/miniconda3/envs/ms/bin/python -B -m pytest .codex/hooks/tests -q -s
node --test .pi/tests/coordexp-rtk.test.mjs
```

Tests cover counterexample RED/GREEN, actual git/rg/pytest exit codes, quoting,
interpreter identity, raw collection denominator and positional argv, bounded
lookup failures, and installed Pi SDK discovery/dispatch through the real Bash
consumer with a permission block. Fixture tests print UTF-8 stdout/stderr byte
counts separately for raw and actual runs, with a `rewritten` indicator; those are local presentation measurements,
not tokenizer measurements or a promise of token/cost savings. Small outputs may
expand. Timing text and fixture paths can change the byte counts across runs.

Observed small-fixture stdout+stderr bytes before the usage-error exclusion:
Git status 67→66; clean diff 0→1; matching rg 11→11; missing-file rg 95→95;
passing pytest 112→17; failing pytest 530→159; empty-selection pytest 23→24.
Raw and rewritten exit statuses agreed (git 0; rg 0/1/2; pytest 0/1/5/4).
These byte changes include intentionally discarded information, not lossless
compression or validated model-token savings.

RTK compression is intentionally **lossy** and may move or erase diagnostics.
The actual RTK 0.48.0 invalid-pytest-option fixture preserved exit 4 but reduced
360 stderr bytes to one stdout newline; unsupported pytest options now remain
raw, and that fixture asserts identical raw stdout/stderr. This does not establish
that every other pytest usage/configuration error retains its diagnostic. Preserve raw command output at its proper artifact owner for
scientific denominators, machine consumers, audits and reproducible evidence.
RTK tee output is not durable research evidence. No model/GPU qualification is
implied by these small fixtures.
