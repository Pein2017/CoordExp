# Command entries and maintenance scope

Start from the exact checkout and task. Model execution belongs to `src/` or the
owning research family, not to a second implementation in this directory. See
[checkout policy](../docs/BRANCH_AND_WORKTREE_POLICY.md) and the
[research capability map](../probes/README.md).

## Current routes

| Task | Entry | Owner / boundary |
|---|---|---|
| Training | `python -m src.train --config ...` | `src.training`; use current `configs/coordexp_infras` configuration, not the old mainline CLI schema. |
| Inference | `python -m src.infer --config ...` | `src.inference`; configuration owns generation and scoring policy. |
| Saved detection evaluation | `python scripts/evaluate_detection.py --artifact-dir ... --out-dir ...` | `src.eval.detection_consumer`; requires a compatible scored artifact. |
| Current knowledge check | `python -B scripts/research/check_research_knowledge.py check --live-only` | `scripts.tools.research_knowledge`; current catalog/state/links without historical migration inputs. |
| Knowledge plus historical evidence plumbing | `python -B scripts/research/check_research_knowledge.py check` | Also checks retained migration coverage and frozen exposure data; no scientific revalidation. |
| Research experiments | The named `probes.<family>` entry | Family documentation and the current unit contract, never an old command alone. |

## Retired inference launch chain

The old `run_infer.py`, `run_infer_eval.sh`,
`analysis/run_ckpt_pair_confidence_eval.sh` and
`pipelines/run_rollout_stability_probe.sh` are removed. Their dependency on the
old `src.infer.pipeline` package was already unsupported. The current YAML
entry is not a flag-compatible replacement for their legacy configurations.
No forwarding aliases are kept. Git retains the historical code; existing
results, source captures and result interpretation were not rewritten.

Saved-output analysis remains in `analysis/`. A reader can interpret an old
format without supporting the launcher that originally produced it.

## Retired training and debug chain

The old `train.sh`, `train_stage2.sh` and both `pipelines/train_task_manager`
entries are removed together: their dependencies on `src.sft`, the old
`ConfigLoader`/rollout preflight and Stage2 launcher no longer exist. No retained
executable caller outside that chain was found. Its tmux queue/step/time stopping
semantics were operator machinery, not a current research scheduler. No running
process was stopped or restarted during retirement.

`postop_confidence.py` and `tools/dump_rollout_text.py` are also retired; they
depended on removed legacy inference/evaluation APIs. Old example comments and
historical runbooks are provenance, not compatible current commands. Saved-result
readers remain; a file such as `confidence_postop_summary.json` does not require
keeping its obsolete producer executable.

## Other retained tooling

`tools/`, `analysis/` and `pipelines/` contain task-specific tools. Presence of a
file is not a promise that every historical CLI runs on the current environment.
In particular, proxy/export and historical data-preparation wrappers are not
the default production or research route. Their old schemas need separately
verified consumers, not automatic import-name substitution. `merge_coord.sh` is an
explicit export operation, not permission to mutate an existing model package.

The existing `scripts/research` strict admission/support adapters still have
real consumers. They remain optional, not a mandatory experiment workflow.
New experimental implementations belong to a meaningful `probes/<family>`;
use a thin CLI only when an operator command actually needs one.

Shared shell primitives remain in `_lib/backbone.sh`. Resolve transfer tools
from the current authorized runtime skill catalog, rather than assuming a
repo-local copy of an agent skill exists. Uploads and model execution require
their own task authority.
