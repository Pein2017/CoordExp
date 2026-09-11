# Frozen implementation task (v2)

## Goal

Implement OpenSpec change `establish-research-probe-infra-base` through tasks
1.1-4.3 in this assigned worktree. Produce a production-quality candidate for
the smallest reusable research-probe mechanics base by composing existing
owners; do not invent a new execution framework.

OpenSpec is the sole authority. Read, in order:

1. the repository `AGENTS.md` instructions;
2. `openspec/changes/establish-research-probe-infra-base/proposal.md`;
3. `openspec/changes/establish-research-probe-infra-base/design.md`;
4. `openspec/changes/establish-research-probe-infra-base/specs/coordexp-infras-research-probe-infra-base/spec.md`;
5. `openspec/changes/establish-research-probe-infra-base/tasks.md`.

## Frozen route and scope

- Tier: production candidate inside a benchmark; no promotion is authorized.
- Work only in the assigned worktree. Do not create or operate on any other
  branch, tag, worktree, process, or external artifact.
- Implement and, only when literally satisfied, check off tasks 1.1-4.3.
  Leave tasks 4.4 and 4.5 unchecked; the benchmark lead owns review and
  submission.
- Authorized write surface:
  - `src/artifacts/__init__.py`
  - `tests/artifacts/test_research_probe_infra_base.py`
  - `tests/artifacts/test_research_probe_admission.py`
  - `docs/RESEARCH_PROBE_INFRA_BASE.md`
  - `docs/AGENT_INDEX.md`
  - `docs/BRANCH_AND_WORKTREE_POLICY.md`
  - `openspec/changes/establish-research-probe-infra-base/tasks.md`
- Read existing owner modules and normative docs as needed. Do not read or edit
  the active Image2299 worktree, and do not restore or inspect a Human13
  worktree. The frozen OpenSpec contains the specimen mapping needed here.
- Preserve all unrelated work and credentials. Do not edit this prompt or its
  sibling manifest.

## Required implementation behavior

- First add the public-import regression and observe it fail for the missing
  exports before editing `src/artifacts/__init__.py`. Record the exact RED
  command and failure.
- Extend only the existing lazy `src.artifacts` facade with the exact symbols
  named by the OpenSpec. Keep implementation in existing deep owners.
- Characterize exclusive publication and independent per-producer journals
  through the public surface without weakening occupied-path failure or adding
  global ordering.
- Add the one canonical guide and the two routing links. Keep scientific
  meaning and lifecycle commands with their existing owners.
- Do not add a wrapper, runner, registry, coordinator, phase DSL, config
  object, compatibility alias, overwrite fallback, dependency, or new
  production module.
- Do not use a GPU. Run Python only through `conda run -n ms` and force the
  requested test suite to CPU with `CUDA_VISIBLE_DEVICES=-1`.
- Do not commit, push, merge, tag, unlock, move, delete, or retire anything.

## Acceptance commands

Run these after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest -q -p no:cacheprovider \
  tests/artifacts/test_research_probe_infra_base.py \
  tests/artifacts/test_evidence_journal.py \
  tests/artifacts/test_research_probe_admission.py

conda run -n ms ruff check \
  src/artifacts/__init__.py \
  tests/artifacts/test_research_probe_infra_base.py

conda run -n ms python -m compileall -q \
  src/artifacts \
  tests/artifacts/test_research_probe_infra_base.py

openspec validate establish-research-probe-infra-base --strict

git diff --check -- \
  openspec/changes/establish-research-probe-infra-base \
  src/artifacts tests/artifacts docs

git status --short
```

Inspect the final diff against the stable journal, admission, inference, and
branch/worktree contracts. The only blocking failure modes are: a hidden
coordinator or wrapper, weakened exclusive publication, global producer
ordering, scientific-semantic leakage, duplicated lifecycle authority, a
failing acceptance command, or a write outside the authorized surface.

## Budget and stop rule

Use one implementation turn, existing dependencies, CPU-only checks, and no
external retry loop. If a non-transient blocker prevents a literal task or an
acceptance command, stop and report it rather than broadening scope. Otherwise
stop as soon as tasks 1.1-4.3 and all acceptance commands pass. Do not perform
task 4.4 review or task 4.5 submission.

## Final response contract

Return only:

1. outcome: `candidate` or `BLOCKED`;
2. changed files;
3. the RED receipt;
4. each acceptance command with exact result/count;
5. any unresolved blocker or scope deviation.

Do not claim lead acceptance, user acceptance, merge readiness, or scientific
evidence.
