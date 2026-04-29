# Tasks

## 1. Spec Synthesis

- [x] Create a clean spec-only worktree from current `main`.
- [x] Inspect `feat/training-runtime-pipeline-integration` and extract only
      the durable OpenSpec ideas.
- [x] Inspect `feat/agent-research-runtime` and extract only the durable
      mission / recipe / run / evidence ideas.
- [x] Draft this consolidated OpenSpec change without porting implementation
      code.
- [x] Validate the OpenSpec change strictly.

## 2. Deprecate Imperfect Implementations

- [x] Remove deprecated worktrees:
      `feat/training-runtime-pipeline-integration`,
      `feat/agent-research-runtime`,
      `agentic/refactor-training-pipeline-architecture`,
      `clean/agent-research-runtime`, and
      `command/agent-research-runtime`.
- [x] Delete the corresponding deprecated branches after their worktrees are
      removed.
- [x] Keep only the new `feat/training-runtime-architecture-spec` worktree as
      the active `feat` branch, plus unrelated explicitly-kept work such as
      `codex/et-rmp-ce`.

## 3. Future Implementation Gates

- [x] Implement the training setup plan and runtime profile as the first code
      slice.
- [x] Add tests before each implementation slice.
- [x] Prove Stage-1 and Stage-2 setup behavior is preserved before moving
      larger blocks out of `src/sft.py`.
- [x] Refresh the change against current `main` Stage-1 set-continuation branch
      runtime docs and route pipeline-manifest required-namespace errors
      through the runtime profile.
- [ ] Do not move math-bearing or artifact-bearing code without dedicated
      parity tests.
- [ ] Introduce mission / recipe / run orchestration only after the training
      setup seam is stable.
