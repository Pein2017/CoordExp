## Context

CoordExp is a research stack, so failures that “continue silently” are usually worse than failing fast: they hide data bugs, can leak mutable state across samples, and make paper-ready runs hard to debug and reproduce.

Current sources of unnecessary engineering (high-signal examples from the audit):
- Blanket `except Exception: pass` blocks in dataset encoding paths (`dense_caption.py`, `unified_fusion_dataset.py`) that can hide sample metadata loss and prompt leakage.
- Private helpers in trainers/pipeline modules with zero call sites that inflate maintenance surface and invite drift.
- Deprecated config knobs that are no-ops but still appear in code/specs, leading to redundant warnings and unclear contracts.

Constraints:
- YAML-first experiments; avoid adding new CLI flags.
- Preserve geometry invariants (never drop/reorder coords); training uses `do_resize=false`.
- Maintain Qwen3-VL chat-template compatibility; do not edit upstream HF model internals.

Data flow (reference shape):
YAML config -> dataset preprocessors/packing -> training/inference -> JSONL artifacts -> eval/vis outputs.

## Goals / Non-Goals

**Goals:**
- Remove dead/unreferenced helper code (verified by zero call sites).
- Define and enforce a narrow “silent failure” policy:
  - core execution paths are fail-fast or explicitly logged,
  - only explicitly-justified I/O sinks suppress exceptions.
- Ensure per-sample prompt injection cannot leak across samples (restore deterministically).
- Reduce contract noise by aligning specs with simplified deprecation handling.

**Non-Goals:**
- Changing training algorithms, packing behavior, geometry semantics, or evaluation metric definitions.
- Introducing new external dependencies.
- Introducing new config knobs or CLI flags.

## Decisions

1. **Policy-first simplification**
   - Add a small local helper (or minimal shared pattern) to make “best-effort vs fail-fast” choices explicit and consistent.
   - Alternative considered: fix call sites ad-hoc. Rejected: encourages inconsistent patterns and regressions.

2. **Fail-fast for core dataset encoding**
   - Remove blanket exception swallowing around sample metadata attachment and prompt injection in dataset `__getitem__` / encoding.
   - Keep best-effort behavior only for truly optional telemetry that cannot affect training inputs/labels/artifacts.

3. **Prompt override restoration is mandatory**
   - Replace “try to set / try to restore” with a deterministic mechanism (e.g., a context manager) that guarantees restoration.
   - Restoration failure is treated as fatal to prevent cross-sample prompt leakage.

4. **Dead code removal is gated by evidence**
   - Only remove helpers proven to have zero in-repo references (repo-wide search + symbol reference scan).
   - Keep public entrypoints stable; avoid removing anything imported from outside `src/`.

5. **Deprecated keys are fail-fast (no legacy support)**
   - Deprecated/legacy keys are removed or rejected at load-time (fail-fast).
   - Warning emission is not a requirement because deprecated keys do not remain accepted inputs.

## Risks / Trade-offs

- [Risk] Fail-fast surfaces issues earlier and can stop long runs that previously continued.
  -> Mitigation: failures become actionable; add targeted tests around dataset encoding and prompt restoration; enforce CI scanning so blanket suppression fails immediately.
- [Risk] Removing dead private helpers could break out-of-repo scripts that reached into internals.
  -> Mitigation: restrict deletions to private helpers with no in-repo call sites; document removals in the change.
- [Risk] Refactoring evaluator prep code could accidentally change metrics.
  -> Mitigation: keep algorithm identical; add a parity test comparing outputs on a small fixture.
