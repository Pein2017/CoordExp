## Context

Current Stage-2 AB Channel-B builds one unified teacher-forced target (rollout prefix + FN injection) but masks CE on the outermost top-level `}` and `<|im_end|>` for stop-neutral behavior.
In B-dominant schedules this can reduce stop/closure supervision and correlate with long-tail over-generation.
Observed smoke behavior (e.g., `smoke_ab_mixed/v0-20260210-153258`) indicates rollout-length growth under stop-neutral without consistent training-time TP-hit improvement.
The requested change is to remove stop-neutral while keeping FP-neutral protections for potentially unlabeled instances.

## Goals / Non-Goals

**Goals:**
- Remove stop-neutral masking from Channel-B behavior.
- Preserve FP-neutral semantics (FP CE and geometry remain masked).
- Keep unified Channel-B pipeline and key-injection determinism intact.
- Update specs/docs/tests so the changed contract is explicit and auditable.
- Avoid introducing new duplicate helper modules during implementation; reuse canonical helper surfaces (e.g., `src/common/*`) established by schema/module refactors.

**Non-Goals:**
- No change to Hungarian matching itself.
- No change to geometry decode/loss definitions.
- No dataset contract expansion beyond current Stage-2 AB scope.

## Decisions

1. Treat stop/closure tokens as supervised structure tokens in Channel-B path.
- Rationale: reintroduces explicit stop signal in the same sequence used for Channel-B learning and addresses rollout-length drift observed under stop-neutral.
- Alternative: retain stop-neutral and rely only on repeat-aware decode guardrails.
- Rejected: decode-only control does not provide training-time stop calibration.

2. Keep FP-neutral unchanged.
- Rationale: unlabeled-instance tolerance depends on FP neutrality, not stop-neutrality.
- Alternative: penalize FP closure/structure tokens to force early stop.
- Rejected: risks suppressing true-but-unlabeled objects.

3. Hard-remove stop-neutral masking (no runtime toggle).
- Rationale: avoids dual semantics and ensures reproducible behavior across runs.
- Alternative: keep stop-neutral behind a default-off legacy toggle.
- Rejected: creates two valid runtime contracts and weakens comparability.

4. Replace stop-neutral skip accounting with closure-supervision drop accounting.
- Rationale: stop-neutral masking is removed, but the trainer still needs a deterministic failure policy when it cannot robustly locate the outermost `}` / `<|im_end|>` markers (typically truncation/misalignment). The correct behavior is to drop the sample and count it explicitly.
- Normative replacement counter: `stage2_ab/channel_b/closure_supervision/N_drop` (emitted via the neutral `src.metrics` payload contracts as a global aggregate).
- Alternative: keep the `stop_neutral/*` counters for historical continuity.
- Rejected: keeps stop-neutral terminology in the post-stop-neutral contract and confuses operator interpretation.

5. Specify lightweight validation signals (non-gating).
- Report training-time TP efficiency and rollout-length/truncation indicators for visibility under the updated contract.
- Fixed-budget F1/recall/FP breakdown remain required reporting metrics, but there is no in-spec pass/fail gate for this change.

6. Require Stage-2 AB training metrics (emitted via the neutral `src.metrics` payload contract) to be global aggregates.
- Rationale: avoids per-rank ambiguity and makes audit signals comparable between single-GPU and distributed runs.
- Alternative: emit rank-local metrics and rely on downstream aggregation.
- Rejected: breaks reproducibility and produces non-comparable dashboards/artifacts.

## Risks / Trade-offs

- [Risk] Stronger stop supervision may reduce recall in some sparse-label datasets -> Mitigation: keep FP-neutral unchanged and validate on matched/FN/FP breakdown metrics.
- [Risk] Historical run comparability shifts due to contract change -> Mitigation: mark as BREAKING in proposal/spec delta and annotate runbook migration notes.
- [Risk] Channel-B may become more sensitive to malformed prefixes if closure supervision is active -> Mitigation: retain strict parsing + deterministic injection tests.

## Migration Plan

Sequencing note:
- If also planning to land `2026-02-11-add-vllm-repeat-aware-logits-processor`, prefer landing that change first to reduce attribution ambiguity (decode-time tail guardrail vs. training-time stop calibration).

1. Update Stage-2 AB spec delta to remove mandatory stop-neutral requirement.
2. Update trainer CE-mask construction to stop masking top-level `}` and `<|im_end|>` in Channel-B path.
3. Remove stop-neutral masking behavior and reject any stop-neutral knobs.
   - `stop_neutral` keys under `stage2_ab.channel_b` MUST fail fast (unknown/unsupported key under the typed schema).
   - Update `configs/stage2_ab/**` so no stop-neutral keys remain.
4. Remove/adjust stop-neutral-specific metrics and tests; add replacement tests for closure supervision behavior.
   - Keep comparability by documenting metric mapping from old runs to new runs.
5. Run a bounded Stage-2 AB smoke focused on FP-neutral integrity and closure supervision behavior.
   - Include rollout-length/truncation indicators for visibility (e.g., `rollout/parse_truncated_rate`), but do not treat them as a hard acceptance gate.
6. Rollback path: revert change commit if closure supervision causes unacceptable regressions.

## Resolved Choices

- Stop-neutral is hard-removed; no runtime legacy toggle is kept.
- Validation focuses on reporting TP-efficiency and truncation-tail indicators (no baseline gate).
