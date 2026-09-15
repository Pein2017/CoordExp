## Context

See proposal.md. Baseline HEAD is 6cf1a53b23ab25107c2e1a407e7dd0171e2cbd2f; the worktree was clean. Existing source identity hashes a fixed directory selection; prompt rendering imports an omitted templates package. The existing supervisor owns a process group and cleanup logic but calls communicate without a timeout. The renderer snapshot failure was independently reproduced before authorization.

## Goals / Non-Goals

Use the current identity/admission and supervision owners. Preserve numerical and receipt contracts; avoid a new source-manifest service or dynamic module instrumentation. This is CPU correctness acceptance, not model qualification or a speedup claim.

## Decisions

1. Extend the existing deterministic source selection to cover its project-local dependency closure and add a regression guard for static local imports, including package initializers. Keep hashing in the current owner. Verify byte drift using a temporary source tree and exercise real admission against stale identity; do not modify checkout sources to manufacture RED.
2. Thread a finite positive child timeout from the existing run CLI to production supervision. Use a conservative documented default of 1800 seconds per child, overridable for slower hosts. This is an operational ceiling, not a measured model SLA. Preserve process-group ownership, TERM/KILL cleanup and failure semantics; bound GPU census commands. No timeout opt-out. Reuse existing log behavior unless a minimal change is necessary for bounded teardown; standalone log redesign is out of scope.
3. Update only the two stale template fingerprints after a structured actual/expected comparison confirms no rendered-content change. Do not regenerate unrelated fixture content or alter renderer semantics.
4. Use one qualification implementation owner for source identity plus supervision because they share code/tests. Use one independent fixture owner. Lead owns these planning artifacts, integration, operator documentation and final acceptance. No nested agents or GPU jobs.

## Risks / Trade-offs

- Expanded source coverage invalidates existing receipts by design; keep old artifacts intact and state that requalification is required.
- Static import coverage does not prove arbitrary dynamic-import closure; inspect current dynamic imports and cover concrete semantic dependencies, without claiming a general Python attestation framework.
- A slow healthy child can hit the ceiling; document the override and preserve explicit timeout failure.
- Tests use lightweight real child/descendant processes and stub only GPU census, so process cleanup is checked without model or GPU cost.

## Migration Plan

Run the affected CPU suites and strict spec validation. Do not admit or reseal old receipts. Deploying this source requires a new qualification run under its new identity; such a run is outside this task. Changes remain uncommitted for user review.
