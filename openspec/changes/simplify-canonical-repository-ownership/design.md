## Context

See proposal.md. Initial clean HEADs: main 52d7dd204449a5d610715666359c91ce243b758e; infrastructure f534953c485feafb45d840cd91b566ae93126ca8; research 249b9b09762457d2f786ab2fbc0c7de677cb5d3f. Main/infrastructure have 273/157 unique commits; infrastructure/research 155/781. They are divergent products, not a linear promotion pipeline.

## Goals / Non-Goals

Retire unsupported leftovers without changing scientific evidence, data, annotations or active consumer code. Restore truthful navigation and test ownership. Branch integration and changes to model/data/qualification semantics are out of scope.

## Decisions

Use AST import resolution, actual callers, current configuration and Git deletion history together. Record selected paths and exact inspected bytes in a bounded receipt, then recheck before mutation. Retire exclusive callers with their absent targets, without aliases.

Preserve data-recovery consumers. public_data/run.sh references scripts/tools/inspect_chat_template.py; an LVIS producer references scripts/analysis/measure_gt_max_new_tokens.py. Their unresolved imports require separate semantic review, not a silent change to data rendering or length budgets.

Retain annotation, data, evaluation and training contracts. Retire tests for absent implementations and permanently quarantined documentation. Small size alone is not a reason to remove a test.

Keep the closed documentation retirement seal unchanged. Correct live owner maps and stale references; this audit belongs in OpenSpec, not a global docs diary.

Research was initially read-only: src/artifacts/git_identity.py checks the entire checkout for cleanliness, and probes/rule_stability/artifacts.py invokes this gate for released native work. Existing eight-rank processes made even a documentation edit potentially disruptive. The continuation rechecked live processes and closed unit state before its separate local ownership/test-layout change; runtime source and admission gates remain untouched.

## Risks / Trade-offs

Concurrent development requires exact target-byte checks and exclusion of unrelated .codex changes. AST evidence alone misses dynamic consumers; inspect references and validate maintained interfaces. Historical records retain their original recovery identity. CPU tests do not qualify model behavior; no model runs are part of validation.

## Migration Plan

Record the bounded retirement inventory; apply guarded main-only changes; update current navigation and test discovery; run offline CPU checks; inspect the exact diff. Infrastructure configuration consolidation has its own local OpenSpec change. Record cross-checkout recommendations and unresolved debt in review.md; do not execute cross-checkout migration.
