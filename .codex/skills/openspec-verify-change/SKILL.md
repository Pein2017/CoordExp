---
name: openspec-verify-change
description: Use when checking that an implemented OpenSpec change matches its artifacts before archive.
---

# OpenSpec Verify Change

Requires `openspec` CLI.

Produce a findings-first verification of artifact/code coherence. This is not a style review.

## Flow

1. Select the change; ask if ambiguous.
2. Run:
   ```bash
   openspec status --change "<name>" --json
   openspec instructions apply --change "<name>" --json
   ```
3. Read the context files, tasks, delta specs, and design if present.
4. Check:
   - all required tasks are done or intentionally deferred;
   - implementation evidence exists for each requirement/scenario;
   - tests or smoke commands exercise the contract;
   - docs/specs/artifact names remain consistent with CoordExp authority order.
5. Run targeted validation when feasible.

## Report

Use:

```text
Findings
Confirmed OK
Required Before Archive
Recommended Follow-Ups
Validation Run
Residual Risk
```

Severity should track contract/eval/reproducibility risk. Do not promote a search miss into a critical finding without implementation evidence.
