---
name: scope-economy-review
description: Reconsider overdesign locally when new wrappers, schemas, runtime layers, or repeated reviews displace the shortest path to the requested outcome.
---

# Scope Economy Review

Use a brief lead-local check, not a mandatory stage or reviewer. This applies
to the lead's own constraints and design as much as a worker's implementation.
Workers who are unsure should ask parent using
[native-agent-team-guidance](../native-agent-team-guide/SKILL.md); they should
not build a speculative alternative architecture before asking.

Ask: what concrete failure does the proposed addition prevent, and why cannot
the existing real path close it more cheaply? Inspect the nearest real caller
or artifact when needed. Green wrapper tests, future scale and available agent
budget do not establish necessity.

- **KEEP:** a demonstrated acceptance, identity, safety or behavior risk needs it.
- **CUT:** it only supports speculative machinery or duplicates an existing path.
- **REORDER:** try the smallest real path before deciding whether abstraction helps.
- **USER_DECISION:** the choice changes user-owned meaning, cost or authority.

Preserve data integrity, required recovery and declared compatibility. Preserve
historical evidence without automatically forbidding every source change;
choose an appropriate versioned boundary. A running job's bound code remains
protected. A derived-output defect does not justify rerunning valid model work.

State a verdict only if it changes the next action; no routine report template,
receipt or extra gate. Use at most the review budget permitted by the governing
contract, and delegate only a named unresolved risk cheaper to test independently.
Stop when that risk is closed. Do not conduct another review of the review.
