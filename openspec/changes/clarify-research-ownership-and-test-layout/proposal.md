# Clarify research ownership and consolidate retained tests

## Why

The current root README still limits research to two numerical operators even
though maintained direction packages and their current callers exist. OpenSpec
context points to a deleted PROJECT_CONTEXT document and prescribes retiring the
infrastructure lane, contrary to the three independently versioned canonical
checkouts. Three retained operator test modules live under probes/tests while the
other current probe contracts live under tests/probes.

## What changes

Correct local ownership/navigation without duplicating implementation details.
Move the three operator test modules byte-for-byte to tests/probes and remove the
broad probes discovery root. Keep high-value numerical and integrity contracts;
small tests are not obsolete merely because they are small. Validate the local
OpenSpec metadata. No observable runtime contract changes; skip_specs is explicit.

## Scope and boundaries

This change is local to research-probes. The prior repository review held research
writes while source-bound workers were active. On continuation, those units were
closed and no training/test worker or GPU compute application was observed. A
parallel visualization commit advanced HEAD from a1244d801 to ca7ec3c285; write
preflight rejected the stale base before effects. Five concurrent readout-audit
knowledge files are outside this change. Their baseline identity is recorded in
.local/repository-upgrade/research-ownership-baseline.json, not copied into docs.

No src/probe implementation, scientific evidence, execution release, clean-source
gate, external artifact, model call, dependency, branch, or other checkout is changed.
Cross-checkout infrastructure promotion remains a recommendation only.
