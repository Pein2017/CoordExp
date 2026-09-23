# Lead ruling 03: bounded localization under the unchanged live reference

2026-09-22. Continue technical localization and an evidence-backed mechanical
repair. [Runtime ruling 02](lead-ruling-02-runtime-reference.md) and
[unit.md](unit.md) retain scientific authority. No reference, tolerance, source,
architecture, objective or LR change is authorized. Scientific fitting is held.

The lead verified live-reference-hold-v1/manifest.json SHA256
`534b6168693ad291457a6f185a2355d18f295518ae839a09b0e7440d9162d204`
and all ten bound files. The recorded errors 0.0 / 0.5 / 1.2265625 remain a failed
three-case qualification. Exact mature tensors and zero new B tensors narrow
the search but do not establish function identity or its failure mechanism.

Start with the short failing case 13004. Find the FIRST divergent computation,
using the actual input/device/autocast/dtype of newly added adapter branches.
Compare each branch with its own base-layer output on the SAME activation;
retain compact evidence at the first post-cast mismatch rather than dumping
all layers. Record module name, shape, B maximum, runtime norm, magnitude,
scale-minus-one, and correction before/after output casting. Check upstream
reference activations agree before assigning causality to that branch.

The current norm repair runs during model setup, before CUDA placement; runtime
PEFT calculates its norm on the execution device/input dtype. A non-unit
magnitude/runtime-norm ratio can contribute (ratio-1)*base_result even with B=0.
This is a hypothesis to measure, not an accepted explanation of the logit drift.
Use the runtime expression, including bias and autocast handling. If newly added
branches are locally identical, follow the first full-model activation mismatch;
request-state or input handling is still possible. A cold replay of the same
failing case before preceding cases is within scope if needed to test that.

Any fix must preserve ordinary DoRA forward AND backward. Do not ship an eval-
only zero-B bypass, skip new branches to pass qualification, modify mature
magnitudes, or reset learned parameters during device moves/reload. If newly
initialized magnitudes must be finalized after placement, distinguish one-time
initialization from restoration of a trained checkpoint and prove persistence.
Retain the old failure as the regression witness and verify the corrected caller
on the actual execution device, with intended gradients intact.

Localize within at most two additional allocated GPU-hours, counted inside the
unchanged package envelope and original wall clock. This is a compute bound,
not a retry-count ceiling. Stop earlier when evidence identifies the defect;
return if the bound is exhausted, the diagnosis remains ambiguous, or a fix
would change the frozen numerical/scientific contract. Preserve every attempt.

Following an in-scope fix, the already-authorized qualification may proceed:
all three unchanged cases must meet the 2e-4 gate, and affected real training/
distributed/save/fresh-reload paths must qualify the corrected producer. Reuse
only unaffected prior evidence. These necessary requalification runs remain
inside the original overall package budget. After COMPLETE qualification passes,
continue bounded fitting automatically under unit.md; otherwise report HOLD.

Report the first-divergence witness, root-cause evidence, minimal changed paths,
fresh checks, cost/job state and stable artifacts directly to the existing lead.
No full-dataset launch, new experimental axis, self-acceptance or publication.
