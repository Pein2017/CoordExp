# Existing-checkpoint transition mechanism Phase Zero

On 2026-07-25 the user approved four existing-checkpoint directions before
any new training: robust free-rollout evaluation, same-prefix decomposition of
continue/stop versus conditional owner selection versus row realization,
complete-action normalization and fixed-budget reanalysis, and entity-level
review of all 85 held-out gained/lost owner references.

The formal owner is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-25-existing-checkpoint-transition-mechanism-decomposition/unit.md`

The unit preserves the original `list all objects` prompt and one free
completion as the decision-owning outcome. Fixed-prefix likelihoods,
teacher-forced candidate rows, and forced continuation are diagnostics. The
primary meaning of forced continuation is to supply exactly the canonical
`<|object_ref_start|>` row opener at an exact shared prefix; masking only the
first terminal choice is a separate historical-comparability control.

The historical Source checkpoint already showed that forcing past a natural
stop can expose a true remaining owner in selected states, but usually adds a
repeat, localization failure, or other low-value row. That history motivates
the current Source-versus-transition test; it does not establish that
transition step 36 learned conditional uncovered-owner selection.

The current goal authorizes bounded implementation, existing-artifact
analysis, short Source/transition inference probes, review-packet generation,
and recursive controls within this unit. It explicitly stops before new
optimizer updates or long training. Completion requires verified evidence or a
bounded stop for all four lanes, a `results.md` claim boundary, and user
discussion of the next training direction.

Before the new unit was created, the previously accumulated dirty work was
verified and split into five commits: research routing policy, Hugging Face
likelihood chunking, row-local owner objectives, experiment configs/scripts,
and the completed prefix-local training/result record. The worktree was clean
at the start of Phase Zero implementation.
