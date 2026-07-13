# Bounded Review Convergence

Read this only when the user explicitly asks for review, revision, and repeated
checking.

## Run the loop

1. State the objective, allowed mutation level, artifact/version, decision,
   evidence that could change it, approval gates, and stop condition. This step
   is complete when every reviewer can inspect the same fixed point.
2. Produce or inspect the first bounded artifact.
3. When subagents are allowed or requested, use two independent lanes by
   default. Give each exact scope, read/write boundary, evidence output, and stop
   condition. Use `implementation_worker` only for an explicitly owned patch.
4. Triage results as P0/P1/P2, non-blocking, wrong, or duplicate. Reject a
   finding only with technical evidence.
5. Give every accepted P0/P1 one disposition: `fix`, `narrow`, `drop`, `probe`,
   or `needs user decision`.
6. Revise only authorized surfaces. Route research meaning, compatibility,
   cost, destructive behavior, and publication decisions to the user.
7. Re-review changed evidence. Default to two review/revision rounds; continue
   only when another round can close a material finding.

## Completion gate

Convergence requires all P0/P1 findings to be resolved, narrowed, dropped,
probed, evidence-rejected, or assigned to the user; required reviewers and
verification must be complete; and claims must not exceed evidence.

End in exactly one state: `approve`, `approved to implement`, `ready for user
approval`, `implemented and verified`, `hold`, `needs user decision`, `probe
required`, or `narrowed/dropped`.

Report the artifact/version, lanes used, accepted and rejected findings,
dispositions and revisions, verification, remaining gate, and exact stop state.
