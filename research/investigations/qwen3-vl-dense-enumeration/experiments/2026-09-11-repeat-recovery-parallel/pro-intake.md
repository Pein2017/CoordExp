# Pro reply: intake and bounded lead ruling

Source: the user-provided [complete reply](/data/CoordExp/.codex/attachments/fd94b2e0-e10a-4b0f-bdbb-d196bd3e8319/pasted-text.txt).
The user explicitly asked to finish the current task first and allowed adjustment.
This note is advice intake, not model evidence or a new launch packet.

## Adopted now

Keep both frozen evidence panels and stop rules unchanged. Do not train or alter
KV, attention, tokens, matching, or the unchanged Stable50 policy in this phase.
The reply identifies no reason to invalidate the current native-prefix witness
or counterbalanced history-reweighting acquisition.

For a subsequent bounded training study, replace the earlier proposed
negative-only versus negative-plus-positive comparison with:

- A: verified complete next-row positive learning plus compatible protection;
- B: the identical positive learning and protection plus duplicate negative learning.

This isolates the incremental value of the added negative objective in the
presence of a known useful direction. It does not separately identify why the
old coordinate-only negative pilot failed. The selected negative objective must
be named explicitly; a new event estimator is not a replication of the old
four-coordinate geometric-mean unlikelihood loss.

Only rows admitted as distinct real instances **and** supported by a newly
generated useful successor under the actual native history can supply the
proposed positive witnesses. Candidate existence or forced-row insertion alone
does not establish autonomous recovery or suffix preservation.

## Mathematical boundary retained

For a fixed token history h and next complete action A sampled from the current
policy, define D_h(A) as a valid later row with native-pixel class-blind
IoU > 0.95 to any earlier valid row. Count each later row once, not every pair.
Then the gradient of R_D = E[D_h(A)] is E[D_h(A) grad log pi(A|h)].
The local sample proxy sum_j D_h(A_j) log pi_theta(A_j|h) / K has the correct
Monte Carlo gradient at its sampling snapshot, assuming genuine policy sampling,
the stated action space, and fixed h. A greedy hard-negative bank does not
meet that unbiased-sampling claim. Repeated updates on stale samples need their
own declared approximation or refresh; this is not automatically the gradient
of total duplicate burden on an end-to-end rollout whose prefix distribution
also changes. Invalid, EOS and censored outcomes remain in the denominator and
readout, rather than conditioning the reported risk on valid rows.

Crucially D_h(invalid) = D_h(EOS) = 0. Even this correctly estimated local risk
does not prohibit replacement errors or premature termination. Complete-row
positive likelihood gives a useful direction but still cannot guarantee its
argmax realization, absence of alternative bad paths, or useful later output.
Those are empirical acceptance obligations, not consequences of using CE or KL.

## Subsequent interfaces to verify, not extra work in this phase

1. Fixed h, no forced c: does the current model choose a correct complete row
   and continue usefully?
2. Fixed h+c: does the current model retain the witnessed successor ability?
3. Original image and prompt: does zero-intervention generation improve jointly
   in owner coverage/preservation, duplicates, invalid geometry and cap burden?

Recompute prefix representations with current parameters; freezing token IDs
does not authorize reusing old K/V. Protect compatible h+c successors and normal
reference trajectories, not the old bad branch at the very decision being
changed. KL is a function-change proxy, not an owner ledger guarantee.

An at-most-once bounded prefix refresh is a possible later design component,
not a current authorization to collect more cases. It is useful only if fixed-h
progress fails to reach the actual changed natural histories. Re-admit c against
the new history rather than assuming the same owner set preserves its label.

No hyperparameters, training budget, new sampler, or refresh cohort are frozen
by this intake. Finish and interpret the two current panels before committing
to that next implementation.
