# Scalable K-trajectory successor direction

## Why the route changed

The completed Human-13 on-policy successor proved the live update and rollback
spine, but all sixteen fixed-dose proposals exchanged newly gained H owners for
protected G owners.  More candidate aliases therefore do not address the main
uncertainty.  The next question is whether K trajectories can estimate a
useful whole-picture update and whether a small, separate greedy compiler can
turn that stochastic improvement into clean-greedy coverage without erasing G.

Three external advisory analyses were compared as non-authoritative inputs:

- `/data/CoordExp/agent_discussion/pro.md`
- `/data/CoordExp/agent_discussion/fable.md`
- `/data/CoordExp/agent_discussion/qwen38.md`

The preferred scientific backbone is the Pro-style fresh on-policy first-hit
set objective.  Qwen's viable-versus-bad margin is retained only as a sparse
greedy-compilation module.  Fable's masked tilted distillation is useful as a
lower-cost biased control, not as an unbiased policy objective.

## Durable statistical distinctions

- K trajectories are repeated Monte Carlo paths within an image.  They improve
  support discovery and estimator precision, but `13×K` does not create
  `13×K` independent images.
- A shared union reward gives tied group advantages and cannot compile the
  union into one completion.  Equal self-sample CE is either zero in expectation
  under the same policy or distills a different sampling processor.
- Trajectory-own first-hit set utility supplies global credit; a sparse
  argmax-oriented margin supplies greedy compilation.  Neither ordinary
  single-owner CE nor local argmax surgery alone answers both needs.
- Existing `rp=1.10`, truncated-top-p K16 artifacts are support evidence.  Each
  proposed policy contract instead requires fresh `top_p=1` samples and exact
  history-conditioned processed-policy log probabilities at its own RP.
- A G witness margin is only a local surrogate.  The bounded C arm now measures
  whether an optimizer-metric projection built from those witnesses changes
  realized retention; dual-surface clean greedy, not witness satisfaction,
  owns the outcome.
- Reviewer-suggested numeric G-loss allowances are not evidence-derived.  Report
  the Pareto-safe `G loss=0` surface and the full named-owner gain/loss frontier;
  any exchange budget remains a user decision.

## Small-first evidence ladder

1. Human-13 mechanism screen: a matched `rp∈{1.0,1.10}` policy crossover.  For
   each RP, use a fresh K ledger, fixed dose, and nested arms for trajectory
   credit, greedy compiler, and G preservation; one predeclared update followed
   by clean-greedy audits at both RP surfaces; repeat over three shared seeds.
   This is `2×3×3=18` update/audit proposals, not an adaptive loop.
2. Width confirmation: if the mechanism is reproducible and not driven by a
   few images or output burden, expand to a stratified 50–100-image screen before
   increasing epochs.
3. Scale training: only after the wider paired result preserves the causal
   ordering of the three components and yields a usable H-gain/G-loss frontier.

The approved planned route now lives in:

- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md`;
- `openspec/changes/add-human13-k-trajectory-rp-crossover-screen/`;
- `docs/superpowers/specs/2026-08-14-human13-k-trajectory-rp-crossover-screen-design.md`; and
- `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md`.

These are planned contracts, not executed evidence.  Fable-5-xhigh returned
`PROCEED` after requiring exogenous owner weights, exact RP replay parity,
one-sided STOP credit, RP-specific Source baselines, and fixed-learning-rate
proposal accounting; all five are absorbed in the formal route.
GPT-5.6-sol-max and an independent coherence re-review then returned `PASS`
with no remaining P0/P1; neither verdict grants implementation or execution.

Hard questions that could change the objective, interpretation, scope, or
go/no-go verdict are routed through the repository `agent-routing` discipline.
Fable-5-xhigh/max and GPT-5.6-sol-max are peer read-only principal researchers;
the lead reconciles their advice and the user owns the large direction.

## Resolved user decisions

- The stochastic objective is owner-balanced first-hit coverage over trusted
  `G∪H` with a small fixed positive-β utility; costs and preservation remain
  separate.  Legacy-M rows receive no direct credit and their own policy tokens
  are masked, while remaining causal history for later trusted actions.
- Zero matching-surface Source-baseline owner loss is the primary pass/width-
  expansion surface; legacy `G_loss=0` is a necessary reported subset.  A
  positive total owner change with any named baseline loss does not authorize
  rolling the update forward.
- Both `rp=1.0` and `rp=1.10` are optimization-policy contracts.  They share
  `top_p=1`, one fixed temperature rule, and exact history-dependent processed
  log-probability reconstruction.  Every proposal is evaluated under both RP
  greedy surfaces, yielding contract-local and RP-robust dispositions.
- Use K16, three shared acquisition seeds, three nested active arms, and one
  update per arm/seed/policy.  No per-image optimize-until-satisfied behavior or
  post-outcome dose selection is allowed.
- Use disjoint `30001..30016` seeds for the production-shaped mechanics
  vertical; its owner outcomes are excluded from the fixed matrix and cannot
  change the frozen settings.
