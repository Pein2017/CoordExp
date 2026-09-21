# Decision checkpoint after mature untied/readout comparison

Date: 2026-09-18. Owner: research lead. Status: consultation pending; no new
experiment, training, implementation, or feedback model budget released.
This records the next decision, not an executable protocol or a consultation prompt.

## Accepted evidence and retained boundaries

The [natural result](results.md), [active-readout result](../2026-09-18-untied-active-readout-geometry/results.md)
and [lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/lead-acceptance.json)
own the frozen populations, metrics, artifacts and counters. All580 natural outputs
and442 readout slots were independently reconstructed; all owned model jobs ended.
Candidate bytes and hashes remain preserved. No source annotation was changed.

Observed recurrence and endpoint output-norm peaks persist with independent input
and output special-token deltas. Tying is therefore unnecessary for those observed
phenomena. The comparison includes a new0.01 axis loss and different training
history, so it does not estimate an untie-only training effect. Fixed norm
equalization substantially improves the new package on selected difficult scenes,
but has mixed effects across the old package and strata. Both sentinel within-model
coverage intervals include zero. Most sampled endpoint winners survive equal norms.

The [gradient unit](../2026-09-18-untied-gradient-path-accounting/results.md) remains
technical-invalid: the real no_grad axis-value bug was repaired and independently
tested, while its separate CE finite-difference discrepancy remains unexplained.
This is not evidence of a training-gradient defect. The
[feedback unit](../2026-09-18-untied-recurrence-feedback/results.md) has zero model
calls and admission-HOLD on its selected physical owner/control contrast.

## User hypotheses, preserved as hypotheses

- Discrete coordinate choices may create undesirable sensitivity: adjacent bins
  can cause radically different subsequent trajectories.
- Special-token learning, input/output tying, ordering or uneven supervision may
  leave a bias that becomes visible in dense, weak or ambiguous image regions.
- Many annotation-unmatched predictions are credible real objects; missing matches
  can instead reflect changed extents or inconsistent annotation granularity.
- Suppressing physical duplication may unlock coverage. The desired explanation
  concerns why natural SFT produces recurrence, beyond an operational repetition
  penalty, coordinate mask or post-hoc NMS remedy.
- Existing high-confidence annotations and8 GPUs are available. Checkpoint-stage
  sweeps remain deferred because immature decoding would confound interpretation.

## Lead judgment and strongest alternatives

The useful next distinction is readout preference versus autoregressive
maintenance. At a fixed hidden state, norm equalization tests output-row magnitude.
After an emitted choice, the history and future model states change. The accepted
interventions identify control points, but do not yet explain why those states
persist or reappear. Norm-resistant choices do not by themselves implicate input
embeddings, cache, or a missing ledger.

The provisional working account is context-conditioned spatial preference plus a
readout-scale contribution, sustained on some trajectories by feedback. A strong
alternative is that ambiguous visual evidence and next-token likelihood prefer
similar local descriptions without a special feedback-maintenance mechanism.
Training-distribution or objective effects may contribute to either account; their
historical origin has not been isolated.

Earlier [common-tail history evidence](../2026-09-18-covered-history-common-tail/results.md)
shows opposite relative effects at the first coordinate and full row. The
[native donut branch search](../2026-09-18-native-coordinate-branch-completion/results.md)
improves likelihood while still realizing the repeated owner, and does not retain
a higher-likelihood new-owner row. Broader likelihood search, top-k admission and
physical novelty are consequently separate propositions. Prior targeted residual
patches and native cache rebuilding can rescue selected routes, but partial patches
and other cache changes fail to recover coverage. Another generic layer scan would
need a new discriminating prediction.

## Proposed next decision, not released work

First distinguish two endpoints: a clearly defined numerical/structural recurrence
and repeated emission of one physical owner. Literal geometry recurrence, invalid
loops, or repeated very small/full-image boxes can support a numerical-dynamics
study without a complete physical owner ledger. They cannot independently support
physical de-duplication or recall claims. Even literal box identity in crowded
scenes is not universal proof of physical owner identity.

The preferred first step, if consultation supports it, is a bounded saved-artifact
analysis of onset and earlier healthy slots, coordinate roles and history length,
separating norm-sensitive from norm-resistant choices. Its purpose is to select a
specific falsifiable mechanism contrast, not to infer cause from activation norms
or attention pictures. Any model intervention would then need fixed natural
prefixes, matched controls, a predicted selective effect and a free-continuation
outcome. A numerical recurrence experiment may have weaker physical admission than
the stopped covered-versus-new-owner experiment; it must receive its own contract,
not silently reuse that HOLD unit.

The decision-changing uncertainty is whether prior-row feedback selectively
maintains repeated-versus-new preference, or whether similar sensitivity appears
at healthy boundaries too. A changed bin followed by another extent of the same
owner, another loop, or EOS is a counterexample to treating local escape as useful
recovery. Coveredness must also be separated from spatial progression/order before
claiming an abstract coverage-memory mechanism.

External consultation should rank this inference-first direction against a narrow
gradient-gate repair, a genuinely matched training intervention, or another more
identifiable direction. Ask for at most a few sequential discriminators with
contrasting predictions and stop rules, including reasons to abandon the current
framing. No new experiment starts merely because the existing package closed or
external advice is received.

## Continuation and archive status

The human-facing consultation is response-only. Local accepted records and this
decision checkpoint are retained; no prompt file, Notion update or managed-memory
write is created. Resume from the user's returned advice and reconcile it against
the owning results before freezing any successor.
