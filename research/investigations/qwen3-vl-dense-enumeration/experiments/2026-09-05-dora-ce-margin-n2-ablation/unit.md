---
title: Fixed N2 Cross-Entropy versus Squared-Margin Adam Add-On
description: One fresh cross-entropy Adam run against the sealed same-runner-family squared-margin Adam control, with teacher forcing and a common cold generation gate.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-05-dora-ce-margin-n2-ablation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# One cross-entropy add-on

## Terminal disposition

The [verified result](results.md) closes the one authorized add-on. CE reaches
the common canonical gate at step 66 and cold natural coverage 65/65 at all
three IoUs with no hard debt; the sealed squared-margin Adam control remained
incomplete at step 300. The fixed-profile verdict is
`CE_COMPLETE_MARGIN_INCOMPLETE_FIXED_PROFILE`. Both train and cold processes
have exited. No further experiment is active; discuss the next question first.

## Question and authority

From exact step-2444 Source, does changing only the Adam training objective
from squared worst-competitor margin loss to full-vocabulary cross-entropy
reach the fixed N2 complete-fit gate within the same numerical schedule and
process-entry budget?

The user authorized one quick cross-entropy (CE) run on 2026-09-05, then a
discussion before any next experiment. This is a new bounded unit, not a
reopening of the closed [QP comparison](../2026-09-05-dora-prox-linear-n2/results.md).
The lead owns semantics, the launch packet, and acceptance; one bounded Luna
builder owns the loss selector/tests and a fresh Luna executor owns the frozen
train/cold commands. No new top-level task or scheduling layer is needed.

The hypothesis under test is that the squared-margin objective supplies a
practical benefit over CE independent of the quadratic-program solver. The
strongest alternative is that CE already fits this panel at least as well;
the historical CE success is supporting context, not the fresh comparison arm.

## Frozen specimen and changed factor

Inherit exact checkpoint, adapter/config/panel/image identities, template,
canonical ordering, and runtime from the [source packet](../2026-09-01-human13-dora-magnitude-finite-overfit/n2-launch-packet-v1.json)
and [parent unit](../2026-09-05-dora-prox-linear-n2/unit.md).

- Worktree: `/data/CoordExp/.worktrees/dora-prox-linear-n2`; pre-change commit
  `380eb5cc0399f306d02e5fcc99989ab28fa012de`.
- Source adapter SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
- Images 6040 and 16228; 65 annotated owners; 136 + 456 = 592 positions,
  including every canonical output token and end-of-sequence token (EOS).
- Training uses fixed correct canonical prefixes (teacher forcing), never
  sampled or model-generated prefixes. Both images form one mathematical batch.
- Train only 196 shared language DoRA magnitude vectors / 573,440 scalars;
  freeze A/B, base, vision, aligner, embedding delta, and output head.
- Existing FP32 unmerged Hugging Face runtime, actual selected-token output
  wrapper, and complete vocabulary of 152,670 tokens remain unchanged.
- AdamW: lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0, seed=0,
  no scheduler or clipping; a fresh optimizer and adapter from Source.
- Sole training change: per-image native cross-entropy with reduction=sum,
  divided by the same total 592 positions; accumulate both backwards before
  one Adam step. The control uses squared worst-competitor deficit / (2*592).
- Keep the actual per-step full-vocabulary margin scans and stopping predicate
  unchanged. Receipt `metrics.loss` remains the common squared-margin metric,
  not CE; the actual training objective must be explicitly labeled separately.

## Fixed control and evidence

Reuse the sealed margin-Adam arm, without retraining or editing its artifacts:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-prox-linear-n2/paired-v1/`

- `adam/terminal.json` SHA-256:
  `4239019cf24d3c3bae8b7188b7e2cd897afdbf3ed6bda29d1f4be84ef329d36f`.
- `adam-cold/terminal.json` SHA-256:
  `70fc6ba08edbeeb6190de7747941124b0fd7c28cb5022a87222ec4bc313807e0`.
- Original runner SHA-256:
  `fe5eaafe21e964dcb1fa5f28b3c1caf7ffb0ecd8beba52505a98b1357b5d2b71`.

Primary outcome is complete fit: all canonical margins >=0.00998, followed by
adapter-only persistence, independent-process exact cold readback of all 588
adapter tensors, and natural greedy RP1.0 coverage 65/65 at IoU50/60/80, natural
EOS, and zero confirmed duplicate/malformed/invalid-box/cap debt. Valid unmatched
predictions remain unknown and nonblocking. This exact-route target is a
sufficient-condition fixed-panel probe, not a general ordering requirement.

Report steps/process time to success, or bounded failure with violated-position
count, minimum margin, and natural behavior. Compare common margin metrics, not
CE values against margin-loss values. Norm comparisons require equal success.
This reuses a historical sealed control rather than simultaneous randomized
arms: a fixed-profile optimization comparison, not a hardware-normalized speed
benchmark, seed-repeatability result, best-tuned loss ranking, or held-out recall
claim. Exact Source metrics and inherited implementation bindings must match.

## Execution and stop

One new training attempt, on GPU 0, at most 300 updates or 1800 process-entry
seconds; one independent cold verification at most 600 seconds. No separate
GPU smoke: the real finite/save/cold seam was already qualified in the parent;
the only new loss branch is checked against native CE loss and gradients before
launch. The new run still has to pass the complete real persistence/cold gate.
Do not rerun QP, margin-Adam, or larger panels, change learning rate or objective,
add self-rollout, or retry after observing scientific results. A technical failure
is preserved and reported before deciding on any replacement; it is not a
negative scientific result. No background GPU workload is stopped for occupancy.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-ce-margin-n2-ablation/run-v1/`.
The lead freezes exact commands and implementation/input hashes in
`launch-v1.json` after candidate inspection. Only the log parent is created by
the launcher; the runner exclusively creates `adam` and `adam-cold` directories.
Reuse the parent's receipt summarizer for the comparison reducer. Close this
unit after one terminal comparison and discuss the next question with the user.
