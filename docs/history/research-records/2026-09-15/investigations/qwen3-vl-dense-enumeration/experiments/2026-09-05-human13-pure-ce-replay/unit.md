---
title: Fresh Human13 Pure Cross-Entropy Magnitude-Only Replay
description: Independently repeat the existing pure-CE Human13 recipe from Source and test all 392 owners in cold natural generation.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-05-human13-pure-ce-replay
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# One independent Human13 replay

## Terminal disposition

The [verified result](results.md) closes the replay: 140 pure-CE AdamW updates
from Source, minimum cold margin 0.0950431824, and independent cold RP1.0
392/392 at IoU50/60/80 with 13 natural EOS, 13 exact routes, and no hard debt.
The new candidate payload is byte-identical by SHA-256 to the historical
same-seed N13 CE candidate. One actual GPU training chain completed; its
pre-producer v1 failure remains preserved. No experiment remains running.

## Question and authorization

From the unchanged step-2444 Source, does one fresh pure cross-entropy (CE)
AdamW magnitude-only run again fit all thirteen fixed Human13 images and yield
392/392 annotated owners under independent-process natural greedy decoding?

The user explicitly requested this replay after the fresh N2 CE result. The
historical [Human13 CE result](../2026-09-01-human13-dora-magnitude-finite-overfit/results.md)
already reports 392/392; this is an independent execution of its identical
training recipe, not the first known CE success or a new algorithm. The main
agent owns the scientific contract and acceptance. One Luna scout checks the
existing Source/history receipts; a fresh Luna executor owns the fixed commands.

The strongest alternative is that the historical success fails fresh replay or
does not survive cold natural decoding. Passing removes an exclusive
fixed-panel overfit-capability advantage for external readout quadratic
programming (QP). It does not settle QP's compute, norm, or generalization
comparisons, which require matched evidence of their own.

## Specimen and intervention

Reuse the exact source/config/data/image identities, token routes, and runtime
in the [historical N13 launch](../2026-09-01-human13-dora-magnitude-finite-overfit/n13-launch-packet-v1.json).
Its N4 prerequisite was historically satisfied; neither N4 nor any learned
adapter initializes this replay.

- Worktree `/data/CoordExp/.worktrees/dora-prox-linear-n2`, initial commit
  `18050cecba8a9dcfc7f51c71bb8bbdf5214cf432`.
- Source adapter SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
- Images 1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923,
  14038, 14439, 16228; 392 annotated owners and 3,637 full canonical positions,
  including end-of-sequence (EOS).
- Fixed correct-prefix teacher forcing, with one shared adapter and one joint
  update after accumulating all thirteen images. No self-rollout training,
  reinforcement learning, external readout patch, or per-image adapter.
- Train only the 196 language DoRA magnitude vectors / 573,440 scalars.
  Freeze A/B, base, vision, aligner, embedding delta, and output head.
- Native full-vocabulary CE, reduction=sum per image divided by total 3,637.
  AdamW lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0, seed=0,
  no scheduler, at most 300 updates; full-vocabulary margin scans every 10 steps.
- Existing FP32 unmerged Hugging Face runtime and selected-token output wrapper.

## Shortest execution path and gates

Use `scripts/research/run_human13_dora_magnitude_finite_overfit.py` unchanged:
its training, CPU materialization, and independent cold readback commands are
already implemented and were used for the historical N13 result. The runner,
surface helper, runtime, and tests still have the historical packet's exact
hashes. No new model code or extra GPU mechanics smoke is required.

Training must produce `FINITE_MARGIN_PASS`, every canonical target beating the
full vocabulary by >=0.00998, with Source restored. Then materialize one standard
unmerged adapter and cold-read it in a new process. Require all 588 saved/live
adapter tensors to match exactly and all 3,637 cold margins to pass.

The lead reconstructs the current natural gate from per-image primitives:
RP1.0 owner matches sum to 392 at each IoU50/60/80; every image has natural EOS,
no confirmed duplicate/malformed/invalid-box/cap debt, and the complete canonical
route is reproduced. Valid unmatched predictions are unknown/nonblocking.
The legacy verifier's IoU50-only aggregate and unmatched-as-debt rule are not
the lead's authority. It also produces thirteen RP1.10 generations natively;
retain them as ungated monitors rather than modifying the reused verifier.

Source natural baselines are historical immutable captures, not a new baseline
generation. Verify their declared hashes before using them. Report fixed-panel
conditioning and data membership explicitly: success is overfit, not held-out
recall or proof of learning a general dense-detection algorithm.

## Budget and stop

Exactly one fresh training invocation, one CPU materialization on training pass,
and one cold verification on valid materialization. GPU 0, one model process
at a time, no distributed launch. Maximum 300 training updates; outer process
wall ceiling 3,600 seconds for training, 120 seconds for materialization, and
1,200 seconds for readback. `timeout` sends INT at the ceiling, with 60 seconds
to exit before forced termination. Preserve partial logs and label an interrupted
native run technically incomplete; do not fabricate acceptance receipts.
Expected cost is tens of single-GPU minutes, not a parameter sweep.

Freeze the commands and hashes in [launch-v2.json (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/launch-v2.json) (the pre-producer replacement
described below). Active output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-human13-pure-ce-replay/run-v2/`.
The launcher may create this new log/candidate parent only; materialization owns
the `adapter` directory. This native runner differs from the N2 add-on's
exclusive-directory entry and its exact output rules must be preserved.

Stop after the one replay and its bounded interpretation, whether positive,
negative, or technically incomplete. No retry, new seed, changed loss/rate,
additional images, QP run, generalization evaluation, or Source cleanup is
authorized by this unit. The earlier N2 units remain closed.

## Pre-producer launcher correction

The exact `launch-v1.json` invocation exited 0 in 0.076 seconds, leaving an empty
`run-v1/train.log`, no model process, no candidate, and no acceptance receipt.
The lead reproduced its complete envelope with a print-only Python sentinel:
bare `env` resolves to `/root/.local/bin/env` and did not execute Python. The
same envelope with `/usr/bin/env` printed the sentinel successfully. No global
environment file was changed. Preserve the original empty log as technical
provenance; it is not a training result.

`launch-v2.json` makes only this exact launcher correction and selects a new
output root. One actual scientific training invocation remains permitted;
this is not a retry after a model outcome or a change to objective, budget,
data, optimizer, or claim. A fresh executor owns this recovery boundary.

The lead independently verified the historical Source natural receipt hashes,
Source adapter path, `mode=source`, absent QP payload, and RP1.0. The derived
input manifest is [source-baseline.json (same archive ref)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/source-baseline.json): N2 coverage is
30/27/8 of 65, and Human13 is 173/159/91 of 392 at IoU50/60/80. These are
historical baselines reverified from receipts, not fresh baseline generations.
