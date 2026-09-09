---
title: From layerwise readout to checkpoint-state causal transfer
description: Bidirectional current-state versus full-prefix grafting, then fixed Human13 replication of local coordinate decisions.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-08-logit-lens-causal-transfer
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

## Active sentence and user authority

On exact overfit-generated image/prefix anchors, does transplanting Source
versus overfit residual states at a fixed depth change the receiver's final
coordinate-token preference, compared with native/self-patch and equal-norm
random-delta controls, and does the transfer profile repeat across Human13?

The user asked for an independent research series from the preceding Logit
Lens result, permitted Human13 expansion, and requested principles/dynamics
and an explanatory analysis framework. This unit replaces the prior diagnostic's
no-intervention stop only for this new research; the completed diagnostic and
its evidence remain unchanged. The independent row-feedback experiment is not
owned, paused or gated here. Lead owns the scientific contrast, model selection
and synthesis; `dynamics_runner` owns the new experiment-local runner/tests.

Product: bounded local causal-transfer evidence plus a falsifiable descriptive
framework, not architecture promotion, training, a general LLM mechanism or
owner-coverage intervention. No new checkpoint, optimizer, training, model edit,
Tuned Lens fit, framework installation, commit/publication or cleanup.

## Identity and conditioning

Use the same Source step-2444 and Human13 magnitude-CE overfit adapters, common
base/head/norm/embedding delta, FP32/SDPA and native HF materialization as the
[parent diagnostic](../2026-09-08-image2299-logit-lens/unit.md). Adapter SHA256s:

- Source: `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`.
- Overfit: `8a5ebfcacfa92be4b873fea4439fc25570a9c415be2245e20fd1da94c9ff4070`.
- Common selected embedding delta: `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`.

Worktree `/data/CoordExp/.worktrees/image2299-logit-lens`, branch
`probe/image2299-logit-lens`; existing completed probe files/records are
owned prior work, not reset or rewritten. Reuse its helper file unchanged
and bind its hash in every launch. Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-causal-transfer`.

Pilot: exact `run-v2/trajectory-overfit.json` from the parent output root,
file SHA256 `537079135d3b12a3bfd72778ea63352112cde60775405e702abbbfa71efce063`;
415 generated token IDs, 46 closed rows. Bind prompt/media/tokenizer identities.
Use its 12 selected coordinate decisions (first/middle/last row, four coordinates
each). These are generated-prefix replay sites; no GT-canonical next-owner
assumption. The whole teacher-forced prefix includes the original coordinates;
patching one position does not constitute a new autoregressive box trajectory.

## Stage A: Image2299 causal-transfer pilot

Capture native baseline residuals/logits at 1-based decoder block outputs
`8,16,24,26,27,28`, with full no-cache recomputation. For every site freeze
`a_i`, overfit native top-1 token, and `s_i`, Source native top-1 token, *before*
intervention. Record equal-token or noncoordinate endpoints separately, without
silently changing the 12-site population. Define final preference
`m_i = logit(a_i) - logit(s_i)`. This measures checkpoint discrimination, not
coordinate truth, geometric accuracy or owner recovery.

Both donor/receiver directions, overfit→Source and Source→overfit:

1. **Current state:** replace only residual at the prediction position `t`.
   Each intervention is isolated; simultaneously replacing several prediction
   positions would contaminate later sites and is forbidden.
2. **Full causal-prefix state:** replace all positions through `t`, including
   image, prompt, generated history and current position. A single full-sequence
   graft followed by selected-position readout is causally equivalent for each
   site under the verified autoregressive mask. This is not history-only routing.
3. **Random-delta current state:** add four fixed seeded random unit directions,
   each scaled to `||h_d(t)-h_r(t)||`, to receiver `h_r(t)`. Do not replace the
   state with a random vector. Record seed and delta norm identities.

Run these at nonfinal blocks 8,16,24,26,27. Block 28 current-state graft must
reproduce donor final logits with the common norm/head; this is an algebraic
positive control only, not evidence of a discovered late computation. No
redundant block-28 random or full-prefix arms. All later blocks use receiver
parameters. No cache, attention mask, rotary position, image or token changes.
No DeepStack intervention is made (the selected blocks are after blocks 1–3).

### Metrics and falsifiable framework

Persist per-site raw margins, donor/receiver tokens, top-k, donor-token wins,
delta norms and output distribution diagnostics. Report direction separately.
Aggregate transfer fraction:

`R = sum_i(m_patch_i - m_receiver_i) / sum_i(m_donor_i - m_receiver_i)`.

Exclude equal-token sites only from this normalization, show their count, and
never clip R. Raw deltas remain primary when the denominator is small. Record
donorward counts, four-control mean/range, and full-prefix-minus-current transfer.
Do not average the two directions into a single success label.

Operational framework: final decision is a receiver-suffix function of a
current residual **and contextual residual states**. The layerwise lens is
only its fixed-head readout. Candidate predictions:

- Portable current-state account: current graft approaches full graft and
  donor endpoint before the final block, exceeding random controls.
- Context-state dependence: full graft materially exceeds current-only at
  earlier depths. This cannot isolate generated history from image/prompt.
- Receiver-suffix dependence/co-adaptation: early grafts remain receiver-like,
  or directions differ. This does not prove missing information or distinguish
  computational change from incompatible representation alignment.

Full-state four-corner margins (native Source, native overfit, both hybrid
directions) allow an exact two-factor interaction description; no additive
or Markov interpretation is assumed in advance. Cross-checkpoint stitching
tests functional compatibility in this shared nominal basis, not identical
natural algorithms. Large patches may create off-distribution hybrids.

## Stage B: fixed replication, not a layer search

After Stage A mechanics are lead-accepted and cost fits the remaining budget,
replicate nonfinal blocks **16,24,27** on the remaining 12 Human13 images.
This fixed layer set is declared before Stage A results, not selected by the
largest observed effect. For each image generate one overfit-native greedy
RP1.0 trajectory (cap 768, EOS/cap explicitly recorded), select the four
coordinate decisions of its middle completed row using the same deterministic
selector, and replay the identical prefix on both checkpoints.

Use the same two patch scopes, directions and four seeded random controls;
block 28 and self-graft remain mechanics controls. Preserve broad/eligible/
executed/analyzable image and site counts. If a row is absent or a required
endpoint is not a coordinate, retain the exclusion reason, not a replacement
image/row chosen for favorable effects. This is replication across **training
images**, not held-out generalization. Summarize images equally, also expose
raw site counts, directional asymmetry and outliers. No cohort expansion beyond
Human13, depth sweep or seed search. A valid pilot null may still merit this
fixed replication; expansion is not restricted to a statistically positive
finding. Lead makes the bounded cost decision once from measured execution.

## Execution, acceptance and stop

One GPU0 process at a time, reconcile existing jobs, default `ms` environment.
At most 20 minutes GPU execution for Stage A, at most 40 further minutes for
Stage B; total 60 minutes. At most two technical attempts per stage, no silent
relaunch or changed dtype to force a pass. Peak device allocation below 48 GiB;
record resident models, complete forward counts, actual image encoding passes,
wall time, host RSS, payload bytes and model/config/prefix/code hashes. Full
states may be transient but only compact selected tensors need be persisted.

Hooks-off, self-patch, token-position alignment and final donor reconstruction
must pass fixed atol=2e-4/rtol=2e-4; tests must reject shifted/corrupted patching.
Full-state patch and current-state patch must preserve stated scope. No purely
mock acceptance: real pilot captures, interventions, serialized metrics and
cold artifact readback are required. A failed mechanical path leaves affected
science unanswered; preserve failure receipts. Parent helper remains unchanged.

One machine-readable execution receipt per stage owns counters and identities.
Lead replays the smallest deterministic tests and artifact reductions, then
synthesizes observations, remaining alternatives, and the framework's failed
or supported predictions. Stop after pilot+fixed replication+synthesis, or
the technical/resource boundary. A further mechanism/behavior experiment is
a separately proposed successor, not an automatic third search round.

## Literature context (not evidence for this model)

- Zhang and Nanda, [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042): intervention and metric choices affect conclusions.
- Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255): distinguish the interpretation of patching directions and scopes.
- Smith et al., [Functional Alignment Can Mislead](https://proceedings.mlr.press/v267/smith25a.html): functional stitching should not be equated with informational similarity.

## Current evidence

Stage A is lead-accepted through `run-v1/evaluation-v2/receipt.json` under the
output root. All 744 raw site records are retained, covering 650 decoder
forwards and two image encodes in 319.86 seconds. The process failed only at
terminal aggregation: each executed random replacement had passed the original
absolute-plus-relative norm predicate, but the final aggregator mistakenly
required an absolute-only bound. A CPU-only, versioned evaluator rechecked all
480 random controls using the original predicate, preserving the failed
receipt, exact launched-source snapshot and unchanged raw data. No GPU replay
or causal-parity tolerance change was made. Self-patch pass details were not
persisted individually in this pilot; their status is explicitly inferred from
the fail-fast execution path reaching later interventions, not cold remeasured.

Lead independently replayed six tests, rebuilt transfer fractions/four-corner
identities from raw records, verified patch-scope guards, original norm checks,
all raw hashes and the original runner snapshot. The failed receipt's stale
trace-row count (732) is superseded only by the cold verified complete count
(744), not silently edited. The one bounded scientific advisory pass is complete.

Stage B admission: the actual pilot cost projects the fixed 12-image replication
within its 40-minute cap (approximately 20 minutes including new generations).
No positive-only selection was applied. The worker is authorized to implement
the frozen multi-image path and execute Stage B once, with per-self-patch
checks durably recorded and the original combined norm predicate preserved.
The late current-state transfer exceeds the control in one direction; the
reverse control is also large, so direction-specific interpretation and the
state-radius caveat in [framework](framework.md) remain mandatory.

Stage B is now lead-accepted through `stage-b-v1/receipt.json`: all 12 intended
images completed, 48 population sites/47 normalized-ratio sites, with the one
equal-token endpoint exclusion explicitly retained. All per-image self checks
are persisted. Lead independently checked 1,824 trace rows, 1,152 original norm
predicates, raw and per-image receipt hashes, seven focused tests and exact
four-corner reductions. Stage B took 1060.47 seconds, within its bound.
See [results](results.md) and [lead acceptance](lead-acceptance.json).
The fixed series stop is reached; geometry/third-token reductions use retained
data only. No norm intervention, additional sweep or natural-behavior test was
launched. Source-gate, primary metrics and parity tolerances were not relaxed.
