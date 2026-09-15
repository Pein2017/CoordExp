---
title: Verified full-row positive learning with and without repeat-event credit
description: A bounded paired learning test after native-history witness acquisition.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_bounded_probe
unit_id: 2026-09-11-positive-branch-vs-repeat-event
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: verified
updated: 2026-09-11
---

## Frozen question and authorization

From identical unchanged Stable50 parameters, does adding a sampled next-action
duplicate-event gradient to the same verified complete-row positive learning
improve original-prompt owner/burden outcomes after32updates, under identical
compatible protection and the original greedy3084-token deployment budget?

The user authorizes independent continued research on8GPUs, accepts the Pro
reply as adjustable advice, and explicitly cancelled a temporary network-switch
pause. The preceding two evidence panels remain closed. This is a new paired
learning phase, not another arm of the old coordinate-geometric UL pilot.

Strongest alternative: positive learning alone supplies the needed direction;
the added negative term merely weakens positive learning or moves probability
to invalid, malformed, censored or premature-EOS alternatives. Neither loss
movement nor fewer strict duplicates alone can decide the contrast.

## Specimens and compatible protection

Freeze one preselected positive row per image, without choosing by later
training outcomes:351017-c01 table;417044-c01 donut;477415-c02 stage chair.
Each uses its exact original native token prefix h from the accepted witness
packet. Supervise the entire literal c row, including description, four
coordinates and structural delimiters through box_end. Do not target h, the
free suffix, or EOS automatically. The native h contains past errors and is a
conditioning state, not a clean positive trajectory.

Only the first fresh, root-visually-admitted successor row w supplies local
protection under h+c: respectively the held wineglass [281,361,362,502],
distinct donut [276,253,349,308], and stage woman [249,333,362,609]. Preserve the
Stable50 full-vocabulary conditional distributions at the w-token states, not
unverified later suffixes or the old bad h->repeat branch. This protects a
local conditional capability, not an exhaustive owner ledger.

Normal protection reuses the exact existing56 Stable50 reference trajectories
and their admitted KL masks from the closed dedup pilot. They are disjoint from
the three active images. Keep the known360573 invalid-row exclusion: these
references are not all parser-clean, and no new mask or automatic neutrality
rule is inferred. Both arms share identical reference bytes and masks.

The input owner binds literal prompt/h/c/w IDs, image/model/config identities,
all selected positions and source hashes. The source packet's raw config still
names Source in adapter.path: effective execution MUST override it with the
separately bound Stable50 anchor_adapter, not accidentally train from Source.

## Paired objectives

Let L+ be the mean across the three records of negative **summed** log
probability of the complete c row. Let Kcond be mean-per-token full-vocabulary
KL(Stable50 || current) within each admitted w row, averaged across three
records. Let Knormal be the existing masked mean-per-token KL per normal
reference, averaged across56references.

    A: L+ + 10*Kcond + 100*Knormal
    B: L+ + 10*Kcond + 100*Knormal + L_event

All coefficients are fixed; there is no lambda sweep or adaptive positive-row
replacement. Both arms independently start from identical Stable50 bytes and
fresh AdamW state: lr1e-5, betas(0.9,0.999), eps1e-8, weight_decay0,
foreach=False, global clip1.0,32updates. Train only the established588
language-DoRA tensors /18,006,016scalars; keep FP32/SDPA, model eval mode,
unmerged DoRA, frozen vision/embeddings/LM head and current checkpoint format.

### Event estimator

Before every B update, sample one raw-softmax suffix per h on each of8ranks:
temperature1, top_p1, top_k0, RP1, use_model_defaults=False, native EOS151645,
max64generated suffix tokens. Use deterministic distinct step/rank/case seeds.
No forced c appears in negative sampling. K=8independent samples per h globally.

The unchanged native wrapper supports scalar EOS only. Preserve its real EOS
and post-truncate each generated suffix at the earliest box_end151649 or real
EOS, including that token; otherwise retain the64-token censored action.
Later discarded tokens cannot affect the sampled stopped prefix. Retain the
whole generated suffix separately for cost/provenance. Do not reuse a helper
that strips EOS. Stop rules and terminal classes are explicit and tested.

D=1 only for a geometry-valid next object row whose native-pixel bbox has
class-blind IoU>0.95 to any earlier valid h row. Count it once, regardless of
how many prior rows match. Invalid, malformed, EOS and censored outcomes are
retained with D=0, not removed from the denominator. The event objective is
sum D*sum_token(log pi_current(action|h))/24 over all three histories and all
eight ranks. Desc/wrappers/coordinates are part of the complete sampled action;
this is not four-coordinate geometric-mean unlikelihood and does not enumerate
a bbox neighborhood. Every sampled action is replayed, including D=0 through
a differentiable zero.

At each sampling snapshot, this is a Monte Carlo gradient for fixed-h next-action
duplicate risk. It is not an unbiased gradient of full root-rollout duplicate
burden, and does not forbid invalid/EOS alternatives. Recollect every update;
no stale-bank optimization or greedy-as-unbiased claim is allowed. All sampling,
parsing, labels and reference probabilities stop gradient; current prefix
representations are recomputed and remain differentiable through DoRA.

## Eight-rank execution and measured first gate

Execute arms sequentially on all8GPUs rather than change the proven collective
topology to two4rank groups. Every rank replays all three positives and all
three conditional references; these shared means need no world-size multiplier.
Normal56references shard seven per rank, each scaled100*8/56 before DDP mean.
B's three rank-owned event terms use local sum/3, giving denominator24 after
DDP averaging. A has13local backwards; B has16. Accumulate with no_sync until
one final synchronized backward; clip and step once. No extra division or
DDP compensation is permitted on the replicated terms.

First build the smallest real producer and run a separate2update smoke for
each arm on8ranks, at most600seconds per invocation. The B smoke must prove
post-update resampling, and both must export and cold-reload a checkpoint for
a real score-forward consumer check. These four smoke updates are plumbing
evidence, excluded from the fixed32update endpoints, and counted as cost.
Do not tune objectives or doses from their quality. Exact full-run resource
envelopes will be set from the measured smoke before launching32updates.

Per invocation: one model load/rank, detached reference preparation once,
no persistent sampling K/V across updates, bounded CPU full-vocabulary caches,
and observed model/image forwards, samples/generated/retained action tokens,
backward/sync/optimizer counts, CUDA allocated/reserved, RSS and wall time.
Initial smoke limits: CUDA allocated/reserved <=24GiB and RSS <=24GiB per rank;
at most48negative samples and3072raw sampled tokens in B's2update smoke.
Preserve technical failures; root owns any bounded same-semantics repair.

The implementation records actual event class counts and denominators, all
positive/protected positions, each objective component, finite gradients,
clip norm, tensor/optimizer/checkpoint identities, per-c fixed-route log scores
and token margins. Capture the positive-gradient projection on the actual
optimizer update on rank0, naming its replicated-positive-gradient assumption.
This distinguishes interference with the positive update from lack of a useful
conditional direction; it is not a new optimization arm.

## Endpoint evidence and stop

At the fixed endpoint, read both adapters on the same exposed384 original
images/prompts with unchanged native greedy/RP1/cap3084. Report paired gains
and losses, TP50/TP80 and F1, strict later-row repeat, invalid/malformed/raw-row
burden, and cap/EOS changes. Unknown unmatched predictions remain unknown,
not automatic hallucinations. This exposed screen is not independent transfer.

Also read all three fixed h conditions without c, and h+c conditions, using
the same original total action cap. Distinguish natural row choice, local
successor survival and original-prompt behavior. A successful conditional test
cannot replace the original-prompt result. No exhaustive owner-completeness
claim follows from the limited visual witnesses or incomplete annotations.

After both fixed endpoints and reads, stop. No weight/dose sweep, new positives,
prefix refresh, extra cases, stronger matcher, critic, architecture/KV/token
changes, or checkpoint promotion is authorized by this phase. A prefix refresh
may be proposed only as a separately bounded next decision if fixed-h learning
does not transfer to changed natural histories. Negative findings are final
for this contrast and budget, not a reason to extend it.

## Ownership

Root owns science, visual labels, scope/budget, runtime grants and acceptance.
Luna-max owns the disjoint CPU input-preparation package. Sol-high owns the
task-local trainer and real producer checks. The previous native-witness owner
owns the separately assigned endpoint wrapper and execution after root grant.
Shared runtime/source modules and all closed evidence artifacts remain immutable.

## Execution checkpoint

CPU input, trainer, and endpoint preparation are lead-accepted. The first
eight-rank A smoke stopped before any forward or update because Swift and infras
embedding receipt envelopes have different fingerprints despite identical
payload files and complete semantic identity. The failed `smoke-A/` is preserved;
the correction compares that exact common identity, without changing execution.

Both `smoke-A-retry1/` and `smoke-B/` completed two updates on eight ranks and
passed exact-score cold reload. B retained all48 sampled actions, but none met
the strict duplicate event: D=0 throughout, and A/B adapters were byte-identical.
This validates the zero-event path and identifies a possible signal-supply
limitation; it is not evidence of negative-learning efficacy or inefficacy.

The raw root's `full-pair-grant.json` authorizes sequential independent A32 and
B32 from unchanged Stable50. Same-arm measured resource envelopes are accepted
at `trainer-preparation/resource-envelope-{A,B}.json`:1200/2100seconds per rank,
24GiB memory, exact forward bounds. After each sealed full endpoint, cold reload
must pass. Both full runs and cold reloads are now lead-accepted. B acquired
zero strict-repeat events in768 samples; all32 updates and final adapters are
identical to A. A source-record check confirms each initial Stable50 greedy
h-only successor immediately repeats with IoU1.0, so the inactive negative
signal is not simply due to prefixes lying before the repeated row.

The requested research/delegation discussion pause was honored after training.
The user has since resumed autonomous overnight research and flat L1 delegation.
[Results](results.md) retain that checkpoint and its signal-supply limits.

### Resumed endpoint execution amendment

Execute the frozen384 natural plus6conditional read once, using the actual A
receipt, adapter and arm label. Exact A/B payload identity makes a duplicated
quality contrast uninformative. B is **not independently evaluated**, and no B
records may be relabeled or fabricated. The read now decides the shared learned
checkpoint's natural behavior versus the retained Stable50 baseline, not the
efficacy of the absent negative gradient. The user authorizes independent research
choices; `endpoint-A-grant.json` binds this explicit compute/scope amendment.
All input, parser, decoder, rank and resource settings remain frozen. Root accepts
the fresh output before selecting another experiment; no architecture, GT or
token changes follow automatically.

### Closeout

The amended A-only384+6 endpoint and concentration reducer are lead-accepted.
Natural owner TP50 falls25 despite lower repeats and slightly higher F1; a new
dev cap coexists with diffuse preservation losses, including net27 lost owners
on the56 explicitly protected references. [Results](results.md) own the exact
metrics, posthoc exclusions and visual limits. No promotion or further call is
pending in this unit. The autonomous goal continues through separately frozen
checkpoint-history and reference-margin probes, not a dose extension here.
