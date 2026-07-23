# Current Project Memory

Last verified for the active dense-enumeration route: 2026-07-23.

Last verified for the physical-owner duplication unit: 2026-07-22T21:25:00Z.

## Closed result: physical-owner duplication

The physical-owner duplication unit is complete. Equal-update Source-only
controls show that the duplicate-cleaned trajectory training recipe, rather
than extra optimizer steps alone, produces the only favorable bounded result.
Relative to its 6-update Source control it is `+4` matched unique owners and `-51`
strict duplicate candidates on train-256, including `+3` owners and `-48`
duplicates on 240 never-trained images. On the disjoint twelve-image
human-refined panel it is `+3` owners and `-9` duplicates versus the equal-update
control, and `+4` owners and `-3` duplicates versus frozen Source.

These controls repeat Source rows, so they do not match event composition or
Source exposure and cannot isolate cleaned semantics as the sole cause. The
complete recipe is promising but not uniformly safe: image `10707` loses a
laptop and
develops a repeated remote row. Recovery-positive and local rejection reduce
duplicates but often lose owners; the combined local-plus-cleaned profile is
rejected. Do not automatically promote the current expanded overlap queue:
most extra immediate-recovery cases are concentrated in one image, and many
longer candidates contain unresolved intervening rows. If that closed
duplication-specific branch is reopened later, its only justified expansion is
more exact self-rollout trajectories plus row-level physical-owner review; it
is not the active program route.

Authoritative result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`

Parallel breadth-screen and Pi-worker state below was not revalidated by this
closeout.

Last verified for the breadth screen: 2026-07-23T03:57:46Z. A fresh unified
vLLM sampled-only panel completed across eight workers at `production-v2`:
2,432 images, 38,912 sampled trajectories, 38,912 natural closures, and zero
length stops. The output image set exactly matches the candidate pool; every
image has one unique `sample_index` 0 through 15, and all 152 persisted artifact
hashes match their manifests. This panel estimates sampled object support. It
does not contain a matched greedy baseline and cannot support a sampled-rescue-
over-greedy claim by itself.

The parallel Pi worker-ablation execution state was last verified on
`2026-07-23`; the port-9090 infrastructure rerun and stateful Remote Procedure
Call thread pilot are complete, and no Pi process is live. The investigation
used no graphics-processing unit.

## Active objective

The user has corrected the next Qwen3-VL dense-enumeration treatment to true,
order-independent set-level supervision. The previous same-prefix single-row
comparison remains useful as a diagnostic or auxiliary term, but it is no
longer the primary treatment. At an actual model-produced prefix, every
verified uncovered physical owner is a valid next owner; candidate
continuations are judged by their final unique-owner sets, and trajectories
that exchange different owners without set inclusion are incomparable rather
than positive versus negative. Unknown or potentially unlabeled entities
receive no negative gradient. A compact covered-set or task-state carrier
remains a later fallback rather than a current architecture commitment.

## Closed parallel objective: physical-owner duplication

The user authorized an independent long-running goal that proceeded in
parallel with the interrupted breadth screen:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/unit.md`

The frozen question is whether generated history causes repeated selection of
one physical instance and whether training can reject the repeated row while
recovering into a later valid, uncovered owner. Official unmatched predictions
remain neutral unless review confirms a category error or entity hallucination;
verified false negatives and physical-owner duplicates are actionable signals.

The dedicated implementation contract is:

`openspec/changes/add-physical-owner-duplicate-rejection-and-recovery-training/`

The OpenSpec proposal, design, capability spec, and tasks are complete and
strict validation passes. Planned controls and treatments are frozen Source,
Source-preservation-only, recovery-positive-only, local duplicate rejection
and recovery, duplicate-cleaned counterfactual trajectory imitation, and their
combined profile. Diagnosis and training proceed in parallel; causal
results control interpretation rather than canceling a valid training smoke.

The initial 256-image census found 113 annotation-anchored repeated-owner rows
in 38 trajectories. Thirty-four trajectories later recover an uncovered
annotated owner. The safest initial subset is 34 near-exact repeated rows in 12
images. The existing 144 geometry-derived ambiguous overlap candidates remain
excluded from automatic duplication supervision.

## Parallel bounded investigation: Pi external worker

Pi `0.81.1` is installed in a non-global output prefix. Four read-only Stage 0
task fixtures and hidden verifiers are frozen under:

`/data/CoordExp/outputs/research/pi-lightweight-worker-ablation/2026-07-22-stage0-frozen-task-harness-screen/fixtures-v1/`

The research owner is:

`research/investigations/pi-lightweight-worker-ablation/experiments/2026-07-22-stage0-frozen-task-harness-screen/unit.md`

The user completed worktree-local OAuth, specified the required port-9090
proxy, and authorized reopening the twelve Pi cells. All worktree-local Bash
and Pi entry points now export the proxy through `127.0.0.1:9090`. The rerun
completed with positive token usage in all cells and no workspace mutation.

Luna, Terra, and Sol all pass artifact inventory and mechanical aggregation.
The audited Task 2 verifier passes Luna and Terra; Pi Sol and native Sol share
one exact `HFBackendSession` capitalization error. All Pi Task 3 cells have the
correct verdict, identifiers, and logical argument but fail the original
lexical limitations checks. Native Sol remains three of four under the
original frozen verifiers.

Stage 0 supports a larger frozen benchmark for mechanically verifiable tasks,
not a default route. Native Codex lacks comparable token and cost receipts, so
no total-cost advantage or causal harness effect is established.

A separately authorized pilot added
`.pi-worker/home/pi-worker/pi_worker_thread.py` as the main Pi integration
path. It is a
Codex-independent JSONL supervisor over Pi RPC that preserves one conversation
across turns and process restarts. A three-turn Luna-medium smoke reused prior
results twice without new tool calls, including after restart with the same
session identifier. Treat Pi as a logical child below a native Codex foreman,
not as a native Multi-Agent Version 2 node. Bind each thread to one prepared
sandbox and do not permit concurrent shared writes.

The default interactive `pi` command now has a separate matched-context setup:
Pi natively loads the root `AGENTS.md`, discovers all 26 worktree skills, and
receives the same Serena MCP launch contract as Codex through twelve direct
tools. A real Luna-medium smoke verified all three surfaces against this exact
worktree and reported 10,887 first-request input tokens. Use that value as the
Pi initial-background baseline for the later direct Codex CLI comparison. The
stateful supervisor's isolation flags were not changed.

See `memories/notes/2026-07-23-pi-stage0-proxy9090-rerun-result.md`,
`memories/notes/2026-07-23-pi-stateful-rpc-thread-pilot.md`, and
`memories/notes/2026-07-23-pi-default-cli-context-parity.md`.

Historical harness-comparison note: two same-HEAD forks named
`codex-research-probes` and `pi-research-probes` were prepared on 2026-07-23,
their Serena language configuration and Pi RTK hook parity were verified, and
their shared-output risk was recorded. Both worktree directories have since
been removed and neither appears in the current Git worktree registry. Do not
use their old paths as live execution targets. Preserve the bounded setup and
receipt history in
`memories/notes/2026-07-23-codex-pi-research-probe-forks.md`.

## Most recent closed evidence

The constant-dose image-breadth screen is complete. Both arms contain 496
sampled-route events and 496 Source-preservation events and run 31 optimizer
updates under two matched seeds. The broad arm uses 496 images; the
concentrated arm uses 162 nested images with exact object-count-band by
selection-rank matching.

At the primary repetition penalty 1.0, broad does not beat concentrated on the
124-image pre-admission held-out complete-case cohort. Broad minus concentrated
is `0/-12` by seed at Intersection over Union 0.30 and `-3/-10` at 0.50; the
mean paired intervals include zero. Broad image exposure therefore has no
established held-out advantage at this fixed dose.

The training signal nevertheless produces strongly owner-enriched recovery on
gradient images. Sampled-route treatment owners missed by Source are recovered
approximately 36 to 41 percent of the time, versus approximately 8 to 12
percent for non-selected missed owners. Source-preservation events retain
approximately 95 to 97 percent of the Source owners they name. This is
consistent with owner-specific uptake, but it is not a same-owner untreated
counterfactual, and the loss does not define how to gain an owner while
preserving the complete final owner set.

The corrected repetition-penalty-1.10 sensitivity panel exchanges roughly 65
to 100 owner identities per arm and seed relative to repetition penalty 1.0 and
usually lowers absolute matched-owner count. Its apparently stronger
treatment-versus-Source deltas are partly caused by a weaker Source baseline.
Treat repetition penalty as a trajectory intervention, not a harmless
duplicate-only adjustment. The earlier `repetition-penalty-1p10-v1` attempt is
invalid because the live collector actually used 1.0.

The 118-image Source-route-preservation screen is complete. Its treatment was
learnable but narrow:

* selected Source-missed owners were recovered approximately four to six times
  as often as non-selected missed owners;
* admitted images could gain owners and box quality and standard detection
  metrics improved;
* non-admitted images lost owners at every milestone;
* more updates did not repair transfer.

The result supports route-conditioned owner redistribution, not safe final-set
expansion. The two live alternatives are insufficient image breadth versus an
intrinsic owner-exchange limitation of positive complete-row imitation.

## Frozen current unit

The unit is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`

The 2,432-image label-only pool was frozen before route inspection and split by
image identity and object-count band into:

* 2,048 training-candidate images, 512 per band;
* 256 development images, 64 per band;
* 128 held-out images, 32 per band.

Each executed treatment arm contains 496 sampled-route rows plus 496 exact
Source-preservation rows: 992 unique events, 31 optimizer updates, and effective
event batch size 32. The broad arm uses 496 physical images; the concentrated
arm uses a nested 162-image subset. The twelve human-refined images remain
safety evidence only and never supply gradients.

The selector must record one of three matching modes:

1. exact object-count-band by selection-rank matching;
2. coarse object-count-band by rank-one, rank-two, rank-three, and
   rank-four-or-deeper matching;
3. policy-only rank-one fallback.

Interpretation must follow the emitted mode. Only exact matching supports a
breadth-effect claim at fixed ordinal-rank distribution. Coarse matching has a
weaker claim, and policy-only fallback cannot identify physical image breadth
as the cause.

## Breadth-screen closeout state

The authoritative trajectory panel is complete at:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-vllm/production-v2/`

It was regenerated for all 2,432 images with one unified vLLM backend rather
than mixing the interrupted Hugging Face panel with new outputs. Its request
semantics are sixteen sampled trajectories per image, temperature 0.4, nucleus
probability 0.95, repetition penalty 1.0, 1,024 generated tokens, 4,096 maximum
prompt-plus-generation tokens, and `sample_index` 0 through 15. Request seed is
not an experimental variable.

A targeted four-image comparison showed that greedy decoding at repetition
penalty 1.0 enters stable repeated-row loops and consumes the full generation
allowance, while all 64 low-temperature trajectories close naturally. The
production panel is therefore sampled-only. Preserve those greedy failures as
mechanism evidence; do not enlarge their limit or treat their truncation as a
complete greedy reference.

The verified receipt and claim boundary are in:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-vllm-receipt.md`

The finite `Source@B16` baseline was qualified and frozen as repetition penalty
1.0 with at most sixteen complete object rows or an earlier natural image-end.
The StateBanks, four training runs, development milestone selection, held-out
evaluation, gradient-cohort attribution, human-refined check, and corrected
repetition-penalty sensitivity panel are complete. The primary result is in:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md`

The owner-attribution receipt is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/gradient-owner-ledger-v1/treatment-owner-attribution-v2.json`

The common repetition-penalty comparison is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/repetition-penalty-1p10-v2/heldout-owner-ledgers-v1/repetition-penalty-sensitivity-comparison.json`

## Current understanding

Established or bounded-supported:

* Low-temperature sampling exposes real physical owners that greedy decoding
  misses.
* Prefix order is causally active; no order-free covered-object ledger has been
  demonstrated.
* Entity discovery and complete box geometry are separate outcomes.
* Fixed-prefix coordinate correction can be learned without improving clean
  rollout.
* Positive complete-row imitation can strongly favor selected owners while
  exchanging away other Source owners.
* Standard mean Average Precision can improve even when unique physical-owner
  coverage does not.
* Under repetition penalty 1.0, dense-image greedy decoding can enter stable
  exact-row duplication loops even when matched low-temperature trajectories
  terminate naturally.
* At fixed event and update dose, expanding from 162 to 496 gradient images
  does not establish better pre-admission held-out owner coverage.
* Sampled-route event owners are recovered about four times as often as
  non-selected missed owners on gradient images, providing strongly enriched
  recovery consistent with owner-specific uptake.
* Repetition penalty 1.10 substantially changes owner identity and may lower
  absolute owner coverage even when treatment-minus-Source looks better.

Tentative or unresolved:

* Whether grouped set-level supervision over nondominated trajectories can
  convert sampled object support into safe clean-greedy final-set expansion.
* Whether the existing sampled trajectories contain enough strict owner-set
  dominance relations to produce a useful gradient without recollection.
* Whether native prefix state is sufficient once supervision is made more
  set-aligned, or a compact covered-set or task-state carrier is necessary.

Rejected or held:

* Do not interpret longer output, terminal suppression, selected-owner
  recovery, or a mean Average Precision gain alone as set expansion.
* Do not scale the previous 118-image or current 496-image positive-only
  treatment unchanged.
* Do not make one sampled owner, geometry-sorted owner, complete row, or
  trajectory the unique teacher target for the next treatment.
* Do not add a second cohort, external detector, object slot, terminal
  suppression, canonical supervised-fine-tuning mixture, or online refresh to
  this pilot.

## Immediate next actions

1. Design the next bounded treatment as a 256-image grouped set-level screen.
   First close a read-only admission census using only the 2,004 Source-eligible
   members of the frozen 2,048-image training split. Map each continuation to
   its verified final physical-owner set, consume every valid strict set-
   inclusion edge, keep owner-exchange and unknown trajectories neutral, and
   reject groups that collapse to one maximal serialization. Then freeze the
   training unit and evaluate clean greedy gained, retained, and lost owners.
   Keep a compact task-state carrier as the fallback branch if native prefix
   state cannot use the stronger set-level signal.
2. Treat the constant-dose breadth screen as closed. Do not spend the next run
   on more image breadth, epochs, or repetition-penalty tuning.
3. Preserve separate-thread work by intent and verify the live worktree before
   staging, committing, or resuming any unit.
4. Treat the physical-owner duplication unit as closed bounded evidence; do not
   promote the combined profile or automatically expand the current overlap
   queue. If it is reopened, use a separate unit with more exact self-rollout
   trajectories and row-level physical-owner review.
5. Treat Pi Stage 0 and the stateful-thread pilot as closed; any later default-
   route claim requires a new frozen benchmark.

## Minimum reading path

1. `handoff/2026-07-23-qwen3-vl-dense-enumeration-comprehensive-research-flow.md`
2. `handoff/from-side-chat.md`
3. `research/investigations/qwen3-vl-dense-enumeration/compass.md`
4. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md`
5. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`
6. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-vllm-receipt.md`
7. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/results.md`

The comprehensive handoff is the fresh-session entry point and states the
current set-level treatment correction. The breadth and duplication documents
remain bounded evidence; neither owns the next training route. Reopen the
experiment units and receipts only when their exact claim or implementation
details matter.
