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

## Closed result: trajectory owner-set admission census

The read-only root-state trajectory owner-set admission census is complete and
training promotion is rejected. Among the exact 2,004 Source-eligible training
images, only eight satisfy the frozen primary predicate, versus 256 required.
The eight image identifiers are `2434`, `16796`, `69532`, `174740`, `207431`,
`256151`, `294679`, and `569960`. Do not launch the grouped set-level training
screen and do not use a unique-row or unique-trajectory fallback.

The production artifact contains 2,004 records and 34,068 candidates. It has
9,427 eligible and 24,641 excluded candidates, 134 strict owner-set edges, 105
admissible edges, and 709 images with at least two eligible candidates. Of 378
fully adjudicable images, only four pass; 355 have neither an admissible edge
nor a same-class multiple-first-owner alias. This establishes structural
scarcity in that selected subset. It does not establish population-wide
structural absence because 1,626 images are censored by excluded candidates.

Three blind advanced-model reviews approved the artifact and stop decision.
They agreed that no conclusion-critical ambiguity remains for `8 < 256`, but
that the present evidence cannot choose censoring or structural scarcity as
the sole causal explanation. Any causal-resolution work belongs to a separate
successor research unit. With four fully adjudicable admissions fixed, the
censored population would need at least 252 admissions out of 1,626, or 15.50
percent, to make a 256-image screen feasible. Treat that as a falsifiable
future threshold, not a current estimate or a preselected method.

Authoritative result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-admission-census/results.md`

Production artifact:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census/production-v1`

No successor direction was selected before this closeout. The persistent goal
permits choosing one dynamically from this result and then leading a separate
exploration to a decision-grade checkpoint.

## Active successor: frozen-panel owner-ledger salvage gate

After the admission census closed, independent science, quantitative,
infrastructure, and adversarial lanes selected a train-only image-first owner-
ledger salvage futility gate as the next highest-value discriminator. The
frozen question is whether the 1,622 censored, nonpassing training images can
contribute the 248 additional primary admissions required by the unchanged
17-candidate panel.

Stage Zero is authorized. It may use only immutable exact-B16 parser, projected-
token, row-order, and generated-category facts to build a no-false-negative
category-capacity over-approximation. It must emit a witness or replayable
impossibility certificate for every image and pass independent full replay.
Owner-based reconnaissance pools of 576 and 606 were withdrawn because global
matching can reassign previously resolved rows. A later 1,106-image category-
only scout is non-authoritative until production reconstruction and audit.

The frozen unit includes the empty-lower-owner-set corner, permits the lower-
first preservation witness to reuse either high alias, accepts valid parser
drops only after the exact B16 boundary, and treats every unresolved review
case as a potential success across owner-universe, assignment, safety, edge,
frontier, and admission state. Only after Stage Zero and all reviewer,
adjudication, replay, and outcome-classifier artifacts are frozen and approved
may one canonical 64-byte entropy journal use exact rejection and ordered-sample
unranking to select cumulative looks of 16 and 32 images. An independent
advanced-model fixed-point review approved the contract after requiring pre-
entropy binding of every possible-pool image and replay input, a mechanically
unique one-draw journal, exact downstream look-root binding, and statistical-
decision-blind adjudication. No entropy, seed, or sample exists yet. Training,
model inference, graphics-processing-unit work, development and held-out route
semantics, loss selection, and
architecture selection remain unauthorized.

A final Stage Zero semantic gate found and corrected a shared B16 chronology
defect: generated-row ordinals had been mixed with character offsets, so future
unmatched text before the sixteenth complete row could be misclassified as
post-boundary. An independent fix audit approved the correction. Exact read-
only differential replay over all 34,068 train-U routes found no triggering
stored case and reproduced all 2,004 census records exactly, including the 8
admissions, 1,622-image successor population, and 709 images with at least two
eligible candidates. Stage Zero category state and exhaustive pool membership
also remained unchanged at 1,106 possible and 516 impossible images. The
completed census therefore remains authoritative and does not require rerun.

After a final independent admission gate approved exactly one production
materialization, the immutable `stage-zero-v1` artifact completed successfully.
It contains the exact 1,622-image population partition: 1,106 possible and 516
impossible images, with the pre-authorized ordered-image hashes reproduced.
An independent audit then replayed all 1,106 witnesses and 516 impossibility
certificates in both forward and reverse order without a semantic mismatch.
However, that audit imported the frozen source snapshot as root and created 54
undeclared `.pyc` files in 13 `__pycache__` directories. No declared artifact
is missing or mutated, and no sibling staging, failed, or quarantine residue
exists, but exact root inventory is currently violated. AgentGuard prohibits
agent cleanup, so the audit verdict is `HOLD` until the user manually removes
those 13 derived cache directories and a fresh bytecode-disabled audit proves
the exact 64-file, 19-directory receipt-bound inventory before writing the
closed approval receipt. No entropy, sample, real image review, model
inference, graphics-processing-unit work, or training has begun.

The selection and blind-review machinery has otherwise reached an independent
fixed point. The selector, review assembler, tests, reviewer-visible packet,
and real no-op smoke evidence are frozen read-only. Independent audits closed
production entropy injection, exact one-draw resume, full official-owner
binding, reviewer sequential-information blindness, pre-entropy member and
Stage Zero chronology, closed Stage Zero audit binding, exact no-addition
census equality, and terminal regeneration of added-owner replay before
outcome classification. The combined suite passes 111 tests. A durable
bytecode-disabled smoke exactly reproduces all 17 route rows and complete
census records for admitted image `2434` and geometry-exclusion image `831`.
Therefore the audit-created Stage Zero cache inventory is the sole current
external blocker before the independent Stage Zero approval receipt and the
single canonical entropy acquisition.

Stage Zero artifact:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-adjudication-salvage-gate/stage-zero-v1`

Authoritative contract:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-adjudication-salvage-gate/unit.md`

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
* The frozen root-state census finds only 8 primary multiple-positive strict
  owner-set groups among 2,004 Source-eligible training images; the proposed
  256-image training screen is not feasible from current automatic evidence.
* Structural scarcity is real inside the 378 fully adjudicable images, while
  1,626 censored images prevent a population-wide causal attribution.

Tentative or unresolved:

* Whether train-only adjudication or improved matcher and trusted-geometry
  coverage would raise the censored population's primary-admission rate to the
  15.50-percent feasibility threshold.
* Whether the censored population shares the strict-edge and same-class
  first-owner-alias scarcity observed in the fully adjudicable subset.
* Whether a separately frozen exploration intervention can create enough
  strict owner-set dominance and natural first-owner diversity without
  weakening the primary predicate.
* Whether native prefix state is sufficient once supervision is made more
  set-aligned, or a compact covered-set or task-state carrier is necessary.

Rejected or held:

* Do not interpret longer output, terminal suppression, selected-owner
  recovery, or a mean Average Precision gain alone as set expansion.
* Do not scale the previous 118-image or current 496-image positive-only
  treatment unchanged.
* Do not launch the proposed 256-image grouped set-level screen from the
  current census, count the two image-level-only frontier cases, or reopen the
  admission predicate after seeing the shortfall.
* Do not make one sampled owner, geometry-sorted owner, complete row, or
  trajectory the unique teacher target for the next treatment.
* Do not add a second cohort, external detector, object slot, terminal
  suppression, canonical supervised-fine-tuning mixture, or online refresh to
  this pilot.

## Immediate next actions

1. Treat the admission census as closed and the 256-image training promotion as
   stopped. Dynamically select a separate successor question from the observed
   censoring-versus-structural boundary; freeze its falsifiable contract before
   inspecting new outcomes. Do not precommit the method merely because
   adjudication, matcher coverage, and improved exploration are all plausible.
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

1. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-admission-census/results.md`
2. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-admission-census/unit.md`
3. `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census/production-v1/receipt.json`
4. `handoff/2026-07-23-qwen3-vl-dense-enumeration-comprehensive-research-flow.md`
5. `handoff/from-side-chat.md`
6. `research/investigations/qwen3-vl-dense-enumeration/compass.md`
7. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/results.md`

The completed admission-census result now owns the current route. The
comprehensive handoff remains provenance for how the question was frozen, but
its conditional 256-image training path was not admitted. The breadth and
duplication documents remain bounded evidence; none owns the successor route.
