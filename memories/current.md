# Current Project Memory

Last verified for the active dense-enumeration route: 2026-07-25.

Last verified for the physical-owner duplication unit: 2026-07-22T21:25:00Z.

## Live-router authority rule

The user established that every current, next, fresh-session, or minimum-
reading-path pointer must target a tracked owning research unit, result,
decision, compass or index, current doc, or stable spec. Handoffs, agent-review
outputs, reviewer packets, audit scratch, transcripts, memory notes, and
temporary artifacts are provenance only and may not own the live route. The
formal rule is recorded in `AGENTS.md` and the research graph contract.

## Active goal: existing-checkpoint transition mechanism Phase Zero

The formal live route is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-25-existing-checkpoint-transition-mechanism-decomposition/unit.md`

The user approved four no-new-training lanes: robust evaluation of the current
transfer artifacts; same-prefix separation of continue/stop, conditional owner
selection, and row realization; complete-action score normalization and fixed-
budget reanalysis; and entity-level audit of all 46 gained plus 39 lost held-
out owner references. Fixed-prefix and forced-continuation results remain
diagnostics. The original `list all objects` prompt, one free completion, and
final gained/retained/lost physical-owner set remain the decision-owning
outcome.

The primary forced-continuation intervention supplies exactly the canonical
row opener at a shared prefix. Masking only the first terminal choice is a
separate historical-comparability control. The Source-produced prefix bank is
primary for the fixed-state comparison; treatment-produced prefixes are added
only if state visitation becomes decision-relevant. The goal stops after all
four lanes close and before any new training.

## Predecessor result: prefix-local owner-set training and transfer

The formal live route is the prefix-local unit and its updated result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/`

All four 1,440-event runs completed 90 finite applied optimizer updates. The
governing task remains the original `list all objects` prompt followed by one
autoregressive completion; selected-owner recovery that exchanges away another
Source owner is not success.

The 40-checkpoint 512-token dose curve shows that complete-action pairwise
training at learning rate `1e-5` and grouped owner-conditioned training develop
increasing output expansion with dose. Their final checkpoints have thirteen
and twelve short-horizon length stops. The lower-rate pairwise arm stays between
one and four, while the first-divergence transition control retains Source's
single short-horizon stop. A 512 stop is only an early warning: Source naturally
closes all 64 images at the 3084-token final horizon.

Final comparisons must match batch policy. A same-checkpoint replay found raw
greedy-text differences on 57 of 64 images between batch sizes 16 and 4. The
mixed-batch owner table is rejected. Under the authoritative matched batch-4,
3084-token, repetition-penalty-1.0 policy, Source has 389 matched owners and 64
natural stops. Only first-divergence transition step 36 has zero treatment
length stops. It gains 21 owners, loses 18, and is therefore only `+3` net,
with 155 fewer predictions and 14 fewer strict duplicate candidates. Other
evaluated checkpoints have one to ten length stops; the pairwise and grouped
final checkpoints add roughly 2,450 predictions, so their larger apparent
owner gains are invalid as usable set expansion.

The matched comparison is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/owner-comparisons-64-long-promotion-max3084-matched-b4-v3/`

The separately approved matched transfer gate is also complete. Development
and heldout are disjoint 256-image and 128-image cohorts. Source and transition
step 36 use the original prompt, one completion, batch size 4, 3084 generated
tokens, repetition penalty 1.0, scoring enabled, the same eight ranks, and an
identical per-split shard-plan fingerprint.

On development, Source has 1,452 matched owners and treatment has 1,520:
gained/lost is `128/60`, net is `+68`, predictions are `+350`, strict duplicate
candidates are `-14`, and length stops are `1/3`. On heldout, Source has 729
matched owners and treatment has 736: gained/lost is `46/39`, net is `+7`,
predictions are `-118`, strict duplicate candidates are `-13`, common-owner
mean Intersection over Union is `+0.006447`, and length stops are `0/1`.
Heldout has 21 net-positive images, 17 net-negative, and 90 unchanged. A post-
hoc paired image bootstrap interval crosses zero on heldout. The correct claim
is promising directional transfer with a favorable quality profile, not a
robust usable-improvement or architecture-promotion result.

The authoritative transfer comparison is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/owner-comparisons-transition-step36-transfer-max3084-matched-b4-v2/`

HF policy likelihood extraction was corrected to call Transformers transition
scoring in 32-step chunks. The old batch-16 path failed on a 16.23-GiB transient
allocation; the same 3084-step real replay now completes, and all 12 current
HF session tests pass. This is a runtime correction, not evidence that batch
sizes are behaviorally interchangeable.

The planned zero-update influence matrix, genuine on-policy and hybrid arms,
matched ordinary row cross-entropy, and Source-preservation-only controls remain
incomplete. The external one-row-at-a-time or textual covered-set branch remains
pending. No full-pool, successor-direction, or final-architecture work has been
launched. The approved stop boundary is met: discuss whether the small and
uncertain held-out gain warrants broader confirmation or a different mechanism.
Detailed transfer reasoning is in
`memories/notes/2026-07-25-transition-step36-matched-transfer-evaluation.md`.

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

The read-only root-state trajectory owner-set admission census is complete.
Among the exact 2,004 Source-eligible training images, only eight satisfy its
frozen primary predicate, versus 256 required by the exact all-positive
natural-alias pairwise design that the census tested.
The eight image identifiers are `2434`, `16796`, `69532`, `174740`, `207431`,
`256151`, `294679`, and `569960`. Do not launch that exact grouped set-level
screen and do not silently use a unique-row or unique-trajectory fallback.

The production artifact contains 2,004 records and 34,068 candidates. It has
9,427 eligible and 24,641 excluded candidates, 134 strict owner-set edges, 105
admissible edges, 76 images with at least one admissible edge, and 709 images
with at least two eligible candidates. Of 378
fully adjudicable images, only four pass; 355 have neither an admissible edge
nor a same-class multiple-first-owner alias. This establishes structural
scarcity in that selected subset. It does not establish population-wide
structural absence because 1,626 images are censored by excluded candidates.

The census predicate is a strict composite: a nondominated exact-owner-set
class must have multiple natural token serializations with different first
owners and must be the higher side of an admissible, first-owner-preserving
strict inclusion edge. Independent reviews later established that `8 < 256`
decisively stops this exact design, but not every possible 256-image owner-set
training design. A real strict edge or other nonzero set-level contrast is
needed for an image claimed to train set expansion under the frozen pairwise
loss. The natural-alias condition is a conservative guard for a strong natural
order-independence claim, not a universal prerequisite for all owner-set
training. The originating proposal did not state that all 256 images must pass
the later exact numerical composite predicate. Any alternative must explicitly
separate signal, preservation/background, and matched-control images and narrow
its claims rather than treating arbitrary images as equivalent signal.

Authoritative result:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-admission-census/results.md`

Production artifact:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-23-trajectory-owner-set-admission-census/production-v1`

## Paused successor: frozen-panel owner-ledger salvage gate

After the admission census closed, independent science, quantitative,
infrastructure, and adversarial lanes selected a train-only image-first owner-
ledger salvage futility gate as the next highest-value discriminator. The
frozen question is whether the 1,622 censored, nonpassing training images can
contribute the 248 additional primary admissions required by the unchanged
17-candidate panel.

Stage Zero used only immutable exact-B16 parser, projected-token, row-order,
and generated-category facts to build a no-false-negative category-capacity
over-approximation. Its one production materialization and independent full
replay are complete. The clean immutable artifact exactly partitions the 1,622
images into 1,106 possible and 516 impossible cases.

The selector, blind-review assembler, outcome finalizer, tests, reviewer packet,
and real no-op smoke are frozen read-only and independently approved. The user
manually removed the audit-created bytecode caches; the recovered Stage Zero
root now has the exact receipt-bound inventory, and the closed independent audit
receipt is approved. There is no remaining mechanical sample-selection blocker.
No entropy, seed, sample, real image review, model inference,
graphics-processing-unit work, or training has occurred.

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

The successor is now deliberately paused before entropy because its estimand
may not match the user's intended training design. Continuing it would ask
whether the 1,106 possible censored nonpassers can supply 248 more admissions
for the exact strict design. If the intended 256-image screen is a mixed
cohort, that is the wrong next question. One plausible mixed design would use
the 76 edge-positive images as the explicit set-expansion signal stratum, the
eight exact-primary images as the strongest natural order-alias subset, and
additional preservation, background, and matched-dose control strata. Such a
design cannot claim that every one of the 256 images carries the same set-level
signal.

The user treats 256 as a total training-set size rather than 256 images that all
satisfy the strict composite predicate and has now selected a broader prefix-
level, multiparadigm successor direction. Do not acquire entropy or resume the
248-admission salvage sample. That estimand remains paused because it answers
the wrong design question. The formal current route and mixed-design boundary
are recorded in the investigation `compass.md`; the selected successor still
requires a new concise formal unit before execution.

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

## Closed breadth-screen unit

The closed unit is:

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

* How many trusted prefix-level owner events survive row-local rather than
  whole-trajectory censoring across the existing training artifact.
* Whether a local owner-conditioned loss can raise an uncovered owner relative
  to confirmed harmful actions without systematic shared-parameter suppression
  of other valid or Source owners.
* Whether genuine on-policy final-set utility and its hybrid with local credit
  can produce clean-greedy net owner gain rather than another owner exchange.
* Whether targeted train-only adjudication materially expands conclusion-
  bearing event supply. The exact 248-admission salvage estimand is no longer
  the immediate question.

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
* Keep the external one-row-at-a-time prompt, textual covered-set prompt, and
  controller-assisted generation branch pending. It may be revisited as a
  separately claimed diagnostic or teacher, but it is not the immediate route.
* Do not reject a semantically valid paradigm merely because it requires more
  compute. Do reject selected-owner exchange, longer output alone, or local
  exact-prefix actuation alone as evidence of final-set improvement.

## Immediate next actions

1. Materialize the robust development/heldout reanalysis from persisted raw,
   scored, stop, and owner-comparison artifacts; retain all-image results and
   report natural-stop, trimmed, leverage, owner-yield, duplicate-rate, fixed-
   token-budget, and matching-threshold sensitivities separately.
2. Reuse the existing exact-prefix scoring, forced-row, terminal-suppression,
   owner matcher, and visualization surfaces for one Source/transition real
   smoke before expanding to four to eight cases.
3. Recompute complete-action candidate summaries as sequence sums, token means,
   and equal-weight description/schema and coordinate group means. Keep the
   trained sequence-sum semantics visible and use a shared singleton projection
   across banks where available.
4. Build an arm-blinded review packet for the 85 geometry-derived held-out
   owner changes and record entity/category and geometry dispositions without
   treating official unmatched output as automatic hallucination.
5. Close the unit with observed, supported, ruled-out, unresolved, and not-
   claimed sections. Stop for discussion before any new optimizer update,
   replication training, or architecture work.

## Minimum reading path

1. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-25-existing-checkpoint-transition-mechanism-decomposition/unit.md`
2. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/results.md`
3. `research/investigations/qwen3-vl-dense-enumeration/compass.md`
4. `research/investigations/qwen3-vl-dense-enumeration/experiments/index.md`

The existing-checkpoint unit owns the active route. The predecessor result
owns the executed checkpoints and transfer evidence. Handoffs, review outputs,
and memory notes remain provenance rather than live route authorities.
