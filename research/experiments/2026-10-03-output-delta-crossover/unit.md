# Output-delta crossover between the two one-update endpoints

## Decision and computation

At a fixed saved DoRA/input configuration, does replacing the output-delta
table move natural invalidity, recurrence and known-owner coverage toward the
donor endpoint? Or do the differences remain with the DoRA/input configuration
or depend on its interaction with the output table?

The autonomous research grant remains active. A new GPT-6.1-Sol/high worker owns
execution; the lead owns this intervention, release and acceptance. The accepted
[mass-versus-ranking result](../2026-10-03-mass-versus-ranking/lead-ruling-02.md)
repairs6/6 illegal prefixes under both objectives. Gmass preserves more original
known owners but has greater invalidity, recurrence and length, concentrated in
images13348,351017,7511. Its output-delta update L2 is about8× larger even though
combined update norms are similar. That difference motivates this test; it does
not establish the cause. Parameter directions or row norms alone cannot decide
the behavioral contribution because hidden states and subsequent histories differ.

For a selected output tokenj, the readout includes
`z_j = (W_base,j + output_delta_j) dot h(history, image; DoRA, input_delta)`.
The frozen model uses additive, untied special-token deltas. Swapping the output
table changes this readout at the same body; natural greedy generation also
propagates the change through later token histories. The intervention therefore
includes those later history effects and does not isolate only a local logit effect.

Define R as accepted Gmax1 and M as accepted Gmass1. "Body" here means all
remaining model parameters: identical frozen base/vision/projector plus the
chosen endpoint's DoRA and input delta. Freeze the full2×2:

| Arm | Body source | Output delta source |
|---|---|---|
| R_R | R | R; original accepted checkpoint |
| R_M | R | M; new hybrid |
| M_R | M | R; new hybrid |
| M_M | M | M; original accepted checkpoint |

Use all1004 rows of the saved FP32[1004,2048] `output_embed_delta` tensor.
Do not select rows, scale, blend, zero, interpolate, retrain, or change inputs.
`input_embed_delta` stays with the body. Both endpoints share the same anchor,
token mapping and additive parameterization, so replacing the saved complete
output table also replaces only its one-update increment relative to that anchor.

## Frozen identity and CPU preparation

Inherit exact input/media/frontend, ten supplied contexts,18 images/570 evaluator
labels, native generation, norm OFF and evaluator from accepted round07:
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/mass-versus-ranking-07/released-contract-01.json`,
SHA256 `67e98ce0c666c15a08087be7282b95397b2d751f77eefc7b783f6d9632b55243`.
Its `lead-acceptance-01.json` SHA256 is
`ff045f47565bdce6c92e1c8e08eca207876d0b591a20c2db660491acb1825d3f`.
The acceptance binds both parents' exact five payloads:

- R: round06 `short-dose-ranking-06/package-01/train/checkpoint-1`, weight
  `b2fd7427a6f698a0b64a47aa0643cbd6b0dea0fbf3eb0029f11b1d352c8bf1c9`.
- M: round07 `mass-versus-ranking-07/package-01/train/checkpoint-1`, weight
  `9fc7b57baee2168412001b97c4ac547f03d276dde1216cc3c406ac6aa7548b08`.

Prepare two immutable hybrid checkpoints on CPU with maintained serializer and
inspection conventions. DoRA payload/config/model-card bytes come from the body;
input delta values come from the body and output delta values from the donor.
Require identical exact tensor keys, FP32 dtype, shapes, token IDs/strings/order,
base model/config/tokenizer identities, additive semantics and tying flagfalse.
The original parent checkpoints are read-only inputs and need no republishing.
Each hybrid has its own five-payload identity and provenance naming both parents.
Do not relabel an original identity or rewrite a historical receipt.

CPU hybrid outputs belong to canonical
`outputs/research/physical-fn-recovery/2026-10-03/output-delta-crossover-08/`.
Publish and bind them before the qualified candidate; the released execution
package reads these immutable artifacts. Code always comes from maintained
source. Parent or hybrid checkpoints are never executable source.

Use actual assembly/readback and runtime payload inspection to check the new
serialization boundary without loading a model. A wrong input/output key swap,
wrong donor/arm, changed metadata mapping, copied parent identity or modified
payload must fail. Check four serial native callers and final consumer with CPU
doubles; reject false source/counters/owner credit and wrong factorial direction.
Reuse unchanged frontend/score/evaluator/cleanup evidence. No broad old-suite
rerun, new serialization framework or edits to predecessor/shared runtime.
The first scheduled hybrid native phase is the real composition/runtime seam;
no separate HF or native smoke is authorized.

## Fixed readout and falsifiable outcomes

After separate exact lead release, run fresh native engines in fixed order
R_R,R_M,M_R,M_M. Each scores/emits the same ten supplied contexts, then generates
all18 images from empty history in inherited image order. No optimizer or HF
model forwards occur. Fresh original endpoints are the controls. Historical
anchor-relative outcomes are descriptive only; no fresh anchor is needed for
this factorial question. Do not reuse old endpoint outcomes as its controls.

For every arm retain all ten literal emissions, legality, full-vocabulary legal
mass and max margin by site/inherited split. Those split names index historical
context selection; this unit trains nothing. Report legal/illegal transitions
for each fixed-body output swap. Zero error denominators remain absent contrast;
do not manufacture a baseline error or retry a tie.

For each arm report per-image and aggregate category-correct and geometry-only
known-owner sets, invalid rows, literal complete/valid repeats, near-repeat
occurrence pairs, malformed output, unmatched rows, category disagreements,
length and stop. Keep counts with length/cap: longer outputs create more chances
for invalid/repeated rows. Unknown/unmatched remains neutral. Near-pair counts
multiply row occurrences and do not count physical entities.

For each scalar natural readoutY, compute these fixed contrasts:

- Output effect at R body: `d_R = Y(R_M) - Y(R_R)`.
- Output effect at M body: `d_M = Y(M_M) - Y(M_R)`.
- Body-output interaction: `d_M - d_R`.

Also retain body effects at fixed output, `M_R - R_R` and `M_M - R_M`, and the
original-endpoint contrast`M_M - R_R`. For these five directed edges preserve
per-image category/geometry owner gained/lost/retained IDs, not only net counts.
They are counterfactual endpoint comparisons, not a training trajectory.
Saved-token divergence/visitation can remain descriptive, with no extra scores.

The output-block explanation predicts that R_M increases invalidity/recurrence
relative toR_R and the reciprocal replacement M_R reduces them relative toM_M.
Similar directions at both bodies support a transferable contribution of this
particular output-table difference. If hybrids remain close to their own body
originals, the saved body difference is the stronger explanation at this contrast.
Opposite or very different effects support interaction; there is no universal
"better head" inference. Report disagreement among metrics and the opposing
scene7511 response; do not select the favorable arm, scene or metric afterward.

Large supplied-prefix changes with little natural change separate conditional
sensitivity from natural effects. Natural changes with stable supplied-prefix
legality leave other histories unresolved. The experiment identifies the effect
of exchanging this parameter block on these saved configurations, including
history changes; it does not isolate individual DoRA/input mechanisms, prove
physical false-negative recovery, or establish a general causal account of
stopping or owner loss. This is one observation per arm/input on a fitted cohort,
not population evidence or a training-replicate/repeatability estimate.

## Release, resources and stop

CPU preparation currently authorizes no real model loading or GPU work. The lead
selects/qualifies clean execution source and releases one exact package to this
same worker. The worker owns all four native phases, final readback, cleanup and
direct report. Successful phases continue automatically; a scientific null,
mixed outcome or lost conditional legality still completes the four fixed arms.

Bounds:0 optimizer steps,0 training/HF forwards,40 one-token native scores and72
natural generations capped at3084,112 native requests and at most222088 new tokens.
One GPU/rank/sequence at a time,context4456,2GiB KV. Each of four phases has1800
active seconds plus one30-second owned-group cleanup window;7200 total active
seconds. Preserve native startup/capture defaults. Internal initialization
forwards and child CUDA allocation remain explicitly unmeasured; startup cost
stays inside the phase windows. Record wall,parent/reaped-child RSS and bytes.
Expected retained hybrids are about180MB and native raw artifacts about230MB;
budget1GiB for new retained hybrid/native evidence, excluding read-only parents.
Final readback stays CPU-only and uses saved artifacts, without another generation.

Identity, nonfinite, resource or execution failure stops with partial evidence
and owned-group cleanup, without relaunch/retrying completed requests. No added
anchor, repeated control, dose, threshold, context, label, or extra model call.
Stop after the four fixed arms and readback; worker cannot schedule another unit.

Worker owns `probes/output_delta_crossover.py`, its focused test in`tests/probes/`,
this unit's state/results, canonical output and `.local/scratch/output-delta-crossover-08/`.
Native outputs belong to the lead-selected execution checkout. Lead owns this
protocol/index/catalog, release, scientific interpretation and acceptance.
