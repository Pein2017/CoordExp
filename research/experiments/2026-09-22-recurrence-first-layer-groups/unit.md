# First-layer anchor aging versus repeated-pool growth

2026-09-22. Root continues the user's autonomous mechanism grant, fixed mature
untied+axis step2444, with Luna-max execution and root acceptance. This is a
new bounded computational contrast after the accepted native-trajectory unit.

## Question and prediction

Which score group accounts for native first-layer attention-head output movement
from the established plateau (N=15 complete repeats, row42 x2) to the numerical
exit query (N=62, row89 x2): fixed pre-run anchors aging relative to Q, or the
growing repeated pool? This is a first-layer attention computation estimand.
It does not identify final-logit causality, recurrence onset or exit timing.

The previous assay verifies exact constant layer0 pre-Q/pre-K/V across repeats,
while its output varies. A generic repeated-pool log-mass identity alone cannot
tell which source drives the normalized head output. Root's advance prediction
is that anchor aging contributes more than pool growth to the late net head
displacement; the strongest alternative is pool growth dominance.

Freeze val7511 only, original target batch2, native full width2127. Query input
indices are1703 and2126 (original action offsets384 and807). Coordinate next
tokens are38 and999. No new history, generation, checkpoint, image or query scan.

## Single native capture

Reuse maintained native preparation and actual-consumer attestors. Exactly one
full original forward, selected logits at the two queries. Capture actual layer0
cached post-K/V for target batch's whole prefix, pre-Q and actual rotary phases
at the two queries, and actual o_proj input reshaped as two16-head outputs.
Save embedding/native input identities, all28 actual mask/cache-slot/LM consumer
attestations via the already qualified helper, source/model/media bindings and
exact counters. Save original arrays before postprocessing. No model activation
is modified; do not save other layers' caches or full attention matrices.

Both complete152670-logit vectors must match accepted trajectory indices15/62
within2e-4, with identical global winners. Replay both native layer0 attention
head outputs from the saved Q/K/V, actual causal visibility and scale1/sqrt(128);
maximum absolute error must be<=2e-4. Actual current-prefix V and repeated-row
per-role V must agree with the qualified stationary template. These checks
precede interpretation; errors are preserved, not tuned away.

## Frozen two-factor CPU contrast

Group A: all pre-run keys at physical positions[0,1563), including image, prompt
and pre-repeat output. Group R: complete repeats, old[1563,1698) and
new[1563,2121). Group C: current-row prefix, old[1698,1704) and
new[2121,2127), six tokens. At layer0 anchor K/V are identical stored keys for
both queries; their score changes come from relative query phase. Group R has
15 versus62 identical per-role templates at the corresponding relative ages.

Compute native scores sA_old/new, sR_old/new and sC_old/new in FP64 from the
captured actual post-Q/post-K; use the corresponding actual V. Construct all
four cells O_ab=softmax(concat(sA_a,sR_b,sC_new)) @ concat(VA,VR_b,VC_new).
Hold C at its new native values in every cell. Old-old with new C must match
old native output<=2e-4; new-new must match new native<=2e-4. Also retain the
two exact native reconstructions using their own C and report C-only effect.
If C drift violates the gate, report technical/numerical nonqualification;
do not silently treat this as a clean two-factor experiment.

Flatten all16 head outputs into one2048-vector; no favorable head selection.
Let D=O11-O00. Define symmetric contributions
A=0.5*((O10-O00)+(O11-O01)) and R=0.5*((O01-O00)+(O11-O10)).
They sum to D; report interaction O11-O10-O01+O00 separately. Primary readout
is signed projection pA=<A,D>/||D||^2 and pR=<R,D>/||D||^2, including any
negative or>1 values. Root's prediction passes only if pA>0.5; pA<=0.5 rejects
it. Report norms and perpendicular components so cancellation is not hidden.
If ||D||<=64*eps32*max(||O00||,||O11||), declare the contrast unresolved.
Report per-head values descriptively, with no additional head-based verdict.

This is an exact local score-group computation under an explicitly fixed C;
its attribution convention is not a model-wide causal fraction. A larger
contribution does not establish necessity for the final numerical exit.

## Ownership, cost and stop

Execution worker /root/luna_temporal, gpt-5.6-luna/max. Root owns records, CPU
factorial analysis and acceptance. Worker owns only the new maintained producer
probes/training_set_completion/recurrence_first_layer_groups.py and raw output
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-first-layer-groups/attempt-001/.
Reuse helpers; do not edit accepted producers or unrelated codebook work.

Budget1 model/1 vision forward,10minutes, GPU4,64MiB tensor payload. No waiting
for expected stress occupancy. Preserve any failed attempt; no automatic retry,
new layer, donor, endpoint or follow-up model arm. Stop after independent CPU
decomposition and root acceptance or report a concrete qualification gap.
