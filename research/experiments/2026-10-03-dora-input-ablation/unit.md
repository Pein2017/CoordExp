# Separating DoRA and input-delta increments with output fixed

## Decision and computation

Can either saved component of the body update reproduce its conditional repair
while retaining more anchor known owners or avoiding additional recurrence?
Accepted [round09](../2026-10-03-endpoint-block-ablation/lead-ruling-02.md) shows
that body-only M_A repairs6/6 anchor errors with lower burden than full Gmass,
but loses13 anchor category owners versus12 under full Gmass. The output table
is unnecessary for those six conditional repairs; the remaining DoRA/input
contribution to repair and preservation is unresolved.

The autonomous grant remains active. A NEW GPT-6.1-Sol/high worker owns bounded
execution; lead owns meaning, release, acceptance and any subsequent question.
Current authority is CPU preparation only.

Input deltas change the embedding of coordinate/wrapper tokens read in history.
DoRA changes the language transformations applied during the forward pass.
With output held fixed, the readout is
`z_j=(W_base,j+anchor_output_delta_j) dot h(history,image;DoRA,input_delta)`.
Natural effects include any later generated-history changes. Parameter norms
alone cannot establish either component's behavioral sufficiency.

| Arm | DoRA | Input delta | Output delta |
|---|---|---|---|
| A_A | Anchor | Anchor | Anchor |
| A_I | Anchor | Gmass1 | Anchor |
| D_A | Gmass1 | Anchor | Anchor |
| D_I | Gmass1 | Gmass1 | Anchor |

These are components of one jointly trained saved endpoint, not independently
trained blocks. Freeze all1004 input/output rows, FP32[1004,2048], additive and
untied. No training, blending, scaling, zeroing, row selection, new contexts or
evaluator changes. Common base/vision/projector remain unchanged.

## Exact parents and CPU boundary

Inherit all inputs, ten contexts,18 images/570 evaluator labels, frontend,
native generation, norm OFF and evaluator from round09's released contract:
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/endpoint-block-ablation-09/released-contract-01.json`,
SHA256`eb75dcb988e3de0833802c0852d179032cd92ff7cd61511155736672a3136948`.
Its immutable `lead-acceptance-01.json` SHA256 is
`8e97a2dd07d6a632673e0e165c9da6bd52a3cc09e648e8eef5b02c3ee6d61f5b`.

Reuse the accepted checkpoints bound by that acceptance's `weights`:

- Parent A: round09 A_A, the round07 checkpoint-0 export, weight identity
  `8dcd444f01ef806b743fd5d0f25518ac305e251f2f1dfdb2609fcc1993b80eb3`.
- Parent B: round09 M_A, the canonical body-only hybrid, weight identity
  `a5456cec3f59102d83f60b53302266f5a0e05d926e121118f838ef6d54a666d1`.

Current A_A reuses A; current D_I reuses B. Do not rewrite B's historical M_A
arm/provenance fields or any old identity. Current phase labels belong to this
unit's contract; old payload identities and provenance remain intact. Reuse the
accepted original-versus-export equality proof and distinct original identity.

Publish only A_I and D_A as new immutable canonical hybrids under
`outputs/research/physical-fn-recovery/2026-10-03/dora-input-ablation-10/`.
A_I takes adapter/config/model-card bytes from A and input values from B;
D_A takes those adapter bytes from B and input values from A. Both output
tensors must equal A exactly. Verify common output values and exact metadata,
keys/shapes/dtypes, ordered token mapping, base/config/tokenizer identities and
untied/additive semantics. Whole delta-file byte copying is sufficient only
after proving both parent output tensors equal the same anchor table; it must
not silently bring in the original Gmass output update. Reuse existing payload
formats/inspectors. No model construction or re-export is needed.

Give new hybrids their own five-payload identities and explicit DoRA/input/output
parent provenance. Bind absolute published checkpoint locators before qualification.
Use maintained imports; no predecessor/shared-source edits, global monkeypatching,
generic serialization framework or code execution from artifacts.

CPU checks exercise actual assembly and four-arm caller/final consumer. Reject
wrong DoRA/input donors, any changed output tensor, original-Gmass substitution,
mapping drift, historical-label confusion, copied identity/modified payload,
false source/counts/owner or repair credit and reversed contrast signs. An
independent small oracle covers fresh-anchor errors, baseline-legal loss, zero
error denominators, ties and antagonism. Reuse unchanged earlier checks and exact
export evidence; no broad suite or redundant original/export comparison.

## Fixed readout and falsifiable outcomes

After separate exact release, run four fresh serial native engines in order
A_A,A_I,D_A,D_I. Each scores/emits the same ten contexts, then generates all18
images from empty history. Fresh A_A and D_I are controls; previous outcomes
are descriptive only. First scheduled A_I is the new hybrid runtime seam. Retain
its startup/weight/device/norm/request evidence without extra smoke or a gate.

Fresh A_A literal emissions define repair denominators; preserve all four
outcomes, legal mass and max margin at every context. For baseline errors:

- A_I and D_I repair, D_A fails: input increment sufficient on anchor DoRA and
  required with this updated DoRA at this context.
- D_A and D_I repair, A_I fails: reciprocal DoRA result.
- Both isolated components and D_I repair: either component sufficient here.
- Only D_I repairs: complementary interaction.
- An isolated component repairs while D_I fails: antagonism, naming which one(s).
- None repairs: no repair in this contrast.

Baseline-legal retention/loss is separate; zero errors are an absent repair
contrast. Preserve ties and inherited site/split names without implying new
training or independent generalization. Do not extend necessity/sufficiency
beyond these endpoint/context observations.

Retain full per-image category-correct and geometry-only known-owner sets and
the existing invalidity/repetition/malformed/unmatched/category/length/stop vector.
Five directed edges are A_A→A_I, D_A→D_I, A_A→D_A, A_I→D_I, A_A→D_I.
Report gained/lost/retained IDs for each, not only net counts. Scalar input
effects are `d_A=Y(A_I)-Y(A_A)` and `d_D=Y(D_I)-Y(D_A)`; interaction is
`d_D-d_A`. Retain both fixed-input DoRA effects and joint-minus-anchor.
Per-owner presence across all four arms may be projected from those same sets;
it requires no new inference. Keep opposing scene7511 with the full18 readout.

A useful separation preserves actual conditional repair while improving a
preservation/burden component relative to D_I; report all opposing costs too.
The strongest alternative is joint action or exchanged owner losses, rather
than a removable harmful component. Fewer rows or a net owner gain alone do not
answer this. Unknown/unmatched remains annotation-relative unknown; repeat pairs
are not physical entities. No physical-FN or restricted-block training claim.

## Resource boundary and stop

CPU preparation stops at a clean qualified candidate. Lead selects clean
execution source and issues one exact release to this same worker. The worker
then owns all fixed phases, existing final consumer/readback, cleanup and report.
Scientific null/mixed/worse results still finish all four arms.

Bounds remain0 optimizer/training/HF forwards,40 one-token scores+72 natural
generations=112 requests,maximum222088 new tokens. Cap3084,context4456,one
GPU/rank/sequence,KV2GiB; four1800-active-second phases, each with one30-second
cleanup window,7200 active seconds total. Startup/capture defaults remain;
internal forwards and child CUDA stay explicitly unmeasured. Record wall time,
parent/reaped-child RSS and bytes. New hybrid/native budget1GiB excludes read-only
parents (about180MB hybrids and230MB native expected). Final readback is CPU-only.

Identity/nonfinite/resource/execution failure preserves partial evidence and
cleans owned groups without relaunch or repeating completed requests. No added
control, smoke, dose, context, label, threshold or model call. Stop after readback.
If neither isolated component offers useful separation, close this block-ablation
branch; do not automatically subdivide DoRA by layer or input by token rows. A
positive result identifies a saved-parameter candidate, not a trained recipe.
Any different subsequent question remains a separate lead decision.

Worker owns `probes/dora_input_ablation.py`, its focused test in`tests/probes/`,
this unit's state/results, canonical outputs and `.local/scratch/dora-input-ablation-10/`.
Lead owns this protocol/index/catalog/rulings, source selection and release.
