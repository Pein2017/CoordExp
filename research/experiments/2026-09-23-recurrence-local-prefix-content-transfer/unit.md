# Cross-image transfer of current-row contextual prefix content

2026-09-23. Root owns science/acceptance. User authorizes continued autonomous
research and expansion when useful. This is a new bounded discriminator on the
same mature untied+axis step2444 checkpoint, not a reopening of a phase scan.

## Question and interpretation

At three independent native first-coordinate differences following exact-row
repeats, does substituting an earlier same-token CURRENT-ROW PREFIX's contextual
K/V content, at the destination's unchanged native rotary phases, restore the
repeated coordinate under native and identity-recomputation controls?

The accepted local-prefix-phase control gives579/579/38 and identifies an
introduced local-coherence confound. Late incoming state plus coherent current
and prefix phases is still insufficient for late999 on earlier memory. Remaining
history differences include contextual local-prefix content and the older pool.
The earlier fixed-template assay changed PRECEDING COMPLETE rows on two other
images; FIRST and recent donors differed, so it could not isolate distributed
replacement extent. Here the affected keys precede the query WITHIN ITS CURRENT
row, and only the target query reads the replacement. No query phase/state is
transplanted. The second repeated row is the fixed donor, avoiding the first-row
entry transient. This does not erase upstream positional influences in pre-K/V.

Candidate explanation: the current row's contextual prefix is a local carrier
of decision-relevant history. Strongest alternative: older history and/or the
current residual dominate these native first differences, so local-prefix
substitution retains their native winners. Root's advance prediction is decisive
restoration of the repeated coordinate in at least2 of3 independent images.
Full-vocabulary argmax is primary; top-two gap<=0.001 is inconclusive. Preserve
native winner, repeated coordinate, third coordinate, noncoordinate and near-tie
outcomes separately. A log-odds shift without the predicted categorical result
does not rescue the prediction. Mixed/negative outcomes close this branch without
donor, layer, head, strength or additional-image rescue scans.

Even success establishes only conditional local causal influence at these
native landmarks. It does not identify a unique circuit, content free of earlier
position effects, a count-only mechanism, natural exit timing or physical recovery.
"Exit" here means the first different coordinate after an EXACT-token row run;
the image may continue repeating nearby boxes or later reach its generation cap.

## Frozen cohort and source identity

Source: 2026-09-18-untied-highconfidence18-natural, untied-original only, original
native FP32/SDPA batches, prompts/media, adapter and independent embedding deltas.
Use its event-selection-order.json. Exclude already probed images7511 and269858;
take the first three distinct remaining images with an adjacent different complete
row after>=3 consecutive exact-token rows and a coordinate first difference.
For each take the earliest eligible run. Do not select by intervention outcomes,
eventual EOS, visually presumed owner validity or largest effect.

| Image | Group / batch | Run rows inclusive | Donor row | Recipient row / role | Raw action | Repeated -> native bin |
|---|---|---|---:|---|---:|---|
| val14038 | refined-02 /2 |8..10|9|11 /x1|108|795 ->801|
| train351017 | refined-03 /2 |3..10|4|11 /y2|106|86 ->85|
| train417044 | refined-03 /3 |16..30|17|31 /y1|315|197 ->159|

Rows can have different token lengths. For a first differing token at row offset
d, the query input is offset d-1; substitute only row positions[0,d-1), excluding
the query's own key. Expected prefix counts are3,6,5. Bind exact token equality,
raw/trace alignment, source/destination offsets, actual padded prompt width and
physical query = prompt_width + raw_action -1 in selection.json. Reconstruct the
ORIGINAL full batch prefix through that input using the maintained native request
and replay helpers. Do not run singleton batches, retokenize rendered output,
change checkpoint, truncate companions differently or use code from outputs.
The selection's model_identity_sha256 hashes UTF-8 json.dumps(identity,
sort_keys=True) with Python's default separators. Compare the loaded effective
identity to the saved receipt itself; a serialization difference is not a model
identity waiver.

## Three cells per image

1. native: fresh full native replay; capture original selected logits and prefix
   pre-K/V/phases. It is the production-shaped admission, not an extra smoke.
2. identity: full native attention plus selected-query recomputation with native
   K/V and the explicit native mask row; replace only that attention output.
3. donor_content: same recomputation, replacing destination prefix K by LIVE
   donor normalized pre-K rotated at destination-native cos/sin and V by LIVE
   donor V. Do this in all28 layers. Earlier queries stay native; current Q and
   later states adapt normally. No edits to inputs, positions or the full-call
   history/cache. Donor positions precede the target and cannot depend on its
   intervention. No generation or cache reuse between cells.

Use exact text-module identity to scope hooks. Live k_norm layout is[B,S,8,128];
v_proj produces[B,S,1024], reshaped to8 heads. The actual masked SDPA path uses
16 query heads and repeats each KV head twice. Selected query shape[1,16,1,128];
keep all native key slots, explicit row mask (including left padding), scale and
is_causal=False. Other queries' outputs must equal the original full call exactly
at each replacement boundary. No global historical K/V edits are allowed.

## Gates, evidence and stop

CPU qualification must exercise the installed SDPA caller and actual replacement
consumer with unequal dimensions, GQA, left padding and variable prefix lengths.
Use independent paired-coordinate rotation/attention expectations. A wrong phase
or V donor and a wrong-query/global-history modification must fail the SAME
production identity/consumer checks, not just an assumed patch-index guard.

Before first model call, bind producer/imported installed sources, checkpoint
identity, panel/config, original raw/trace/runtime receipts and literal prefix
mapping. Use relative archive names for installed dependency captures. Freeze a
source-to-cell manifest; keep maintained source in probes/, archives at their
existing owner. Save one final tensor payload per cell and bind its hash AFTER
all fields are written; do not enrich an already-bound tensor in place.

Native target global top-two IDs/logits and logsumexp must reproduce the saved
natural trace within2e-4. Identity full-vocabulary logits for the whole batch must
match fresh native within2e-4 with identical target winner. Stop before donor
treatment if either gate fails. All treatment companions match fresh native.
Verify native prompt/input/media identities, masks/slots/positions at their real
consumers, and expected selected LM query. Layer0 donor/destination pre-K and V
must be identical for the same-token prefix; incoming layer1 at the target must
match native within2e-4, excluding a first-layer state substitution explanation.

Save live donor/destination prefix pre-K/V, destination phases, actual selected
prefix K/V and query Q, attention-output target slices and full-vocabulary logits.
Bind full-call and unchanged non-prefix K/V by hashes and exact runtime checks;
do not duplicate whole-sequence K/V across layers beyond the96MiB payload bound.
Independent FP64
rotation readback<=2e-4, exact donor V, unchanged non-prefix K/V, original full-call
inputs before/after selected computation, exact off-target output replacement,
actual o_proj consumption and source hashes qualify all28 layers. Save raw
evidence/errors before numerical gates. Full raw payload stays on disk; report
top10 and probabilities/ranks of repeat/native/global competitors.

Exactly9 model/9 vision calls, one GPU4,15minutes after model setup begins,
96MiB tensor payload total. Expected252 full text-attention and168 single-query
kernels; no hidden extra model smoke or repeated vision prefill. Execute first
image's three cells, report the first qualified production result, then continue
the other two automatically under this contract. On failure preserve its receipt
and stop without automatic retry or dropping/replacing a case. Root decides any
bounded repair; technical invalidity is not a scientific null.

Worker owns only probes/training_set_completion/recurrence_local_prefix_content.py
and this unit's output attempt-001. Root owns selection, this unit, independent
readback and acceptance. Reuse native reconstruction/helpers, not old hardcoded
phase globals or a generic hook framework. No peer codebook/three-loss model work,
training, new generation, publication or unrelated cleanup. Stop at stable
candidate; only root marks lead-accepted and chooses a successor.

Before the first launch, root took exclusive producer and runtime ownership
from gpt-6-luna/max after the installed SDPA CPU qualification passed. The worker
is read-only for its final handoff; root executes the same frozen nine-cell plan.
This changes execution ownership only, not the cohort, contrast or stop rule.
