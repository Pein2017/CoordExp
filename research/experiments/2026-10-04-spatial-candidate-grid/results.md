# Fixed spatial queries produce nine additional visual bottle identities

**Lead-accepted; closed.** On image351017, the32 predetermined x1 queries
produce a conservative count of **nine distinct, visually supported bottle
entities absent from the saved native B16 baseline**. Root inspected every
candidate and all14 unique baseline geometries in the full scene and enlarged
original-pixel context. These are assistant visual judgments, not human ground
truth. The complete [visual ledger](visual-ledger.json) retains every slot,
reason, original geometry, source and unresolved correspondence.

This is a finite supplied-cue candidate recipe on a full-label training image.
Only query-coordinate selection is GT-coordinate-free. The native bottle header
and preceding person history are supplied, and the32 independent continuations
are not one ordinary greedy trajectory. The result does not establish natural
physical-FN reduction, held-out transfer, complete physical recall or safe
training targets.

## Primary identities and geometry

| Query | x1 | Visible identity | Geometry limitation |
|---|---:|---|---|
|06|203|Pale-capped bottle with a large light label behind the left head|Lower-body occlusion|
|08|265|Brown bottle with a green/light neck band in the left rear cluster|Narrow body strip|
|12|390|Dark wine bottle with a white label behind the tabletop menu|Small edge clipping|
|13|421|Left tall-necked bottle of the rear pair|Extra background; lower body truncated|
|14|453|Separate right tall-necked bottle of that pair|Lateral/lower clipping|
|16|515|Closed silver-topped transparent cylindrical bottle on the table|Lateral clipping and overlapping glassware|
|17|546|Dark red-necked bottle with bright vertical label beside the rear head|Small border/context error|
|20|640|Exposed neck/cap and side of a bottle beside the rear head|Heavy occlusion; weakest-resolution credited case|
|22|703|Slim red-capped dark bottle beside the right shelf post|Background above; partial body width|

Visible neck/cap/body continuity and separate scene positions establish these
identities. The adjacent13/14 have separate necks and caps;16 has a closed top
and continuous cylindrical body, unlike the surrounding stemmed glasses. A
focused independent Astra review opened all nine crops and the full scene and
found no category, identity, duplicate or baseline-correspondence counterexample.
Root retained the final judgments and the substantial visibility limitation on20.

Five of the nine have maximum saved norm1000 bottle-annotation IoU below0.5.
That is geometric context, not a physical identity test. Conversely, bottle
pixels in a multi-object region do not automatically identify a single owner.
No annotation-unmatched candidate was declared false solely from that status.

## Complete denominator and costs

| Pixel-support assessment | Slots | Interpretation |
|---|---:|---|
|Definite bottle support|13|Nine resolved new identities; four partial/multiple-identity cases get no primary credit|
|Definite non-bottle entity|3|Shelf support07 and wine glasses18/23|
|Ambiguous category|7|00/04/09/15/19/24/25 remain unresolved|
|No identifiable supporting entity|9|01/02/03/26 and background regions27..31|
|Total|32|Every prescribed query remains in the denominator|

The supplied bottle header means these are errors in pixel support for the
conditioned category, not errors from a freely generated category decision.
The four definite-bottle cases with unresolved single identity are05/10/11/21.
Together with the seven category-ambiguous cases,11 candidates retain category
or identity uncertainty. There are no confirmed physical duplicates among the
nine credited identities; duplicate burden across unresolved candidates remains
unknown. Five overlapping right-edge background outputs are repeated unsupported
regions, not five counted bottle duplicates.

All32 queries complete their selected row. Thirty-one original boxes are
numerically valid. Query03 is the original zero-width `[109,0,109,44]`, retained
as invalid without repair; its annotation comparisons are null. Numerical
validity therefore does not establish visual support. The sole post-hoc exact
x1/annotation-coordinate coincidence is query18=578, whose output is a wine-glass
bowl; no coincidence selected or adjusted a query.

The historical baseline retains3084 tokens,308 complete rows,307 bottle
occurrences in14 exact category/geometry display groups, and the unavailable
tail at actions3079..3083. Every baseline box lies in the extreme upper-left
region. That region contains partially obscured neck-like silhouettes; Root
keeps individual category/identity judgments ambiguous and does **not** declare
the baseline physically empty. The nine credited bottles are identifiable at
separate scene locations. No absolute baseline bottle census is needed or made.

## Execution and acceptance evidence

One invocation at execution source
`e6dda6de37ddb320145abb2aaa97e971027b1657` used the unchanged implementation
`18a9526eab66c3e279028ae03d143865670a04be` and B16 checkpoint. The19-action
historical fidelity anchor and all32 grid queries completed:33 requests,
627 actions,494 forced and133 free, one model load,33 prefills and594 subsequent
cached predictions. All stopped at the literal19-action budget. The fresh
anchor exactly matches output IDs, all pre-force median winners and recorded
selected likelihoods; it qualifies only that short prefix, not the full saved
historical trajectory.

Native and saved-data gallery commands both exited0. Native external/producer
time was83.759/80.437s; generation67.688s. Peak native RSS was13,805,785,088 bytes
and allocated CUDA4,874,622,976 bytes. Gallery rendering took19.369s.
The expanded retained execution package was213,606,486 bytes, below256MiB.
All33 resource snapshots passed. No training, backward, optimizer, warmup,
retry, reload, extra query or duplicate consumer occurred.

The automatic consumer ran once and admitted the original detached-FP32 CPU
canonical summaries. Optional GPU-minus-CPU log-normalizer differences reached
one FP32-scale increment; they remained descriptive, with no changed tolerance
or cross-device equality gate. Original native likelihoods and vectors remain
preserved. A read-only process-inventory parsing error was corrected and its
failed evidence retained; neither native execution nor source was changed.

The final gallery contains all32 candidate and14 baseline contextual views,
full-scene pixels and indexed overviews, with full49-object annotation pools
and original invalid coordinates. Root opened all32 candidate views. Native
baseline PNGs are byte-identical to the14 actual historical-baseline views Root
had already opened, so those judgments were reused. Root independently checked
the final consumer/record/prefix/denominator/manifest boundary without rerunning
the model, tests, gallery or scientific consumer. All owned native/gallery jobs
settled and the source holder was released.

Artifact owner:
`outputs/research/physical-fn-recovery/2026-10-04/spatial-candidate-grid/`.

- `native-candidate-01.json`: SHA256 `37d99e233d54d5243ebccbf1c4984327e7af133ef7a29367f7d642406ca20d28`.
- `native-01/readback.json`: SHA256 `a444d3d3e504ef0af5a172e427bae42291e0b346cbfcb9a1cfc788e44a66af54`.
- `native-01/gallery/index.json`: SHA256 `e1b1dc927e0a1d7fc04457198f5b6ca39b65c7f296da9ba1b4a33e42aafcc7cf`.
- `lead-native-acceptance-01.json`: SHA256 `3913d29626a6fbceffb738f95a0fe38b4198ee399e9202f924a5622d24174238`.
- `native-projection-01.json`, `package-settlement-01.json` and the complete visual ledger retain the detailed expansion path.

## Inference and remaining question

A fixed coarse x1 rule can reach useful visual bottle candidates beyond the
native corner cluster on this image. Exact GT-derived x1 cues are therefore
not necessary for every useful candidate in this selected state. The rule also
produces category, unsupported-region and localization failures, so conditional
regional access alone does not supply a reliable entity selector.

A post-hoc saved-data check gives no support for pruning this batch by a simple
completion-confidence score. The score is the mean actual full-vocabulary
median-policy log probability of the three free coordinates at actions15..17,
excluding forced x1 and closure. Against13 definite-bottle-support slots and12
definite-negative slots, with seven ambiguous slots excluded, its pairwise AUC
is63/156=0.403846. Its top8 retain3/9 known new identities and include four
definite negatives; top16 retain4/9 and include seven. Unsupported background
queries29/30 rank first and second. This was not prespecified in the native
unit and fits no threshold or classifier; it is an exploratory same-panel
observation, not a general calibration result. The exact32 scores, ranking,
tie convention and source/ledger bindings are preserved in
`posthoc-free-coordinate-score-01.json` under the artifact owner above.

This does not identify an internal attention route, prove that the naturally
selected first coordinate has an intended owner, or show transfer beyond the
training image. Candidate validation/selection, held-out transfer and eventual
ordinary-policy realization remain separate questions. The frozen unit and its
single release are consumed and closed; any successor requires a separate Root
selection under the user's continuing grant.
