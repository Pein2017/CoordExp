# Endpoint loop: readout norm advantage and conditional state alignment

**Lead-accepted; bounded conditional readout/state result.** Independent CPU recomputation verified all144 slot decisions/decomposition terms, all aggregate counts, exact equality to27 saved native-logit rows, common effective readout tensors, canonical teacher counts, source bindings and completed jobs. [Lead receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/lead-acceptance.json) and [CPU verifier](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/lead_verify.py) retain acceptance; candidate research records and original result/terminal bytes are preserved. No learned repair, generation, or causal intervention was executed. The local coordinate-substitution series remains closed.

## Decision

Fixed output-row norms contribute to endpoint preference, but cannot account for it alone. Effective coord_999 and coord_0 rows have the largest and second-largest norms among1000 coordinates. Removing row-norm differences and bias offline changes17 of69 native endpoint wins to interior wins;52 endpoint wins survive. All9 cells have identical effective readout rows, yet the same readout yields interior decisions throughout the accepted escape and strong endpoint alignment in repetition/relapse. This is conditional state/readout evidence, not an identified training-origin mechanism.

Norms: coord_999=1.395369907, coord_0=1.357017008, median=1.312550512, range=[1.280969147,1.395369907]. There is NO output bias. Tied base storage and shared additive delta were verified; effective input and output rows are numerically identical, but separately saved/hashed. Base endpoint norms are1.317094207/1.325757027; delta norms0.211102530/0.209660739. These are descriptions, not causal decomposition of training origin.

## Frozen conditional panel

144 coordinate slots =36 rows across9 checkpoint/image cells. All checkpoint comparisons use the SAME literal histories and within-row preceding tokens. P rows are native P histories; other checkpoints are conditioned/off-policy (no claim that they naturally reach those states). C01 relapse paths include supplied row137, hence conditional even for R16 and off-policy for the others. Own-path versus cross-checkpoint questions are not pooled as natural failure rates.

| Image / literal path | Slots | Native endpoint wins | Equal-norm endpoint wins |
|---|---:|---:|---:|
|477415 / P-row1|12|3|3|
|477415 / P-row4|12|5|3|
|477415 / P-row16|12|11|9|
|477415 / P-row138|12|0|0|
|477415 / C01-free46|12|8|3|
|477415 / C01-free62|12|12|10|
|351017 / P-row1|12|6|3|
|351017 / P-row2|12|6|6|
|351017 / P-row16|12|6|6|
|417044 / P-row1|12|6|3|
|417044 / P-row2|12|3|3|
|417044 / P-row16|12|3|3|

Across Bnormalized64/P16/R16 respectively, native endpoint wins24/24/21 of48; equal-norm wins17/18/17. There are33 total coordinate argmax changes, including interior-to-interior changes. These selected-slot counts are descriptive, not independent statistical trials.

The shared477415 escape has0/12 endpoint winners both before and after normalization. Established-repeat slots retain18/20 endpoint wins; all999 relapse retains10/12. At relapse onset,8 endpoint wins fall to3. Thus norms can tip weakly separated decisions while context-specific angular alignment sustains many others.

The panel contains first-row, geometrically valid border examples. In351017 and417044, y2=999 becomes990/989 under equal norms in all3 checkpoints; six norm-sensitive endpoint wins therefore occur in legitimate geometric border examples, not only pathological loops. This does NOT certify physical owner identity, but illustrates why removing endpoint preference is not automatically beneficial. There is no clean non-border x1 baseline in this selected panel; no extra images or favorable search were added.

## Exact decomposition and technical scope

Captured h is the actual lm_head input AFTER Qwen final RMS normalization and all language DoRA effects. The actual output wrapper computes base_linear(h) plus h @ shared_delta. W is base+delta, not the misleading base-only `.weight` property. For each coordinate c: z_c=||h|| ||W_c|| cos(h,W_c)+b_c. The reducer reconstructs this in FP64 from saved FP32 tensors and native coordinate logits, then computes all1000 cosine scores and a common-median-norm/no-bias counterfactual. With observed bias absent, h norm scales logits but cannot change coordinate argmax. Input-embedding influence remains upstream in h.

Each slot table row includes endpoint and strongest interior competitor, native/equal-norm argmax and margins, norm/cosine/bias terms, endpoint ranks and reconstruction error. All1000 rows, native coordinate logits and final hidden vectors are retained, so the counterfactual is independently reproducible rather than restricted to top-k candidates. No full-vocabulary decoder was modified; coordinate-only rankings cannot establish owner recovery. All144 native full-vocabulary argmaxes were coordinate tokens.

Maximum affine reconstruction error=2.59920025e-05; maximum2*error/native coordinate margin=0.009159; reconstructed argmaxes agree in144/144 slots. Reused27 row files agree with saved native coordinate logits exactly (max error0). This validates the extraction seam without rerunning accepted cache/full/no-op controls. New rows use the same full-prefix FP32/SDPA bs1 path; they are not claimed as heterogeneous-bs4 free-generation replay.

Endpoint raw IDs are151670 (0) and152669 (999); all1000 IDs were verified through the tokenizer. Literal raw sequences already contain thin corner/full-canvas/zero-extent/inverted-extent/repeated endpoint rows before parsing or drawing. Prior accepted consumers distinguish these forms and cap; no clipping, rasterization or post-hoc coordinate normalization creates the measured token concentrations. A border coordinate by itself is valid and not a loop. The C01 relapse rows46/62 are respectively chair[999,776,999,781] and chair[999,999,999,999]; raw tokens and offsets are bound in panel.json.

## Bounded exposure audit

| Current canonical train256 role | Tokens | bin0 | bin999 |
|---|---:|---:|---:|
|x1|1955|84|0|
|y1|1955|38|0|
|x2|1955|0|135|
|y2|1955|0|69|

Endpoints are the most frequent individual bins in their conventional roles here. Opposite-role endpoints are absent in this current teacher inventory, while pathological loops use them. This is1955 annotated canonical boxes, not presentation-weighted completion exposure, rejected-output exposure, complete optimization history or pretraining counts. It cannot explain training origin or make unmatched objects false positives. No dataset cleaning or new labels.

## Ranked hypotheses and lead-selected next causal contrast

| Rank / hypothesis | Prediction | Evidence for / against | Discriminating evidence |
|---|---|---|---|
|1 Repeated-history feedback/state attractor|Recent trajectory can sustain endpoint alignment despite equal readout norms; changing its state can release or relapse|Selective bridges and late relapse;52 normalized endpoint wins. Token edits alone do not locate a cache mechanism.|Cross the accepted looping/escape histories with one matched natural donor image; test visual dependence of maintenance.|
|2 Output readout prior|Endpoint row scale can tip otherwise close decisions|Endpoint norms rank1/2;17 wins depend on row norms. Against sufficiency:52 survive and escape has0.|Already discriminated against norm/bias-only sufficiency; actual repair efficacy remains untested.|
|3 Visual-conditioning dependence|Adequate visual evidence changes conditional alignment/exit|Old image/history and KV results support plausibility; this panel keeps images fixed, so cannot isolate it.|Separate future matched visual-conditioning intervention would be needed; not requested in parallel.|
|4 Generic off-manifold/format attractor|Invalid/supplied histories fall into a generic repetitive schema state|Invalid natural/supplied trajectories relapse; however natural loops and valid-coordinate repeats exist too.|Validity/format-matched state controls are necessary; current evidence does not reduce all loops to malformed input.|

**Lead decision after acceptance:** first cross the accepted R16/C00 looping history and R16/C10 category-only escape history with one deterministically selected natural donor image that has an existing healthy R16 native-output receipt and matching executed image grid/prompt positions. Reuse the original-image cells after admitting any necessary history-supply runtime seam; add only two donor-image continuations. Fix each literal prefix, checkpoint, readout, batch topology, free horizon and unrestricted EOS. Score only free suffixes against the corresponding image's known-owner bank; foreign supplied history receives no owner credit. A separate frozen unit owns this maintenance question and its runtime/cost boundaries.

If image substitution releases C00, current visual conditioning contributes to maintenance. If C00 remains trapped while C10 produces donor-specific content, history-dependent access to visual conditioning becomes more plausible. A null contrast excludes only this donor. Neither outcome identifies initial-entry causality or removes the image/history mismatch limitation. No donor sweep follows.

The worker's proposed late-relapse K-direction transplant is deferred. Equal K-vector norms do not equalize donor-to-receiver angular displacement or perturbation magnitude, and K-only transfer creates a synthetic K/V pairing. Another successful cache patch would establish conditional actuation without cleanly separating repeated-history maintenance from changed visual routing.

Success must be sustained free trusted-owner coverage with retention, recurrence, invalid/repeat/UNKNOWN burden and EOS over the unchanged horizon; a first-token change or attention plot is insufficient. The ultimate objective remains ordinary greedy physical FN reduction with precision and old-owner retention, not an inference-time oracle patch.

## Cost, evidence and closure

Executed36 model forwards/36 vision calls,0 generated tokens;340.569 allocated GPU-seconds (0.09460 GPU-hours), peak reserved13067354112 bytes, tensor payload296741724 bytes. Launch-to-closeout elapsed243.7s, within both2-hour ceilings. The first real cell was reused as part of the panel; remaining8 cells ran independently on8 GPUs. No model failure, rerun, control generation, training or new intervention. All owned jobs ended.

Codegraph was queried on the exact worktree; load-bearing selected-delta and installed Qwen final-normalization/head paths were verified against source and native reconstruction. Saved27 logit files were inspected before forward launch; they lacked hidden states, so the36 frozen extraction calls supply the missing decomposition quantities.

- [panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/panel.json)
- [result.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/result.json)
- [slots.tsv](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/slots.tsv)
- [teacher-exposure.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/teacher-exposure.json)
- [producer.py](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/producer.py)
- [reduce.py](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/reduce.py)
- [run.sh](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/run.sh)
- [launch.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/launch.json)
- [terminal.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-endpoint-loop-readout-state/terminal.json)

Predecessors: [accepted mechanism](../2026-09-16-corner-loop-mechanism/results.md), [accepted factorial](../2026-09-16-corner-loop-bridge-factorial/results.md), [accepted y1/y2 series close](../2026-09-16-corner-loop-y1-one-bin-control/results.md). The mechanism predecessor links older small-owner/history/KV controls; those evidence boundaries remain in force.

Technical status: valid bounded extraction; scientific status: lead-accepted conditional norm contribution plus surviving state alignment. Training-origin causes and causal repair remain unanswered. This extraction unit is closed; the separate image/history package is lead-authorized under the user's overnight grant.
