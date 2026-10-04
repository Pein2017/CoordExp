# Coordinate readout and early recurrence diagnostic

## Question, authority and boundary

The user authorized this diagnostic on 2026-10-04: “好的，请继续让 worker 来做这一轮探索”. Does the current B0/B16 coordinate vocabulary retain useful distinctions between nearby values in the actual computation, and how do selected invalid/copying decisions respond to controlled history changes? This is descriptive diagnosis, with zero training. Root owns scientific interpretation and release; `/root/coord_readout_impl` owns implementation and the bounded execution package; `/root/coord_binding_reasoning` owns the scientific brief and interpretation support.

Hypothesis: similar neighboring coordinate rows contribute to small decision margins or weak sensitivity to coordinate history. Strongest alternative: the contextual hidden state/slot choice is wrong even though the vocabulary can distinguish those coordinates. Static cosine alone cannot distinguish these. Actual natural-context scores, valid narrow controls, and single-token history interventions can falsify complete indistinguishability locally. They cannot uniquely establish instance binding or identify a training cause. Monotone coordinate quantization can yield equality, but cannot by itself reverse ordered endpoints.

## Frozen inputs and computation

Use B0 and B16 from `outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01/checkpoint-{0,16}`. Saved raw and companion analysis records are under rank7/image351017, rank4/image7511 and rank3/image13348 at the corresponding version. Bind exact filenames and source identities in the input packet. Reuse original full-label/prompt/media/grid/runtime identities, with labels from `research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json`. These are prescribed endpoints, not checkpoint selection. Do not mutate historical evidence or use historical execution instructions as authority.

The base is frozen and tied; FP32 input/output deltas are independent. Coordinate IDs are151670..152669; checkpoint delta rows4..1003 correspond to these1000 values. Read only selected base rows and delta payloads on CPU. Report analytic FP32 input rows and actual BF16 input-addition rows separately. Compute output effective rows, FP64 norms, maintained lower-median factors and normalized directions. Report adjacent cosine/distance distributions, exact duplicate groups, B0-to-B16 same-coordinate changes, and pairs120/121 plus observed values and their immediate neighbors. Output effective-row algebra is descriptive: actual logits separately accumulate and round base and delta terms. No geometry pass threshold.

Use maintained native model loading without an optimizer, actual frontend/media qualification, `generate_continuations`, cached singleton transitions, eval/inference mode, BF16 autocast and `MedianPolicy`. Do not use a full-prefix replay/prefill as a numerical substitute. Capture raw scores before normalization and actual policy scores after normalization but before forcing. For an N-action request, force literal history at positions0..N-2; leave the final action free. Record pre-force argmax and requested-token raw/policy log probabilities at every step, even where forced. Never label post-force likelihoods as natural likelihoods; never mutate the raw scores in place.

At named observation positions preserve full1000 coordinate raw/policy scores with token ordering, full-vocabulary log-normalizers, top5 non-coordinate scores, EOS/object-start scores, argmax, coordinate role and actual prefix identity. Derive neighbor margins, best-legal versus best-illegal margins, coordinate mass, and intervention differences (exact equality, max/RMS log-probability differences, argmax changes). Illegal/valid geometry, copying and EOS are separate observations. No requirement that an intervention improve a metric or move in a specified direction.

## Exact request matrix

All P indices and action positions are zero-based. Run B16 first, then B0; two model loads total. Natural means the checkpoint's own saved prefix, with every pre-force argmax checked; B0/B16-history is a conditional score comparison. Observation sets follow literal saved analysis selectors, verified in preparation.

| Checkpoint / condition | History and observations | Actions |
|---|---|---:|
| B16 natural351017 | Bottle P1 coords14..17; exact copy P2 coords24..27; P3/P4 coords34..37/44..47; boundaries9,19,29,39,49 |50|
| B16 natural7511 | Valid narrow P45 coords410..413 `[670,577,677,597]`; zero-width P46 coords419..422 `[684,577,684,597]`; reversed-x P48 coords437..439 with x2=677 after `[684,577]` |440|
| B16 natural13348 | Normal worker P25 coords231..234 `[544,632,559,682]`; reversed-y P26 coords240..243 `[584,629,599,627]` |244|
| B16 zero-width input-1 |7511 change only action419 from684 to683; observe x2/action421 |422|
| B16 zero-width input+1 |Same,684 to685 |422|
| B16 normal input-1 |13348 change only P25 x1/action231 from544 to543; observe x2/action233 |234|
| B16 normal input+1 |Same,544 to545 |234|
| B16 copy input-1 |351017 change only first bottle x2/action16 from23 to22; retain other original actions through28; observe boundary19,coords24..27,boundary29 |30|
| B16 copy input+1 |Same,23 to24 |30|
| B16 trusted current-row prefix |13348 P25 actions231..233 become `[546,633,556]`; observe pre-force x2/action233 and y2/action234; previous history unchanged |235|
| B0 natural351017 |Own saved prefix; bottle coords44..47 and boundaries9,18,28,39,49 |50|
| B0 natural7511 |Own saved prefix; valid narrow P3 coords31..34 `[44,539,53,553]`, P4 coords40..43 `[68,538,79,553]` |45|
| B0 natural13348 |Own saved prefix; worker P5 coords50..53 `[545,632,558,685]`, actual EOS/action55 |56|
| B0 B16-history351017 |Unedited B16 history and B16 observation positions |50|
| B0 B16-history7511 |Unedited B16 history and B16 observation positions |440|
| B0 B16-history13348 |Unedited B16 history and B16 observation positions |244|

The trusted worker is annotation191150 with box `[546,633,556,682]`; verify its unique saved-row/label binding in preparation. This is a normal-worker control, not a GT repair assigned to invalid P26. Do not assign owners to invalid boxes or boundary bottles by IoU. Other narrow controls are geometrically valid, not necessarily correct detections.

## Fidelity, resources and stopping

Natural controls must reproduce their own saved token and pre-force argmax at every step, mismatch count exactly0. Compare saved/current likelihood differences descriptively without inventing a new tolerance. Run a checkpoint's natural controls before its dependent conditions; B0 cross-history requires the corresponding B16 and B0 controls. A natural mismatch marks that image/checkpoint's dependent cells HOLD; preserve completed evidence and continue only independent cells. Changed or unchanged counterfactual argmax, continued copying and failure to improve are results, not fidelity failures. Wrong identity, malformed selectors, nonfinite scores, unsupported policy/runtime, unexpected edits, ambiguous trusted owner, OOM or execution failure cause explicit technical HOLD, without retry.

Exactly16 requested cells, at most3226 emitted actions including forced actions, six single-token ±1 interventions and one trusted-row substitution. One serial GPU, two checkpoint loads, singleton batches; at most440 actions and1760 prompt-plus-generated tokens/request. Zero optimizer/backward/training, full-scene rollout, warmup, extra forward, export or automatic retry. Operational request-boundary limits:15minutes producer wall,32GiB peak RSS,12GiB peak CUDA allocation,32MiB retained diagnostic payload. Stop starting requests after a resource limit is exceeded; preserve actual counts, timing, memory and partial results. Shared GPU stress is expected; launch directly and react only to concrete failure/conflict.

Implementation belongs in `probes/coordinate_readout.py` with focused tests. Artifacts belong under `outputs/research/physical-fn-recovery/2026-10-04/coordinate-readout-audit/`. A CPU positive/mutation consumer check qualifies the candidate; root binds a clean source commit and exact input/runtime packet before one native invocation. CPU qualification does not establish numerical fidelity. Consumer output must enumerate every requested cell, including skipped/HOLD, bind original/edited prefixes and observation positions, separate technical fidelity from scientific measurements, and preserve raw evidence. Worker may repair implementation within this frozen package before release; no model relaunch without a new decision.

Stop after native acquisition, saved consumer report and lead acceptance/partial/HOLD ruling. No automatic next probe or training follows. Claims are limited to these coordinate rows and supplied contexts: no global competence, unique instance-binding cause, natural-loop escape, physical-FN recovery, checkpoint promotion or learned-stability claim.
