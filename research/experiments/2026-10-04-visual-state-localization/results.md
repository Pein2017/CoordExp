# Visual-effect transport: accepted partial result and score adjudication

## Lead disposition

Lead closes this unit with partial native evidence and a separate, CPU-derived
adjudication of the frozen median endpoint score. The single released run
acquired all20 requests/2550 actions, but exited2:18 cells passed the original
consumer and two remained attempted-invalid. Both original middle paired
contrasts remain HOLD. Their statuses, validator, likelihoods and raw artifacts
have not been changed. This is not a20-cell technical PASS.

The additional adjudication accepts only the middle **median coordinate
differences** computed from unchanged captured vectors. It does not restore
whole-cell qualification, raw normalized probabilities or invalid-cell box
evidence. No new model call, source repair, tolerance, retry or duplicate saved
consumer occurred. All processes are settled and the source holder is released.
No successor is scheduled.

## What the selected states show

The strongest finding is the person control. Removing the target region changes
the freely generated box substantially, whereas the matched background
perturbation leaves it near the designated person. Transplanting only the
target-masked current-token residual after block2 into a clean receiver leaves
the box near the original target too.

| Person condition | Accepted free box | Designated annotation IoU |
|---|---|---:|
| Clean |`[544,632,559,682]`|0.653333|
| Target image perturbation |`[684,607,701,638]`|0|
| Background image perturbation |`[545,632,558,682]`|0.753846|
| Target residual after block2 |`[544,632,558,682]`|0.700000|
| Background residual after block2 |`[543,632,559,682]`|0.612500|

The target-masked image's median x1 winner is684; the designated546 drops from
full-vocabulary rank5 to423. It does not lose to endpoint0. Endpoint0 is a
fixed distant reference for this person, not its natural top competitor.
The target-masked box has zero overlap with every supplied person annotation;
this is not a physical identity or false-positive judgment.

Let `m=s(t)-s(0)` in the actual unforced median score channel, with t186 for the
bottle and546 for the person. Image rows compare perturbed and clean image
requests. Residual rows compare clean receivers patched with the corresponding
donor and an unpatched clean request. The clean m values are−5.690679550 for
the bottle and18.559119225 for the person.

| State / condition | Target minus clean | Background minus clean | Target minus background | Evidence |
|---|---:|---:|---:|---|
| Bottle / image |+0.230543137|+0.125484467|+0.105058670|Original consumer|
| Bottle / postblock2 |+0.125484467|+0.125484467|0|Original consumer|
| Bottle / postblock13 |+0.125484467|+0.125484467|0|Separate median-score adjudication|
| Bottle / postblock27 |+0.230543137|+0.125484467|+0.105058670|Construction control|
| Person / image |−18.296557426|+1.126668215|−19.423225641|Original consumer|
| Person / postblock2 |−0.045141459|−0.075235844|+0.030094385|Original consumer|
| Person / postblock13 |−9.321802139|+0.480574608|−9.802376747|Separate median-score adjudication|
| Person / postblock27 |−18.296557426|+1.126668215|−19.423225641|Construction control|

For this person state, the postblock13 current-token residual transports a
substantial change in the fixed score competition under the clean receiver
context, while the postblock2 transplant carries little of that endpoint
effect. This is a useful difference between two tested interfaces. It does not
locate where the information originated or identify a defective layer.
The final-boundary result is exact by construction and serves as a transport
control, not another independent localization result.

The receiver retains its original prefix KV and the current-token KV computed
through the replaced layer. Later attention can therefore reintroduce clean
image context. KV-driven reconstruction, information distributed across tokens
and caches, and nonlinear donor/receiver interaction remain compatible
explanations. Small early endpoint effects do not demonstrate absent early
visual information, and these contrasts are not fractions of uniquely mediated
owner information. Only two sparse non-final boundaries were tested.

The bottle is less informative for this localization question. All admitted
branches still choose x1=0 and have zero overlap with the designated bottle.
Target/background image effects differ by only0.105059 on this fixed median
contrast; both early transplants and the adjudicated middle transplants change
it equally. Equal scalar changes do not imply equal hidden states or complete
score distributions. This does not show that bottle visual information is
absent or explain why a supplied x1 helps in the preceding unit.

## Why the run failed, and what the adjudication accepts

The frozen validator recomputes compact summaries on CPU from saved full-score
vectors and requires exact equality with the native observation. The two
failures are isolated to raw `full_log_normalizer`:

| Cell | Native GPU FP32 | CPU from the saved vector | Native minus CPU |
|---|---:|---:|---:|
| bottle-background-middle |24.365333557128906|24.365331649780273|+0.0000019073486328125|
| person-target-middle |19.604457855224609|19.604459762573242|−0.0000019073486328125|

Each difference is one FP32 ULP. All raw/median vectors are finite; coordinate
scores, argmax, noncoordinate top5 and other compact fields match. Every median
compact field matches. The saved native-head/raw binding, actual token, prefix,
cache and hook-removal evidence also remains valid. The CPU-only Worker
diagnosis used the existing two tensor artifacts and exited0 without CUDA or
model execution. The original validator still rejects both cells.

Lead's acceptance reason is dependency separation, **not the small error size**.
The primary median score difference does not use the raw normalizer. Source
inspection confirms that the compact normalizer is an observation: greedy
selection uses the incoming scores; residual replacement uses captured states.
It did not feed back into either failed acquisition. Those middle cells have no
dependent requests and all20 requests were acquired.

Before inspecting the missing middle values, Lead fixed a narrower adjudication:
read only the six bound median vectors needed for the two clean/target/background
comparisons; retain all original execution controls and reject any score or
identity mismatch. The CPU reader verified file/tensor hashes, dtype, finite
values, exact coordinate summaries and argmax, cell/request/action/prefix/cache
pairing, donor identity and unchanged-receiver attestations. Accepted partner
and clean margins exactly reproduce the original consumer's convention.

Six in-memory counterexamples changed the designated coordinate score while
retaining the old identity and compact record; all were rejected. Removing all
raw fields left the derived values unchanged. This calculation imported no
Torch and ran in0.086513s; it invoked neither the model nor the saved consumer.
The resulting evidence is separately stored in `lead-score-adjudication-01.json`.
The invalid cells' raw probabilities and free-box outcomes are not admitted by
this ruling. The original run remains technically incomplete.

A possible future implementation repair would retain the observed GPU
normalizer and separately bind a canonical CPU compact summary for saved
comparison, preserving strict tensor/control equality. No such repair or new
release was executed here.

## Controls, denominators and costs

Both clean controls exactly reproduce every saved token and pre-force median
winner. Raw historical likelihood differences are0; median differences of
5.364418e−7 and7.152557e−7 remain descriptive. Both no-op controls exactly match
the complete sequences, score-step/likelihood channels and selected h/raw/median
vectors. All four final controls exactly transfer donor normalized h and full
raw/median vectors, with maximum difference0 and no tolerance. The bottle
background-final tail differs from its donor at x2 despite exact x1 transfer;
their different retained caches permit that continuation difference.

All18 originally admitted selected rows are complete, valid and coordinate-only,
with no selected semantic divergence or EOS. Every request ends at its literal
budget. The person-prefix duplicate is already present in the common supplied
history. Annotation overlap is descriptive; none of these supplied-prefix or
intervened trajectories earns natural physical-recovery credit.

Actual work was one B16 load,20 prefills and2530 cached predictions, totaling
2450 forced plus100 free actions. The producer counts2550 acquired actions;
the original consumer counts2295 admitted actions/90 free actions, excluding
the two acquired-invalid histories of19+236 actions. There are no skipped cells,
missing acquisitions or extra native calls. The automatic saved consumer ran
once. No training, backward, optimizer, warmup, replay or export occurred.

External wall was278.244087s; producer wall274.897874s; load5.530297s and summed
acquisition262.424905s. Peak RSS was13,844,262,912B, allocated CUDA5,084,301,312B,
and measured retained run payload30,028,893B. All frozen bounds passed.
Native PID/PGID472292 and supervisor472289 are absent; descendant/group scans
and the saved GPU compute query are empty. Cleanup and source/input/release
verification succeeded. The one-shot grant is consumed.

## Evidence and identity

Execution source was `06528f900ca01c8fccd664801514ba0c3797a3d0`; implementation
was `fe3ac14c97617ed07292860339355f9fa674386f`. Neither was changed during the
run. This closure changes records only. The [unit](unit.md) retains the exact
selected states, pixel controls, receiver semantics, estimand and limits.

All artifacts below are under
`outputs/research/physical-fn-recovery/2026-10-04/visual-state-localization/`:

- `native-01/{qualification,conditions,readback,terminal}.json`, per-cell
  JSON/safetensors and both `*-invalid.json` preserve the original run.
- `native-01-summary.json` and `native-01-report.md` retain all cell outcomes,
  costs, controls and same-category overlaps.
- `native-01-saved-score-discrepancy-01.json` isolates the two raw-normalizer
  differences; `lead-score-adjudication-01.json` binds the separate median-only
  calculation and falsification evidence.
- `lead-native-acceptance-01.json` records this disposition and exact evidence
  hashes. Owner, source/process, GPU and final-handoff settlement receipts retain
  actual exits, unchanged source and released ownership.

Original readback SHA256:
`bd42ecd6dba0de361c67d6fea3cd9f35662898ebe715294a1b8f35ba2e98ee84`.
Summary SHA256:
`3a3b63f4264fc10b83accb72678f4dddc4e11f35bd6008cd6539f990bf16624c`.
Input packet SHA256:
`263433c607e0c13256d00cdab7f8c7f41a368081b2ba709ab778fb8b4c63d160`.
Release SHA256:
`6e1b51b11eab3efd15fbc79e778b7102f989c0da12786974706025ed7edcc0fa`.

These two selected states do not estimate prevalence, physical identity,
training cause, complete visual mediation or a unique layer defect. The user
owns any further research direction; no follow-on model work is scheduled.
