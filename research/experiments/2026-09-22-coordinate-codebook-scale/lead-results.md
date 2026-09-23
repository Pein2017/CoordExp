# Accepted result: broader fitting, failed promotion eligibility

2026-09-22. **Lead-accepted evidence; package closed.** The fixed epoch32 recipe
improves training natural completion and coordinate fidelity, but fails the
prospective format/recurrence limits. No full-data promotion, additional fitting
or successor is authorized. Validation coverage/CE regression is descriptive,
not the reason for this decision. Authority remains [unit.md](unit.md).

| Training1024,9519 targets | Source | Epoch32 |
| --- | ---: | ---: |
| Class-consistent IoU50 matches | 5580 (58.62%) | 7574 (79.57%) |
| Class-consistent IoU80 matches | 3515 (36.93%) | 6450 (67.76%) |
| Clean known-positive completions | 320/1024 | 613/1024 |

This is broader-panel learning, not complete fitting or address-component
causality. Ordinary refitting remains sufficient. Validation IoU50 is
1224 to1155/2033 and clean104 to87/256. UNKNOWN remains annotation-unmatched,
not verified false. The [candidate report](candidate-results.md) preserves the
complete tables, failed attempts, runtime boundaries and replay commands.

## Prospective failure and paired severity

New failures count source-negative to epoch32-positive images. Training new
bad/cap/owner-recurrent/severe counts are75/6/47/1 against51/10/51/10; validation
counts are41/2/17/0 against12/2/12/2. Thus training bad incidence and validation
bad/owner-recurrent incidence fail. These frozen operational limits were not
retuned after outcomes. Passing severe/cap limits does not erase those failures.

An independent raw-cell check reproduced every guardrail count from2560
source/final cells, including IoU owner assignments and consecutive runs.
Existing source-bad images account for parser-drop reductions7946 to44 on
training and2635 to18 on validation. Newly bad images contribute301/95 drops
and147540/87746 dropped-span characters. Aggregate improvement coexists with
failures spreading to previously unaffected images. Raw examples include a
literal fourth-coordinate `9`, reversed y coordinates, nonconsecutive owner
revisits, and a seven-row same-annotation-owner run despite natural EOS.
These are saved parser/annotation-proxy classifications, not physical review
or an identified recurrence mechanism.

## Where learning remains incomplete

At the final training endpoint, ordinary407/407 images are clean and all965
targets match at IoU80. Middle images are92/97 clean; dense images114/520 clean,
with5973/7914 IoU50 and4857/7914 IoU80 matches. Dense images therefore account
for406/411 remaining non-clean training cases. This localizes the remaining
fit deficit descriptively; it does not distinguish visual crowding, enumeration
length, supervision difficulty, insufficient exposure or gradient interference.

On the same frozen96-image sentinel, clean counts are source21,epoch4=24,
epoch16=37,epoch32=53; IoU80 matches are396,398,512,683/1294. These are diagnostic
curves, not alternative checkpoint selection. Continued improvement between
the saved endpoints is not proof that more epochs would pass the guardrails.

The retained32 reach9/32 clean here versus32/32 in the accepted small-panel
predecessor. A fresh count of its frozen seed1729 schedule finds273 presentations
for29 images and274 for3 images (8739 total); this scale run has32 per image.
This was not a matched-exposure scaling test. Lower exposure and the expanded
co-training population remain confounded, so the retained32 result cannot alone
establish lost capacity or catastrophic forgetting. The saved comparison is
`lead-saved-comparisons-v1.json` under the output root.

The direction remains worth discussing, but this recipe has not earned an
automatic scale-up. The next decision would need to separate the dense-scene
fit deficit from output-format preservation; no new experiment is released by
this interpretation.

## Independent acceptance and closure

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale`.
Candidate `candidate-v1/manifest.json` SHA256:
`e3178f1e8862710d3bdb073b688c65b45b6e34d16c70866abedf16111c26f4f6`.
The lead checked30 artifact bindings and both current/captured identities of
10 changed paths. Source identity checks do not imply whole-repository review.

All2752 analytical cells are complete:2720 new,32 qualified source reuse,
zero missing/mutated. The lead's fresh CPU reduction is byte-identical,
SHA256 `c46f78a140f009d38aec07c938634eecd9000cde1134142836deff387696af3e`,
at `reductions/lead-final-replay-v1.json`. Fresh payload readback checks all2720
new cells against saved source590/trained903 named tensors and exact payload
bindings. It is byte-identical, SHA256
`b11edda5fe6729e87c0d74f862928e7a53173fb39eecd0813207e0484348f31c`,
at `lead-payload-readback-v1.json`. Reused source cells retain prior acceptance;
they are not new replications. The focused scale-reducer test passes on fresh
execution; prior instrumentation acceptance remains in force. No whole-repo
test-suite pass is claimed.

The lead recomputed all22 terminal producer intervals:21 exit0 and one retained
instrumentation failure;66392.10316848755 allocated GPU-seconds (18.442251GPUh),
8886.383215427399 wall seconds (2.468440h), original start1790076495.1041024.
All24 recorded producer/supervisor PIDs are absent on fresh inspection. There
were1968 logged optimizer calls and32768 image presentations; the first zero-LR
call can advance Adam state without changing parameters. Acceptance used no
model calls. Earlier numerical-resume limits and historical data exposure remain
unchanged:959/1024 training and248/256 validation identities occur in mature SFT.

The machine-readable `lead-acceptance-v1.json` records final checks and this
bounded disposition. Worker: preserve all candidate/failed bytes and keep this
package stopped. No further execution or acknowledgement-only return is needed.
