# Matched-history diagnostic: finite result

Lead-accepted diagnostic, not a latent-ledger identification or a training gate.
Root viewed the two image/owner panels and the original spoon image. Both cups
are distinct; the second spoon is truncated by the image edge and much smaller,
which remains a spatial/extent confound.

The same four supplied histories were run under Stable50 and N16,16 full
continuations total, with32 literal a/b row scores. Both arms retain FP32/SDPA,
the same media/prompt and3084 total assistant tokens, including supplied history.
Root reran the cold consumer read-only and obtained exact equality with all16
saved identity/terminal/score/burden results. Forced rows earn no free credit.

Let d(h)=mean-token logP(a|h)-mean-token logP(b|h), for the exact literal rows.
The following are d(H+a+n+S)-d(H+b+n+S), and the corresponding newer-position
contrast d(H+n+a+S)-d(H+n+b+S).

| Image | Model | Older-position contrast | Newer-position contrast |
|---|---|---:|---:|
|210457 cups|Stable50|-.576475|-.585068|
|210457 cups|N16|-.780482|-.781755|
|219546 spoons|Stable50|-.525586|-.592886|
|219546 spoons|N16|-.781012|-.848718|

## Observation versus interpretation

- Both models respond to the changed earlier a/b row despite identical finalS.
  The negative contrast favors a relatively more when b rather than a was
  supplied. It is inconsistent with a strictly last-S-only account on this panel.
- N16 nevertheless prefers the learned cup a over b in all four histories
  (d=.967/1.748/.887/1.669) and freely emits a in all four, even when a is supplied.
  Stable50 immediately ends in all four cup histories.
- On spoons, N16 freely emits a only in the two b-supplied histories. It emits
  neither target spoon in the two a-supplied histories. Exclusion behavior is
  therefore not uniformly absent across the two scenes.
- N16 cup continuations have2/1/3/1 valid free rows; the two a-supplied histories
  each have one strict repeat against supplied history. One also emits b twice
  with alias geometry, illustrating why0.95 alone is not physical revisit count.
- Spoon N16 has0 strict repeats versus10 total for Stable50, but one N16
  history has a geometry-invalid output. This is not a clean broad quality claim.

A leading hypothesis is an increased preference for the learned target that
can outweigh history-conditioned modulation, rather than complete failure to
respond to older rows. Identity is entangled with coordinates/spatial cursor
and literal-row likelihood is not marginalized owner probability. These short,
artificially ordered histories neither prove a semantic coverage state nor
explain every late natural drift loop. Keep the planned common-successor versus
additional observed-repeat margin contrast; do not open a KV/architecture arm.

## Evidence and technical accounting

Raw result:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/history/launch/result.json

Successful arm runtimes sum198.23seconds (.0551 allocated GPU-hours), with1970
model forwards and48 image forwards, including32 row-score forwards. Root
caught a launcher slot error before execution; the same owner corrected it.
A subsequent image-root failure loaded both models but made zero forwards;
it is preserved separately under launch-failed-image-root and is not a
scientific null. No extra image or successful scientific cell was substituted.
GPUs6–7 were released to the new confirmation baseline after completion.
