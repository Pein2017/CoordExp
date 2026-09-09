# Completed EOS-numerator control

Scientific disposition: positive on the registered dev IoU50 owner count,
with severe termination/parser debt and a later high-IoU regression.
Technical disposition: completed training, byte-bound checkpoints, all three
native natural reads, and lead-accepted reduction. No architecture promotion.

| Fixed panel / checkpoint | Source IoU50/60/80 owners | EOS-zero | Same-step full-CE | EOS-zero length caps |
|---|---|---|---|---|
| Dev128 / step64 | 614 / 585 / 451 | 629 / 594 / 456 | 609 / 578 / 444 | 35 /128 |
| Dev128 / step256 | 614 / 585 / 451 | 620 / 581 / 428 | 596 / 563 / 418 | 70 /128 |
| Train256 / step256 | 1259 / 1190 / 908 | 1427 / 1353 / 1079 | 1394 / 1317 / 1046 | 140 /256 |

Dev has891 annotated owners; train has1955. Source dev has no length caps;
Source train has4. The cap remains3084 generated tokens in every arm. No capped
row is excluded. At dev step64 the IoU50 change is40 gains/25 losses; atstep256
it is40 gains/34 losses. The registered primary therefore improves by15 and6
owners versus Source, and20 and24 versus same-step full CE. This is not merely
an increase in output count without recovered owners.

The associated cost is substantial. Dev prediction count grows from1103 to2419
at64 and3449 at256. Dropped predictions grow from58 to9875 and19635. Terminal
strict physical duplicate candidates are24 versus9 for Source. There is one
geometrically invalid prediction at each dev point, even though native parser
and score failure counters are zero; these are different diagnostics. Terminal
valid unmatched predictions are2829, still unknown rather than hallucinations.
Terminal train has7376 predictions and39328 dropped predictions. See the
aggregate for density bands, matching and common-owner geometry.

| Forward dev CE diagnostic | Optimized EOS-zero numerator | Common full-transcript CE |
|---|---|---|
| Step64 | 1.4296793891 | 1.4788048826 |
| Step256 | 1.5427957056 | 1.6215256066 |

Both diagnostics use the same logits and unchanged original segment denominator;
the latter restores EOS only in the reported full-transcript diagnostic. The
trained intervention is not transcript censoring or a changed denominator.

Inference: terminal-EOS supervision is outcome-relevant for this exact CE
recipe, but removing its numerator does not isolate better grounding from the
extra matching opportunities caused by longer output. The later-curve IoU80
loss and large debt prevent a clean quality-improvement interpretation. This
single-seed, historically used dev panel does not establish repeatability,
untouched-test generalization or complete-scene precision. The finite unit is
closed; the accepted RLOO reward and its course are not changed in response.

## Evidence identity

- Aggregate: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/analysis-eos_zero-v1/aggregate.json
- SHA-256:b7df0a7a819736d87fda1af8eb266ba91bb088e34276e48bfdac3f16a3010a5a.
- Checkpoint acceptance: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/checkpoint-acceptance-eos_zero-v1.json.
- All256 optimizer updates are finite/applied; all four checkpoints retain the
  588-key full DoRA surface and byte-identical Source selected embeddings.
  Natural reads are original unmerged composition, FP32/SDPA, world8/B2/greedy.
- The shared wave1 shell exited1 because its earlier density reducer failed.
  EOS inference and its corrected reducer completed successfully; the unrelated
  density reduction was separately repaired without rerunning inference.
