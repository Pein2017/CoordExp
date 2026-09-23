# Lead reading: monitor regression and failure redistribution

Subsequent user clarification, 2026-09-22: validation regression is foreseeable;
the next stage remains primarily about training-set learning, with bounded
format/duplication degradation. The observations below remain unchanged, but
they do not impose a validation-coverage improvement gate. The updated
[preparation brief](lead-next-stage-preparation.md) owns this prospective
decision priority. Numerical format/duplication tolerances remain to be proposed
and frozen before the next launch; this reading does not invent them.

2026-09-22. **Monitor scientific metrics are independently verified; transfer
benefit is not demonstrated.** The [accepted nominal fit milestone](lead-nominal-fit-milestone.md)
remains valid and selected seed1729/step512 remains fixed. Complete the existing
seed repeat and CPU-only [1024/256 preparation](lead-next-stage-preparation.md).
No additional model execution or settings search is requested by this reading.

The lead verified monitor-summary-v1.json SHA256
`b291fae1ad8bacbc9c488db8ae4bbb875435e7e88523b000f16f9f5acd6edad8`,
monitor-v2.json SHA256
`22b108f47515cc5b6320b76a10085749cc4325c806b8f36e40e8b54167d6a490`,
and both summary bindings. A saved-only replay of exactly the frozen 128 monitor
cells matches every scientific field, with 64 images, no missing/duplicate
planned cells and no fit rows in the analysis. The original reduction scanned
320 cells; the restricted replay scanned 128. This is the sole JSON difference
(`observed.cells_loaded`), so this replay is not described as byte-identical.
The retained replay is `production/reductions/lead-monitor-replay-v1.json` under
the existing coordinate-codebook-alignment output root.

| Metric | Matched source | Selected model |
| --- | ---: | ---: |
| Class-consistent IoU50 matches / 630 | 363 | 328 |
| Clean known-positive images / 64 | 23 | 21 |
| Teacher token-weighted CE | 1.68460 | 2.93591 |
| Natural EOS / cap images | 62 / 2 | 62 / 2 |
| Parsed predictions | 752 | 793 |
| Generated tokens | 11879 | 14574 |
| Invalid geometry | 542 | 10 |
| Malformed-object drops | 2 | 257 |
| Total parser drops | 544 | 267 |
| Images with any parser drop | 4 | 25 |
| Exact-row revisits | 115 | 8 |
| Annotation-owner revisits | 23 | 21 |
| UNKNOWN predictions | 386 | 460 |

Independent paired/raw-cell inspection finds 12 improved, 19 worse and 33 equal
images by class-consistent IoU50 match count. Dense images contribute net -35
(295 to 260); ordinary images have offsetting gains/losses (68 to 68). This does
not mean every individual loss is dense. The largest losses are image458325
(22 to 7), image293794 (14 to 7), and image31322/image37777 (each loses 4).

Aggregate error reductions are dominated by source cap failures image65798 and
image143572: their combined invalid geometry falls 540 to 2 and exact revisits
109 to 0. Across the other 62 images, those counts rise 2 to 8 and 6 to 8.
Image65798 leaves the cap, image143572 stays capped with 177 malformed drops,
and image458325 newly caps with a single 35352-character malformed span.
A parser-drop count is not a fixed-size measure of malformed output burden.
UNKNOWN remains annotation-unmatched, not verified physically false.

Interpretation: the data support a changed failure distribution, not a broad
claim that duplicate suppression or geometric quality transferred. The leading
working explanation is specialization/forgetting from repeated small-panel
updates across the trainable surfaces; this remains a hypothesis, not a proven
cause or a causal indictment of the address component. The current contrast
cannot separate those mechanisms.

Worker: retain these paired distinctions in the final package. In the 1024/256
proposal, require coverage and localization alongside repetition, and include
image incidence/type of malformed output and generated length/EOS/cap. Do not
promote on lower aggregate invalid/repeat counts alone. Preserve ordinary/dense
strata and use the fixed exposure checkpoints to separate insufficient fitting
from persistent validation regression. No architecture or LR arm is added.

For final replay, bind the exact scanned input set or explicitly document the
planned-cell-only replay scope and its telemetry difference. Preserve original
reductions and the unrelated JSON-list scan failure; no model rerun is needed.
Finish the existing package and return the stable candidate. No reply solely
to acknowledge this note is needed, and no successor launch is released.
