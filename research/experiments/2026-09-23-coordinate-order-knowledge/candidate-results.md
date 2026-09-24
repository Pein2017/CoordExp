# Candidate: source coordinate-token order use is inconsistent at illegal x2/y2 histories

2026-09-23. Worker candidate, **lead acceptance pending**. The source-only
diagnostic completed the frozen 12-image/64-pair plan. It is a fixed-prefix
readout, not a natural-generation intervention or a new training result.

Source is the accepted mature promoted-live untied+axis001 step2444 at Git HEAD
`6b6883529c6d6996f1826231406dd47e96a2abcf`, HF FP32/SDPA, unchanged
vision positions and no codebook. The plan binds the original source adapter,
both selected-row payloads, base config, exact 1,000 coordinate-token IDs
151670..152669, image bytes/grids, processor-generated prompt token IDs, saved
native token traces and each causal prefix. The first real x2 replay on image632
matched the saved greedy winner and selected logprob within 3.58e-6; repeated
full-vocabulary logits matched exactly (declared tolerance 2e-4). All 70 native
queries and 512 text comparisons completed; 13 proposed teacher-row
correspondences remain HOLD because unique annotation ownership was not
defensible. No illegal native case obtained such a teacher match.

The five predeclared illegal rows are y reversal (632), x equality (885,5586),
x reversal (7281), and x equality (18380). Each earliest complete illegal box
has a reconstructible token history and no *completed* malformed/parser-drop
span before that query. None repeats an identical prior parsed row; this does
not resolve annotation or physical owner recurrence. Source caps on 885,5586,
18380 and later malformed output on 18380 are future-history outcomes, not
corruption of these earlier prefixes. Seven deterministic healthy controls are
6471,17899,18491,7574,7511,1000,16228. Image7281 has a close control in
annotation count and prefix position; 632 and 5586 lack close controls on both
dimensions, and 18380's closest-length control has fewer annotations. The
matched-panel contrast is descriptive, not a randomized history intervention.

At the original illegal coordinate queries, coordinate-family mass is near one,
but the conditional illegal-family mass varies substantially:

| Image/role | Threshold | Emitted bin | Illegal mass within coordinate family |
|---|---:|---:|---:|
| 632 y2 | 476 | 473 | 0.7557 |
| 885 x2 | 0 | 0 | 0.0232 |
| 5586 x2 | 0 | 0 | 0.0251 |
| 7281 x2 | 849 | 846 | 0.9526 |
| 18380 x2 | 563 | 563 | 0.3543 |

Thus a low *total* illegal mass can coexist with an illegal full-vocabulary
argmax (the two zero-equality cases). Merely moving the threshold changes
illegal mass even if logits do not move; every per-query curve therefore stores
the frozen-native-logit rethresholding null and reports actual redistribution
separately. For +1/+4 same-axis shifts on the five abnormal x2 queries,
redistributed illegal mass is below that null in 9/10 eligible comparisons,
and below the equal-sized wrong-axis control in 8/10. This shows some local
axis-linked sensitivity, not a reliable order gate: among the three illegal
role candidate crossings with v-1/v support, only 7281 x2 both lowers the
crossing candidate's logprob and lowers it relative to still-legal neighbors.
At 632 y2 the crossing candidate's logprob rises, and at 18380 x2 its small
drop is not selective relative to neighbors. Zero-bin equalities on 885/5586
cannot have a legal-side v-1 control. Raising x1 from 0 to 1 flips x2 to a
legal winner on both, but the matched wrong-axis change also flips 5586.
Full curves, wrong-axis effects, family mass, full-vocabulary winners and
candidate margins are retained per query; no pooled sign hides reversals.

The text-only interface assay used 64 frozen unequal pairs, both input orders,
both A/B answer placements and coordinate-token versus three-digit decimal
spellings. Raw correct-label likelihood wins are 128/256 coordinate-token cells
and 150/256 decimal cells. After averaging the four counterbalanced cells per
pair, the correct answer has a positive margin on 37/64 coordinate-token pairs
(adjacent 9/19, decimal-boundary 8/18, broad 20/27), versus 64/64 decimal
pairs. Both interfaces show answer-label bias at the raw-cell level. The decimal
control's zero-padded spelling and this one prompt are interface limitations;
failure of the special-token text assay cannot prove absent internal ordinal
knowledge.

These observations weaken a strong claim that the mature source reliably
enforces strict ordinal order at the observed illegal queries. They are
compatible with partial relational sensitivity, ordinary coordinate copying
or translation, and failure to recruit order knowledge under abnormal
generated histories. The five illegal cases, unequal control support, and
teacher correspondence HOLDs cannot distinguish those mechanisms or establish
natural rollout rescue. No physical-owner conclusion follows. The lead retains
the decision whether ordinal supervision, history robustness or a deterministic
decode constraint is the next useful intervention; this package authorizes none.

Evidence root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-coordinate-order-knowledge`.
Frozen `selection/plan-v1.json` SHA256
`9364ad3342849d93447833de502c80bd91f3f482f9d830f9f3da3a43adf2edaf`.
Saved-only `reduction-v1.json` and fresh `reduction-replay-v1.json` are
byte-identical, SHA256
`c293238ee7572c9d7a0e88f7f8cc07cf0fc46510318f7919c224932fbf8b046f`.
The source replay receipt is `qualification/native-replay-v1.json` SHA256
`7b43d09588e00796e65f791f9b32732b0833791cf7ba6b2bd9bbec9f4c18c4c5`.

Replay and checks from the named checkout:

```sh
python -B -m probes.training_set_completion.coordinate_order_knowledge.probe reduce --output /absolute/fresh/reduction.json
python -B -m pytest -q probes/training_set_completion/coordinate_order_knowledge/test_probe.py
python -B scripts/research/check_research_knowledge.py check
python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs
git diff --check
```

The single producer is terminal exit0, 885 model forwards, 600.620086 model-wall
seconds and allocated GPU-seconds (one GPU, 0.166839 GPU-hours). Artifacts are
about 36 MiB, below the 2 GiB ceiling. No optimizer updates or other model
conditions ran. The direct candidate manifest binds all 582 cell files and the
source captures; the saved plan and prior source artifacts remain unchanged.
Four focused probe tests, both-root output layout and `git diff --check` pass.
The global knowledge check currently fails on two peer-owned
`2026-09-23-recurrence-chair-history-position` state/catalog links; it reports
no error for this probe. Those peer files were not edited here.
