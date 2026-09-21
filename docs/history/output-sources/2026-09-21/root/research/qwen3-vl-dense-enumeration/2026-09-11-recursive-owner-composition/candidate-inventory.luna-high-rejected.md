# Stable50 candidate inventory (bounded, not lead-accepted)

**Evidence identity.** Current `positive7-support50-81` native evaluation uses manifest/consumer/reduction plus training receipt. The Stable50 assignment is read from `consumer.json` `score["50"].matches` (global native matching), not old proposal inventories or reduction source baselines. Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`; candidate DoRA fingerprint: `024e46a512491b15d8715218c9fe7970707e7449f8b354b40ae37122c2c7ac9b`; unchanged Source adapter: `5a59270e1ed1bf2062590fd0eb820e534e39e8914c7a20d91d642a8f2a99452d`; Source embedding: `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`. Manifest split counts are `{'positive7': 7, 'dev128': 128, 'support50': 50, 'remaining199': 199}`; train256 is `remaining199 + support50 + positive7`, with protected positive7 excluded.

**Frozen filter.** Nonprotected train256 only; ≥2 missing GT owners; exactly 2 is the executed primary composition; GT count 3–12; `cap=0`, `strict_repeats=0`, `parser_drops=0`, `invalid_predictions=0`; complete generated length ≤256 tokens and `stop_reason=im_end`. Missing/preserved IDs and boxes are geo-sorted `(x1,y1,x2,y2,annotation_id)`. Quality ranking is deterministic: descending minimum missing-box area fraction, ascending missing-pair IoU, ascending generated length, ascending numeric image ID. This is a shortlist/triage filter, not visual owner adjudication.

**Counts.** Among 249 nonprotected train256 images: 108 have ≥2 missing owners; 97 are clean; 57 pass clean+moderate+short; 27 pass exactly2 (primary); 30 pass ≥3 (nonexecuted separate pool). Selected 24 exact2 candidates (max24): 20 remaining199 and 4 support50; support50 overlap is disclosed, not held out.

**Selected candidates (rank: image, split, GT, missing IDs, preserved count, tokens, min area).**
1. `184490` `remaining199` GT=4 miss=116077,115120 preserved=2 tokens=29 min_area=0.1349
2. `344033` `remaining199` GT=4 miss=1635210,2006537 preserved=2 tokens=19 min_area=0.08106
3. `391492` `remaining199` GT=3 miss=1055184,1057674 preserved=1 tokens=11 min_area=0.04961
4. `463272` `remaining199` GT=12 miss=1221845,1270782 preserved=10 tokens=166 min_area=0.04409
5. `323322` `support50` GT=7 miss=1956103,1608021 preserved=5 tokens=50 min_area=0.04208
6. `381996` `remaining199` GT=7 miss=1210921,1208840 preserved=5 tokens=57 min_area=0.01774
7. `307814` `remaining199` GT=8 miss=35704,1635293 preserved=6 tokens=66 min_area=0.01738
8. `200597` `remaining199` GT=4 miss=1721708,1936290 preserved=2 tokens=28 min_area=0.01313
9. `412516` `remaining199` GT=4 miss=1992576,96592 preserved=2 tokens=52 min_area=0.01174
10. `531929` `remaining199` GT=5 miss=422596,2230653 preserved=3 tokens=51 min_area=0.01134
11. `32124` `remaining199` GT=9 miss=1170611,1839359 preserved=7 tokens=87 min_area=0.01038
12. `535579` `remaining199` GT=6 miss=1162376,1443195 preserved=4 tokens=63 min_area=0.009259
13. `548288` `remaining199` GT=8 miss=694643,523660 preserved=6 tokens=75 min_area=0.005018
14. `114139` `support50` GT=6 miss=383950,1484692 preserved=4 tokens=41 min_area=0.004111
15. `264919` `remaining199` GT=7 miss=1238427,325416 preserved=5 tokens=65 min_area=0.001998
16. `355180` `remaining199` GT=9 miss=1491387,1491960 preserved=7 tokens=118 min_area=0.001447
17. `330923` `remaining199` GT=6 miss=1347921,350019 preserved=4 tokens=55 min_area=0.0009887
18. `207467` `remaining199` GT=8 miss=343103,342830 preserved=6 tokens=70 min_area=0.0008439
19. `498406` `remaining199` GT=4 miss=1859096,631144 preserved=2 tokens=41 min_area=0.000704
20. `330665` `support50` GT=7 miss=329822,325301 preserved=5 tokens=67 min_area=0.0006496
21. `354398` `remaining199` GT=5 miss=242627,1841408 preserved=3 tokens=28 min_area=0.0003844
22. `161635` `remaining199` GT=5 miss=1478651,2092288 preserved=3 tokens=48 min_area=0.000286
23. `507362` `support50` GT=11 miss=412372,412430 preserved=9 tokens=101 min_area=0.0002697
24. `254292` `remaining199` GT=12 miss=1574313,1574153 preserved=10 tokens=101 min_area=0.0002071

Machine-readable candidates preserve image paths, GT annotation IDs/categories/bboxes, Stable50 native matches/IoUs, missed/preserved IDs, burden counters and deterministic sort keys. No continuation, held-out confirmation512, visual adjudication or lead acceptance is claimed.
