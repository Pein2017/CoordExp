# Older cache origin prevails in the matched AF/FF readout

Lead accepted, 2026-09-24. The prospective latest-contextual-origin prediction fails; its predeclared older-origin comparator passes in both directions. [Acceptance](lead-acceptance-v1.json), [independent lead verification](supporting/lead-verification-v1.json) and [candidate](candidate-results.md) bind the exact evidence.

At original train351017/refined-03 target2, AF and FF differ only in the older written coordinate record; both have latest F and the same naturally emitted person header. This experiment replays that header and reads the x1 distribution. It does not generate a complete row. All original batch companions, images, positions, native attention and model identity remain fixed.

Two prefills supply native AF/FF caches. Swapping the latest nine post-RoPE K and V positions at all28 layers creates both cross combinations while keeping each older row and the common pre-object cache fixed. Full-vocabulary anchor TV is0.7934679513597854.

| Older / latest contextual origin | TV to latest anchor | TV to older anchor | Ratio latest / older | Frozen outcome |
|---|---:|---:|---:|---|
| AF / FF |0.5301360282|0.3423659836|0.668125 /0.431481|Older origin|
| FF / AF |0.7824285544|0.0964808024|0.986087 /0.121594|Older origin|

Both satisfy older ratio<0.5 and latest-minus-older ratio>0.1, beyond the1e-6 guard. Both fail the prospective latest-origin criterion. Independent NumPy FP64 reduction reproduces all distances from the bound raw vectors. The FF/AF hybrid nevertheless changes the greedy winner from151671 to151670: categorical coincidence is distinct from distributional proximity.

The retained interpretation is that directly available older-row K/V can carry a substantial part of this conditional history effect, beyond information already present in the latest F row. Latest context also matters: each swap causes nonzero distribution change, with an asymmetric response. The result does not isolate an attention head/current query, distinguish key versus value, measure a natural mediation fraction, establish an owner ledger or explain general repetition onset. Current-header states adapt under each cache. Prior AA nonpass and F physical HOLD remain unchanged.

Technical acceptance covers all10 saved four-row vocabulary vectors and inputs; original source/history/common header; both fresh reference matches; cached/full and sham parity; actual28-layer selected donor/complement masks/K/V, unchanged companion suffix and finally-restored cache hashes. Fresh references match saved vectors exactly; maximum cached/full errors are5.340576171875e-5 and5.7697296142578125e-5, under2e-4. Sham and companion logit differences are0. First-layer latest K/V are equal between prefills, with later contextual differences. Original post-block hidden-state differences were not confused with incoming K/V.

One terminal job:10model/4vision/0generated, no retry, parent outer82.67631235718727s; cumulative sequence0.560139886111397GPUh. Lead verification added no model/CUDA calls. The finite unit is closed. A separate older-K/V component comparison will distinguish key-initiated retrieval changes from value-initiated content effects with the latest cache held fixed; no additional call is authorized here.
