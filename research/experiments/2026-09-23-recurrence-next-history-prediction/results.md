# Frozen next-history forecast: accepted negative

Lead-accepted; [acceptance](lead-acceptance.json), [worker detail](candidate-results.md), [protocol](unit.md).

The affine native-step prediction misses both global winners and is worse than persistence on both full-vocabulary KL comparisons. Actual y1 winners are 408 and 999, versus frozen affine 413 and 0. Persistence KL is 0.001790/0.006103; affine 0.699424/0.345621; position-step 1.707692/0.733020. Position-step hits one winner despite its larger distribution error.

Lead CPU readback and independent KL recomputation passed. Derived TV between previous and next native-history conditional distributions is 0.024349/0.046103. Thus small distribution displacement can accompany a categorical flip; it does not establish saturation or cancellation. This closes the affine forecast without coefficient or horizon repair. Five model/vision calls, no generation, 10.508065 allocated GPU seconds; cumulative 0.066430621 GPU-hours of 8. No physical recovery, third physical arrival or natural-exit claim.
