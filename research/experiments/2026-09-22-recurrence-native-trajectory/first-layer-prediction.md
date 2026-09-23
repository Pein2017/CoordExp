# Conditional first-layer prediction, before trajectory readback

Root, 2026-09-22. This is a supplementary mathematical check using the already
authorized capture, with zero additional model calls. It does not change the
frozen geometric-settling criterion or establish a whole-burst mechanism.

For a fixed coordinate query role, assume identical pre-RoPE Q and identical
per-role pre-K/V across the exact repeated rows at layer0. These equalities
are directly checkable in the native tensors; they are not assumed at deeper
layers. Static MRoPE shifts all three text axes by9 between rows.

Reindex complete previous repeated rows by relative age d=1..N. For head h
and token role j, their score is a fixed function s_hj(d), independent of N.
Thus the unnormalized mass of complete repeated rows is

    Z_h(N) = sum_(d=1..N) sum_(j=1..9) exp(s_hj(d)).

Consequently Z_h(N+1)-Z_h(N)>0 in exact arithmetic. At the next row, the newest
row and query move together; reindexing preserves the existing recent-age
terms and adds one OLDEST relative-age term. This is an identity about a
stationary first-layer template, not a claim that the physically added row is
the oldest row. Fixed current-prefix roles also retain their relative scores.

Meanwhile fixed image/prompt/pre-run anchors become9 positions older relative
to Q. Their rotary scores and combined mass need not change monotonically.
Therefore normalized repeated-row mass, head output and final coordinate logits
need not be monotone, even with a stationary first-layer repeated template.
Later-layer Q/K/V can then change through ordinary contextual processing.

Check source equalities first. With captured Q, pre-K and actual rotary phases,
compute per-head log Z over the COMPLETE repeated rows available before each
query. Retain all16 heads and every native query, including the empty N=0
boundary as log Z=-infinity. Compare actual FP32-coefficient results with an
age-reindexed reference made from the final query's scores; report errors and
any apparent decrease with its numerical scale. Do not fit frequencies or
choose favorable heads. A coefficient/score discrepancy is not automatically
a scientific rejection of the exact-arithmetic identity.

This check can demonstrate how changing context arises before any learned
row-specific first-layer K/V variation. It cannot determine the normalized
mass without the other keys, attribute the final winner, distinguish an onset
transient from a gradual higher-layer process, or predict the exit time.
