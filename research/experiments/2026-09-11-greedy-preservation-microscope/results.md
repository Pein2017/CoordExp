# Low average KL did not preserve greedy coordinate decisions

Status: **completed, lead-accepted, no promotion**. Root freshly replayed the
actual56-case consumer and checked counts, numerical exceptions, source masks
and resource totals. There were112 score/image forwards, no generation or
training in this unit.

## Observations

On the exact56 original Stable50 reference trajectories (6056 action tokens,
6047 protected positions), final positive32 KL has mean-of-image-means
**0.00583332**. Token-weighted mean is0.00754456, median0.00041132,
p900.0181489 and maximum0.555361. The means use different weights; they are
not interchangeable. Top1/5/10% of protected positions contain23.4/52.7/68.5%
of total KL, so the shift is concentrated but not a single-token anomaly.

There are588 source-literal argmax flips, **586 inside the protected mask**:
578 coordinate tokens and8 description/content tokens. The other2 flips are
in the explicitly unprotected invalid row of360573. No protected wrapper or
EOS token flips on the original teacher-forced paths. This local fact does
not guarantee unchanged natural stopping after earlier coordinate decisions
change the prefix.

55/56 complete natural outputs differ. In **all55**, the actual first
divergence is protected, Stable50's literal token is its replay argmax, and
A's replay argmax equals the observed new natural token at that shared prefix.
53/55 first divergences are coordinate tokens. Their KL mean/median is
0.0229880/0.0115849; source-margin median changes from+0.0433197 to-0.0902557.
Only2/55 are within the frozen epsilon1e-3 near-tie band. These are not merely
unexplained replay mismatches or numerical ties.

The15 images with owner loss contain317 flips and mean image KL0.00824196;
the41 without owner loss contain271 flips and mean0.00495211. Crucially,
40/41 no-loss images also flip. Thus a changed token is **not itself an owner
error**. Existing natural owner losses33/gains6 are an association with these
decision changes, not a causal certificate for each individual flip.

## Interpretation and next contrast

Mean token KL is an inadequate surrogate for preserving these greedy
decisions: source probability mass can remain close while coordinate ranking
crosses the argmax boundary. Later prefix changes can then matter, but the
initial changes already occur at the genuinely shared source/natural prefix.
The microscope does not prove that constraining one token would rescue an
owner, nor that all coordinate changes should be prohibited.

The next [margin-preserved positive-branch unit](../2026-09-11-margin-preserved-positive-branch/unit.md)
tests one explicit learning intervention rather than increasing KL coverage
or its coefficient blindly. It applies a one-sided worst-token floor over
all eligible original-mask positions, not a retrospective first-flip patch.
That unit owns the new formula, cost and natural-output acceptance. Stronger
preservation may merely suppress useful positive updates, so both owner
retention and the three learned repairs remain decision-bearing.

## Exceptions, identity and cost

All source-literal nonargmax, unprotected-first-divergence, unscored-divergence
and current-replay/natural-next-token mismatch counts are0. There are17 source
and10 current near-tie positions.156 raw FP32 KL values are slightly negative
(minimum-2.746e-7); they remain unclamped as roundoff evidence, not substantive
negative divergences. Full records live in `execution/exception-summary.json`.

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-greedy-preservation-microscope`.
Input `preparation/input.json` SHA256
`a69a2b650ccc08bc3e1c457db19b9e96af70ba2cee49aaf9cce2439ec3d6d68b`;
producer SHA256
`2c7b8c646a4d39a679a546155aac0858067e64538fcd5fe6b463d4c2bee5f735`;
accepted `execution/consumer.json` SHA256
`4728296deadbec32736486699858d0ff0d413121afc7355706ad22c371436466`.

All8 workers exited0;16 sequential model loads,112 model/image forwards,
0generation/backwards/collectives. Sum rank lifecycle652.969s (0.181380 GPUh),
maximum93.875s; peak CUDA allocated9,825,687,040/reserved12,769,558,528 bytes,
RSS13,996,089,344 bytes and reference cache841,517,040 bytes are within bounds.
One attempt, no retry; owned processes released. The unit is closed.
