# Reciprocal x1 changes do not exchange the complete localization

Lead-accepted 2026-09-24. [Acceptance](lead-acceptance-v1.json); [independent CPU verification](supporting/lead-verification-v1.json). The six-arm unit is closed: technically valid, both reciprocal primary clauses and both exact free-suffix secondaries **NONPASS**.

| History / x1 selection | Complete person box | Frozen region |
|---|---|---|
| AF_native | [0, 0, 29, 86] | fragment_F |
| FF_native | [1, 13, 536, 999] | broad_A |
| AF_sham | [0, 0, 29, 86] | fragment_F |
| FF_sham | [1, 13, 536, 999] | broad_A |
| AF_flip | [1, 0, 47, 86] | fragment_F |
| FF_flip | [0, 13, 536, 999] | broad_A |

The treatments changed only current x1 at step4, after independently greedy headers; subsequent coordinates and terminator were greedy. AF changed x1 from0 to1 and x2 from29 to47 but remained a fragment. FF changed x1 from1 to0 and retained the broad row suffix. Identity-write shams reproduce native exactly. Neither supplied x1 receives prediction credit.

The first-coordinate choice is therefore insufficient to exchange these two complete outcomes under their retained histories. This is not proof that x1 never matters: the earlier [first-coordinate mediation](../2026-09-19-first-coordinate-mediation/results.md) remains a different, positive conditional route. No second row or burst was tested.

A retrospective CPU contrast now matches the entire current header plus x1 across histories. At x1=0, AF versus FF y1 distributions differ by TV0.447602; at x1=1, TV0.517370. Margins z(y1=13)−z(y1=0) are respectively AF−3.073736 / FF+0.023079 and AF−2.899521 / FF+0.743477. Only the older historical coordinate slots differ in these matched inputs. This strengthens the case for history-dependent computation beyond the first x1 decision, without separating direct historical reading from contextualized current-header or latest-record states. These are exploratory descriptors, not a new prospective test.

All54 full vectors/inputs,18 accepted reference steps,1512 actual attention entries, original source traces, same-base historical/companion states and exact supplied-token consumers passed independent CPU inspection. Accepted-vector and sham/state differences were0; source trace maximum error5.340576171875e-5. All six rows are complete canonical person rows with valid geometry. F/F2 and the widened fragment remain physical UNKNOWN/HOLD; broad A repeats an already reported person.

Exactly54 model/54 vision/54 logical emissions:50 greedy,2 identity writes,2 flips,0 reuse. One terminal successful job; outer147.84622029215097seconds. Sequence charge0.7676192350225765GPU-hours; this is accounting, not a time ceiling. No retry or successor model call.

Next decision: assess a finite read-access partition at the original one-record first-collapse state. Distinguish history access by current header queries from access by subsequent coordinate-token queries, with native/sham/full-block references and an F-history control. CPU geometry/source feasibility comes first; this result does not admit another GPU unit.
