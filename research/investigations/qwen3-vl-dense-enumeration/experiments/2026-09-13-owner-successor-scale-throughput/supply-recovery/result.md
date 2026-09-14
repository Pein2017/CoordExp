# Exact unfinished-call recovery and finite supply closeout

Status: acquisition technically complete; all physical admission and training
decisions remain root-owned. No singleton trust, successor trust, or negative
label is inferred from geometric matches or the machine local-w flag.

## Why recovery was necessary

The original density-round-robin pool order aliased modulo4 execution shards:
one GPU received almost exclusively dense10–19 images and another20+ images.
The frozen scientific population/order was valid; execution assignment was
imbalanced. Root authorized recovery after observing sealed sparse/medium
shards and two live dense stragglers.

Only verified owned workers1082723 and1082724 were interrupted with SIGINT.
Their failed terminal receipts preserve elapsed time/counters and
`KeyboardInterrupt`; successful shards2/3 were not modified. Snapshot cold
validation recovered3373 complete image records and339 conditional outcomes,
plus the two earlier slice outcomes. Remaining work was723 natural calls and
two known conditional calls, with future nominations still governed by the
same fixed rules. No partial row was accepted as success.

The counted recovery slice executed only `natural:538126` and
`368397:h1:c0`, with2 image forwards/402 model forwards and successful cold
consumption. The full continuation used explicit per-rank image assignments,
balancing density-object loads2110/2110/2110/2100/2100/2100/2100/2100.
This scheduling projection never changed pool order, nominations or admission
priority. Original producer and recovery producer byte snapshots are preserved.

## Final exact union

Output recovery root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/supply-recovery/`.

- `completion.json`:4096 frozen train images;3766 unique new natural calls
  plus330 reused N16 anchor rows;589 unique conditional outcomes; no pending
  natural or conditional calls and no completed-call rerun.
- `ordered-jobs.json`, `ordered-results.json`, `image-records.json` preserve
  canonical frozen-pool/history/candidate order and complete literal records
  for the separate physical-decision join.
- Machine immediate-w statuses:502 candidate,68 no valid free row,
  16 strict-duplicate first rows,3 nonliteral first free content. These counts
  are **not physically accepted packages**.
-589 nominations cover198 images and301 exact repeat histories.
- All8 balanced-continuation worker exits and cold rank consumers passed.
  Rank consumer directories explicitly mark read-only source projections;
  their symlinks do not represent additional GPU runs.

## Artifacts and costs

`verification-and-cost.json` binds fresh file/card/denominator checks and all
terminal receipts. Total counted cost, including interrupted attempts:
18 model loads,483065 model forwards,482909 committed generated tokens and
4357 image forwards.4355 unique completed calls plus2 interrupted incomplete
attempts explain the image-forward difference;156 forwards have no committed
token output. Summed worker time is10.513 GPU-hours. Maximum balanced-remainder
worker time is1954.38 seconds; no clean matched speedup claim is made.

The completion receipt references250 original/recovery slice and new
continuation cards. `preserved-cards-manifest.json` adds262 durable original
shard0/1 cards.512 PNGs are owned and freshly verified here. The remaining77
original sealed shard2/3 outcomes stay in the separate physical-admission
owner's projection; those sources were left immutable. Missing w, group c and
uncertain first-owner interpretations remain preserved rather than replaced.

No additional inference, pool growth, trust relaxation, fitting or promotion
is authorized or performed by this closeout.
