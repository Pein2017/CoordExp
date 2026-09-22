# Learning-v1 natural-output transition census

Status: complete CPU diagnostic on sealed v1; not a reward rewrite or v2 gate.

## Result

V1 has substantial discrete-greedy inertia, but its unchanged downstream owner
totals do **not** mean the natural outputs were all unchanged.

| Source comparison | Exact raw token sequences | Changed | First changed coordinate | Description | Structure/EOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| immediate train16 | 12 | 4 | 4 | 0 | 0 |
| immediate dev64 | 48 | 16 | 16 | 0 | 0 |
| downstream train16 | 9 | 7 | 7 | 0 | 0 |
| downstream dev64 | 44 | 20 | 20 | 0 | 0 |

Exact sequence means exact `generated_token_ids` list equality, independently
cross-checked against exact `raw_generated_text`. All `80` Source rows and all
`160` updated rows have both raw lineages and valid token hashes; none were
invented or marked available by inference. Across the `47` changed arm/source
comparisons, the first unequal actual schema literal is always a coordinate
token. There is no first description or structure/EOS divergence.

The immediate arm is bit-identical to Source on `60/80` trajectories (`75%`).
The downstream arm is identical on `53/80` (`66.25%`). The two updated arms are
also identical to each other on `9/16` train and `47/64` dev trajectories.
Thus the strongest alternative—real parameter change that often does not cross
a greedy token boundary—is directly supported, but it is not the whole result.

## Changed output versus unchanged owners

Every downstream trajectory retains the exact Source category-agnostic matched
owner set: `16/16` train and `64/64` dev. Nevertheless `7` train and `20` dev
trajectories change tokens and parsed predictions. Of those `27`:

- `6` train and `16` dev rows keep prediction count and description sequence
  fixed but change prediction geometry;
- the remaining `1` train and `4` dev rows change prediction count or parsing
  burden after an initial coordinate divergence;
- positionally aligned same-description predictions contain `22` changed boxes
  on train (median old/new pixel IoU `0.813`) and `89` on dev (median `0.714`).

This is real geometric movement, but not pervasive across the entire panel:
`53/80` downstream trajectories remain bit-exact. The aggregate `295 -> 295`
dev owner count conceals `20` changed trajectories and `89` aligned box changes,
while total valid predictions move `574 -> 575`.

Immediate credit moves fewer trajectories (`20/80`) but crosses owner boundaries
in three: train image532132 gains one owner, while dev images255904 and131580
lose owners `1194977` and `1656471`. The other `17` changed immediate trajectories
retain their owner sets. This restates the sealed `0 gained / 2 lost` dev result
at trajectory level rather than changing its metric.

Dropped rows remain in the accounting. Dev Source and immediate each have three
dropped predictions across two images; downstream has two across the same two
images. In image561545, downstream has `47` valid predictions versus Source's
`46` at the same `481` raw tokens because dropped predictions fall from `2` to
`1`; token length alone is therefore not prediction burden.

## Three examples

1. **Downstream train532132:** first change is coordinate literal
   `<|coord_315|> -> <|coord_322|>`. Output grows `344 -> 354` tokens and
   `35 -> 36` valid predictions, yet the exact 14-owner set is unchanged.
2. **Immediate dev255904:** first coordinate changes
   `<|coord_269|> -> <|coord_276|>` with identical length (`48` tokens) and
   prediction count (`5`), but owner `1194977` is lost. Geometry can cross an
   owner threshold without a length/structure change.
3. **Downstream dev226097:** first coordinate changes by one bin
   (`491 -> 492`); the cascade grows `476 -> 496` tokens and `48 -> 50`
   predictions while retaining the exact three-owner set. Extra unmatched
   predictions remain annotation-relative unknown, not physical false owners.

## Interpretation

Observation: downstream v1 changes natural greedy outputs on `27/80` sealed
trajectories, exclusively through coordinate-first divergence, without changing
any category-agnostic owner set. Inference: the null owner endpoint cannot be
described simply as “the update never moved natural decode.” It is a mixture of
majority discrete-output inertia and changed coordinates that usually remain
inside the same owner-assignment basin.

For future diagnosis, exact-token equality can identify insufficient natural
movement, while changed coordinate outputs with stable owners point instead to
an owner-boundary/credit-effect limitation. This distinction is descriptive;
it neither retrofits v1 nor decides whether F1 credit should help v2.

## Reproduction

```bash
cd /data/CoordExp/.worktrees/self-rollout-behavior
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/transition_census/reduce_transitions.py
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/transition_census/verify_transitions.py
```

Machine-readable artifacts:

- `transition_census/analysis.json`, SHA256
  `779c1c61a765acdb584d408667345b9555108ce71c29fb289840d6d0f940bd24`
- `transition_census/verification.json`, SHA256
  `01b667cb372fd1c1d6c0b11e21460623ce9718137a21673735d37916a22a9aba`

The fresh consumer verifies `51` sealed results/source/receipt/row-artifact
bindings, all `240` raw lineages, exact counts and divergence taxonomy, the
`27` changed-downstream/same-owner-set cases, and the three examples. Stop rule
reached: no generation, GPU use, new classifier, reward change, or v2 decision.
