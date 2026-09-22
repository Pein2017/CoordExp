# Completed Source256 CE versus refreshed RLOO

Scientific disposition: small positive terminal development IoU50 difference
for RLOO at this fixed dose, not uniform improvement across IoU thresholds or
evidence of equal-compute superiority. Technical disposition: lead-accepted
four updates per arm, all four fresh banks, persistent states and ten cold
evaluations. The registered four-round stop is reached; no model is promoted.

## Natural annotated-owner coverage

Counts are globally one-to-one category-consistent matched GT owners at
IoU50 /60 /80. Development always includes128 images /891 owners; training
includes256 images /1955 owners. No capped or imperfect output is filtered.

| Development checkpoint | Canonical CE | Refreshed RLOO |
|---|---|---|
| Original Source | 614 /585 /451 | 614 /585 /451 |
| Round1 | 619 /587 /449 | 619 /589 /450 |
| Round2 | 616 /587 /449 | 616 /587 /448 |
| Round3 | 613 /588 /452 | 616 /585 /447 |
| Round4, registered terminal | 612 /589 /452 | 618 /589 /451 |

| Terminal training panel | IoU50 /60 /80 owners |
|---|---|
| Source | 1259 /1190 /908 |
| CE | 1264 /1199 /912 |
| RLOO | 1278 /1209 /923 |

Terminal RLOO minus Source development is+4/+4/0; RLOO minus CE is+6/0/-1.
The primary IoU50 advantage over CE is0.67 percentage points, with11 owner
gains and5 losses. Versus Source it has10 gains and6 losses. CE versus Source
has5 gains and7 losses. RLOO's relative terminal advantage over CE is confined
to the4-7-owner band (+2) and16+-owner band (+4); the other two bands are zero.
The608 owners common to RLOO and Source have mean IoU change -0.000122, so
these counts do not demonstrate broad localization improvement. Terminal
training RLOO minus CE is+14/+10/+11, still a small change at four updates.

All terminal development outputs stop naturally. The CE round3 read has one
length cap; every other development read has zero. Terminal training has four
caps in each arm, as does Source. All ten new reads have zero parser and score
failures. Terminal development prediction counts are1160 CE /1137 RLOO versus
1103 Source; strict physical duplicate candidates are9 in all three. Dropped
predictions are8 /6 versus58; unmatched valid predictions are548 /519 versus489.
Unmatched means unknown under incomplete labels, not hallucination or proven FP.

## Execution and unequal costs

Both chains start at original unmerged Source step2444, update all588 A/B/m
tensors with frozen selected embeddings, and use FP32/SDPA. Each round uses
the same256 images and one persistent AdamW update at lr2.5e-6. CE averages
native GT token NLL per image including EOS. RLOO uses K4 complete actions,
per-image TP50/GT reward and leave-one-out advantage times summed action
log-probability. The reward is image-normalized; reported owner counts above
are pooled counts, not a relabeling of the training objective.

| Resource, four rounds | CE | RLOO |
|---|---|---|
| Replay forward /backward calls | 1024 /1024 | 4096 /4096 |
| Supervised /replayed action tokens | 74980 | 297272 |
| Sum reported rank0 update seconds | 211.90 | 640.22 |
| Peak per-rank CUDA reserved GB, decimal | 39.47 | 56.67 |
| Additional sampled trajectories | 0 | 4096 |

RLOO banks retain all1024 cells each, including79/92/89/85 flat groups and
342/393/373/359 zero-advantage actions. Rebuilding every bank from retained raw
rollouts exactly reproduces all prompt/media/token/EOS/reward/advantage fields.
All4096 sampled actions end naturally; actual maximum action length is569 and
maximum prompt-plus-action length1904, below the frozen3084 action cap. Each
bank uses its immediately preceding own-arm adapter and fresh registered seeds.
Mean sampled rewards are0.68731/0.69542/0.69208/0.70689; different seeds and
policies make this descriptive, not a paired improvement test.

Summed per-bank maximum sampler elapsed time is3977.42 seconds, additional
to RLOO replay; it is not measured controller wall time. The joint course runs
14:32:57-16:54:40UTC, excluding the earlier first bank, and includes shared cold
evaluation. No FLOP or equal-wall comparison is claimed. RLOO gradients are
clipped in all four rounds (raw norms2.14-2.29); CE norms0.37-0.40 are unclipped.
Identical LR and clip threshold do not imply equal effective parameter movement.

## Interpretation and stop

Observation: RLOO provides a small primary fixed-panel gain without the severe
termination debt seen in the separate EOS-zero control. Its advantage is not
monotonic and does not persist at every IoU threshold. Both four-step chains
remain close to Source; this is not the separate256-step CE training course.

Strongest alternatives are unequal replay/sampling expenditure and seed-level
variation around a tiny effect. Neither is resolved here. A new, explicitly
authorized matched-resource replicated contrast would discriminate those
alternatives; none is launched. This single training/sampling seed schedule on
a historically used development panel establishes neither repeatability,
untouched-test transfer nor an intrinsic CE limitation. No best-round selection,
extra rounds, reward change or architecture promotion follows from this result.

## Evidence and reproduction

- Authoritative aggregate: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/analysis-v1/aggregate.json
- SHA-256:366dfcf0d4643e1542db3e68c238e849b3290f8028a5d2e1ae0057b22f394f9d.
 - [CPU reducer (preserved from `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-source256-ce-rloo/reduce_results_v1.py) reuses native identity validation and
  owner comparison; run with `conda run -n ms python <reducer> --out <new-dir>`.
  It verifies all sealed runtime hashes, eight plans/receipts, eight saved
  588-state AdamW documents, all rank counters, all raw banks and12 native
  artifact sets (ten new plus two immutable Source reads). It performs no GPU work.
- Consumer sensitivity checks reject an actual-plan missing action and an
  incorrect optimizer round. Same-artifact comparison at all three thresholds
  gives identical summaries and zero owner gains/losses. Full reduction passed.
- All raw comparisons, density bands, common-owner geometry, decode diagnostics,
  exact config/adapter/artifact hashes and measured costs are in the aggregate.
