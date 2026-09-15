# COCO227: both CE reductions reach sustained clean completion at saved step16

## Decision

**Lead-accepted bounded result.** From the same accepted A-final256 adapter,
with the same227-owner teacher and fresh AdamW, both sample-equal CE (S) and
global-active-token-equal CE (T) first reach clean227-owner completion at saved
step16 and remain clean at32/64/128/256. Both final256 outputs have FN0, F1=1,
old218 retained218/218 and new9 acquired9/9. All227 owners also match at IoU>=.8.

This round establishes successful acquisition of the nine admitted additions
without incumbent-owner loss. It finds no normalization advantage on the
predeclared sustained-clean milestone or final quality. Retain sample-equal CE
as the operational default; this tie does not establish universal equivalence.
Step8 exposes a tradeoff, not a winner: S covers one additional new owner, while
T has fewer parser-dropped and physically unresolved rows.

Protocol: [unit.md](unit.md). Full evidence root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization`.

## Natural-greedy evidence

| Arm / new updates | Joint227 | New9 | Old218 retained | F1 | Parser-dropped | Physically unresolved valid rows |
|---|---:|---:|---:|---:|---:|---:|
| Common source0, reused | 218/227 | 0/9 | 218/218 | .979775 | 0 | 0 |
| S / 8 | 222/227 | 4/9 | 218/218 | .950749 | 36 | 18 |
| T / 8 | 221/227 | 3/9 | 218/218 | .967177 | 6 | 9 |
| S / 16,32,64,128,256 | 227/227 | 9/9 | 218/218 | 1 | 0 | 0 |
| T / 16,32,64,128,256 | 227/227 | 9/9 | 218/218 | 1 | 0 | 0 |

S8 has240 valid predictions and four geometry-invalid rows among its36 parser
drops; T8 has230 valid predictions and no geometry-invalid rows. Physically
unresolved rows are not confirmed false positives. All132 new requests ended
naturally at EOS. At step16 and every later saved point both arms have exactly
227 valid rows, all scoped descriptions correct, and zero malformed/geometry,
non-COCO, duplicate, physical-unknown/error or stop/cap burden.

Old218 and new9 partition the same joint227 one-to-one assignment. Final S/T
cover the same owner sets: retained227 joint owners, no gained/lost owners
between arms. Across the saved trajectory no original218 owner was lost.
The milestone is sampled: no exact completion time between updates8 and16 is
established. Later unsaved updates were not independently evaluated.

## Scope and interpretation

The source0 outputs cover none of the new9. Both final endpoints cover227/248
in the preserved current-known ledger; its21 remaining owners are19 category-
unknown owners and two known non-COCO owners. The historical232 ledger remains
218/232: these nine additions belong to the later ledger, not the historical
232 population. No owner list was retroactively redefined.

The strict227-owner COCO-80 training stage is complete. The full historical
physical-object investigation is not complete. No class guesses or new physical
reviews were performed. The evidence covers11 fitted images, one seed and the
frozen dose; seven additions are donuts on one image. It does not establish
held-out performance, population-wide equivalence, or an EOS/FN mechanism.
The two CE scalar values have different image weights and are not directly
comparable measures of training quality.

## Technical acceptance and cost

- Both arms completed256 updates,2816 logical image exposures each, with
  matching fresh initialization, geometry and optimizer recipe. Root replayed
  all512 CE/geometry formulas with zero error and confirmed identical initial
  per-image terms. All12 saved adapters and AdamW states are finite, with588
  parameter states each and correct optimizer steps.
- Training qualification consumed132 exposures across six two-update runs.
  Microbatch2/3 were faster but failed the frozen parity gate; both scientific
  arms used microbatch1 with activation checkpointing. This is not proof of a
  batching implementation bug; detailed normalized losses and raw-sum/norm
  gate failures are preserved in the preparation evidence.
- Readback batch3 passed33-request qualification with exact-token agreement
  to live serial and retained source0; measured generation time fell11.16%.
- The original scheduler failed because runtime Python lacks `os.pidfd_open`.
  Training had finished and S8 was already running. A separate portable
  recovery entry reused its11 completed rows and generated121 missing rows,
  preserving frozen sources, checkpoints and the original7200s phase deadline.
  The original failure receipt remains; recovery has its own successful terminal.
- Root checked all132 distinct request keys, prompt/media/grid/greedy settings,
 222 bound files and actual generation counts: no repeated requests. All12
  endpoint scores were freshly replayed and matched the candidate at both
  aggregate and per-image levels.
- Main launch to all readbacks:2412.49s (40.21min), including scheduler repair.
  S/T training elapsed1473.66/1329.27s; recovery controller338.65s. The original
  readback phase took913.40s, within7200s. New readbacks generated26780 tokens.
  These are wall-time and exposure measurements, not GPU-utilization claims.

## Reproduction and evidence

[Trajectory figure](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/evaluation-final-v1/acquisition-retention-parser-unknown-v2.png).

- Scoring entry and candidate summary:
  `evaluation-final-v1/score_final.py`, `summary.json`, `checkpoint-summary.csv`.
- Detailed scores: `evaluation-final-v1/scores/`.
- Root score replay and acceptance:
  `lead-admission-v1/scientific-score-acceptance.json` and `*-score-replay.json`.
- Training validation: `completed-training-validation.json`,
  `full-update-accounting.json`, `all-checkpoint-finiteness.json` under
  `lead-admission-v1/`.
- Runtime validation: `lead-admission-v1/full-readback-validation.json`,
  `runtime-cost-summary.json`; recovery `readback-recovery-v1/terminal.json`.
- The initial candidate figure's combined display count is not a scientific
  error metric. The replacement figure separates parser drops from physically
  unresolved rows; use its v2 plot receipt for display provenance.

## Stop and continuation

This bounded batch is closed. Do not extend dose/seeds or infer a CE winner.
The next decision-changing work is category adjudication of the19 unresolved
ledger owners under the explicit COCO-80 boundary, while retaining the two
non-COCO owners in historical records. This is a proposed next research stage,
not an automatic new annotation or GPU authorization. No publication or Git
commit was performed.

### Later continuation ruling (2026-09-15)

The subsequent user-approved [22-image contract](../2026-09-15-coco22-cumulative-expansion/unit.md) supersedes the proposed priority of adjudicating all19 unknown-class owners before expansion. It allows verified extra real COCO80 outputs, requires manual review and JSONL `unlabeled` writeback, removes hour caps, and conditionally authorizes Source. This changes the next stage, not the completed227-owner experiment or its frozen criteria.
