# Mixing this earlier-row order does not improve robust target recovery

**Fixed experiment complete and lead-accepted. No checkpoint promotion.**
Both 32-update arms, both cold reloads,
both exposed384 natural reads and four supplied-history reads per model finished.
The [unit](unit.md) stop is reached; all history GPUs have been released.

## Decision

The P/Q exposure mixture supplies **no additional target-recovery or tested
history-robustness capability** over P-only learning on these two cases. P-only
already recovers both trusted targets naturally and under the untrained Q
permutation. Mixture achieves the same target success, slightly more aggregate
annotation owners, but substantially greater repeat/invalid/cap burden than
P-only. Neither arm preserves Stable50's total owner count. Do not promote the
mixture as a robustness treatment from this result.

This is a bounded negative for the incremental mixture benefit, not a claim
that history diversity is generally useless. There are two trained targets,
one Q transformation per target, one fixed training schedule and an already
exposed preservation panel. No independent transfer estimate is made.

## Actual contrast and identity

The finite exposed candidate bank contained seven rows. Three had the accepted
complete c+w package; table case351017 had only two history rows and could not
support a nontrivial earlier-order swap while keeping the last row fixed.
It remained excluded rather than being padded/refreshed to meet an assumed
8–16-case count.

Selected targets are donut1083135 on417044 and stage chair1583762 on477415.
Their literal c rows, images and owner identities are identical across P/Q.
P histories contain eight/seven complete rows; Q exchanges only the first two,
holding complete row-token multiset, annotation-relative covered-owner set,
token/row count and final row fixed. Unknown/native-error rows never become
positive labels. Q is counterfactual supplied conditioning, not natural history.

Both arms start from unchanged Stable50 with fresh AdamW and the same two-rank
qualified FP32/SDPA language-DoRA route. Fixed P gets32 exposures/target; mixed
alternates P and Q for16+16, with identical total320 donut and288 chair target
tokens. Complete-row summed NLL/2, fixed ORIGINAL P+c→w KL10, unchanged
normal56 KL100 and C-style normal margin10 are common. Donut w remains a
visually admitted literal protection witness without an invented GT owner.

Shared qualification was performed once centrally, not repeated per lane.
Both full adapters pass the shared receipt verifier and independent cold
reloads with zero score deltas on all four P/Q target routes. P-only and mixed
final literal target-hit counts match: donut9/10 and chair9/9 in both P and Q.
Mixture has lower teacher NLL on all four routes, but that is not extra natural
recovery. Both fresh Stable50 P continuations reconstruct the retained natural
3084-token outputs exactly, including identical score dictionaries.

## Target recovery and supplied-history read

| Read | Stable50 | Fixed P | Mixed P/Q |
|---|---:|---:|---:|
| Natural target recovery at IoU50/60 | 0/2 | 2/2 | 2/2 |
| Supplied P and Q free target recovery at IoU50/60 | 0/4 | 4/4 | 4/4 |
| Natural TP50 across the two target images | 3 | 27 | 27 |
| Natural target-image retained/gained/lost vs Stable50 | — | 3/24/0 | 3/24/0 |
| Natural target-image strict repeats | 295 | 0 | 0 |
| Natural target-image geometry-invalid/other malformed | 331/2 | 0/0 | 0/0 |
| Natural target-image caps | 2 | 0 | 0 |

The donut target also passes IoU80 in every learned natural/P/Q read. The chair
target passes IoU50/60 but not80 in those reads; its trained c geometry itself
has IoU≈.750 to that annotation. This is not a demand for exact-row coordinates.

Every learned target natural/P/Q read reaches EOS with zero strict repeats,
invalid geometry or malformed drops. Fixed P has24/24 free donut rows under
P/Q and25/24 free chair rows; mixture has27/25 and24/24 respectively. Valid-row
count is not an extra owner count. Different valid next owners or coordinates
were allowed. Supplied history rows receive no free recovery credit.

## Natural union384: preservation and burden are not interchangeable

All reads preserve original prompt, native geometry/parser/global matching,
greedy/RP1, EOS and cap3084. Owner and FP metrics are annotation-relative.

| Metric | Stable50 | Fixed P | Mixed P/Q |
|---|---:|---:|---:|
| TP50 | 1899 | 1891 | 1894 |
| FP50 | 1522 | 1269 | 1424 |
| FN50 | 947 | 955 | 952 |
| F1@50 | .606032 | .629704 | .614536 |
| TP60 | 1787 | 1784 | 1788 |
| TP80 | 1354 | 1358 | 1355 |
| Raw row starts | 4213 | 3599 | 3975 |
| Valid parsed rows | 3421 | 3160 | 3318 |
| Strict later-row repeats | 582 | 347 | 456 |
| Geometry-invalid drops | 788 | 437 | 654 |
| Other malformed drops | 4 | 2 | 3 |
| Caps | 4 | 2 | 3 |

Mixture versus fixed P gains24 and loses21 IoU50 owners, retaining1870: net+3.
That small owner increase accompanies+155 FP,+109 repeats,+217 invalid
geometries,+218 total parser drops and one extra cap. Fixed P versus Stable50
gains55/loses63; mixture gains54/loses59. Fewer total repeats than Stable50
does not establish loss-free or broadly improved owner enumeration.

### Verified disjoint decomposition of TP50 change versus Stable50

| Disjoint population | Images | Stable50 TP50 | Fixed P change | Mixed P/Q change |
|---|---:|---:|---:|---:|
| Trained target images | 2 | 3 | +24 | +24 |
| Normal protection references | 56 | 416 | -5 | -5 |
| Remaining nonreference, nontarget images | 326 | 1480 | -27 | -24 |
| Union | 384 | 1899 | -8 | -5 |

The earlier nonreference328 subtotal of -3/0 includes the two target images;
it is **not** a third disjoint component. The326 remaining images worsen under
both treatments. Their repeats are287→347/455 and invalid geometries
456→435/652 (Stable50→fixed/mixed). On exposed dev128, TP50 is614→596/604;
this is not fresh holdout.

The changed cap set illustrates why aggregate TP alone is inadequate. Mixture
escapes fixed P's59571 cap (TP50 0→12), but creates caps on360071 (7→1) and
502725 (4→3);351017 caps in both. These are descriptive members of the unchanged
primary denominator, not exclusions or new visual labels.

## Interpretation and stop

Observed: trusted complete-row learning from P transferred to the tested Q
without Q exposures; mixture did not improve the target success count and
shifted preservation failures elsewhere. Thus this experiment does not support
the hypothesis that these repairs require explicit training on both earlier
orders. It also does not justify exact-next-owner/coordinate invariance losses.

Strongest remaining alternative: this single swap is too mild or too narrow to
represent broader history shift. The experiment did not test longer history,
different covered sets, target refresh, new prefixes or a natural-history
distribution. Those are different contrasts, not reasons to extend this fixed
run. Both learned arms use common margin protection, so this is not a margin
ablation or a replication of the prior three-target C32 trajectory.

The frozen scientific stop is complete. No automatic target/Q/dose/coefficient
search, architecture change, fresh-data training, GT edit or model promotion.

## Execution and verification

- One driver, separate full-arm roots/fresh optimizer states; fixed on GPUs2,3
  and mixed on0,1 under root's explicit parallel allocation amendment. All
  endpoints used2,3. No failed model launch, retry or intermediate selection.
- Both32-update receipts freshly verified;1090/1082 actual model forwards per
  rank in each arm,1024 backwards/rank and32 syncs/rank. Twelve total model
  loads including training, cold checks and endpoints;90,352 total model
  forwards. The780 actual endpoint continuations are4 baseline supplied reads
  plus388 for each learned arm; retained384 Stable50 natural rows were not rerun.
- Allocated rank-lifecycle time: training6756.049s, cold19.721s,
  endpoints1.902365 GPU-hours; total **3.784523 GPU-hours**. This is allocated
  lifecycle accounting, not measured device utilization; central shared
  qualification belongs to its separate receipt.
- Thirteen focused tests pass, including target-row mutation, changed-final-row,
  unequal exposure, incomplete-row rejection, no forced target credit and
  invalid/repeat denominator checks. Actual endpoint consumers reparse token-
  verified text and full owner/burden ledgers; final paired reduction reproduces
  byte-for-byte as a JSON object under `--verify`.
- One first CPU-only reducer invocation lacked workspace PYTHONPATH and failed
  before computation; corrected invocation succeeded without data/code/model
  changes. No scientific row was dropped and no model work was repeated.

## Authoritative artifacts

Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/`.

- [Paired reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/paired-reduction.json)
  owns the exact disjoint counts, per-owner identities, target reads and costs.
- [Candidate acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/candidate-acceptance.json)
  links packet, model, cold, endpoint, driver and final reduction receipts.
- [Lead-accepted closeout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/lead-accepted-closeout.json)
  records root's independent paired-reducer replay, target-ledger inspection and
  explicit scientific acceptance without promoting the candidate snapshot.
- [Driver receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/driver-receipt.json)
  preserves every phase's command, PID, log, exit and timing.
- [Exact retained baseline replay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/endpoint-Stable50/retained-baseline-replay.json).

CPU replay:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/summarize.py --verify
```
