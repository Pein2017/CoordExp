# Fixed versus order-diverse history learning

Status: **fixed execution complete and lead-accepted; no promotion**.
Owner: `/root/astra_xhigh_history_learning`. Portfolio/root owns launch grants.

Root accepted the single shared qualification and packet SHA
`29bc9a7131829757afbc3a59862f80cb2731188be712b51db3da2856965a75b1`,
then explicitly granted independent parallel fits on2,3 and0,1. Both32-step
fits/cold checks and the complete declared endpoint reads finished. Endpoints
remained on2,3. [Results](results.md) own the outcome and completed stop; no
GPU worker remains and no continuation/promotion is implied.

## Frozen question

From unchanged Stable50, does splitting the same complete trusted target-row
exposure across two histories with identical complete-row multiset and final
row improve natural target recovery and owner preservation relative to repeating
the original history, under identical fixed protection?

Strongest alternative: training simply specializes to its exact supplied
histories, with no useful natural recovery or with displaced burden. A different
valid next owner/coordinate string is not an error. Natural original-prompt
output, not next-row token equality or teacher-forced NLL, owns the decision.

## Finite source and admission

The already exposed native-witness bank has seven candidate rows on four images.
The unchanged accepted complete c+w learning package contains three targets.
`351017-c01` has only two history rows; fixing the final row leaves no earlier
order contrast, so it is **HOLD_insufficient_history**, not padded or refreshed.
The first wave contains exactly **two** targets:

| Case | Trusted target | Existing history rows | Annotation owner |
|---|---|---:|---|
| 417044-c01 | donut `[243,388,303,440]` | 8 | 1083135 |
| 477415-c02 | stage chair `[223,424,309,600]` | 7 | 1583762 |

P is the stored original native history. Q swaps its first two complete rows,
leaving every token within each row, total token count, row count and final row
unchanged. No truncation, insertion, target refresh or alternative Q search.
Q is a **counterfactual supplied history**, not a naturally reached history.
Identical row multiset and annotation-relative covered-owner set are checked;
unknown/native-error rows remain conditioning, not new labels or certified
physical coverage. The target is annotation-backed and absent from the covered
set in both histories. This is a two-case exposed learning study, not an
8–16-case population estimate or independent transfer panel.

The fixed protection successor for 417044 is visually admitted but GT-unmatched
at IoU50. Its original literal row remains a conditional KL witness only; it
receives no newly invented annotation owner or natural recovery credit.

## Exposure and loss

Two independent arms start from identical Stable50 adapter bytes and fresh
AdamW. `fixed_P` uses P for both targets at every update. `mixed_PQ` uses P on
odd one-based updates and Q on even updates. Both take exactly 32 updates and
32 complete-row exposures per target; mixed divides those into 16 P + 16 Q.

L+ is the mean over the two selected records of **summed** complete c-row NLL,
including description, coordinate and structural tokens through box_end. No
history, future suffix or EOS token is targeted. Conditional KL is fixed at
the two ORIGINAL P+c→w states in both arms on every update, mean-per-token
within each w and then mean over two records. Q's source distribution, bad
fork and unknown suffix are not protected.

    L = L+ + 10*Kcond(P+c→w) + 100*Knormal56 + 10*Rnormal56

Normal reference bytes/masks and eligible margin floors are the unchanged
accepted C inputs. Each normal KL is a masked token mean. Each reference
margin penalty is the maximum one-sided floor violation using the current
full-vocabulary non-target competitor; average over all 56 references.
The known 360573 invalid-row mask and 17 KL-only near-tie states are retained.
Margin10 is a common fixed preservation background, not the contrast.

Shared execution: FP32/SDPA, unmerged model.eval(), frozen vision/embeddings/head,
588 language-DoRA tensors /18,006,016 scalars, AdamW lr1e-5,
betas(.9,.999), eps1e-8, weight_decay0, foreach=False, clip1.0.
Positive/conditional means replicate on both ranks; normal references shard
28 per rank with world-size compensation before DDP average. One synchronized
backward/update. No copied trainer or old accepted producer modification.

## Evidence and bounded route

Physical GPUs 2,3 are reserved. The shared owner qualifies the two-rank engine
ONCE; this lane does not duplicate a two-update smoke. The first lane execution
must show real initial score/target alignment and consumed P/Q identities,
then complete each fixed 32-update arm and independent cold reload.

Primary endpoint: unchanged exposed union384 original images/prompts under
greedy/RP1, native EOS, cap3084, native pixel geometry/parser/global matching.
Read both final arms once; bind the existing Stable50 raw baseline/decode
identity. Report target recovery, all gained/lost/retained annotation owners
at IoU50/60/80, TP/FP/FN/F1 and burden on targets, reference56, other images,
train256 and exposed dev128. No fresh transfer images are consumed.

Secondary endpoint: free continuation from each fixed P and Q for Stable50 and
both final arms. Four histories/model, original total 3084 action-token cap
minus supplied prefix length. Score free owner recovery separately from
supplied rows; retain free-to-history and free-to-free strict repeats, every
invalid/malformed row, raw row starts, EOS/caps. Same next owner or exact
coordinates are not required. Target-row teacher scores are diagnostics only.

Initial work estimate is roughly 2–4 allocated GPU-hours, not a portfolio cap.
Training has 2 replicated positives +2 replicated conditional references +28
sharded normal references per rank/update: 32 backwards and one sync. Shared
engine owns exact forward/cache/load/resource counts; endpoints retain every
generation/forward, GPU/RSS peak, elapsed time, exit and raw artifact.

Stop after the two fixed 32-update endpoints and their declared reads. No dose,
coefficient, target, Q-history or checkpoint-selection sweep. Missing gain is
a bounded negative; technical failure is not a scientific null. Any materially
changed target/history/objective returns to root before model work. No GT edit,
architecture change, checkpoint/default promotion, Git cleanup or memory write.

## Historical constraints and source paths

- `2026-07-19-same-covered-set-prefix-order-equivalence/results.md`: same set and
  final row can change the valid next owner; final set outcomes were unresolved.
- `2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/results.md`:
  old/refresh package contrast changed targets too; the actual 35 prefix-only
  cases gave interaction .00261 with CI touching zero.
- `2026-09-11-positive-progress-matched-control/results.md`: C's preservation
  advantage concentrates on reference56; matched positive NLL is not equal
  histories/distributions, and exposed dev128 is not fresh transfer.
- Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/history/`.
- Maintained lane code: `probes/parallel_owner_research/history.py`; training
  interface belongs to sibling `shared-execution/interface.md` and `training.py`.
