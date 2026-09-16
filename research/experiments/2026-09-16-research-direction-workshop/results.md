# Workshop synthesis: reduce FN by learning useful complete behavior

## Decision summary

The user prioritizes physical recall on unlabeled or incompletely labeled data;
mAP follows. Keep COCO 22 as an accepted fixed-teacher baseline. Do not repeat
small-panel overfitting as new algorithmic progress, demand an exhaustively
annotated training set before learning, or launch a broad KV/binding investigation.

Preferred next question: **at fixed trusted owner information, does learning
corrections to complete continuations from the student's actual histories reduce
natural greedy FN more than canonical fixed-teacher SFT, without exchanging
owners or introducing false instances?** This is a candidate learning comparison,
not an executed or validated algorithm. Continuous refresh is a later factor.

## What was already settled

| Evidence | Conclusion and limit |
|---|---|
| [Human 13 pure CE](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-05-human13-pure-ce-replay/results.md) |140 magnitude-only updates gave 392/392 at IoU 80 and no unmatched/structural debt. Finite-panel SFT solvability is established; population recall is not. |
| [COCO 22](../2026-09-15-coco22-cumulative-expansion/results.md) |Fixed teacher, full old-data replay,376/376 at saved 128 and 256. A cumulative fitting baseline, not refresh or self-rollout trajectory evidence. |
| [Natural opportunity census](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-09-natural-candidate-opportunity/results.md) |Useful complete sampled outputs exist in a bounded bank; a union can also exceed every individual route. Support and executable routes are different quantities. |
| [Fixed-witness access](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-fixed-witness-route-access/results.md) |Complete likelihood rose for 32/43 but zero original first-fork argmax crossings. Raising route likelihood does not guarantee greedy entry; the exact witness path is not necessarily the only valid owner route. |
| [Single sampled-route imitation](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/results.md) and [multiple routes/preservation](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/results.md) |Learned selected owners/geometry can coexist with lost incumbents and lower aggregate physical coverage. mAP improvement can hide that exchange. |
| [Prefix-local/on-policy predecessor](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/results.md) |Some directional gain, but broad output growth, preservation and cap tradeoffs; not a general owner-set learning solution. New prefix learning must specify the changed credit/consequence. |
| [Later continuation](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/results.md) |First row and immediate successor can succeed while later owners diverge or repeat. Whole continuation matters. |
| [KV amplitude controls](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/instance-state/amplitude-control/results.md) |Wrong-owner magnitude-matched perturbation can also help. Effective KV interventions do not establish a uniquely owner-specific ledger or a cache implementation defect. |

These findings justify measuring support -> natural access -> complete consequence.
They do not imply that one special module must be added. At an identical image,
token history, positions, parameters and aligned execution, a cache/recompute
discrepancy would justify a cache-bug investigation. Prefix dependence or a useful
activation intervention alone does not.

## What COCO 22 can and cannot attribute

Root recomputed the join between frozen-bank provenance and the cold step 0 score:

| New-image target source | Targets | Step 0 matched | Step 0 FN |
|---|---:|---:|---:|
| Visually verified original GT |142|49|93|
| Admitted unlabeled extras |7|2|5|
| Total |149|51|98|

All 11 new routes are explicitly `synthetic_teacher=true`, serialized in fixed
geo_sorted_xy order. Exploration helped discover some objects, but training did
not preserve sampled complete transcripts.149 new tasks are not 149 newly found
unlabeled owners.93 of 98 baseline misses already had admitted original GT.

This is descriptive provenance, not attribution: adding a label can influence
other rows through shared parameters, conditioning and EOS. There is no matched
control estimating the necessity of the 7 extras, the effect of the teacher's
route construction, or self-rollout SFT. The training changes and optimizer dose
were bundled. See [the exact CPU projection](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-research-direction-workshop/supervision-partition.json).

## Separate three research questions

1. **Information:** same learner/serialization and budget, with versus without
   additional verified owners. This tests a supervision package. Removing rows
   also changes prefixes and EOS; it does not isolate a pure EOS cause. Masking
   their loss while retaining their literal rows leaks the withheld information.
2. **Learning from routes/histories:** same trusted owner bank, canonical fixed
   route versus a treatment using actual student histories and corrected complete
   consequences. No extra labels available only to the treatment.
3. **Refresh:** once a treatment has value, keep its rules/budget fixed and compare
   one frozen acquisition to recomputed acquisition after learning. New owner
   supply, selection, compute and reviewer information must be accounted for.

Do not combine these into one uncontrolled fixed-versus-refresh comparison and
then attribute its gain to whichever factor sounds most promising.

## Adviser disagreement and lead judgment

The supervision adviser prefers canonical versus actual verified sampled-route
SFT with identical owner IDs and row strings. This is a clean route-value contrast
when such complete transcripts exist. If they do not, concatenating the union
does not create an actual sampled witness. It can also collapse into another
serialization-order comparison; existing July imitation results must be addressed.

The mechanism adviser prefers fixed-bank current-prefix correction with complete
suffix and incumbent accounting, and recommends skipping broad new KV diagnostics.
Root favors this **full-consequence learning** direction, while retaining the
first adviser's same-information constraint. The actual rule for corrected
suffixes, prefix masking, valid EOS and old-owner preservation must be specified
before implementation; "refresh" is not a substitute for that definition.

## Smallest useful next execution, proposed only

- Start both arms from one checkpoint with room to improve on the chosen task,
  before its specialized complete fit. Do not start from COCO 22 final 256 and
  interpret a ceiling tie as evidence about the method.
- Acquire a bounded pool of trusted positives once, from available annotations
  and explorer proposals screened with Co-DETR crop+resize and bounded residual
  review. It need not be an exhaustive scene census. Freeze owner identities and
  give both arms exactly the same positive information.
- **A:** fixed canonical SFT on this bank; this is the established baseline.
- **B:** replace a declared share of baseline exposures with supervised correction
  of full continuations at actual failed student histories, retaining a matched
  preservation component. Begin with one frozen acquisition/correction round,
  not continuous refresh or a seed/architecture matrix.
- Match starting parameters, optimizer, CE/geometry semantics, updates and global
  image exposure; publish any unavoidable target-token/masking differences rather
  than claiming only order changed. Count candidate generation and review costs.
- Require corrected continuations to preserve verified prior owners, avoid
  duplicate/false-instance credit and terminate appropriately. An unknown row in
  the prefix must not be silently promoted to positive supervision. A union alone
  does not certify a suffix after an edited history.
- Incomplete positive banks do not certify physical completeness or the correct
  EOS position. Define whether each termination target is trusted, unknown or
  excluded before implementation; reaching the end of a reviewed subset does
  not itself justify teaching "nothing remains." Unknown objects must not become
  negative examples merely because neither the bank nor Co-DETR contains them.
  Masking EOS can remove one direct signal, but its standalone failure history
  means it is not a complete stopping policy.
- Evaluate original-image, empty-prefix natural greedy: known-reference FN,
  recovered explorer-supported owners, lost incumbents, repeats, false instances,
  malformed/cap debt and cost. Lower bbox precision can be reported separately
  from valid independent-owner discovery; do not let mAP substitute for recall.
- A small independently reviewed evaluation sample must not donate its additional
  labels to the training bank. Without an independent scene census, report
  known-reference recall, not a claim of zero total physical FN. Report unused
  images separately before making any population/generalization claim.
- Stop at the agreed fixed dose. No net-coverage/preservation advantage closes
  this treatment at that budget; do not automatically add dose, K, new labels or
  a mechanism scan. Positive incremental value earns consideration of a later
  refreshed-acquisition contrast.

This is still a proposal: cohort, numeric dose, correction mix and final acceptance
thresholds are not frozen or launched by this workshop. Lack of valid full
consequences leaves the proposed contrast unready; it must not become an
unbounded raw-proposal-review project.

The cheap alternative is a matched COCO 22 control withholding only the 7 extras,
reusing the accepted full-bank arm where identity permits. Its discriminating
baseline signal is only 5 omitted-owner FN on two images. It can answer that
narrow counterfactual, not the broad question of label completeness or explorer
learning; root does not recommend making it the new main program by default.

## Noisy granularity and reviewer corrections

The user accepts dog/chair boxes previously labeled defective, and rejects a
hand-sized region as an independent person. Thus the reviewer itself must be
calibrated to acceptable extent; conservative admission is not arbitrary tightness.
Keep entity existence, category, owner identity, geometry and provenance separate.

For dense images, an individual box may overlap neighbors. An explicitly
annotated group is a group-level target, not automatically an error; it must not
earn an invented atomic-owner count. Report individual, group, part and unknown
granularity separately. Preserve weak-visual/prior-only exclusions and historical
labels. See [shared unmatched terminology](../../../docs/eval/UNMATCHED_REVIEW.md)
and [accepted user adjudication](../2026-09-16-codetr-only-review-proxy/results.md).

## Research-flow repair and durability

The old skill already required predecessor retrieval. This incident was primarily
a failure to use that evidence to challenge direction selection, not proof that
the rule was absent. The small replacement makes its timing/output explicit:
before recommending a direction, state settled findings, unresolved gap, changed
factor and decision-changing outcomes; include successful baselines, not only
failed recipes. Check the user's ultimate objective, label replication/scale
qualification honestly, and reuse the recap within an unchanged contract.

No extra approval, full-history reading or mandatory reviewer is added. The
workshop outcome is accepted as synthesis, not as new algorithm evidence. Project
entry, physical-evaluation interpretation and the user-requested memory note
preserve the vision; the existing 22-image and historical proxy receipts stay intact.
