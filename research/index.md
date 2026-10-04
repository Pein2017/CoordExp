# Current research workset

The goal is unique physical-owner coverage from the original image and empty
history under ordinary greedy generation, with explicit duplication, invalidity,
unsupported-output and stopping costs. See [story](story.md) for the argument map,
[assets](assets.md) for model/data identities and [conventions](CONVENTIONS.md) for
the evidence contract. Implementation entrypoints are in [probes](../probes/README.md).

## Current decision

The [coordinate readout audit](experiments/2026-10-04-coordinate-readout-audit/results.md)
is lead-accepted and closed:16 cached score requests/3226 actions, six exact
natural-fidelity controls and zero training. All1000 coordinate rows remain
distinct at B0/B16; adjacent similarity barely changes, and a one-bin input
change can switch a neighboring output winner. B0 given the same B16 histories
also produces the selected zero-width/reversed-coordinate decisions. Raw-score
ties and normalization affect local winners, but do not identify a unique cause.
The result supports conditional score competition, not global row collapse,
instance-binding attribution or free-loop escape. No further unit is scheduled.

The [first-row history cross](experiments/2026-10-03-first-row-history-cross/lead-ruling-02.md)
is lead-accepted and closed. Four requests/292actions with exact native-history
controls show that replacing the first person's three differing coordinate
tokens with GT does not recover the intended next bottle at A0 or A16. At A16,
the correct teacher first row and freely correct bottle header still lead to
x1=0 instead of186 before any wrong generated coordinate or repeated row.
Other coordinates do change, so this is a local negative for that specific
prefix-mismatch explanation, not history invariance or an identified training
cause. No checkpoint is promoted; this historical unit remains closed.
[Results](experiments/2026-10-03-first-row-history-cross/results.md) retain all
four continuations, short-budget censoring, costs and limitations.

The [rule-stability IoU90 contrast](experiments/2026-10-03-rule-stability-iou90/lead-ruling-04.md)
is lead-accepted and complete. At the prescribed 16-update endpoint, B has
187 fewer strict-IoU>.9 duplicate events, 115 fewer invalid rows and 2 more
matched annotations than A. Both worsen repetition and lose baseline owners:
A retains/gains/loses 223/63/32, B 230/58/25. B's longest duplicate burst is
worse. This is mixed training-internal evidence, with approximate replay and
no checkpoint promotion or physical-FN claim. All jobs are settled and the
source holder is released. [State](experiments/2026-10-03-rule-stability-iou90/state.json)
and [results](experiments/2026-10-03-rule-stability-iou90/results.md) retain the
completed package and its limitations.

The pre-row detection auxiliary unit is [lead-accepted and closed](experiments/2026-10-03-pre-row-detection-aux/results.md). The frozen two-arm16 screen completed technically, but B covered225 of570 annotation owners versus233 for A and247 at the anchor. Geometry invalidity improved while valid repetition increased; both arms had the same2/8 conditional successes. Connected auxiliary gradients and falling training loss did not establish native coverage benefit. This is a bounded negative with mixed burdens, not a physical-FN or general auxiliary-method verdict. All jobs are cleaned up; no further model call, retuning, extension or next unit is scheduled.

Round11 remains closed. The accepted [known-owner visual audit](experiments/2026-10-03-known-owner-visual-audit/lead-ruling-01.md) reviewed all23 saved A_A→D_I credit transitions against ten source images:3 candidate localized additions,3 candidate removals,8 plausibly persistent entities with changed geometry,1 unchanged-candidate assignment exchange and8 unresolved identity cases. These are visual candidate judgments, not GT or a physical-FN score. Unlabeled/both-missed entities and retained-credit representation changes remain outside the panel. All round11 evidence is CPU-only and grants no new model calls.

The accepted [DoRA/input ablation](experiments/2026-10-03-dora-input-ablation/lead-ruling-02.md) shows that saved DoRA alone repairs6/6 supplied errors with input/output held anchor. Removing the input increment lowers some repetition/length costs but increases near pairs and anchor-owner losses13→24. Input-only repairs3/6 at ties and loses two baseline-legal decisions. This closes the saved block-ablation branch with mixed preservation costs; no layer/token-row subdivision is scheduled.

The accepted [endpoint ablation](experiments/2026-10-03-endpoint-block-ablation/lead-ruling-02.md) retains6/6 conditional repairs with the body increment alone and lowers natural burden relative to the full update, but loses one additional incumbent known owner (13 versus12 anchor losses). Output-only repairs3/6 at literal ties and preserves original owner sets while increasing burden. This is partial separation with a preservation tradeoff, not physical recovery.

The accepted [output-delta crossover](experiments/2026-10-03-output-delta-crossover/lead-ruling-02.md) shows that the Gmass output table increases aggregate invalidity, repeats and length on both trained bodies, with strong body-dependent interaction and opposing scene effects. All four arms remain10/10 legal on supplied contexts, so this contrast cannot identify which update block repairs the original errors. Known-owner exchanges and metric disagreement preclude a universal head preference or physical-recovery claim.

The accepted [mass-versus-ranking contrast](experiments/2026-10-03-mass-versus-ranking/lead-ruling-02.md) repairs6/6 illegal contexts under both one-update objectives. Gmass preserves more original known owners (12 losses versus22), but has more invalidity, recurrence and length, concentrated in three scenes. Neither objective improves aggregate known coverage over the anchor. Similar total parameter update norms conceal an approximately8× output-delta norm difference; this motivates a block intervention, not a causal conclusion.

The accepted [short-dose ranking contrast](experiments/2026-10-03-short-dose-ranking/lead-ruling-02.md) repairs6/6 illegal contexts after both1 and4 updates, but natural category-correct coverage falls274→267→261 with owner exchange and increasing valid recurrence. The first update already has preservation costs; updates2–4 add costs without conditional-legality benefit. These are selected conditional repair and annotation-relative coverage results, not physical recovery.

The accepted [saved-output attribution](experiments/2026-10-03-owner-transition-attribution/lead-ruling-01.md) locates46 of59 earlier losses below the unchanged IoU threshold and13 without positive same-category overlap; none is an eligible same-category row displaced by assignment. The accepted [prefix-exposure contrast](experiments/2026-10-03-prefix-exposure-ranking/lead-ruling-02.md) repairs6/6 illegal contexts in both arms while exchanging known owners; broader exposure adds no legality benefit at that dose. The [completed-row crossover](experiments/2026-10-03-completed-row-crossover/lead-ruling-02.md) and [matched coordinate branches](experiments/2026-10-03-matched-coordinate-branches/results.md) show conditional owner exchanges, not validated positive training targets. The first [greedy-prefix unit](experiments/2026-10-03-greedy-prefix-branching/results.md) retains its native-fidelity HOLD.

The [full-label self-rollout unit](experiments/2026-10-02-full-label-self-rollout-fit/unit.md) completed the original observation, LR-profile comparisons, rollout-balancing qualification, the balanced lower-LR pair, and the same-checkpoint coordinate-norm comparison on 18 images / 570 labels. Lower LR trades better baseline retention and fewer endpoint bursts for fewer new owners; median norm improves some geometry/length outcomes but increases valid repeats and exchanges covered owners. Neither establishes stable full fitting or a general improvement; norm OFF remains the default. Full-label fitting is an intermediate diagnostic toward reducing true physical false negatives on incomplete or unlabeled data.

Predecessor evidence starts with the [discussion and accepted results](experiments/2026-10-02-full-label-self-rollout-fit/results.md#current-discussion-for-the-next-lead), then [scope and computation](experiments/2026-10-02-full-label-self-rollout-fit/unit.md) and [machine state/receipts](experiments/2026-10-02-full-label-self-rollout-fit/state.json). Its released jobs are terminal and cleaned. New work receives fresh source/input identity and a concrete lead-owned unit contract; historical launch instructions do not reopen execution.

### Predecessor evidence

These are completed evidence owners, not a launch queue. Their detailed contrasts and limitations remain at the linked owners:

- [Hidden human-annotation recovery](experiments/2026-09-26-hidden-human-annotation-recovery/unit.md): fixed-bank sampling expands candidate support, but verification/admission and physical identity remain unresolved.
- [Iterative noisy-positive recovery](experiments/2026-09-27-iterative-positive-recovery/results.md): two candidate policies did not produce sustained net natural-recovery benefit.
- [Rollout row credit](experiments/2026-09-27-rollout-row-credit/unit.md) and [fresh online row credit](experiments/2026-09-27-online-row-credit/unit.md): conditional learning and quieter outputs did not reliably improve hidden-owner recovery or preserve incumbents.
- [Self-prefix known-FN bridge](experiments/2026-09-29-self-prefix-known-fn/unit.md): LOCAL/CHAIN, geometry, repetition and correction-only studies exposed objective-coverage and owner-exchange risks; the latest native qualification is technical evidence, not efficacy acceptance or paired8 authority.
- [History/repetition](questions/history-repetition-stopping.md) owns the earlier NONPASS report-quality pilot and the limits of local readout/attention interventions.

The [catalog](experiments/catalog.jsonl) retains exact source and recovery locators; prior intake recovery is at Git `108dede0154abfd90a54d18234d9e0bac780a3ba`.

## Authoritative questions

| Decision | Owner |
|---|---|
| Feasible fitting versus transferable behavior | [Capacity/readout](questions/capacity-and-readout.md) |
| Candidate support versus route value | [Discovery](questions/discovery-and-route-value.md) |
| Teacher likelihood versus native realization | [Greedy compilation](questions/greedy-compilation.md) |
| Gains, source losses and credit | [Preservation](questions/preservation-and-credit.md) |
| What counts as a physical owner | [Physical evaluation](questions/physical-evaluation.md) |
| Visual designation versus endogenous use | [Visual use](questions/visual-designation-and-causal-use.md) |
| Repetition and stopping mechanisms | [History/repetition](questions/history-repetition-stopping.md) |
| Technical validity versus scientific outcome | [Runtime/evidence](questions/runtime-and-evidence.md) |

[The existing catalog](experiments/catalog.jsonl) indexes the distilled history and current units.
It preserves exact historical Git paths, evidence labels and recorded artifact
locators, not a second live state ledger. Closed stage directories are absent
from HEAD. Recover original detail with `git show <commit>:<reading_entry>`;
recovery is not continuation qualification. [Glossary](glossary.md) disambiguates
metrics and evidence levels. External artifacts were not moved or deleted.

The [COCO/LVIS proxy review](questions/physical-evaluation.md#cocolvis-proxy-review-missing-target-is-a-different-population) distills the September 27 target-absent audit and the separate weighted/hard export boundary; it is not a new training result.
