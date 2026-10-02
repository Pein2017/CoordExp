# Current research workset

The goal is unique physical-owner coverage from the original image and empty
history under ordinary greedy generation, with explicit duplication, invalidity,
unsupported-output and stopping costs. See [story](story.md) for the argument map,
[assets](assets.md) for model/data identities and [conventions](CONVENTIONS.md) for
the evidence contract. Implementation entrypoints are in [probes](../probes/README.md).

## Current decision

Research is **active under autonomous lead iteration**, authorized by the user on 2026-10-03 (Asia/Saigon), with no cumulative time or compute ceiling. Each round uses a new GPT-6.1-Sol/high execution worker; the lead chooses the question, accepts evidence and decides the next unit. The [short-dose ranking unit](experiments/2026-10-03-short-dose-ranking/unit.md) now prepares a fixed1/4-update comparison to separate early repair from added-dose preservation costs. The accepted [saved-output attribution](experiments/2026-10-03-owner-transition-attribution/lead-ruling-01.md) locates46 of59 earlier losses below the unchanged IoU threshold and13 without positive same-category overlap; none is an eligible same-category row displaced by assignment. The accepted [prefix-exposure ranking result](experiments/2026-10-03-prefix-exposure-ranking/lead-ruling-02.md) repairs6/6 illegal contexts in both arms, including held-out neighbors, but exchanges31/29 and24/30 known owners in natural generation; broader exposure shows no additional legality benefit at this dose. The accepted [completed-row crossover](experiments/2026-10-03-completed-row-crossover/lead-ruling-02.md) shows that72 is unnecessary for one fresh conditional owner exchange, but B/H gain3 known owners and lose2; neither is a validated positive training target. The earlier [matched coordinate branches](experiments/2026-10-03-matched-coordinate-branches/results.md) found a different3-gain/1-loss conditional exchange; historical suffixes changed at the new prefill boundary. The first [greedy-prefix unit](experiments/2026-10-03-greedy-prefix-branching/results.md) retains its native-fidelity HOLD. These findings do not establish physical or empty-history recovery. This authority supersedes the earlier global pause, not historical consumed run grants.

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
