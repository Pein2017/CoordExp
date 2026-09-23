# Lead-verified nominal fit milestone

2026-09-22. **The nominal seed-1729 fit passes the frozen finite-panel feasibility
target.** This milestone is lead-verified; the full package remains open for the
authorized monitor and seed-2718 repeat. Keep the selected nominal recipe, with
no conditional LR fits. [unit.md](unit.md) and the existing runtime rulings retain
authority; no larger-data training or new ablation is authorized here.

| Condition | Clean images / 32 | Class-consistent IoU50 matches / 679 | Class-consistent IoU80 matches / 679 |
| --- | ---: | ---: | ---: |
| Matched live source | 4 | 264 | 158 |
| Nominal step 32 | 5 | 341 | 220 |
| Nominal step 128 | 21 | 584 | 526 |
| Nominal step 512 | 32 | 679 | 679 |

Final native output has 679 predictions, zero exact or annotation-owner revisits,
zero invalid geometry/parser drops/UNKNOWN, no cap, and natural EOS on all 32
images. These are known-annotation matches, not new exhaustive physical-owner
adjudication. The clean predicate itself permits annotation-UNKNOWN predictions;
this final candidate separately has none. Early step-32 exact revisits increased
from 426 to 491 before falling to 13 and then zero; early behavior was not used
as a reason to abandon this overfit-capable schedule.

The lead verified selection and training bindings (5 and 33 respectively), then
reduced all 128 raw saved cells against the four frozen 32-image queues. The
fresh saved-only replay is byte-identical to the worker reduction. Prompts agree
across the four conditions for each image. The maintained evaluator generates
from empty extensions before computing teacher metrics.

A bounded independent check found all 32 final identities exactly match the
admitted fit panel (13 human, 5 refined, 7 ordinary, 7 dense same-class), reparsed
all final outputs, and independently checked one-to-one assignments, IoUs,
descriptions, geometry and repetition. Missing/duplicate/malformed/cap mutations
reject clean eligibility. Focused reducer tests: 7 passed. No lead or reviewer
model calls were made.

Stable evidence below
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-alignment/`:

- `production/fit-selection-v1.json`: SHA256 `71ef63e350ceeea7f2bdf4b7538c94d93effb8986df46ce1653908ee7aac42dd`.
- `production/reductions/nominal-complete-v1.json` and `production/reductions/lead-nominal-replay-v1.json`: identical SHA256 `8fd34aa81bfe77c490a012a8edff626c78647204529243faed32c04531b1d70e`.
- `production/nominal-training-complete-v1.json`: SHA256 `5e89258645d80bb654261dec541e5c6e5229527f04c6a42bb608c057664c4325`; 512 finite/applied updates from the declared fresh source.

Decision: this recipe is trainable enough to retain for the planned next
decision. The evidence establishes same-panel natural-completion feasibility,
not transfer, superiority to ordinary SFT, causal benefit of the address
component, or the mechanism of duplication bursts. Multiple trainable surfaces
changed together. The source-versus-trained contrast cannot isolate them.

Worker: continue the already-authorized 64-image source/selected monitor and
feasible seed-2718 repeat under the original clock and envelope. Keep seed-1729
and its selection fixed; do not use monitor results to replace settings or
discard this fit result. No extra approval checkpoint or acknowledgement-only
reply is needed. Return the stable full-package candidate and all costs for
final acceptance. The reported milestone cost is 1.9503 wall-hours and 3.3644
allocated GPU-hours; this is a worker snapshot, not a fresh live-cost audit.
