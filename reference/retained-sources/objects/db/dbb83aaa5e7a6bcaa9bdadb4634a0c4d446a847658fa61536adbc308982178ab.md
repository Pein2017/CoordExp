# Readout common-component continuation

Candidate completed 2026-09-18T20:46:45.687245+00:00. Root acceptance pending.

Shared-only does not account for full normalization across this panel; centered-only also fails to reproduce it. Mixed/path-dependent numerical effects, no established physical recovery.

## Evidence and contrasts

The fixed25 original completed-row histories were each continued under original/full/shared/centered readout policies. All100 cells completed; no source history was edited. All25 fresh original controls match their saved native suffix overlap, and independent CPU reconstruction matches every saved shadow-coordinate score exactly across24,088 steps.

At the first original-versus-full divergence, all25 changes are coordinate-to-coordinate. On that SAME original head state, shared-only selects the full winner in8/25 and centered-only in4/25. Neither component produces an exactly identical full-policy continuation on any of25 boundaries. Shared and centered trajectories coincide in6/25. These are finite selected-case comparisons, not independent trials.

| Model / boundary | Policy | n | Native near-return | Alternate repeat | EOS | Invalid rows |
|---|---|---:|---:|---:|---:|---:|
| tied / failure | original | 5 | 5 | 4 | 1 | 13 |
| tied / failure | full | 5 | 3 | 4 | 1 | 3 |
| tied / failure | shared | 5 | 4 | 5 | 1 | 1 |
| tied / failure | centered | 5 | 5 | 5 | 1 | 1 |
| tied / healthy | original | 7 | 3 | 4 | 2 | 12 |
| tied / healthy | full | 7 | 0 | 4 | 3 | 2 |
| tied / healthy | shared | 7 | 2 | 4 | 3 | 1 |
| tied / healthy | centered | 7 | 1 | 5 | 3 | 2 |
| untied / failure | original | 6 | 6 | 6 | 1 | 8 |
| untied / failure | full | 6 | 4 | 5 | 3 | 1 |
| untied / failure | shared | 6 | 5 | 6 | 1 | 0 |
| untied / failure | centered | 6 | 5 | 6 | 1 | 0 |
| untied / healthy | original | 7 | 4 | 6 | 2 | 30 |
| untied / healthy | full | 7 | 1 | 4 | 4 | 3 |
| untied / healthy | shared | 7 | 3 | 4 | 3 | 17 |
| untied / healthy | centered | 7 | 3 | 6 | 2 | 19 |

`healthy` in inherited IDs means a nonrecurrent/pre-onset proxy, not a globally healthy scene. Return/alternate/EOS categories overlap. All cells have zero malformed openers and zero512-token caps; other stops are32 completed target rows.

## Decision-bearing examples

Untied417044 onset: longest near-run14 under original,1 under full (29 rows then EOS),15 under shared and15 under centered (both32-row stops). Full therefore has a distinct numerical effect here; this is not independently verified owner recovery. Untied7511 similarly changes11→3 under full with EOS, while either component yields17; full also creates one invalid row. Conversely untied632 worsens18→27 under full (shared25, centered23), and untied5586 stays at30 in every arm. Tied417044 improves17→6 under full, but shared25/centered20 worsen it. Thus neither a universal benefit nor a shared-only explanation survives.

## Technical and archival status

One false padding guard rejected existing masked prompt padding before forward on affected cells. Preserved12 completed scaleout cells and all failures; corrected guard checks exact unpadded appended target suffix;84 remaining cells executed once. Initial launch-wrapper pre-model failure preserved. No tolerance/operator/history change.

The suffix guard correction does not remove or alter native prompt padding/masks. All25 original-source replays pass after combining preserved and corrected outputs. The old producer-path hash is recovered by its exact prelaunch v1 snapshot;338 unique bindings checked with no unresolved gaps. The vacuous producer selfcheck decomposition expression is not relied upon: the independent verifier asserts the FP64 identity and all four saved-score formulas at every step.

Cost: 24,088 batch forwards, 100 vision passes, 3521.249 allocated GPU-seconds (0.978 hours), 819,402,781 reported tensor bytes including repeated shard readout payloads. All8 GPUs were used for scaleout. All producers and the assigned Luna-max child ended. No cleanup, training, additional case or successor was performed.

## Reproduction and limits

Knowledge checker: lifecycle/evidence axes valid; the sole remaining error is state result not catalogued. Catalog and broad routing are root-owned and intentionally unchanged pending acceptance.

Artifacts and exact commands: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component/ARTIFACTS.md`. Full per-boundary metrics and150 paired comparisons: `reduction.json`; all68 observed original-policy first forks with four same-state shadow decisions are included. Model/config/input/prefix identities reside in plan, cell and shard receipts; raw target tokens and every-step head/coordinate tensors are retained.

- Target outputs only; companion outputs uninterpretable. Original prompt masks/padding retained; appended target history has no padding.
- Repeated boxes are numerical proxies, not verified physical duplicate owners; EOS or alternate recurrence is not recovery.
- 11 onset and14 pre-onset/nonrecurrent proxies on7 selected images are correlated, not prevalence samples.
- Analytic centering convention, not a native module; interaction and generic decision-boundary/path effects remain.
- Mature tied versus untied+axis is a model-package comparison, not untie-only causality.
- Later histories differ; controlled shadow scores refer to one identical head state. No training-origin or deployment claim.
