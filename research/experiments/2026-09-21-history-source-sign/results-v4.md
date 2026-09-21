# History-source sign: corrected worker candidate v4

Status: `candidate`; lead scientific acceptance pending. This reading supersedes only the fork-endpoint interpretation in `results.md`; every v3-bound file remains immutable. Lane B, finite-set masses, control correction, release identities, debts and cost are unchanged.

## Correction

The saved `first_candidate_fork` vectors are taken at the native row boundary. Their top two tokens are row opener `151646` and EOS `151645`; they are not an A/N owner-candidate fork. The previously reported deltas remain valid only as **row-boundary opener-versus-EOS margin changes** and provide no evidence about A/N coordinate choice.

| Boundary | Condition | Top-two token IDs | Opener−EOS margin | Cut−own-native |
|---|---|---|---:|---:|
| failure | native | `[151646, 151645]` | 8.674028 | — |
| failure | cut_A | `[151646, 151645]` | 8.160093 | -0.513935 |
| failure | cut_C | `[151646, 151645]` | 8.845682 | +0.171654 |
| control | native | `[151646, 151645]` | 8.919956 | — |
| control | cut_A | `[151646, 151645]` | 8.422977 | -0.496979 |
| control | cut_C | `[151646, 151645]` | 9.523714 | +0.603758 |

## Pairwise first-divergence diagnostics

For each frozen coherent A/N pair, the saved per-row token log probabilities are compared at the first unequal token under their exact common prefix. The margin is `logP(A_j)−logP(N_j)`. All tokens and prefix log probabilities align; the largest common-prefix difference is zero, below the frozen `2e-4` tolerance. No pair was selected from outcomes and no pooled statistic is introduced.

All 12 failure pairs diverge at x1 and their common prefixes occur on the retained native greedy first-row route. Seven corrected-control pairs diverge at a native-reachable x1 prefix. The control `A-annotation` versus `N-annotation--142` pair diverges at y1 after off-native x1 token `152455`, so it is labeled an off-native conditional comparison.

| Boundary | A candidate | N candidate | Role / native-prefix status | Native A−N | Δ cut-A | Δ cut-C |
|---|---|---|---|---:|---:|---:|
| failure | `A-native-source` | `N-annotation--127` | x1 / native_greedy_reachable | +0.820332 | -0.077024 | -0.243164 |
| failure | `A-native-source` | `N-annotation--120` | x1 / native_greedy_reachable | +1.298613 | -0.156906 | -0.281410 |
| failure | `A-native-source` | `N-annotation--142` | x1 / native_greedy_reachable | +0.748325 | -0.080252 | -0.133896 |
| failure | `A-native-source` | `N-annotation--128` | x1 / native_greedy_reachable | +0.793846 | -0.098188 | -0.104128 |
| failure | `A-actual-greedy` | `N-annotation--127` | x1 / native_greedy_reachable | +1.234556 | -0.468464 | -0.715786 |
| failure | `A-actual-greedy` | `N-annotation--120` | x1 / native_greedy_reachable | +1.712837 | -0.548346 | -0.754032 |
| failure | `A-actual-greedy` | `N-annotation--142` | x1 / native_greedy_reachable | +1.162550 | -0.471691 | -0.606518 |
| failure | `A-actual-greedy` | `N-annotation--128` | x1 / native_greedy_reachable | +1.208071 | -0.489628 | -0.576750 |
| failure | `A-annotation` | `N-annotation--127` | x1 / native_greedy_reachable | -0.277887 | +0.004116 | -0.052105 |
| failure | `A-annotation` | `N-annotation--120` | x1 / native_greedy_reachable | +0.200394 | -0.075766 | -0.090351 |
| failure | `A-annotation` | `N-annotation--142` | x1 / native_greedy_reachable | -0.349894 | +0.000889 | +0.057163 |
| failure | `A-annotation` | `N-annotation--128` | x1 / native_greedy_reachable | -0.304373 | -0.017048 | +0.086931 |
| control | `A-native-source` | `N-annotation--127` | x1 / native_greedy_reachable | +0.658918 | -0.855671 | -0.011190 |
| control | `A-native-source` | `N-annotation--120` | x1 / native_greedy_reachable | +1.112783 | -0.986143 | -0.015907 |
| control | `A-native-source` | `N-annotation--125` | x1 / native_greedy_reachable | +0.983065 | -0.801399 | -0.013628 |
| control | `A-native-source` | `N-annotation--142` | x1 / native_greedy_reachable | +0.652449 | -0.699516 | -0.005020 |
| control | `A-annotation` | `N-annotation--127` | x1 / native_greedy_reachable | +0.006470 | -0.156155 | -0.006170 |
| control | `A-annotation` | `N-annotation--120` | x1 / native_greedy_reachable | +0.460335 | -0.286627 | -0.010887 |
| control | `A-annotation` | `N-annotation--125` | x1 / native_greedy_reachable | +0.330616 | -0.101883 | -0.008608 |
| control | `A-annotation` | `N-annotation--142` | y1 / off_native_conditional | +2.363649 | +0.198236 | +0.005987 |

## Corrected interpretation

- At the failure boundary, cutting A decreases A-versus-N at every native-source and actual-greedy A x1 comparison. Cutting C also decreases every one of those margins, often by more. The annotation-extent A comparisons have mixed signs. This is not a clean target-selective A-source effect.
- In the corrected matched control, cutting A decreases all seven native-reachable A-versus-N margins. Cutting C changes those seven margins only slightly. The single y1 pair is off-native and remains a conditional diagnostic.
- Cross-boundary differences remain matched observational contrasts. The control A set remains a posthoc corrected sensitivity, and extent dependence is material. The package therefore supports local source-dependent coordinate competition but does not establish a failure-specific sign, an abstract coveredness inhibitor, complete-row preference, or physical owner reselection.
- The earlier same-sign opener/EOS comparison is excluded from this conclusion. Actual cut-row owner identities remain `HOLD`; pairwise candidate margins do not resolve them.

## Denominators, unchanged evidence and debt

- Pair diagnostics: 12 failure pairs plus 8 corrected-control pairs, each under native, cut-A and cut-C = 60 condition-level diagnostics.
- Control exclusion remains exactly `A-actual-greedy` SHA `0739bb69023add13cbffe7cecf0656bf6980c9bf4f9c811c2e7d3e5398f2a442`; no row was added or substituted.
- Lane A remains one failure trio and one matched-control trio on tied image 14038; 10/11 failure trajectories remain identity `HOLD`.
- Qualification overrun and posthoc control-candidate correction debts remain unchanged. Lane B remains 0/3 positive finite N witnesses.
- Cost remains 594 forwards, 1553.3640300408006 summed GPU-seconds, six completed free cells plus one zero-forward reserved failed slot, and 47,117,719 receipt-retained bytes.

## Stable correction evidence

- Pairwise derivation: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/first-divergence-v1.json`
- Byte-identical replay: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/first-divergence-v1-replay.json`
- Prior immutable candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/candidate-manifest-v3.json`
- Corrected candidate: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/candidate-manifest-v4.json`

```sh
PYTHONPATH=. python probes/training_set_completion/history_source_sign/derive_first_divergence.py --selfcheck
cmp /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/first-divergence-v1.json /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-history-source-sign/first-divergence-v1-replay.json
```
