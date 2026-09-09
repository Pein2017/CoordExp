# Five-route native A replay benchmark: final bounded result

Status: complete. Root accepts Astra-low and Astra-medium for the frozen
engineering fixture. Sol-medium, Sol-high and Sol-xhigh remain HOLD for
different reasons below. All five stop rules are reached; no further native
invocation is authorized by this record.

## Task and comparison boundary

Identical [frozen brief](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-a-owner-visual-followup/benchmark-brief-v2.md), fresh no-history workers,
disjoint directories and assigned A100s, no helpers for candidates. Four fixed
native A cells, exact tokens/media/source, two ragged mixed-image B2 batches,
independent singleton scores, coordinate-only synthetic credits, full direct
logit-gradient parity, real DoRA backward, unchanged parameters and a cold
consumer. No optimization step, training reward adoption or new generation.

The retired CPU-only brief and interrupted early Sol-medium dispatch are not
part of this matched sample. Its interrupted usage is unavailable, not zero.
The source/cwd canonical-config seam was a shared practical entry difficulty;
this task is not a pure algorithm-reasoning test.

## Results and measured worker cost

| Route | Root result | Task start to final worker completion | Worker USD estimate | Root-requested correction |
|---|---|---:|---:|---:|
| Sol-medium | HOLD: full direct-logit gradient parity unproven | 14.76 min | $1.372054 | 1 |
| Sol-high | HOLD: model-open run failed before forward | 19.24 min | $2.382810 | 1 |
| Sol-xhigh | HOLD: final guard prevented artifact persistence | 23.97 min | $2.333318 | 1 |
| Astra-low | lead-accepted | 8.06 min | $2.338574 | 0 |
| Astra-medium | lead-accepted after correction | 12.20 min | $2.735716 | 1 |

Prices are the **user-supplied estimate basis**, not an independently verified
invoice: Astra input/cached-input/output $10/$1/$50 per million; Sol $4/$0.40/$20.
Uncached input is input minus cached input. Reasoning is already in output and
is not charged twice. All observed cache-write counts are zero.

These amounts include every recorded candidate response, failed entry and
repair in the matched task. They exclude shared lead preparation/acceptance,
the accounting and review helpers, and GPU dollar charges. They therefore are
**not all-in end-to-end acceptance cost**. Wall includes candidate waits for
root correction authorization but excludes root's later acceptance work.
There was no lead takeover of candidate implementation. Common acceptance
included native evidence inspection, source-only golden reconstruction and
fresh CPU consumer replay; Sol-medium additionally required one bounded
gradient-evidence falsification review and root reproduction.

Total five-worker estimated spend is $11.1624708. Sol-medium is 41.33% cheaper
than Astra-low in candidate dollars, but has not met the same full contract.
Sol-high/xhigh cost approximately the same as Astra-low without a qualified
delivery. Among the two qualified deliveries, Astra-low is faster and 14.52%
cheaper. **Choose Astra-low provisionally for this native replay/gradient seam;
do not infer global Sol retirement from one task per route.**

## Exact qualification findings

Both Astra implementations persist seven actual forwards: four existing
singleton references, two real B2 forwards with shapes [2,1400] and [2,1401],
and one cell0 native model-autograd forward. Model-open qualification timers
are 14.764 and 15.498 seconds, respectively. Real process exits and cold CPU
consumers are zero. Maximum batch/singleton chosen-score discrepancy is
1.0347366e-4; full direct-gradient discrepancy is 2.6822090e-6, under the frozen
atol=rtol=2e-4. Root independently recomputes analytic full-vector gradients
from persisted real logits (maximum analytic/saved discrepancy 3.72529e-9),
checks the exact source sampler anchor (maximum score discrepancy 8.20160e-5),
and replays all three required corruption rejections. The named native DoRA
gradient L2 is 28.88529023; zero-credit gradients are exactly zero and parameter
byte hashes are unchanged.

Root's independent FP64 accumulation from raw FP32 scores gives singleton loss
8.712872743606567 and batch loss 8.712882488965988 for both Astra candidates and
Sol-medium. The source-only sampler anchor is 8.712883859872818. Differences
between stored scalar loss rounding and this sum are within the frozen bound.

Sol-medium also has seven real forwards, matching raw scores, correct observed
excluded-row zeros, and real nonzero/zero DoRA backward with unchanged parameter
bytes. But it retains batch row maxima/chosen gradient components and derives
only the singleton chosen component. This silently narrows the requested full
direct-logit gradient parity surface. Root and one bounded reviewer reproduce
a counterexample: two distributions with identical chosen probability and
identical gradient row maximum can differ by 0.2496835 in a nonchosen gradient,
far above the approximately 0.0002499 tolerance. No full native distributions
were saved, so this cannot be closed by a CPU-only repair. The HOLD concerns
missing qualification, not evidence that its actual native gradients are wrong.

Sol-high first corrects a mistyped bank hash, then its sole model-open attempt
incorrectly equates encoded JPEG bytes with executed RGB pixels and fails
before any forward. Sol-xhigh first corrects canonical cwd loading, then fails
an over-strict in-process zero-allocation guard before persisting its tensors
and artifact. Reaching that post-close line is control-flow evidence, not a
substitute for the absent native/cold-consumer artifact. Neither gets a retry
after the frozen actual model-open qualification is spent. Failure PIDs that
were not captured remain explicitly unavailable.

Astra-low self-corrected two pre-model CPU configuration failures. The other
four routes received one root-authorized bundled correction each. Sol-medium
also fixed its CPU consumer's file/RGB-hash confusion without a new model load.
Those repairs and waits remain in the reported totals. Only Astra-medium has
separate persisted task turns for initial versus correction: $1.940790 plus
$0.794926. Same-turn repair token splits for the other routes are unavailable;
their complete per-response receipts remain retained.

## Evidence and root replay

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior/a-owner-followup-v1/benchmark/`.

- Candidate implementations/results: this directory's `candidates/SLUG/` and
  the matching output directory. Never rerun the model commands.
- `lead-acceptance/source-golden.json`: source-only four-cell coordinates and
  cached raw action-score anchor; not a substitute for native replay.
- `lead-acceptance/{astra-low,astra-medium,sol-medium}/`: original worker CPU
  receipts preserved, fresh root CPU logs/PIDs/exits, source-score checks;
  Sol-medium also has `gradient-counterexample.json`.
- `lead-acceptance/five-route-accounting-v1.json`: exact five source rollout
  files, hashes, model/effort, unique response receipts, task boundaries and
  pricing breakdown. All five are terminal with zero integrity warnings.
- `lead-acceptance/accounting-independent-check.json`: root independently sums
  modern per-response usage, checks terminal cumulative counters, Decimal
  pricing arithmetic and exact start/completion times without the helper parser.

Root CPU-only score checks: `python benchmark/verify_scores.py SLUG` from the
unit directory, for each of astra-low, astra-medium and sol-medium. Root's
counterexample is `python benchmark/gradient_summary_counterexample.py`.
The accounting helper's two deterministic tests pass. Source A verification
also freshly reparses/matches 160 rows and 16 midpoints from unchanged bank
hashes. GPU0-4 were observed at 3/0/0/0/0 MiB after all candidate native runs;
no ongoing candidate GPU work is required.

This benchmark qualifies an engineering component, not an owner-recovery
training result. Feedback stays pending; no new A experiment is launched.
