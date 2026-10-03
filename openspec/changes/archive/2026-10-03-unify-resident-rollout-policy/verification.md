# Verification

Implementation was qualified in the isolated `vllm-rollout-policy` worktree
based on `249b9b0`. It was subsequently merged into canonical `research-probes`
after A/B completion and explicit source-holder release. Historical execution
source and receipts remain unchanged; see the integration closure below.

## Original CPU candidate and its limits

- Shared arithmetic and replay: 63 checks passed (exit 0). Original policy and
  replay baselines passed before the move. Mutations that detach factors, change
  effective-addition precision, cast factors early, or add decode scalar reads
  each failed their relevant check. The moved placeholder helper and retained
  experiment diagnostics are byte-identical.
- Resident backend: 51 checks passed (exit 0), with two real-checkpoint/CUDA cases
  deliberately excluded from CPU selection. A pre-change non-coordinate action
  demonstrated the duplicate-channel bug: returned raw log probability
  -0.551445 versus required -2.169846; the corrected consumer passes.
- Trace layer: 31 checks passed (exit 0), including complete unique action
  positions, request reordering, discarded prefill, async suffix limits, PAD/EOS,
  snapshot failures, trace-off state, and exception cleanup.
- One focused Astra review checked the installed vLLM 0.29 sampler/runner seam.
  It found a blocking per-step host-to-device row-index construction. A strengthened
  existing check failed on that implementation, and device-resident slicing
  passed. Lead inspection confirmed removal of the offending construction and
  preserved row/action selection. That review checked the older runner's source;
  the native attempt below demonstrated that it did not match the selected runtime.
- Qualification entry and existing device-admission consumer: 13 checks passed
  (exit 0). The real two-rank Gloo entry with substituted model computation uses
  uneven 2/1 image ownership and matches an independent single-process global
  image-mean gradient reference. Failure records, child settlement, and distributed
  cleanup are exercised. No CPU test loads a real research model.

Raw CPU receipts are under this worktree's
`outputs/runtime-optimization/unify-resident-rollout-policy/`:
`shared-policy/result.json`, `backend/receipt.json`, and `trace/receipt.json`.
They retain actual commands, exits, original failures, and bounded repaired checks.
The entry check log is `qualification-cpu/pytest-final.log`.

## First native attempt: failed before resident generation

Commit `2eb8a44c2802852cfe241602b185dcaa91d230b9` was released once through
`native-release-01.json`. The actual command exited 1 because the active V2
sampler has no `register_forward_hook`. Both ranks saved the original exception.
Six HF requests / 320 emitted actions completed; no vLLM action, replay,
backward, optimizer update, or refresh occurred. The observed HF acquisition
critical path was 47.306 seconds. This is not a completed speed comparison.

Both resident children exited normally, both distributed groups were destroyed,
and the execution owner confirmed all observed owned processes absent. Evidence:
`qualification-native-01-failure-summary.json`,
`qualification-native-01-command-exit.json`, and the full adjacent log. Preserve
these failed-attempt costs and receipts. The V2 repair reopens trace/backend
acceptance; it requires a new source/output binding before the same failed
overall qualification is rerun.

## Repaired V2 candidate

The repaired parent boundary rejects a mismatched begin snapshot, reordered
request IDs, or missing/malformed fully qualified runner and sampler names before
generation. Nine pre-change counterexamples failed; all 32 focused backend checks
pass after repair (exits 1/0). The lead inspected the ACK consumer and confirmed
the runtime names are retained in `paired_trace` receipts. Evidence:
`backend/begin-ack-receipt.json`. This CPU evidence does not qualify the V2 trace
implementation or the native runtime.

The replacement observes V2's plain callable sampler, returning its original
output after exactly one native call. It binds live request slots at admission,
snapshots GPU action positions and `num_sampled`, discards incomplete prefill
before action validation, and restores the original sampler on every abort or
finalization. Final compact counts distinguish retained rows, emitted actions,
discarded prefill, budget overshoot, and verified async post-EOS work.

The final trace suite passes 46 checks (exit 0). Both concrete failures are
preserved: the original plain-sampler hook failure and a faithful V2 fixture
rejected by the original decode-width guard. One focused Astra pass found that
the installed runner passes total `decode_query_len` into the misleadingly named
`sampler.num_speculative_tokens`; ordinary non-speculative decoding has value 1.
The corrected guard requires 1 and retains the no-speculation, one-logit-row,
TP=1/PP=1, and no-sharding bounds. The lead checked this exact installed-source
counterexample and its repair. No other blocking mapping, cleanup, per-step
synchronization, or storage counterexample was found. Evidence:
`trace/v2-repair/receipt.json` and its original RED/GREEN logs.

The parent now preserves exact `captured_rows` from that finalizer. Its existing
consumer check failed before the field was carried and passed after repair;
the fixture distinguishes emitted counts [2,3] from retained counts [3,4].
Evidence: `backend/captured-rows-receipt.json`. Unchanged shared-policy and
qualification-entry checks were reused rather than repeated. The lead accepts
the repaired CPU candidate; native acceptance remains pending.

## Native qualification and lead acceptance

The second release bound commit `b4f9b3b82b46f093ed6e29bd5445664308af2e54`
and tree `2d46f97f2de175ee37bee7ccd15b22e36faed5f6` through
`native-release-02.json`. The same failed overall qualification ran once with
fresh output `qualification-native-02`; command exit was 0. It completed exactly
21 requests / 1,008 emitted actions, three differentiable replay/backwards /
96 targets, one logical update, and four rank refreshes. Both ranks had 590 finite
nonzero gradients with one SUM reduction and global image denominator 3. Their
updated snapshot identities match, and all three 16-action restoration checks
match token IDs and both likelihood channels exactly. Final HF bytes retain the
updated snapshot.

Receipts identify the actual V2 runner and sampler under `vllm.v1.worker.gpu`.
All 688 resident compact rows correspond to emitted actions. This native workload
reached its budgets; it observed no discarded prefill, budget overshoot, or
post-EOS suffix. Those edge branches have CPU counterexamples, not native coverage
from this workload. Non-coordinate actions exhibit distinct raw and normalized
likelihoods. Both resident children exited 0 without forced termination, both
distributed groups were destroyed, and the lead independently confirmed all seven
observed command/rank/child/helper PIDs plus the supervisor absent.

Measured costs use `max` over ranks of each rank's summed relevant parent calls.
Generation includes compact trace retrieval. Refresh includes parent copies,
serialization, GPU acknowledgement, and cache invalidation.

| Work | Emitted actions | Critical-path cost |
| --- | ---: | ---: |
| HF v0 greedy plus sampled acquisition | 320 | 21.657 s |
| Resident vLLM v0 greedy plus sampled acquisition | 320 | 2.014 s |
| Resident vLLM v1 greedy plus sampled acquisition, including updated refresh | 320 | 2.590 s |

The comparable v0 ratio is **10.75x** for this bounded two-rank workload. Resident
startup was 37.613 / 39.618 seconds per rank and is excluded from that acquisition
ratio. Updated refresh took 0.767 / 0.912 seconds. Qualification-only updated
snapshot hashing and restoration clones took another 0.173 / 0.247 seconds,
reported separately from core refresh. Do not compare the unfiltered summary's
HF total (320 actions) against its vLLM total (640 actions across v0 and v1).
The first failed attempt's six HF requests / 320 actions remain separately
accounted; its slower HF timing is not used as the comparison baseline.

Across the 96 literal replay targets, HF-versus-vLLM raw/policy log-probability
maximum absolute gaps are 0.318643 / 0.293440; image-mean absolute gaps are
0.032281 / 0.031868. These are observations without an added numerical threshold.
They do not establish exact numerical equivalence or negligible gradient bias.
Parent HF peak allocated memory was 12.53 / 11.75 GB, with parent peak RSS about
13.95 GB each; resident allocation evidence is separately retained in RPC
receipts. This does not qualify eight-rank, long-sequence capacity or throughput.

Evidence under the same output root: `qualification-native-02/receipt.json`,
its rank receipts, `qualification-native-02-command-exit.json`,
`qualification-native-02-summary.json`, the full adjacent log, and
`lead-acceptance.json`. The lead reused unchanged package evidence and checked the
final action/count, runtime, snapshot, restoration, cost-denominator, and cleanup
boundaries without additional model queries.

Standards verdict: **PASS within the declared repository-local V2, TP=1/PP=1
contract**. Research-contract verdict: **technical qualification accepted**;
scientific efficacy and exact HF/vLLM equivalence are not established. No Conda
package was edited. The original qualification preserved canonical frozen source
`249b9b0` through the complete A/B package. Subsequent integration is recorded
below, separately from that historical execution claim.

## Canonical integration and retained evidence

After A/B and its saved comparison exited0, the worker released its source hold.
The lead merged shallow cleanup `c08a7b4e052e81c8944df76337bcc887b1607a45`
and this implementation `00ae7aad1ecdbbcb8788daa259c59fdf1aa91958` without
conflicts, producing merge commits `5ba8b8cae` and `bd67430d5`. The stable spec
now includes both added requirements and all six scenarios; OpenSpec spec
validation passed before archive. Existing qualified source/native results were
not rerun or reidentified as measurements of the merged revision.

Retained current output root:
`/data/CoordExp/.worktrees/research-probes/outputs/runtime-optimization/unify-resident-rollout-policy/`.
All90files/4,679,601bytes were copied and content-verified at retirement. Original
receipt paths/hashes remain unchanged. The adjacent shallow cleanup's8files/
26,124bytes are retained at canonical `outputs/maintenance/20261004-shallow-module-cleanup/`.
The exact source/retained mapping is
`outputs/maintenance/2026-10-03-research-integration/retained-artifacts.json`;
merged consumer checks and lifecycle receipts live alongside it.

The merged consumer boundary is lead-accepted from six selected CPU cases.
The initial driver exited1 (five passes, one import-order identity failure);
the corrected affected check exited0 with one pass. The initial two optional
CUDA metadata probes were denied before initialization, not admitted GPU work.
The failure arose when the test conftest purged src modules after probe aliases
had already been imported; no maintained source repair was needed. Both original
and corrected evidence remain in `consumer-checks/consumer-candidate.json`.
No new model/native qualification was run for integration. The app archived
the vLLM worktree, the shallow worktree was removed, and the fully merged temporary
vLLM branch was deleted. Both commits remain ancestors of canonical research.
`lead-acceptance.json` and `lifecycle.json` in the integration output root own
the final consumer and retirement receipts.
