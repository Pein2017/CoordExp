---
title: S K10-H20 Natural Crossover Independent Review Record
type: investigation-review
status: complete_documentation_certified
updated: 2026-08-07
---

# S K10-H20 Natural Crossover Independent Review Record

This file records the independent audits of this unit. It does not own the
scientific result; that is [results.md](results.md). It does not own launch
authority; that is [launch-gate.md](launch-gate.md).

## Review ledger

| # | Review | Scope | Disposition | P0 | P1 | P2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fable preseal fixed-tree review | preGPU-v5 fixed tree, operator neutrality, C11 keyword gate dispatch repair, finalizer admission semantics, CPU evidence | `PASS` | 0 | 0 | — |
| 2 | Fable postseal receipt-bound review | sealed preGPU-v5 receipt, receipt-bound shard preflights, exact device mapping, absent fresh roots | `PASS` | 0 | 0 | — |
| 3 | Formal scientific audit (read-only, post-execution) | raw artifact integrity, endpoint reconstruction, operator semantics, interpretation, owner documents | `PASS` on evidence integrity; **`HOLD` on unit closure** | 0 | 7 | 5 |
| 4 | Fable post-remediation re-review | documentation-only remediation of review 3 | **`PASS`** — documentation certified | 0 | 0 | 1 |

Reviews 1 and 2 authorized exactly one frozen no-training launch,
`shard-000 → GPU 0`, `shard-001 → GPU 1`, `shard-002 → GPU 7`, under the
exclusive mapping `logs-v6 → execution-v4/evidence-v4`. Their dispositions are
recorded as supplied by the reviewing lane; this file does not re-derive them.

The earlier preGPU-v1 through v4 review lineage — the Sol rejection of the v1
test binding and stale status text, the Sol acceptance of the v2 prelaunch
tree, the Opus/max confirmation of the v3 inventory-ordering cause, the Sol
`PASS` with `P0/P1/P2` all zero on the v3 tree, and the Fable P0 isolation of
the manifest-to-plan projection contract mismatch — remains recorded in
[launch-gate.md](launch-gate.md) and is not restated here.

## Review 3: formal scientific audit

Read-only, no GPU, no scientific code executed. Verdict: **`PASS` on evidence
integrity and on the `unqualified` scientific disposition; `HOLD` on unit
closure and publication.** No finding changes the reported endpoints, the
operator semantics, the identity chain, or the disposition.

### Independently recomputed and confirmed

- **Artifact identities.** All four formal raw SHA-256 values and all four
  semantic self SHA-256 values recomputed under the repo canonical form
  (`sort_keys`, `separators=(",",":")`, `ensure_ascii=True`, self-field
  elided). Also recomputed: plan self hash, all three `result.json` self
  hashes, and all nine `runtime_identity` / `terminal_summary` /
  `aggregate.receipt` self hashes plus the nine raw hashes those receipts
  record.
- **Selection rule.** Recomputed from the frozen v3 source over all 11 events,
  not copied: the intersection yields exactly
  `gt:2299:29`, `gt:13348:14`, `gt:16228:15`, in that order.
- **Source replication.** The fresh implementation reproduces all nine source
  `K01`/`K10`/`H20` endpoint vectors, `9/9`.
- **Endpoint reconstruction.** All 12 cell metric vectors, all 12
  `raw_result_sha256`, all four `component_contrasts`, and the
  `endpoint_summary` / `component_contrasts` / `utilities` receipt hashes
  reproduce from the raw rows.
- **Boundary contract.** For all 12 cells: prefix ends on
  `151649 <|box_end|>` and not on the opener; `opener_injected=false`;
  `synthetic_opener_injections=0`; `use_cache` off; `max_rows=3`;
  `max_row_tokens=256`; seeded prefix equals natural prefix plus one opener;
  `sha256_json(exact_history_token_ids)` equals the plan-bound `prefix_sha256`
  at every cell; M-RoPE identity constant across cells within each event.
- **Operator semantics.** K10 changed cells equal `image_keys − b_exclusive`
  (`874−39`, `1014−3`, `988−4`); H20 changed cells equal the latest-row key
  count (`9`, `10`, `9`); C11 changed cells equal the exact union
  (`844`, `1021`, `993`) with no future or off-scope leakage; all cells use
  float32 additive masks, so there is no representation confound between C10
  and C11. All-layer consumption is attested at runtime for every cell:
  `layer_count=28`, `missing_layers=[]`, `errors=[]`,
  `all_layers_identical=true`, expected mask hash observed at all 28 layers.
- **Hidden K00 parity.** `27/27` steps, `per_forward_max_abs_delta = 0.0`,
  tolerance `1e-4`, in all three shards.
- **Code and test bindings.** All nine parent `source_files` and five
  `test_files` hashes match live files except the two slots the finalization
  authority permits, both recorded correctly with old and new hashes. The 22
  `unchanged_parent_bindings` entries agree with the parent at hash level.
- **Bound CPU suites re-run** read-only with bytecode and cache writing
  disabled: finalizer plus successor-sealer suites `127 passed`; actuator,
  runner, materializer, and pre-GPU sealer suites `92 passed`. Repository state
  unchanged.

### P1 findings

1. **Aggregate tau was deterministically null before launch.** Plan-v1's own
   frozen source endpoint vectors already recorded unmatched rows for the
   baseline arm at two of three events, so those events could never satisfy the
   qualification predicate. The one fixed launch could not have produced a
   non-null aggregate tau. This is a selection-rule design and power failure,
   not an execution defect. Recorded in
   [results.md](results.md); no retry authorized.
2. **`duplicates` under-reports row-content repetition.** The field counts
   strict physical-owner identity repeats only, so a byte-identical unmatched
   row pair reports `duplicates = 0`. This occurs at `gt:16228:15` C11 rows 1
   and 2. Remediated by defining field semantics explicitly alongside every
   published number; artifacts unchanged.
3. **Executed-run provenance seam.** `execution-v4/*/runtime_identity.json`
   embeds the sealed CPU-only preGPU identity (`gpu_used: false`,
   `model_loaded: false`) next to `status: "completed"`. Remediated by
   documenting the seam and naming `logs-v6`, `result.json`,
   `terminal_summary.json`, and `aggregate.receipt.json` as the executed
   provenance; artifacts unchanged.
4. **Owner documents contradicted the artifacts and each other.** `unit.md`
   and `tasks.md` still described the gate as closed with absent roots and no
   launch authority after the launch had completed. This recurred a defect
   class the preGPU-v1 review had already rejected. Remediated in this change.
5. **Selection conditioning of "K10 3/3" was stated nowhere.** Remediated: the
   figure is now published as a deterministic cross-implementation replication
   of the selection condition, explicitly not efficacy and not a base rate.
6. **"History" versus "latest row" is confounded at two of three events.**
   Remediated: recency-only attribution is now explicitly narrowed to
   `gt:16228:15`.
7. **"C11 retains the target" inverts a published boolean.** `target_release`
   is a release indicator and is `true` for C11 at `gt:13348:14` and
   `gt:16228:15`. Remediated: the correct phrasing is that C11 reproduces
   K10's target release at 2/3 and is endpoint-vector-identical to C10 there.

### P2 findings

1. The K00 arm's own `full_logit_parity` carries `passed: true` with
   `status: "reference_captured"`, `candidate_step_count: 0`, and
   `per_forward_max_abs_delta: null`. The measured parity lives on C00; the
   reference arm's flag attests nothing. Documented, artifact unchanged.
2. `plan_sha256` in `result.json`, `aggregate.receipt.json`, and
   `runtime_identity.json` holds the plan's semantic self hash, not the raw
   file hash, which appears only as `input_hashes.plan_file_sha256`.
   Documented, artifact unchanged.
3. `evidence.tau` and `evidence.component_tau` are byte-identical duplicates of
   the same ten-key map. Documented, artifact unchanged.
4. The unit had no `results.md` or `review.md` and was absent from the
   experiment index; `unit.md` frontmatter read `evidence_status: pending`.
   Remediated in this change.
5. Project continuity memory was stale and named the previous unit as active.
   Remediated in this change.

### Scope limits of review 3

- The model was not re-executed; the GPU trajectory is accepted from
  `result.json` and `logs-v6`. The parity, consumption, mask, prefix, and hash
  chains are internally consistent, and the `9/9` source reproduction is strong
  indirect corroboration.
- The upstream v3 unit's own correctness was not re-audited by review 3; only
  that this unit consumed it at the sealed identities and applied the selection
  rule faithfully. Its support-completion lineage was audited separately and
  later; see the upstream reference at the end of this file.
- Images were not inspected. No claim is made about whether unmatched boxes are
  visually supported; that is precisely the open discriminator.
- IoU thresholds and the `coco_ann_id` mapping method were read but not
  re-derived from ground truth.
- `n = 3` events, `3` images, one checkpoint, one substrate, one boundary.

## Review 4: post-remediation re-review — PASS

**Disposition: `PASS`. `P0 = 0`, `P1 = 0`, `P2 = 1`.** Independent Fable
re-review of the documentation-only remediation. The unit's documentation state
is **certified**.

The remediation under review applied documentation-only fixes to review 3's
P1-4, P1-5, P1-6, P1-7, P2-4, and P2-5, and documented P1-1, P1-2, P1-3, P2-1,
P2-2, and P2-3 without artifact modification. No code, test, JSON, receipt, or
external artifact was edited, and nothing was staged, committed, or pushed.

### What review 4 rechecked

1. **Formal identities.** The four raw and four semantic self SHA-256 values
   quoted in the owner documents match the artifacts.
2. **Endpoint tables.** The three per-event tables in [results.md](results.md)
   match `evidence-v4`, including the `gt:2299:29` C11 first token `291`,
   opener rank 4 at log-probability `-6.34561`, and the byte-identical
   `gt:16228:15` C11 row pair with `duplicates = 0`.
3. **Field semantics.** The published definitions of `duplicates`,
   `target_release`, `unmatched`, `complete_rows`, `row_admission`, `STOP`, and
   `mechanically_valid` match the finalizer's actual predicates.
4. **Cross-document agreement.** `unit.md`, `launch-gate.md`, `tasks.md`, the
   experiment index, the compass, the parent 2026-08-06 owners, and project
   continuity memory agree on disposition, authority, and stop rules.
5. **No claim inflation.** No remediation text converts an unqualified endpoint
   into a qualified claim, promotes the `n=1` admission interaction to a
   checkpoint-level route, or reintroduces a training, A3, P4, sweep, or
   promotion path. `tau` remains null rather than zero throughout.

### P2 finding, corrected after review

1. **Unscoped "small boxes" in the static carrier claim.** The remediation
   described the unmatched K10-family rows as "near-identical small boxes" at
   all three events. That is accurate at `gt:13348:14` and `gt:16228:15`
   (roughly `25×25` and `21×21` pixels) but not at `gt:2299:29`, where the two
   unmatched boxes are `[168,393,246,682]` and `[170,491,253,735]`. **Corrected
   after review**: the size reading is now explicitly scoped to the two later
   events. The lead's separate correction to the strongest-alternative section
   is preserved unchanged.

No P0 or P1 survived. The claim boundary, the `unqualified` disposition, the
null `tau` and utilities, the case-level scope, and the closed training, A3,
P4, and sweep decisions are all intact.

## Upstream review reference

The parent unit's support-completion lineage received an independent read-only
audit returning `PASS` with `P0 = 0`, `P1 = 2`, `P2 = 5`, together with a lead
adjudication of the shard-6 transport reruns. That audit is recorded in the
[2026-08-06 review record](../2026-08-06-natural-boundary-routing-history-replication/review.md).

It is cited here **only as upstream provenance for the frozen source cohort**.
It is not evidence about this crossover, does not enter any endpoint,
contrast, `tau`, or utility here, and does not change this unit's
`unqualified` disposition.
