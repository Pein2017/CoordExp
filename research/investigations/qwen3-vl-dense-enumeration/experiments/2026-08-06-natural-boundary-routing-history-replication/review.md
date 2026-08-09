---
title: S-Primary Natural-Boundary Routing and History Replication Review Record
type: investigation-review
status: complete_support_lineage_pass_documentation_certified
updated: 2026-08-07
---

# Independent Failure and Claim Audit

> **Scope note.** Sections through "Serialization-successor launch review"
> audit the v1/v2 failures and the pre-launch successor tree, and are retained
> verbatim. The successor subsequently completed; the post-completion record is
> at the end of this file.

Disposition: `PASS_BOUNDED_TECHNICAL_HOLD`

The independent read-only audit accepts the v2 failure characterization and
the closing claim boundary. It does not accept any S arm or scientific result.

## Artifact audit

The immutable v2 root contains exactly two regular files:

- `failure.json`
- `failure.stderr`

It contains no result, runtime-identity, terminal-summary, or per-arm artifact.
The failure receipt has schema
`s_primary_natural_boundary_gate.v1.failure.v1`, status
`technical_invalidity`, byte size `94`, and stderr SHA-256
`1151d77e2856c2a2b0d25f788f4f2e547e8f38c497940828344dba04bdf261f6`.
The persisted bytes match the receipt's verbatim stderr exactly. The enclosing
`failure.json` SHA-256 is
`3df629611ab3e28afc734f19e2e31872e711dd71eeb29cb087087cb3ee64c89d`.
Neither file nor the output root is a symlink, and no gate process remained at
review time.

## Independent cause trace

The reviewer reproduced the failure without a GPU:

1. The K14 scalar-step factory returns `score_bias` and
   `score_bias_callback` as `FixedDoseScoreBias` diagnostic metadata.
2. The natural-boundary callback seam excludes those fields from model kwargs
   but preserves them under the receipt's `callback_metadata`.
3. The scalar receipt is retained in the arm result.
4. The gate constructs all in-memory arm outputs and then hashes the complete
   result with canonical JSON.
5. Canonicalization rejects the two Python objects with the same persisted
   error.

This trace supports a terminal receipt/materialization failure. It does not
make the unpersisted in-memory trajectories recoverable or admissible.

## Claim and stop-rule audit

The result, launch gate, tasks, experiment index, program compass, and current
project memory agree on all decision-bearing points:

- v1 is a K01 no-op parity technical invalidity;
- v2 is a terminal serialization technical invalidity;
- no S arm, cohort, A3 contrast, crossover, or training conclusion exists;
- the sealed support plan did not run;
- the unit's one exact repair is exhausted;
- further repair requires an explicit user decision superseding that stop
  rule.

The audit therefore passes the documentation and HOLD semantics while keeping
the scientific endpoint unqualified.

## Serialization-successor launch review

An independent Claude Opus/high read-only review examined the exact successor
authority, serialization boundary, K14 persistence regression, `110`-test CPU
receipt, installed-Qwen probe, pre-GPU v4 source/test identities, gate-root
freshness consumer, and old immutable roots. Its disposition is
`PASS_FOR_ONE_S_GATE_V3`; no P0/P1 survived.

The review confirmed that the Python score-bias handles are absent only from
persisted callback metadata, while model kwargs, additive mask, `+2` dose,
selected positions, mass/consumption receipts, K01 tolerance, MATH backend,
tokens, positions, and MRoPE remain unchanged. All nine source and ten test
hashes match the sealed v4 receipt, and the exact gate-v3 root is consumed
before CUDA/model loading.

Residual risk is bounded but explicit: K14B, N10/N20, and a non-empty matrix
contrast do not each have a separate synthetic canonical-write case, although
their JSON-bearing builders are shared with covered siblings. The reviewer did
not treat this as a launch blocker. The PASS authorizes one gate-v3 only;
support and scientific interpretation still require mechanical artifact
validation.

## Post-completion record

The authorized gate v3 ran to completion, the K/N/H cohort executed over eight
shards, and the formal v3 evidence was sealed. The current disposition and the
cohort denominators are owned by [results.md](results.md).

### Cross-checks performed by the downstream 2026-08-07 audit

A read-only formal scientific audit of the downstream
[2026-08-07 S K10-H20 Natural Crossover](../2026-08-07-s-k10-h20-natural-crossover/review.md)
unit independently re-verified the following facts about this unit's evidence,
without re-executing any model:

- `evidence.json` raw SHA-256
  `77f90bc48f8126ee1b071767308db603e1b64c0d19a4ca3e5ad8906cfc837598` and
  recomputed semantic self SHA-256
  `e04cb63540f9e3a1b27b938ada14f694f6b7208564adcaf2c82656915c62bb95` match the
  sealed identities; the receipt hashes
  `fa11800ec7ad832646162f77c9e5a6df71d2e9ca951144e40b2fd060df46bf8e` and
  `80aa70f46899474ee1f8646c02d9c061babfac9bd58534f4cfa1d75aa9df12fd` match.
- The cohort contains `11` events over `8` distinct images; K10 target release
  is `4/11` on `4` images; H20 is `9/11` qualified and `2/11`
  grammar-disruptive; no native STOP occurs in any arm of any event.
- K14 finite salience qualifies on `1` event and `1` image and is correctly
  held.
- A fresh, independently implemented runner, actuator seam, and finalizer
  reproduced this unit's `K01`, `K10`, and `H20` endpoint vectors exactly,
  `9/9`, on the three selected events. This is a determinism receipt for the
  operators and endpoint contract of this unit.

### What that cross-check does not establish

- It does not convert this unit's qualified static direction into a training,
  architecture, wrapper, token, decoder, or production claim; the frozen
  `training_claim_status: hold` stands.
- It does not extend the K10 result beyond oracle routing and target
  transcription sufficiency. The downstream unit found that every K10 cell it
  ran carried exactly two physical-owner-unmatched rows, so target selection
  and general owner grounding remain distinct capabilities.

## Support-completion lineage audit

Disposition: **`PASS`** — `P0 = 0`, `P1 = 2`, `P2 = 5`. Read-only; no model, no
GPU, no scientific code executed. The audit followed the cohort evidence's own
`input_bindings` back through census-v3, the K/N/H execution plan, the support
merge roots, the support execution contract and plan, and all eight shards,
rather than assuming the newest root suffix.

### Verified

- Sealed plan raw `1b7e97af291b58aec50849ae09aa883148d55610c281a9edb543fce46ec21d4c`,
  `plan_content_sha256 30991d2e11b23697bb355b2d3962e77479b9a1c48f3e59dd23c95c1363f73ea6`;
  both `context_ids_sha256` and `plan_content_sha256` recompute exactly.
- Exactly `200` frozen native-FN contexts, all unique; executed union equals
  the plan set with zero missing and zero extra; shards pairwise disjoint;
  per-shard executed order byte-identical to plan order. No outcome-adaptive
  rebuild or reorder.
- Eight shards, all `completed`; realized equals expected in every shard;
  total realized `77,428` scalar forwards; `resumed = 0` and `failures = 0`
  everywhere; all `200` observations status `measured`.
- Authoritative shard-6 receipt raw
  `375364cab328543f3a5e3e88ac6f634cd70067152cf265b92ca99aa8e213d66b` from the
  v5 supervised root; shards `0-5` and `7` from `support-execution-v3`.
- Authoritative merge `support-merge-v6`: ledger raw
  `c3ab9eca420c3ee2965c7ee72bd12208cf34264ec0238c226f8d7d3b399b64f6`, receipt
  raw `9e16375461edd7a7906f7bfff0c720786f3f23ca686222de4eb27050facd66c4`, self
  `7b6dc645889f5883d2c76e9c2463468d3c6caf478b29d1e9de08b1b8d1a3918b`; one
  receipt per shard index bound by path and hash.
- `220` ledger records = `200` completion + `20` retained; `14` verified;
  census-v3 binds the merge by path and hash; `11` events over `8` images; all
  `11` traceable to verified-support entries.
- Gate v3 precedes every support execution; the plan and shard contracts
  precede the gate by design, satisfying "never rebuild it from gate outcomes".
- No model, training, promotion, architecture, or production claim anywhere in
  the lineage; the only `training` and `behavioral_transfer` strings are inside
  explicit negations.

### P1 findings

1. **Disposition label seam.** Three rows carry `support_verified = true` while
   their `disposition` reads `support_measured_not_verified`, because that
   label is the else branch of `verified_support ∧ eligible_except_support ∧
   geometry.launch_eligible`. Admission does not use the label and
   `verified_S_count = 14` is correct, but a disposition tally yields `11/220`
   instead of `14/220`. **Remediated by documentation**: the field scope is now
   stated in [results.md](results.md); the artifact is unchanged.
2. **Shard-6 retry accounting.** Shard-6 ran in three roots, which under a
   literal reading of the one-exact-repair stop rule would leave the merge
   uninterpretable. **Resolved by lead adjudication**, recorded in
   [results.md](results.md) and [launch-gate.md](launch-gate.md): v3 and v4
   shard-6 produced only transport logs with no failure receipt or verbatim
   stderr, so they are unattested transport/process deaths rather than
   established operator or run-root mechanical invalidities; the user
   authorized a transport and supervision relaunch; v5 changed only external
   supervision while retaining plan, census, support rule, and order, with
   `resumed = 0` and all `13,867` forwards recomputed. No exact scientific
   repair allowance was consumed, and the lineage remains interpretable.

### P2 findings

1. `verified_S_count = 14` is native-FN-scoped while the census holds `26` S
   owners with `support_verified = true`; the extra `12` are
   `native_already_covered` owners carrying prior-ledger support. Both correct,
   conflatable. Documented.
2. `support-merge-v5` is orphaned and unmarked in the artifacts; it differs
   from v6 only by `support_rule = null` on the same `20` retained records,
   with no status or count change. Now recorded as superseded in
   [launch-gate.md](launch-gate.md).
3. The sealed plan receipt was never back-annotated:
   `realized_scalar_forward_count`, `realized_batched_forward_count`,
   `measured_wall_time_seconds`, and `execution_receipt_sha256` remain `null`.
   Realized `77,428` is provable only by summing the eight shard receipts.
   Documented.
4. Measured wall time is receipted for shard-6 only (`9,736 s`, from the
   supervisor); the other seven shards report `null`. Closed without rerun as
   non-decision-bearing — see [tasks.md](tasks.md).
5. Self-hash field-name drift across the lineage (`self_sha256`,
   `plan_sha256`, `aggregate_sha256`, `receipt_sha256`, `plan_content_sha256`,
   `receipt_content_sha256`, `content_sha256`), and `gate_sha256` means raw
   while `gate_result_sha256` means self. Makes independent verification
   error-prone. Documented.

All five P2 items and both P1 items are documentation or convention seams. No
artifact was rewritten and no count, admission, or downstream disposition
changed.

### Audit scope limits

No model was executed; support values are accepted as recorded. The audit
verified accounting, identity, ordering, and composition — not the numerical
correctness of any support feature or the calibration thresholds. The OOM
attribution for the v3 and v4 shard-6 deaths is inference from supervisor
counters and failure signature; no kill record survives. The 2026-08-05 prior
support and H0 ledgers were confirmed bound by hash but not re-audited.

## Post-remediation documentation re-review

An independent Fable re-review of the documentation remediation returned
**`PASS`**. The documentation state of this unit is **certified**; see the
downstream unit's [review record](../2026-08-07-s-k10-h20-natural-crossover/review.md)
for its ledger entry and the single wording finding it raised, which has been
corrected.
