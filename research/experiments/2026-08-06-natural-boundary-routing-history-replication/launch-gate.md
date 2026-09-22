# Natural-Boundary Routing and History Replication Launch Gate

Status: `CONSUMED AND CLOSED — gate v3 executed and completed; the K/N/H cohort evidence is sealed; no further launch authority exists`  
Updated: 2026-08-07

The gate history below is retained in full as immutable technical lineage. The
consumption record at the end of this file supersedes every earlier status
line; no earlier line confers current authority.

Historical status at the moment of authorization: `PASS_FOR_ONE_S_GATE_V3; fresh gate-v3 launch authorized; support still held`

Contract receipt:

- path: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/contract-v1/contract.json`
- file SHA-256: `9138b6efbfa10d61083e1b377d2c97babdd1b3db1d8279d96cd5e45a4f776c4c`
- self SHA-256: `a3f4cae883b283000f4ad4aa2889d2c977b05379b76f5724a5e70304d0d87a1d`

Superseding pre-GPU receipt:

- path: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/pre-gpu-receipt-v3/pre-gpu-receipt.json`
- file SHA-256: `f13968e21f56ff635438c86855abf4ef81e6230a7b5671de739b824c219aa5b7`
- self SHA-256: `602f72951d268362da473a235c5a8c6e95f6544c5300969b11b32cc1549c6c3d`
- independent disposition before the repaired second live attempt:
  `PASS_EXACT_REPAIR_FOR_ONE_S_GATE_RERUN`

First live S gate attempt:

- immutable root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/s-gt5001-live-gate-v1`
- disposition: `technical_invalidity`
- failure: K01 full-vocabulary no-op parity observed maximum logit drift
  `0.00011730194`, above the frozen `1e-4` tolerance
- failure receipt SHA-256:
  `ca33c2c74fdc6beff0e76e198a12d6241ecb1d674b7b2ea9f9c8624712b845f5`
- interpretation: instrumentation/control-path failure, not a model-behavior
  null; no later arm, support shard, cohort event, or A3 probe is authorized
  from this attempt
- repair budget: the unit's one exact repair was consumed by the superseding
  MATH-backend parity repair; the tolerance and operator semantics remained
  frozen

Second live S gate attempt:

- immutable root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/s-gt5001-live-gate-v2`
- disposition: `technical_invalidity`
- failure: terminal canonical result hashing rejected the diagnostic
  `FixedDoseScoreBias` object persisted under callback metadata
- failure JSON SHA-256:
  `3df629611ab3e28afc734f19e2e31872e711dd71eeb29cb087087cb3ee64c89d`
- verbatim stderr SHA-256:
  `1151d77e2856c2a2b0d25f788f4f2e547e8f38c497940828344dba04bdf261f6`
- interpretation: instrumentation/materialization failure, not a model result;
  no arm artifact is recoverable from the run root
- consequence: the unit's one exact repair is exhausted. Support, cohort,
  crossover, and A3 execution are closed unless a later user-owned decision
  explicitly supersedes the frozen stop rule.

Serialization successor authority:

- owner: [serialization-successor-authority.md](serialization-successor-authority.md)
- disposition: the user supplied the previously required owning decision and
  authorizes exactly one serialization-only successor exception
- old v1/v2 roots and receipts remain immutable and scientifically unqualified
- GPU launch remains held until a new pre-GPU v4 receipt binds the successor
  authority, exact code/tests, CPU evidence, and the fresh immutable
  `s-gt5001-live-gate-v3` root, followed by independent review

Pre-GPU v4 successor receipt:

- path: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/pre-gpu-receipt-v4/pre-gpu-receipt.json`
- file SHA-256:
  `ad09d12b0fb627bd3705d8b5906cb7168e0d66df67fb56ba64213d8dd0da6cee`
- self SHA-256:
  `4a875dd2af9805a035c60a174ac33b845c103a06814a7fc9404b4fc93100ee42`
- focused evidence: `110 passed`; file SHA-256
  `6c029381f1a5cd211efe876567fae8e8ce0f4960fb65afc0d4f7a410c8c8f0f7`
- installed-Qwen evidence: additive 4D pass-through, all-layer consumption,
  production GQA `16/8/2`, and all-head block23 mass shift passed; file SHA-256
  `0aa1c3f0587e7cafa8738dc3e62734a0e19f704f5a85056c19b5f72ed3a4974a`
- consumer preflight: exact receipt-bound gate-v3 root and all nine source-code
  identities replayed before CUDA/model load
- independent Claude Opus/high disposition: `PASS_FOR_ONE_S_GATE_V3`
- review found no P0/P1; residual canonical-write coverage risk for K14B,
  N10/N20, and non-empty contrasts is low because their JSON-bearing builders
  are shared with covered siblings, but this is recorded rather than promoted
  to exhaustive proof
- authorization is only for one fresh full S gate-v3 launch; support/cohort/A3
  remain held until its artifacts pass mechanical validation

## Required before implementation

- Independent review must confirm that the old seeded/free-row and unmatched
  semantic seams are represented exactly.
- The CPU census must preserve the 784-row universe and expose unassessed
  support rather than converting it to negative evidence.
- S must remain the only decision-owning substrate; A3 work is held until the
  S main matrix produces a specific secondary-contrast question.
- The pre-opener path must remove both the initial injected opener and every
  per-row appended opener.
- Static hard/soft masks and dynamic residual/history masks must have exact
  edge/position receipts and no-op parity tests.

## Required before GPU execution

- Focused and production-shaped tests pass.
- Installed-Qwen CPU probes prove float-additive 4D mask pass-through and
  identical all-layer consumption rather than silent coercion or ignored kwargs.
- Exact unit, upstream evidence, runner/finalizer source-file, and
  conclusion-critical test SHA-256 hashes are sealed in a contract receipt;
  every run's runtime identity repeats the same code hashes.
- Live GPU/process state is inspected; devices and immutable run roots are
  explicit.
- S `gt:5001:15` is the first live event.
- S support completion materializes and hashes the exact 200 owner/prefix/row/
  candidate identities and reproduces the frozen 77,428-forward eight-shard
  plan before consuming GPU time.

No training, A2, wrapper/token change, production decode, architecture
promotion, or old-artifact overwrite is authorized.

## Gate consumption record

Gate v3 executed exactly once and completed at

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication/s-gt5001-live-gate-v3`

on `gt:5001:15`, with `status: completed`, all fifteen arms
`K00, K01, K10, K11, K12, K13, K14T, K14B, N00, N01, N10, N20, H00, H10, H20`,
and result self SHA-256
`9719f2efe77f399ca66985659d902e33087be79bfdd29762643c6045cbed00fc`. Its
case-level mechanical receipt is owned by
[s-gate-v3-evidence.md](s-gate-v3-evidence.md).

The K/N/H cohort then executed over eight shards and was sealed as the formal
v3 evidence:

- `s-k-n-h-evidence-native-fn-supersession-v3/evidence.json`: raw SHA-256
  `77f90bc48f8126ee1b071767308db603e1b64c0d19a4ca3e5ad8906cfc837598`,
  self SHA-256
  `e04cb63540f9e3a1b27b938ada14f694f6b7208564adcaf2c82656915c62bb95`;
- `s-k-n-h-evidence-native-fn-supersession-v3/evidence.receipt.json`: raw
  SHA-256 `fa11800ec7ad832646162f77c9e5a6df71d2e9ca951144e40b2fd060df46bf8e`,
  self SHA-256
  `80aa70f46899474ee1f8646c02d9c061babfac9bd58534f4cfa1d75aa9df12fd`.

The v1 and v2 roots remain immutable, scientifically unqualified technical
failures and are not reinterpreted.

### Post-consumption disposition

- `11` events over `8` images; K10 target release `4/11` on `4` images, which
  clears the frozen checkpoint floor, so the static direction is `qualified`.
- H20 `9/11` qualified and `2/11` grammar-disruptive; H10, N10, and N20 are
  `0/11` qualified. No native STOP in any arm of any event.
- K14 finite salience qualifies on `1` event and `1` image: `HOLD`.
- `training_claim_status: hold`. The evidence's `next_step_flags` set
  `authorize_a3`, `authorize_crossover`, `authorize_p4`, and
  `authorize_training` to `false` and explicitly confer no execution authority.

The qualified static direction is oracle routing and target transcription
sufficiency only. It is not a natural owner slot, an owner field, or a
trainable soft routing mechanism.

### Support-completion gate chronology and root status

Order of authority, verified from artifact timestamps and bindings:

| When | Artifact | Note |
| --- | --- | --- |
| `08-06 03:25` | `support-completion-plan-v1` | sealed **before** the gate by design, so it can never be rebuilt from gate outcomes |
| `08-06 04:23` | `support-execution-contract-v1` (8 shard contracts) | same rationale |
| `08-06 08:06` | **gate v3 completes** | authority precedes every support execution |
| `08-06 08:15 → 17:06` | support executions v1, v2, v3, v4-shard6, v5-shard6 | all start after the gate |
| `08-07 00:00 / 01:01` | `support-merge-v5` / `support-merge-v6` | v6 supersedes v5 |
| `08-07 02:03 / 02:06` | census-v3 / K/N/H execution plan | census binds merge-v6 by path and hash |

Authoritative support roots:

- shards `0-5` and `7`: `support-execution-v3/shard-N/receipt.json`;
- shard `6`: `support-execution-v5-supervised-shard6/shard-6/receipt.json`,
  raw SHA-256
  `375364cab328543f3a5e3e88ac6f634cd70067152cf265b92ca99aa8e213d66b`;
- merge: `support-merge-v6`, ledger raw
  `c3ab9eca420c3ee2965c7ee72bd12208cf34264ec0238c226f8d7d3b399b64f6`, receipt
  raw `9e16375461edd7a7906f7bfff0c720786f3f23ca686222de4eb27050facd66c4`, self
  `7b6dc645889f5883d2c76e9c2463468d3c6caf478b29d1e9de08b1b8d1a3918b`.

Superseded and immutable, referenced by no authoritative artifact:

- `support-execution-v1` (eight `292`-byte launch logs, no receipts);
- `support-execution-v2` (eight zero-byte launch logs, no receipts);
- `support-execution-v3/shard-6` and `support-execution-v4-shard6` (transport
  logs only, no receipt, no failure log, no partial result);
- `support-merge-v5` — **orphaned and superseded**. It binds an identical
  eight-shard set and differs from v6 only in that `20` retained
  already-measured records carry `support_rule = null`. No status, count, or
  verified flag differs, and no artifact references it.

### Shard-6 transport and supervision adjudication

Shard-6 is the largest shard (`33` contexts, `13,867` forwards). Its two
earlier deaths are adjudicated as follows, and this adjudication is closed, not
open:

- The frozen unit requires a persisted failure log or verbatim stderr before a
  new technical invalidity may be declared and the exact repair consumed. The
  v3 and v4 shard-6 roots produced only transport logs ending after model load,
  with no failure receipt, no verbatim stderr, and no partial result. They are
  therefore **unattested transport/process deaths, not established operator or
  run-root mechanical invalidities**, and carry no scientific or technical
  model evidence.
- The user instructed that processes killed by the Codex restart be relaunched
  as needed, then flagged the repeated shard-6 falls, requested Opus/Fable
  review, and authorized a direct GPU relaunch. That is owning authorization
  for a **transport and supervision rerun**, not for any operator or estimand
  change.
- The v5 run changed only external supervision and observability. It retained
  the exact plan, census binding, support rule, and context order, recorded
  `resumed_reused_scalar_forward_count = 0`, recomputed all `13,867` forwards,
  and produced the sole authoritative shard-6 receipt, schema-identical to the
  seven v3 receipts.

**Therefore no exact scientific repair allowance was consumed by v5, and the
support lineage remains interpretable.** Supervisor evidence: OOM-kill counter
`66 → 66` across the run, `exit_code 0`, `elapsed_seconds 9736`. Resource
exhaustion is the most consistent explanation for the earlier deaths but is
inference only; no kill record survives.

An independent read-only support-lineage audit returned `PASS` with `P0 = 0`,
`P1 = 2`, `P2 = 5`; see [review.md](review.md).

### Downstream consumption

This sealed evidence was consumed as the frozen source of the
[2026-08-07 S K10-H20 Natural Crossover](../2026-08-07-s-k10-h20-natural-crossover/launch-gate.md)
unit, which was authorized separately by the user's prior conditional Phase 5
decision. That unit's own gate is likewise consumed and closed, and its
crossover result is source-specific `unqualified`. No result from either gate
authorizes architecture, objective, training, decoder, wrapper, token, or
checkpoint promotion.
