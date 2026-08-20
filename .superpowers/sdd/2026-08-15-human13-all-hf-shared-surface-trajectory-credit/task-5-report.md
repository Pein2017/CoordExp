# Task 5 report — production HF-native one-image lifecycle

## Scope and boundary

The Task-5 bridge now has an explicit experiment-local production owner:
`RepositoryHFNativeOneImageOwner` prepares the Source compiler boundary and
witness on the live BF16/FlashAttention-2 session before K16 acquisition,
admits the exact four sampled and four replay groups, and hands the bound
trajectory/compiler evidence to the split CUDA proposal lifecycle.  The
service owns append-only recovery, phase receipts, private-proposal cleanup,
rollback, and terminal persistence.  It does not claim Task 5 completion until
one real image-1584 update passes dual-RP audit and exact Source reproduction.

The default CLI remains fail-closed when the scientific owner evidence is not
available.  No vLLM publication, CE fallback, hidden session attribute,
adaptive retry, or alternate objective is used.

## Production attempts and evidence

All observed attempts used a fresh sibling successor of the previous
reservation.  The pre-existing stale reservation and each successor
reservation remain intact; none was overwritten or deleted.

| attempt root | result | observed actions |
| --- | --- | --- |
| `one-image-successor-20260820T091903Z` | typed module/persistence failure before model ownership; recovery and Source preflight only | model loads 0, forwards 0, backward 0, optimizer 0, checkpoint/output 0 |
| `one-image-successor-20260820T092649Z` | typed `update_failure`: the explicit leaf config lacked the inference `schema_version`; no source audit or update | model loads 2, forwards 0, backward 0, optimizer 0, checkpoint/output 0 |
| `one-image-successor-20260820T092800Z` | typed `update_failure`: same-model Source witness rejected `gt:1584:2` before K16 | terminal reported model loads 2, forwards 3, backward 0, optimizer 0, checkpoint/output 0; the pre-fix ledger omitted the completed RP=1.1 GPU-1 audit |
| `one-image-successor-20260820T093802Z-diag5` | same typed Source-witness failure; instrumented diagnostic confirmed the same failure | terminal reported model loads 2, forwards 3, backward 0, optimizer 0, checkpoint/output 0; the pre-fix ledger omission remained |
| `one-image-successor-20260820T094100Z-diag7` | same typed Source-witness failure with durable instrumented observation | terminal reported model loads 2, forwards 3, backward 0, optimizer 0, checkpoint/output 0; diagnostic artifact records the exact mismatch below |

The canonical source inference config was used for the latter two attempts.
Their durable terminal and phase receipts are under their respective roots;
no private proposal or checkpoint was created.

The instrumented Source-owner observation is decision-bearing and precise.  It
is durably recorded at
`one-image-successor-20260820T094100Z-diag7/diagnostics/source-witness-mismatch.json`,
which binds the observation to terminal hash
`b9eaaddb8b62e2e04b6a07733d71bc74f9306d964417cf061d5bb2f8b28f8cd2` and phase
ledger hash
`03dbecc6aea606758733b45b142a5c0154668f67dc5dbbc91aa23a8360ddb308`.
The file is explicitly an instrumented observation, not a normal terminal
payload or model-quality result.

```text
RP=1.0, membership=u_intersect_s_1.0, token_index=5
GPU-1 fp32/SDPA Source decode chose token 152285 = <|coord_615|>
GPU-0 BF16/FA2 same-session logits chose token 152286 = <|coord_616|>
training-surface chosen-minus-best margin = -0.125
```

This is a real model-surface disagreement, not a sampler/replay parity
failure.  The existing strict witness contract therefore stops before K16,
backward, optimizer, private checkpoint, dual-RP proposal audit, or rollback.

## Telemetry correction

The first live failure exposed that the outer `ResourceReceipt` still claimed
the planned K16 request/group/backward budget even when acquisition had not
started.  The receipt builder now derives sampled request/group counts from the
closed shared-surface receipt and derives `backward_count` from observed action
counters; partial failures can truthfully report zero K16 work while preserving
the independent Source-owner forward count.  A focused regression covers the
 zero-acquisition case.  This correction does not alter objective math or the
 Source-greedy decision.

The later Source-owner failure exposed a separate observation gap: RP=1.1's
GPU-1 audit had completed before GPU-0 Source-owner preparation failed, but the
old wrapper only counted and receipted audits after the backend returned.  The
service now annotates that primary error with the observed audit, emits a
failed `source_audit_rp_1.1` phase receipt, increments the forward counter once,
and the entry carries that phase into the terminal without replacing the
primary owner error.  Historical terminal JSON is not rewritten; future
attempts use the corrected ledger.

## Verification

- Focused entry, service, native-owner, and witness suites: 92 passed.
- Adjacent CPU/injected Human-13 suites: 385 passed, 2 warnings.
- Target Ruff, compileall, and Pyright error-level diagnostics: clean.
- The Pyright claim is scoped to the touched production modules and focused
  tests; broad legacy fixture checking retains its pre-existing errors.
- The staged native bridge and adjacent CPU/injected suites were green before
  the live attempts; no real update path has passed the Source-witness gate.
- Task 4.6 and Tasks 5.1–5.5 remain unchecked in OpenSpec.

## Claim boundary and stop rule

The current evidence supports only a safe pre-acquisition HOLD: the required
GPU-1 fp32/SDPA Source decode is not greedy on the same GPU-0 BF16/FA2 training
surface at one coordinate token.  It does not support an update, audit, rollback,
or 13-image continuation claim.

Do not fix this by changing the generated token, relaxing the witness margin,
using GPU-1 logits as the training witness, switching the training surface to
fp32/SDPA, or adding a CE/vLLM fallback.  Any of those would change the frozen
research contract.  A subsequent run requires an additive same-contract owner
that reconciles this source identity without weakening the witness rule; until
then the one-image live gate remains HOLD.
