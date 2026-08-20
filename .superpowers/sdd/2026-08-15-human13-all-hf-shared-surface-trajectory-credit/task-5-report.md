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
| `one-image-successor-20260820T113000Z-surface-reconcile-live` | current-tree production entry reached both model surfaces, then the new reconciliation receipt rejected the same Source-witness mismatch | model loads 2, forwards 4 (2 Source-owner + 2 audits), GPU allocations 2, backward 0, optimizer 0, private/checkpoint/output 0 |

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

The current-tree live attempt is durably recorded at
`one-image-successor-20260820T113000Z-surface-reconcile-live/terminal.json`.
Its outer terminal envelope has content hash
`eced6b997bf88b6f4a69dd235d516df2099581c7003d13174e78464c1042e621`, phase
ledger hash `5bac4ff5998bd7d3c57d5769c01e6a4725ff15de786e6c8a9e1fb07a4b9e5c33`,
and recovery-successor hash
`89df6de23610e68555fb1728eb99a5412b3bb910bcbda59691f3107c5ed2b070`.  The
inner terminal receipt records model loads=2, forwards=4, GPU allocations=2,
backward=0, optimizer steps=0, and no private/checkpoint/output creation.  The
`source_audit_rp_1.1` phase carries the admitted=false reconciliation receipt
(`5f0b9545c8d5cafaabfa06ca85c540911ae5a5c3812f0422c2838b652886b901`) and the
same witness error; no K16 work began.

The first attempt after adding the owner also exposed a direct-script import
defect (`scripts.research` was absent from `sys.path`); the entry now installs
the explicit `--repo-root` before dynamic owner imports.  The fix is covered by
the focused regression and was exercised by the current live attempt.  The
earlier parent reservation remains untouched; the live run used `diag7` as its
single append-only lost-owner parent and a fresh sibling successor.

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

## Additive surface reconciliation owner

The Source-witness seam now has an experiment-local reconciliation owner.  It
does not copy logits, replace the BF16/FA2 training surface, use the fp32/SDPA
audit surface as a training witness, change the `1e-6` margin rule, or add a
fallback objective.  It binds the two execution domains explicitly:

- GPU 1 `hf`, batch one, fp32, SDPA runtime identity and both RP audit hashes;
- GPU 0 BF16/FlashAttention-2/eval/no-cache training identity, adapter and
  selected embedding-delta hashes;
- canonical checkpoint/base-model paths, manifest and image identity, prompt
  and tokenizer hashes, and the distinct checkpoint-payload digest domains;
- the exact two sealed Source decodes and token count.

The owner invokes the existing `WitnessMeasurement` checker as its only
scientific decision owner.  Exact agreement returns a content-addressed
admission receipt; identity drift, unserializable runtime evidence, image
lineage drift, a nonzero teacher-forced change, or a checker exception returns
a typed non-admission receipt.  `HFNativeOneImageOwnerError` carries that
receipt, and the Source-audit failure phase serializes it before the service
re-raises the primary error.  The admitted receipt is included in the
pre-acquisition owner digest.

The additive contract was first verified with CPU/injected evidence; after the
direct-script import-path fix, the focused reconciliation/owner/service/entry
set passes 88 tests and the broader current Human-13 CPU/injected matrix passes
393 tests with 2 warnings.  The current live attempt then exercised the real
entry and both model surfaces, but no update path passed the Source-witness
gate.  The known image-1584 Source-vs-training greedy mismatch therefore
remains a scientific HOLD rather than being hidden by the new receipt path.

## Verification

- Focused reconciliation, service, native-owner, and entry suites: 88 passed
  after the direct-script import-path and precedence regressions.
- Adjacent CPU/injected Human-13 suites: 393 passed, 2 warnings.
- Target Ruff, compileall, and Pyright error-level diagnostics: clean.
- The Pyright claim is scoped to the touched production modules and focused
  tests; broad legacy fixture checking retains its pre-existing errors.
- The staged native bridge and adjacent CPU/injected suites were green before
  the live attempt.  The current-tree production entry crossed model loading
  and both Source surfaces but no real update path passed the Source-witness
  gate.
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
