# Task 5 report — production HF-native one-image lifecycle

## Scope and boundary

### Semantic supersession (2026-08-20)

The former cross-surface exact-token/coordinate-alias gate is superseded by
the active OpenSpec decision.  The immutable v2 diagnostic below shows that
BF16/FA2 and fp32/SDPA are distinct policies: BF16 retained every protected G
owner and additionally covered `gt:1584:12`, while several other token and
owner differences were not quantization aliases.  Cross-surface divergence is
now `diagnostic_only`.  Strict identity fields, protected BF16-G presence,
internal BF16 sampler/replay parity, BF16-native witness/compiler inputs,
dual-RP fp32 Source/proposal audit, private proposal, rollback, K16, LR, and
no-promotion semantics remain unchanged.

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

### Reservation lifecycle correction candidate (2026-08-20)

The prelaunch lifecycle review found that the public entry had only a
lost-owner recovery constructor.  A genuinely new immutable root therefore
could not become the primary owner without being misclassified as a successor,
and a failure before reservation could be replaced by the unconditional
post-reservation terminal writer.

The bounded correction now distinguishes two fail-closed modes:

- `fresh_primary` is the public default.  It forbids stale-parent, recovery-
  authority, and stale-PID inputs, creates the unique root and exclusive
  `run-reservation.json` before calling backend preflight or loading a model,
  and fails on any root collision.
- `lost_owner_recovery` requires the mode, exact stale reservation, and
  recovery authority explicitly.  It preserves the append-only parent link
  and the unchanged one-consumer retry ceiling of one.

Both modes issue one hashed `RunReservationIdentity`.  Every durable phase,
the detached resource receipt, and the terminal envelope bind that same
identity.  A pre-reservation failure retains its original exception and never
calls terminal persistence; a post-reservation preflight failure can publish a
typed terminal with zero model/GPU/network/update/output actions.

The first independent localized review returned HOLD with five P1 findings:
historical v1 resource hashes were not reload-stable; a reservation identity
could be paired with the wrong or absent resource root; primary root creation
had a write/fsync stranding window; the public full-panel branch bypassed or
ignored reservation flags; and a post-reservation preflight failure reported
planned K16 counts despite zero observed actions.  One bundled correction now:

- omits the new optional identity key for legacy v1 payloads and proves exact
  historical resource/terminal reload;
- requires the detached output-root receipt to exactly match the admitted
  identity;
- prepares a complete reservation in a same-parent staging directory and
  atomically publishes it with no-replace `renameat2`, attempts cleanup of any
  unpublished staging directory on writer/fsync/collision faults, and preserves
  the primary exception with an attached cleanup-failure note if cleanup also
  fails;
- validates reservation flags in public `main()` before either one-image or
  full-panel GPU observation and forbids ignored recovery inputs on full-panel;
  and
- emits explicit zero request/group/forward/backward counts when a reserved
  run fails before any observed action or shared-surface receipt.

This is a reversible code/test/report candidate at frozen base
`273a8d25a3f24f392d4e19e115b7db9197adfbb5`.  It changes no scientific
objective, BF16/fp32 surface ownership, K16 policy, dual-RP behavior, or
continuation gate.  The focused lifecycle suites passed 90 tests; the adjacent
reconciliation/native-owner/lifecycle/entry set passed 127 tests.  Ruff,
scoped Pyright, compileall, strict OpenSpec validation, and the diff check are
clean.  No live GPU, model, network, K16, update, checkpoint, or output-root
action was run for this correction.  Independent localized lifecycle review
of the corrected frozen diff is still required before lead acceptance.

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
| `one-image-successor-20260820T-reconcile-v2` | fresh current-tree GPU0/GPU1 source-only diagnostic; RP1.0 passed, RP1.1 produced durable diagnostic-only divergence and source-only failed close | model loads 2, forwards 204, GPU allocations 2, backward 0, optimizer 0, sample/replay 0, checkpoint/private/output/network 0 |

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

The fresh v2 diagnostic supersedes that former admission result without
rewriting it.  Its immutable root is
`one-image-successor-20260820T-reconcile-v2/`; the terminal hash is
`4b8015ed953788c5851701127adac00aaf8e594e2a88477c1eba89bd56d7fc87` and the
phase ledger hash is
`10924d7c7ae0eff57e2e3dc431d0d9117e3567d7f2f2b5e1ade72b4bf3161aaa`.
RP1.0 passed; RP1.1 produced diagnostic-only evidence at
`receipts/006-source_audit_rp_1.1.json`.  Source and training each generated
109 tokens.  Source owners were
`{gt:1584:2,gt:1584:4,gt:1584:7,gt:1584:8,gt:1584:11,gt:1584:17,gt:1584:18}`;
training retained those protected G owners and additionally covered
`gt:1584:12` (H).  The receipt records mismatch positions
`[5,16,40,41,52,58,67,68,69,70,77,97]`, including the known `615->616`
coordinate difference and non-coordinate/large-coordinate differences.  No
K16 sampling/replay, backward, update, checkpoint, or proposal audit occurred.
Training and audit sessions each closed once through the source-only failed
close path with sample/replay=0 and a durable shared resource receipt.

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

## Surface-separated baseline owner

The Source-witness seam now has an experiment-local baseline owner.  It does
not copy logits, replace the BF16/FA2 training surface, use the fp32/SDPA audit
surface as a training witness, change the `1e-6` margin rule, or add a fallback
objective.  It binds the two execution domains explicitly:

- GPU 1 `hf`, batch one, fp32, SDPA runtime identity and both RP audit hashes;
- GPU 0 BF16/FlashAttention-2/eval/no-cache training identity, adapter and
  selected embedding-delta hashes;
- canonical checkpoint/base-model paths, manifest and image identity, prompt
  and tokenizer hashes, and the distinct checkpoint-payload digest domains;
- the exact two sealed Source decodes and token count.

The BF16/FA2 surface remains the sole scientific authority for sampling,
replay, trajectory credit, compiler, preservation, and the update path.  It
must build the free-running BF16 Source projection, compiler boundary,
remaining-owner state, WitnessMeasurement/Jacobians, and post-apply margin
probe from its own output.  The fp32/SDPA surface owns only stable owner-level
clean-greedy behavior and freezes paired Source baselines before update.
Canonical parser/matcher receipts are required independently on each surface.
Strict identity is limited to model/checkpoint/adapter/embedding/tokenizer/
prompt/image/manifest and declared processor policy.  Cross-surface token,
coordinate, row, or owner differences are retained as `diagnostic_only`
evidence and never gate the BF16 proposal.  This does not relax BF16/FA2
sampler-to-replay history, token, shape, or processed-log-probability parity.

`HFNativeOneImageOwnerError` carries the reconciliation receipt, and the
Source-audit failure phase serializes it before the service re-raises the
primary error.  The admitted receipt is included in the pre-acquisition owner
digest.

The previous additive contract was first verified with CPU/injected evidence;
the earlier live attempts exercised the former exact-token/coordinate-alias
gate.  Their durable rejections remain historical artifacts.  The fresh v2
root is the evidence that invalidated that cross-surface admission invariant;
it does not yet prove BF16-native witness/compiler admission or an update.

### P1 durable-baseline receipt correction

The preflight reconciliation phase now binds an ordered four-cell canonical
baseline matrix: fp32/SDPA RP1.0 and RP1.1 followed by BF16/FA2 RP1.0 and
RP1.1.  Every cell carries the complete canonical payload, its SHA-256, the
canonical matcher owner map, and its SHA-256.  The matrix is part of the typed
reconciliation receipt, the durable reconciliation phase receipt, and therefore
the terminal phase ledger.  A nested RP1.0 cross-surface divergence remains
`diagnostic_only` but can no longer cause RP1.1 or either BF16 baseline to be
absent from durable evidence.  No historical artifact was changed and this
correction performed no GPU, model, network, K16, backward, or update action.

The four-cell producer/loader/publisher uses
`human13_source_surface_reconciliation.v3`.  A strict schema dispatcher keeps
legacy `v2` receipts loadable only with their exact historical field set: the
immutable v6 `receipts/007-source_surface_reconciliation.json` round-trips
under current code without acquiring `canonical_baselines`.  Conversely, a
legacy payload relabeled as v3, or a v2 payload carrying v3 fields, is rejected;
the two schemas cannot masquerade as one another.

### Fresh v3 preflight and safety-abort provenance

The first fresh immutable preflight under the v3 receipt code is
`one-image-successor-20260820T-preflight-bf16-native-v7/`.  Its terminal is
`preflight_admitted`, with terminal hash
`9d702138f1c7e2209a7bb4b9dfe6badf6e7878b11028f3f8d8ba784f98fee59d` and phase
ledger hash
`5e498091d0a7816de9168846868fd3f66540ef3ae440e46af39ade875d8c1bf6`.
The v3 reconciliation receipt is
`receipts/007-source_surface_reconciliation.json` (inner hash
`89cbe1c761a4f9619233287577eb62fda7c75b1d7a4190db102618517b524c23`).  It
contains all four fp32/BF16 × RP1.0/RP1.1 canonical payloads and owner maps.
The BF16 Source retains all protected G owners and has baseline H owner
`gt:1584:12`; cross-surface disposition remains `diagnostic_only`.

The v7 counters are model loads=2, GPU allocations=2, forwards=230 (228
BF16-native Source-owner/no-cache forwards plus two fp32 audits), with zero
sample/replay groups, backward, optimizer steps, checkpoint/private/output, or
network actions.  Both sessions closed exactly once through source-only close;
the resource receipt records source-owner/no-cache/total forwards=228 and the
GPUs were released.  This is preflight evidence only; no K16 work occurred.

An interrupted successor,
`one-image-successor-20260820T-k16-bf16-native-v1/`, is preserved as an
operator safety abort, not an algorithm outcome.  It was launched before the
committed-HEAD execution constraint was received and was stopped immediately;
its terminal hash is
`67c432d76053721d4b28680be46fe11673aaf188be162c6866d09818ad0efb1f` and its
failure reason is `KeyboardInterrupt`.  It recorded model loads=2, GPU
allocations=2, one Source-audit forward, and zero K16 sample/replay,
backward, optimizer, checkpoint, private, output, or network actions.  It is
excluded from the scientific one-update budget and must not be reused.

### Committed fresh-primary attempt and tokenizer identity blocker

The committed fresh-primary attempt ran from `b18bff23ca14656d7156d648ff11e41b2c9b014f`
at the immutable root
`one-image-successor-20260820T-k16-bf16-native-b18bff2-v1/`.  Its terminal is
`update_failure` (envelope hash
`6c60d0346afab8e20350e7d21db4dbb9a1cd151c347b701a9cab686c16d9f396`, inner
terminal hash
`ddf5d1b6f87d6d447f7b89317f5584d56702c05f04a0b44c509c22f2cc32a2e6`, phase
ledger `64a673d408a55c90206d27f0360403266f530921bc68facefbfd11af8925b2b4`).
The failure occurred before objective construction and backward:
`HFNativeOneImageOwnerError: canonical_projection_failure: canonical replay
projection failed: HFNativeProjectionError: tokenizer snapshot differs from
surface/manifest identity`.

K16 acquisition/replay had already recorded four sampled groups/16 requests,
463 sampled forwards and 463 replay forwards.  Total/no-cache forwards were
1154 (228 Source-owner forwards); model actions were loads=2, GPU allocations=2,
forwards=1156, backward=0, optimizer=0, checkpoint/private/output=0, and
network=0.  Audit and training sessions closed once each and GPUs were released;
no private proposal, update, rollback, or Source reproduction existed to audit.

CPU artifact inspection established a representation mismatch, not different
tokenization semantics.  For the exact base path bound by image-1584, the
`tokenizer.json` SHA-256 is
`ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8`, matching
the manifest and live identity.  The runtime concrete class is
`transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast`, while
`tokenizer_config.json` records the factory/slow class `Qwen2Tokenizer`; the
installed fast class declares exactly that slow class.  The prior predicate
incorrectly required the config class to equal the concrete fast class.

The narrow correction adds one shared exact relation helper in
`src/qwen/tokens.py` and uses it in both Task-3 verified tokenizer runtime
inspection and the HF-native projection attestation.  It accepts only the
concrete runtime class or its declared `slow_tokenizer_class`; existing
installed source/module checks, tokenizer.json hash, manifest concrete class,
backend serialization, and private decoder checks remain strict.  The exact
Qwen2Tokenizer -> Qwen2TokenizerFast relation and an unrelated-class rejection
are covered by a CPU regression.  No scientific objective, surface ownership,
K16, LR, parity, or gate semantics changed, and no second live attempt is
authorized by this correction.

## Verification

- P1 receipt-completeness/schema correction: 108 reconciliation, native-owner,
  production-service, and entry tests passed, including the nested-RP1.0
  divergence counterexample, v2/v3 dispatch, immutable v6 receipt reload, and
  incomplete/tampered baseline checks.
- Focused reconciliation, service, native-owner, and entry suites: 88 passed
  after the direct-script import-path and precedence regressions.
- Adjacent CPU/injected Human-13 suites: 393 passed, 2 warnings.
- Target Ruff, compileall, and Pyright error-level diagnostics: clean.
- The Pyright claim is scoped to the touched production modules and focused
  tests; broad legacy fixture checking retains its pre-existing errors.
- The staged native bridge and adjacent CPU/injected suites were green before
  the live attempt.  The committed-HEAD scientific attempt must repeat the
  normal Source admission inside its fresh root before any K16 work.
- Tokenizer blocker correction: the exact Qwen2 slow/fast relation regression
  was observed RED before the shared helper existed, then the targeted
  Task-3/runtime and HF-native projection checks passed (28 trajectory-credit,
  54 shared-surface, and 13 native-owner tests).  Ruff, compileall, and scoped
  Pyright are clean for the four changed Python paths.
- Task 4.6 and Tasks 5.1–5.5 remain unchecked in OpenSpec.

## Claim boundary and stop rule

The current evidence supports a durable pre-acquisition diagnostic-only
observation, not an update, rollback, or 13-image continuation claim.  The
next run is exactly one fresh committed-HEAD image-1584 K16 attempt; it must
repeat BF16-native Source/witness/compiler admission and durable fp32/SDPA
Source baselines while retaining the v2 divergence evidence.

Do not increase coordinate tolerance, force owner/token agreement, use GPU-1
logits as the BF16 training witness, switch the training surface to fp32/SDPA,
or add a CE/vLLM fallback.  A subsequent run must use a fresh immutable root;
the historical failed roots remain untouched.  Only after independent baseline
admission and strict internal BF16 parity pass may one private K16 update,
dual-RP fp32 audit, rollback, and Source reproduction execute.

## AdamW/Accelerate production-admission correction

The cbb844e-v1 attempt remains an immutable pre-gradient production blocker:
the live adapter rejected the post-prepare `AcceleratedOptimizer` even though
its inner optimizer was a fresh, correctly bound AdamW.  This is not algorithm
evidence and no GPU/model/K16 retry is part of this correction.

The owned correction adds a content-addressed
`human13_adamw_runtime_ownership.v1` receipt at the live assembly boundary.
It admits only one exact `accelerate.optimizer.AcceleratedOptimizer` layer over
one exact `torch.optim.AdamW`, with the scheduler bound to that base, exact
parameter order/object IDs, fixed group/default hyperparameters, empty wrapper
and base state, world-one BF16/no-scaler/sync-neutral semantics, zero runtime
and scheduler counters, and CUDA RNG capture capability.  Nested/foreign
wrappers, AdamW subclasses, foreign schedulers, stale state/gradients/counters,
and device/dtype/sync/scaler drift fail closed.  Proposal capture and
`TrainingStateTransaction` bind the base; the wrapper remains the execution
handle and manual projected apply does not advance scheduler/runtime counters.

The production entry calls the deterministic ownership and BF16 witness/probe
admission after Source freeze and before the first K16 sample.  The adapter
revalidates the same content-addressed context after K16 and before objective
materialization/backward.  Real Accelerate CPU evidence covers the positive
wrapper/base/scheduler relationship, tiny private apply/rollback, and the
fault matrix; it is production admission evidence only.  Focused and adjacent
scientific suites must remain unchanged in objective, LR, K16, parity, surface
ownership, rollback, and dual-RP gate semantics.

Task 4.6 and Tasks 5.1–5.5 remain unchecked.  No live GPU/model/network,
checkpoint, or algorithm update is authorized by this correction.

The localized follow-up correction closes three shared admission gaps found by
review: same-base non-cosine schedulers are rejected; authoritative optimizer
group `betas`/`eps` are checked independently of defaults; and the production
backend hook is mandatory at the service boundary. A successful hook returns
the content-addressed ownership receipt, which is written as the immutable
`pre_acquisition_update_admission` phase before any K16 call. Missing hooks or
unhashed receipts fail closed before acquisition. Real Accelerate CPU tests and
the private apply/rollback vertical remain green; this is lifecycle evidence,
not an algorithm result.
