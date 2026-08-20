# Wave-0 Entry Audit (task 0.4)

| Field | Value |
| --- | --- |
| Change | `standardize-coordexp-swift-supervised-losses` |
| Audited commit | `0cb0ae729389cc2560833243154f7684f43beff8` (tracked tree clean) |
| Date | 2026-08-20 |
| Auditor | Opus entry auditor, spawned by Claude Fable lead (distinct from lead and all builders) |
| Scope | Read-only. The only file written by this audit is this receipt. No commits, no source/test/config edits, no GPU or cache-materializing commands. |
| **STANDARDS verdict** | **PASS-WITH-DISPOSITIONS** |
| **INTENT-CONTRACT verdict** | **PASS-WITH-DISPOSITIONS** |
| **Entry gate** | **CLEARED** — 0 P0, 0 P1. Wave 1 may begin. Dispositions below are binding on Waves 1–5. |

Frozen-target verification was run at audit start and again immediately before
writing this receipt. Both times: `git rev-parse HEAD` = `0cb0ae729…`,
`git status --porcelain --untracked-files=all` = exactly the three Wave-0
receipt files untracked, no tracked modifications. This receipt becomes the
sanctioned fourth untracked path.

---

## Findings

| id | sev | area | description | disposition |
| --- | --- | --- | --- | --- |
| F-1 | P2 | intent-contract (a) — canonical group tuple | A naive top-level `from src.losses.vocab import V1_TOKEN_TYPES` in `src/config/models.py` **fails with a circular ImportError**, empirically reproduced (see command C-7). Cause: importing any `src.losses.*` submodule executes `src/losses/__init__.py`, which imports `src/losses/runner.py`, which does `from src.config.models import LossesConfig` (runner.py:14). The plan's stated approach ("Wave 1 imports `V1_TOKEN_TYPES`, never edits the file", wave-0-baseline.md) is therefore not directly expressible. | Wave 1.2 MUST use one of two cycle-free routes, neither of which edits `src/losses/vocab.py`: (i) a deferred import of `V1_TOKEN_TYPES` **inside** the Pydantic validator body, or (ii) keep the literal tuple in `src/config/models.py` and add a test asserting equality with `src.losses.vocab.V1_TOKEN_TYPES`. Route (iii) — making `src/losses/__init__.py` lazy — is permitted (it is not a determinant owner) but is a larger blast radius and is not recommended. Do NOT resolve this by editing `vocab.py`. |
| F-2 | P2 | intent-contract — fail-closed gate under ablation | `_build_finite_status` (`src/losses/runner.py:1151-1159`) derives per-term finite labels from `term.weighted_loss`, not the raw value. Today `weighted = raw * 0.0` accidentally preserves NaN, but the Wave-2 rewrite that sets weighted to a literal `0` under `zero_weight_ablation` would **silently mask a non-finite gate diagnostic**, contradicting the delta's "Zero-weight gate diagnostic is non-finite" scenario and the stable `Non-Finite Loss And Gradient Gates` requirement. | Wave 2.3 / 3.4: the ablation's finite label MUST derive from the raw diagnostic, never from the weighted-zero value. Task 3.4 already mandates injecting a non-finite gate diagnostic and proving all ranks skip before backward — that test is the RED evidence for this seam and must be observed failing against a weighted-keyed implementation at least once. |
| F-3 | P2 | standards — research-meaning migration scope | Wave 1.3 changes the **enabled** gate weight on live supported configs from `0.2`/`0.25` to `0.1`. Exact scope measured at HEAD (21 supported configs = 5 prod + 16 smoke): gate `0.2` in **3 prod + 2 smoke**, gate `0.25` in **1 smoke**, gate `0.0` in **2 prod + 13 smoke** (→ `zero_weight_ablation`); `coord_gaussian_rps` appears in **2 configs** (1 prod, 1 smoke), both under `losses.protected` (→ `losses.auxiliary`). The `0.1` constant is recorded in proposal.md (Impact names `0.2`/`0.25` explicitly), design.md §1, both spec deltas, and the 2026-08-12 SDD plan — but **no receipt of explicit user authorization for the `0.1` value itself was found** in the change, the handoff (`research/handoffs/2026-08-20-standardize-losses-handoff.md`), or `research/`. Research meaning is user-owned per the repo standing decision list. | Non-blocking: the decision is recorded in the owning authority this audit is measured against. The Wave-0 close-out commit MUST surface the exact migration scope (numbers above) so the user sees the objective-weight change before Wave 1.3, whose task already requires inspecting every config diff. If the user has not seen the `0.1` constant, that is a one-line confirmation, not a re-plan. |
| F-4 | P2 | intent-contract (d) — eval reduction residue | `src/eval/forward.py:16-24` privately imports `LossContextFactory`, `LossRunnerBoundary`, `QwenForwardFn`, `SupervisedMicroStep`, `_default_loss_context`, `_default_qwen_forward`, `_runtime_loss_denominator_gatherer` from `src/training/supervised_trainer.py` (pre-cost audit finding I-1). Two of these (`_default_loss_context`, `_runtime_loss_denominator_gatherer`) are legitimately on this change's Wave-2/Wave-3 loss-construction and reduction path; `_default_qwen_forward` is not — it is the forward-callable default. | **I-1 stays OUT of scope.** Waves 2–4 MAY touch `_default_loss_context` and `_runtime_loss_denominator_gatherer` as loss-path owners, but MUST NOT re-home, rename, or "clean up" `_default_qwen_forward`, and MUST NOT convert this into a trainer/eval seam refactor. That residue belongs to `add-coordexp-swift-training-observability`. Note: `src/training/supervised_trainer.py` and `src/eval/forward.py` are NOT determinant owners, so this work cannot move a fingerprint. |
| F-5 | P3 | standards — baseline receipt accuracy | `wave-0-baseline.md` states the frozen fixtures `tests/fixtures/training_orchestration/{completed_step_rows.json, run_writer/logging.jsonl}` "carry the OLD field family". **They do not.** Both carry only `loss/total`, which the change explicitly preserves (`loss/total` MUST be the sum of weighted objective terms). The `rg` that produced the ~20-file inventory matched on the `total` alternative. Verified by direct read of both fixtures. | Correction, not a defect: the Wave-0 close-out should reference this correction rather than rewriting the baseline receipt. Consequence is positive — see F-6. |
| F-6 | P3 | intent-contract (c) — telemetry rename vs frozen fixtures | Direct consequence of F-5: `tests/training/test_orchestration_compatibility.py` references `loss/` at exactly two sites (lines 924, 1039), both `loss/total`, both preserved by the change. **No compatibility test node requires a declared row-schema flip for the Wave-4 rename, and the frozen fixtures require no regeneration and stay byte-frozen.** The actual per-term (`loss/base_ce`, `loss/token_type_gate`, `loss/coord_gaussian_rps`) consumers in current roots are exactly four files: `tests/losses/test_runner.py`, `tests/runtime/test_train_runtime.py`, `tests/training/test_wave7_exact_resume_compare.py` (synthetic in-test rows at lines 282-302, not fixture-backed), `scripts/analysis/coordexp_swift_length_isolation.py`. Remaining hits are under `docs/history/…` (immutable provenance) and `docs/superpowers/plans/…` (execution notes, superseded). | Wave 4.3's migration surface is these four current-root files. Fixtures MUST remain byte-frozen (assert by sha256 in the Wave-4 gate). `docs/history/**` MUST NOT be edited. The Wave-4 manifest amendment should record "declared flips: none required in `test_orchestration_compatibility.py`" as an explicit finding rather than silently omitting it. |
| F-7 | P3 | standards — stable-spec coherence | The stable requirement `Non-Finite Loss And Gradient Gates` (`openspec/specs/coordexp-swift-supervision-losses/spec.md:247`) is NOT in this change's delta set and keys its scenario on "total loss is NaN or Inf before backward". Under the named gate ablation the gate diagnostic has weighted value `0` and does not enter `loss/total`, so a non-finite gate would not make total non-finite. The change's MODIFIED `Protected Default Losses` adds the covering scenario, so there is no contradiction — but the unmodified stable text will read as an incomplete statement of the fail-closed policy after sync. | Wave 5.5 SHOULD evaluate adding a MODIFIED delta for `Non-Finite Loss And Gradient Gates` so the synced stable spec states that a non-finite **protected diagnostic** (not only `loss/total`) enters the all-rank pre-backward decision. Optional; not a validation failure. |
| F-8 | P3 | standards — handoff line superseded | `research/handoffs/2026-08-20-standardize-losses-handoff.md:47-50` calls the completed-step per-loss row schema (`loss/base_ce`, `loss/token_type_gate`, …) "a protected compatibility surface". No stable spec pins those keys — `grep -rn "loss/" openspec/specs/` returns exactly one hit, an unrelated prose line. The sole owner is `Wide-Step Logging Stream` in `coordexp-swift-training-artifacts`, which this change MODIFIES to define the raw/weighted family. | No conflict. The handoff line is a decompose-era receipt observation, superseded by this change's delta. Record in the close-out so a later reader does not treat the handoff as a blocking contract. |
| F-9 | P3 | standards — command manifest (0.3) gaps | Manifest gaps against the 6-wave plan: (i) the residue/legacy-placement searches required by tasks 1.5, 2.5, 4.5 and 5.5 are invoked by the tasks but **not frozen as argvs**; (ii) there is no `openspec validate --all` entry — only single-change `--strict` — although task 0.1's pin was established with `--all`; (iii) `wave0-entry-baseline.expected` defers counts to the close-out and now has an observed number to pin (`817 passed, 0 failed, 0 skipped`). The `TO-FREEZE-IN-WAVE-{3,5}-PACKET` GPU placeholders are by design (fresh packet + fresh authorization per the spent decompose blanket grant), not gaps. | All three are append-only amendments, correctly permitted by the manifest's own `amendment_rule`. Add (i) and (ii) before the Wave-1 gate; pin (iii) in the Wave-0 close-out. Otherwise the manifest's frozen argvs, amendment rule, cache invariant, and GPU-authorization notes are adequate. |
| F-10 | P3 | standards — determinant baseline is single-config | `wave-0-determinant-baseline.json` records determinants for exactly one config: `configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`. That is the correct and only config bound to the two published cache targets, but the task-5.3 equality claim is therefore scoped to that config. | Task 5.3's recompute MUST use this exact config; the `wave5-determinant-equality` probe should assert the config path alongside the payload equality, so a config substitution cannot silently satisfy the invariant. |

**Counts: 0 P0, 0 P1, 4 P2 (F-1, F-2, F-3, F-4), 6 P3 (F-5 … F-10).**

---

## 1. STANDARDS — PASS-WITH-DISPOSITIONS

The Wave-0 baseline is sound and complete against tasks 0.1–0.3. Every claim
was re-derived independently rather than trusted.

### 0.1 predecessor pins — VERIFIED

- Both archives exist on disk:
  `openspec/changes/archive/2026-08-19-reconcile-coordexp-swift-training-contracts/`
  and `openspec/changes/archive/2026-08-20-decompose-coordexp-swift-training-orchestration/`.
- Recorded commits independently pinned as ancestors of HEAD (not merely
  quoted from the receipt): `eb2dc97ab`, `ebfa78ff1`, `68191f7ea` all return
  ancestor-OK from `git merge-base --is-ancestor` (command C-8).
- `conda run -n ms openspec validate --all` re-run at HEAD: **21 passed,
  0 failed (21 items)**, matching the receipt. Both this change and
  `add-coordexp-swift-training-observability` validate.
- No disagreement found between code, stable specs, and archive dispositions.

### 0.2 cache invariant — VERIFIED INDEPENDENTLY

(a) **Exactly one loss-package determinant owner.**
`PACKING_CACHE_DETERMINANT_OWNERS` (`src/training/pack_cache.py:65-97`) holds
31 determinants over 24 distinct owner files. Programmatic filter over the
recorded baseline confirms `loss-package owners: ['src/losses/vocab.py']` —
i.e. `realized_vocab_groups` is the only one. No owner file appears in this
change's stated edit surface: `src/config/models.py`, `src/losses/runner.py`
+ term modules (`base_ce.py`, `token_type_gate.py`, `coord_gaussian_rps.py`,
`normalizers.py`, `context.py`), the train/eval reduction paths
(`src/training/supervised_trainer.py`, `src/eval/forward.py`), the rank-zero
logging projection, and `configs/coordexp_swift/{prod,smoke}` are all absent
from the owner set. The DO-NOT-EDIT list in `wave-0-baseline.md` is correct
and load-bearing: `_build_determinant_entries` (pack_cache.py:1618-1636)
stamps `owner_source_identity.sha256 = _file_sha256(repo_root / owner)` into
every entry, and `_registry_entries_fingerprint` hashes those entries — so an
edit to any owner's **source bytes** moves the aggregate fingerprint, exactly
as the receipt claims.

(b) **No determinant field derives from the `losses` config section.**
Read of `build_packing_cache_determinants` (pack_cache.py:262-340): the
semantic payload reads `config.template`, `config.packing.*`,
`config.model.processor`, `config.data.train_order`, `config.runtime.seed`,
augmentation determinants, the Qwen component identities/asset inventory,
`_realized_vocab_group_identity(vocab_groups)`,
`micro_step_runtime_config_identity(config)`, and
`_supervised_micro_step_schema_identity()`. **No `losses.*` read.**
`micro_step_runtime_config_identity` (`src/training/cache_contract.py:21-33`)
serializes exactly three keys derived from `config.training.precision` and
`config.model.fa2_branch_proof` — no loss field. `vocab_groups` is built by
`build_token_vocabulary_groups` (`src/losses/vocab.py:155-191`) from
`token_identity` + `tokenizer` + `extra_blocked_token_ids`; the only
`extra_blocked_token_ids` references in `src/` are that function's own
parameter and body, so no loss config reaches it. Conclusion: the Wave-1.3
supported-config `losses:` migration **cannot** move either fingerprint.
Corollary for F-3: the pure-CE smoke config that backs the cache targets is
itself a migration target (gate `0.0` → `zero_weight_ablation`), and this is
still fingerprint-safe.

(c) **Recorded fingerprints equal the on-disk targets.** Read-only listing of
`.cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/` shows exactly
two target directories, `8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f`
and `3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662`, plus
their two zero-byte lock files. These match
`wave-0-determinant-baseline.json` (`splits.train.aggregate_fingerprint`,
`splits["eval.forward"].aggregate_fingerprint`) and the manifest's
`cache_invariant.{train,eval}_fingerprint` character-for-character.
Additional strengthening not claimed by the receipt: re-hashing **all 24
owner files at HEAD** against the recorded `owner_source_identity.sha256`
values (both splits, 31 entries each) produced **zero mismatches**, so the
cache-hit invariant is live at `0cb0ae729`, not merely recorded.

### 0.4 baseline run — REPLAYED, CONFIRMED

The manifest's `wave0-entry-baseline` argv was replayed verbatim inside a
`bash -c 'export …; unset …; …'` wrapper (never `env`-prefixed — the host
hook silently no-ops those). Real pytest output observed:
**`817 passed, 1 warning in 52.92s`, exit 0** — 817 passed / 0 failed /
0 skipped, matching the expected count exactly. The single warning is a
pre-existing PyTorch sparse-CSR beta notice from
`tests/runtime/test_finite_gates.py:358`. No hang; no kill required.

### 0.3 command manifest — ADEQUATE WITH AMENDMENTS

Frozen argvs cover every wave gate; the append-only amendment rule is
well-formed; the cache invariant is stated as a hard stop with both
fingerprints inline; the GPU-authorization notes correctly record that the
2026-08-19 blanket grant is spent and that each GPU action needs a fresh
packet with its own bounds. Gaps are F-9 (residue-search argvs, `--all`
validation entry, unpinned baseline counts) — all P3, all closable by
amendment.

---

## 2. INTENT-CONTRACT — PASS-WITH-DISPOSITIONS

No conflict makes the plan unbuildable, and **no probed path forces an edit to
a determinant owner file.**

**(a) Canonical group tuple.** The new validation *is* expressible in
`src/config/models.py` without touching `src/losses/vocab.py`, but **not by
the plan's literal wording** — see F-1, an empirically reproduced circular
import. Two cycle-free dispositions are given there. Today's
`TokenTypeGateLossConfig` (models.py:279-291) already carries the four group
names as an inline `Literal` and rejects empties/duplicates but permits
subsets and arbitrary order; the new exact-ordered-tuple rule is a
strengthening of that same validator, no new owner. Note `TokenTypeGateLoss`
itself (`src/losses/token_type_gate.py`) never reads the config `groups` — it
resolves per-atom via `context.vocab_groups.allowed_ids(atom.token_type)` —
so the exactness rule is purely a config-layer contract.

**(b) Gate-ablation `detached_diagnostic`.** An existing structure is already
there: `src/losses/runner.py:236-238` computes
`gate_grad_context = torch.no_grad() if self.token_type_gate_weight == 0.0 else nullcontext()`
and wraps the gate's `_compute_token_term_contribution` in it, retaining the
term in the bundle. Wave 2 attaches the named-mode diagnostic to this seam
inside `src/losses/runner.py` — **not a determinant owner**, so the
fingerprint is untouched. The `omit` policy for `coord_gaussian_rps` also has
a partial precedent (`LossRunner.from_config` builds no term at weight `0`,
and `prepare_planned_step` builds no denominator), so Wave 2.3 extends
existing structure rather than inventing it. The genuine hazard on this path
is F-2 (finite label keyed on `weighted_loss`).

**(c) Telemetry rename vs frozen decompose fixtures.** The change can satisfy
"preserve historical JSONL" with **zero** fixture regeneration and **zero**
declared compatibility-node flips — see F-5/F-6. The fixtures MUST stay
byte-frozen and MUST be asserted so at the Wave-4 gate. The four real
per-term consumers in current roots are enumerated in F-6.

**(d) Eval reduction path / I-1 residue.** The residue is real and this
change's reduction/telemetry work genuinely brushes two of the seven private
imports. It should stay out of scope, and F-4 states the exact boundary.

---

## 3. Entry gate

**Wave 1 may begin. 0 P0 / 0 P1.** The four P2 findings are pre-implementation
dispositions bound to specific waves (F-1 → Wave 1.2, F-2 → Waves 2.3/3.4,
F-3 → Wave 1.3, F-4 → Waves 2–4 scope boundary), not entry blockers.

### The Wave-0 close-out commit MUST include

1. The three existing Wave-0 receipts plus this audit receipt, committed
   together — after this commit the tree has no untracked Wave-0 evidence.
2. Tasks 0.1–0.4 checked off in `tasks.md`, with the observed baseline count
   **817 passed / 0 failed / 0 skipped** pinned into the commit message and
   into `wave0-entry-baseline.expected` via a manifest amendment (F-9 iii).
3. A manifest amendment adding the frozen residue-search argvs (tasks 1.5,
   2.5, 4.5, 5.5) and an `openspec validate --all` entry (F-9 i, ii).
4. The F-5 correction referenced explicitly: the frozen orchestration
   fixtures carry only `loss/total` and are unaffected by the Wave-4 rename;
   `wave-0-baseline.md` is not rewritten, the correction lives here.
5. The F-3 migration-scope line, verbatim and user-visible:
   *enabled gate weight standardizes `0.2` (3 prod + 2 smoke) and `0.25`
   (1 smoke) → `0.1`; `0.0` in 2 prod + 13 smoke becomes
   `zero_weight_ablation`; `coord_gaussian_rps` moves out of
   `losses.protected` in 2 configs (1 prod, 1 smoke); 21 supported configs
   total* — with the note that no explicit user-authorization receipt for the
   `0.1` constant was found, so the user should see it before Wave 1.3.
6. The F-1 disposition recorded as a binding Wave-1.2 constraint
   (deferred-import or duplicated-literal-plus-equality-test; never an edit
   to `src/losses/vocab.py`).
7. A restatement of the DO-NOT-EDIT owner list and the stop rule: an edit to
   any of the 24 determinant owners is a blocking contract review, never a
   cache rebuild.

---

## Method deviation (disclosed)

To settle F-1 empirically rather than by inspection, command C-7 **temporarily
wrote to the tracked file `src/config/models.py`** — inserting one import line,
attempting the import in a subprocess, then restoring the original bytes in a
`finally` block. This exceeded the read-only scope of this audit and is
disclosed rather than omitted. Mitigations and evidence:

- The probe printed `restored ok: True`, an in-process byte-equality check of
  the restored file against the captured original.
- `PYTHONDONTWRITEBYTECODE=1` was exported, so no `.pyc` residue was written
  (the known poisoned-bytecode host trap).
- `git status --porcelain` immediately afterwards, and again immediately
  before writing this receipt, shows **no tracked modification** — the tree is
  byte-identical to `0cb0ae729`.
- No other command in this audit mutated any tracked file.

---

## Commands executed (exact, with observed output)

All commands ran with cwd `/data/CoordExp/.worktrees/CoordExp-swift`. Every
Python/pytest/openspec invocation used
`bash -c 'export PYTHONDONTWRITEBYTECODE=1; unset <required_absent>; …'`
per the manifest's `host_note`; outputs were read directly, never inferred
from exit codes.

**C-1 — frozen-target verification (start and end of audit)**

```
git rev-parse HEAD
→ 0cb0ae729389cc2560833243154f7684f43beff8

git status --porcelain --untracked-files=all
→ ?? openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/command-manifest.json
  ?? openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/wave-0-baseline.md
  ?? openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/wave-0-determinant-baseline.json
(no tracked modifications; `git diff --stat` empty)
```

**C-2 — baseline test replay (manifest `wave0-entry-baseline`)**

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; unset COORDEXP_SWIFT_PACK_CACHE_ROOT \
  COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE COORDEXP_SWIFT_EVAL_REDUCTION_MODE \
  COORDEXP_SWIFT_PROFILE_SYNC_TIMINGS RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT; \
  conda run -n ms pytest tests/config tests/losses tests/runtime \
  tests/training/test_reporting.py tests/training/test_pack_cache_determinant_registry.py \
  tests/artifacts tests/eval/test_forward_eval.py -q'

→ 817 passed, 1 warning in 52.92s
→ [exited with code 0]
  (only warning: tests/runtime/test_finite_gates.py:358 sparse-CSR beta UserWarning)
```

**C-3 — strict OpenSpec validation (task 0.1 pin)**

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms openspec validate --all'
→ ✓ change/add-coordexp-swift-training-observability
  ✓ spec/… (19 stable specs, all ✓)
  ✓ change/standardize-coordexp-swift-supervised-losses
  Totals: 21 passed, 0 failed (21 items)
```

**C-4 — determinant owner registry**

```
grep -n "PACKING_CACHE_DETERMINANT_OWNERS" -A 80 src/training/pack_cache.py
→ 31 entries, lines 65-97; sole src/losses/* owner:
  "realized_vocab_groups": "src/losses/vocab.py"
```

**C-5 — determinant payload construction (no `losses` read)**

```
grep -n "def build_packing_cache_determinants" -A 130 src/training/pack_cache.py
→ semantic_payload keys: version, split, dataset, template, packing, processor,
  ordering, augmentation, qwen, realized_vocab_groups,
  micro_step_runtime_config, micro_step_schema  — no losses.* field

cat src/training/cache_contract.py
→ micro_step_runtime_config_identity returns exactly
  {fa2_model_dtype: config.training.precision,
   capture_fa2_branch: config.model.fa2_branch_proof == "every_forward",
   require_fa2_branch_proof: config.model.fa2_branch_proof == "every_forward"}

grep -rn "extra_blocked_token_ids" --include=*.py src/
→ only src/losses/vocab.py:159 (parameter) and :171 (body) — no loss-config source
```

**C-6 — baseline JSON vs on-disk targets vs owner sources at HEAD**

```
ls -1 .cache/coordexp_swift/packing/coordexp-swift-pack-cache-v3/
→ 3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662/
  8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f/
  (+ two 0-byte .lock files)

bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms python -c "<read baseline json, \
  re-hash every owner file at HEAD against owner_source_identity.sha256>"'
→ bytes 405123
  base_commit: 0cb0ae729   schema: coordexp-swift-losses-wave0-determinant-baseline-v1
  baseline_sha256 = b2d9595f07a0a511abf436e24f47b21c5694760c4d308bed4845fd5712931059
  splits/train/aggregate_fingerprint     = 8f11237fc793f921dce4a9570b2f2d053a69ed604f3b42e006a8ac6c07476f2f
  splits/eval.forward/aggregate_fingerprint = 3b30c157d639fdbfe30013a4c05fa17abb10ed2201c673d4409266cc60af6662
  owner files: 24   determinants per split: {'eval.forward': 31, 'train': 31}
  owner-source MISMATCH at HEAD: []
  loss-package owners: ['src/losses/vocab.py']
  config: configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml
```

**C-7 — circular-import probe for F-1 (MUTATING; restored — see Method deviation)**

```
bash -c 'export PYTHONDONTWRITEBYTECODE=1; conda run -n ms python -c "<insert \
  \"from src.losses.vocab import V1_TOKEN_TYPES\" into src/config/models.py, \
  import src.config.models, restore original bytes in finally>"'
→ IMPORT FAILED: ImportError cannot import name 'TrainConfig' from partially
  initialized module 'src.config.models' (most likely due to a circular import)
  (/data/CoordExp/.worktrees/CoordExp-swift/src/config/models.py)
→ restored ok: True

(supporting) bash -c '… conda run -n ms python -c "import src.losses.vocab; \
  print(\"src.config.models\" in sys.modules)"'
→ config.models loaded? True
→ V1_TOKEN_TYPES= ('desc_text', 'schema', 'coordinate', 'eos')
(cause: src/losses/__init__.py imports runner.py, whose line 14 is
 `from src.config.models import LossesConfig`)
```

**C-8 — predecessor commit pins**

```
git merge-base --is-ancestor eb2dc97ab HEAD → eb2dc97ab ANCESTOR-OK
git merge-base --is-ancestor 68191f7ea HEAD → 68191f7ea ANCESTOR-OK
git merge-base --is-ancestor ebfa78ff1 HEAD → ebfa78ff1 ANCESTOR-OK
ls -d openspec/changes/archive/2026-08-19-reconcile-…/ openspec/changes/archive/2026-08-20-decompose-…/
→ both present
```

**C-9 — supported-config migration scope (F-3)**

```
grep -rn -A1 "token_type_gate:" configs/coordexp_swift/{prod,smoke}/*.yaml | grep "weight:"
→ prod  0.2 ×3 | prod  0.0 ×2
  smoke 0.25 ×1 | smoke 0.2 ×2 | smoke 0.0 ×13
grep -rln "coord_gaussian_rps" configs/coordexp_swift/{prod,smoke}/
→ 2 files (1 prod, 1 smoke), both under losses.protected
ls configs/coordexp_swift/prod/*.yaml | wc -l  → 5
ls configs/coordexp_swift/smoke/*.yaml | wc -l → 16
```

**C-10 — telemetry consumers and frozen fixtures (F-5, F-6)**

```
cat tests/fixtures/training_orchestration/completed_step_rows.json
cat tests/fixtures/training_orchestration/run_writer/logging.jsonl
→ both carry only `loss/total` (values 1.5 / 2.0 / 3.0 / 4.0); no per-term key

grep -rn "loss/" tests/training/test_orchestration_compatibility.py
→ line 924: "loss/total": 1.5
  line 1039: "metrics": {"loss/total": 1.0 + step, "acc_top1": 0.5, "acc_top5": 1.0}
  (2 hits, both preserved by the change)

grep -rln "loss/base_ce\|loss/token_type_gate\|loss/coord_gaussian_rps" \
  tests/ src/ scripts/ configs/ docs/
→ current roots: tests/losses/test_runner.py,
  tests/training/test_wave7_exact_resume_compare.py,
  tests/runtime/test_train_runtime.py,
  scripts/analysis/coordexp_swift_length_isolation.py
→ non-current: docs/history/** (3 files, immutable), docs/superpowers/plans/** (2)
```

**C-11 — spec-coherence and code seams (F-2, F-7, F-8, and (b)/(d) probes)**

```
grep -rn "loss/" openspec/specs/
→ 1 hit: coordexp-swift-training-artifacts/spec.md:295 (prose "eval loss/metric mapping")
  — no stable spec pins per-term loss field names

grep -n "losses\|coord_gaussian_rps\|token_type_gate\|protected" \
  openspec/specs/coordexp-swift-config-runtime/spec.md
→ (no output) — stable config-runtime spec has no loss requirement, so the
  delta's `## ADDED Requirements` is correct and non-conflicting

grep -n "^### Requirement" openspec/specs/coordexp-swift-supervision-losses/spec.md
→ 10 requirements; delta MODIFIES 5; `Non-Finite Loss And Gradient Gates`
  (line 247) is unmodified and keys its scenario on total loss (F-7)

sed -n '85,260p' src/losses/runner.py
→ line ~236: gate_grad_context = torch.no_grad() if self.token_type_gate_weight == 0.0
             else nullcontext()   (existing detached seam for (b))
→ from_config: coord term built only when coord_cfg.weight > 0.0 (existing omit precedent)

grep -n "finite_status\|_finite\|isfinite" src/losses/runner.py
→ line 1158: "terms": {term.name: _finite_label(term.weighted_loss) for term in terms}
  (F-2: finite label keyed on weighted, not raw)

grep -n "_default_qwen_forward\|^from\|^import" src/eval/forward.py
→ lines 16-24 import 7 symbols from src.training.supervised_trainer,
  incl. _default_loss_context, _default_qwen_forward,
  _runtime_loss_denominator_gatherer (F-4 / pre-cost finding I-1)
```

No command hung; nothing was killed; the 20-minute kill budget was not
approached (longest command: 52.92s).
