## Context

See `proposal.md` for the intended outcome. This revision replaces the preliminary module/package guesses after a detailed four-part investigation and lead verification. It is a design for implementation, not a claim that refactoring has been performed.

Fresh state: research base HEAD `73b8b3cc2`; self-rollout HEAD `d6de155fb` (`research: add source Rweak row-cross benchmark`) is clean. The earlier 37-untracked-file warning is superseded. The other eight scoped source worktrees are also now clean at their fresh saved tips (including DORA `ba801de51` and COCO `c00043abc`); their earlier dirty snapshots are superseded. The research base still has pre-existing shared configuration deletions outside this change. The user confirmed all worktrees stopped and subsequently explicitly authorized implementation after one independent review; recheck relevant holders immediately before retirement. Other open changes remain separate unless an exact overlapping contract is reconciled.

Investigation covered native model execution, learning foundations, real producer/package dependencies, artifact/validation ownership, and a lead-owned static import survey over the research base. The survey found 19 `src` subpackages plus entry modules; counts and static imports identify questions, not deletion authority. Representative callable consumers and tests were inspected across the named research worktrees. The excluded bridge-cache worktree was not investigated.

## Goals / Non-Goals

**Goals:** remove repeated caller knowledge; make exact differentiable replay and cheap batched continuation straightforward; preserve scientific choices; shrink historical default code; give each maintained experiment and shared behavior one owner; make current research readable without following branch history.

**Non-Goals:** a universal session/trainer, model-agnostic framework, method/plugin registry, new global run context, blanket removal of inherited training, automatic migration of every historical experiment, new loss/matching semantics, v1 evidence-schema migration, shared-skill edits, GPU research or a performance claim without measurement.

## Decisions

### 1. Native operations below existing adapters

Choose Qwen-native operations consumed independently by research, deterministic inference and packed training. Do not widen `HFBackendSession` to expose models/tensors, gradients, arbitrary hooks and optional tracing through one switch-heavy session. Do not make exact research trajectories construct `PackedSequence`, a `Fa2VarlenPlan`, four-row packed positions, or state-bank identities.

Observed boundaries:

- `src/qwen/forward.py::build_qwen_forward_inputs` is genuinely packed-training-shaped. It remains the packed adapter, not the new exact-replay entry.
- `src/qwen/runtime_loading.py::{QwenLoadOptions,load_qwen_components_from_options}` already offers neutral loading and is reused.
- `src/inference/hf_backend.py::teacher_forced_evidence` explicitly uses inference mode and detached CPU logits. DORA and self-rollout need differentiable device-resident scores.
- HF `_load_hf_components` constructs a `SimpleNamespace` config to call adapter/embedding attachment. Add parameter-oriented entry points at the existing attachment owners instead of requiring research to synthesize an inference/train config.
- `src/inference/execution_model.py` merges/folds and serializes export snapshots. Dynamic adapter attachment is different; cache publication, tied-weight checks and merge order remain there.

Responsibility map (new filenames are proposed implementation placements, not mandatory one-file-per-concept abstractions):

| Owner | Promise / inputs | Remains outside |
| --- | --- | --- |
| Existing `qwen/runtime_loading.py`, `adapters/dora.py`, `qwen/special_token_embeddings.py` | Neutral load options; explicit adapter/embedding attachment and selected named parameters; preserve validation/order | Scientific checkpoint-arm eligibility, optimizer policy, run admission |
| New cohesive `qwen/native.py` | Prepare native image/text tensors; literal token extension; exact positions and causal forward rows; left-padded history alignment | Model lifetime, train/eval mode, gradient disabling, scientific token selection |
| New `qwen/generation.py` | Continue prepared histories with explicit policy/budget/trace needs and stable request association | Cohort, intervention, reward, parsing or metric interpretation |
| New `qwen/inspection.py` | Named Qwen sites, bounded position captures, cloning where mutation requires it, guaranteed hook cleanup | Direction/radius formulas, intervention scope, scientific layer-count expectations |
| Existing `inference/hf_backend.py` | Adapt strict inference requests/results to those mechanics; retain session/history and scored-evidence promises | New research execution policy or general training lifecycle |
| Existing `qwen/forward.py`, `positions.py`, FA2 owners | Preserve actual packed callers and packed positions | Mandatory entry for simple un-packed trajectories |

Keep only files that own substantial behavior. Do not add `composition.py`, `replay.py`, facades or contexts merely to mirror this table. No global model-lifetime manager is needed: actual callers already own the model, processor, device and cleanup.

Conceptual caller sketch (not implemented API):

```python
components = load_qwen_components_from_options(options)
# Attach adapter/embedding settings through their existing owners.
history = prepare_exact_history(components.processor, image, prompt_ids, action_ids)
forward_inputs = exact_forward_inputs(components.model, history)
logits = components.model(**forward_inputs).logits
scores = selected_token_logprobs(aligned_rows(logits), target_ids)
loss = objective(scores, selected_positions, credit, explicit_denominator)
# A continuation caller can instead request trace-free generation.
results = generate_continuations(components.model, histories,
                                 policies=policies, budgets=budgets, trace="none")
```

The new operations remove old implementations after callers migrate: HF native-input/position construction, DORA's old-script position helper and trajectory input plumbing, logit-lens exact-forward copying, row-cross padding/budget handling, and sampled rollout's private-field bypass. Retained HF adapter methods delegate only while their existing contracts require them; no permanent old-script compatibility facade.

### 2. Exact continuation is a real first-class consumer

The latest self-rollout `2026-09-09-source-rweak-row-cross/candidates/eng_beta/run.py::execute` is the generation tracer. It performs literal prefix-plus-action continuation, heterogeneous left padding, per-request remaining budgets, terminal EOS fast paths, and untraced generation. Its reducer checks token/parser/owner correspondence. Those requirements were absent from the first draft.

Public native continuation must preserve exact prefix/action IDs, request association, budget accounting and terminal behavior. A request already terminated or with no remaining budget must not trigger model work for that request; trace-free generation must not collect score/logit/hidden-state tensors. Mixed budgets must not silently adopt one common stopping allowance.

Sampled decode follows after the exact-history contract is stable. DORA's `run_current_seeded_sampled_rollouts.py` differs from the base copy: raw-softmax generation, `top_k=0`, fresh generation config and performance records. Preserve the resolved caller policy, including processor effects and raw versus policy probabilities. Fixed seed and fixed batching is the initial reproducibility scope; stronger invariance across batch sizes/order is deferred, not silently assumed.

Relevant existing contracts remain intact: `coordexp-infras-infer-backend-trace` keeps HF evidence models/native tensors private and full logits unavailable; `coordexp-infras-infer-config-runtime` requires deterministic scored inference. New native research operations coexist below those adapters rather than relaxing them. CPU helper tests do not prove real-model token equivalence.

### 3. Learning foundations share computation, not objectives

Add a low-level scoring operation in `src/losses` that accepts already aligned logits rows and target IDs and returns differentiable per-token log probabilities/CE. No implicit mean, sum, detachment, CPU transfer, world-size factor or token-category inference. Adapt packed `LossContext`/base CE to it where behavior agrees; do not require research to construct packed atoms.

DORA CE uses a token mean scaled by world size/image count; its RLOO uses advantage times a token sum scaled by world size/(image count × K). Self owner-outcome distinguishes coordinate-only and full-action sums with its own denominator. These remain separate objective functions/configurations. Literal row alignment and objective reduction are verified separately, including gradients.

At `src/adapters/dora.py`, expose explicit parameter selection so probes stop importing `_is_dora_adapter_trainable_name` and repeating tower exclusions. The caller controls freezing/enabling and expected scientific counts. At `src/optim/parameter_groups.py`, accept resolved groups without dummy token-embedding configuration. Keep full training config as one adapter to those groups; a small probe may still instantiate native AdamW directly. Preserve parameter order, uniqueness, exact trainable coverage, raw AdamW identity versus prepared execution wrapper and continuation state validation.

Keep `TrainRuntime`, `SupervisedTrainer`, planned-step normalization, finite gates and the concrete packed pipeline. They have actual consumers. Direct DDP trajectory learning remains explicit: its forward/backward `no_sync` scope and last synchronized action must not be replaced by a generic train-step framework.

Experimental Human13 arm composition in `src/losses/human13_k_union.py` belongs with the retained Human13 package where used. Keep genuinely reused tensor math at the losses owner; preserve its strict-crossing and streamed-gradient behavior. Do not turn every historical loss into a supported shared loss registry.

### 4. Evaluation ownership and whole-src disposition

Extract `_global_matches` from `scripts/research/compare_clean_rollout_owner_coverage.py` into `src/eval/assignment.py`. Its contract is category-constrained maximum cardinality, then maximum quantized total IoU, with current deterministic tie behavior. Share the low-level pixel IoU operation through existing geometry ownership rather than making evaluation import visualization. Keep category normalization, dedup order, thresholds, malformed handling and metric denominator explicit at callers.

Do not substitute `src/vis/matching.py::match_row`: it is greedy, and the existing counterexample yields one versus two owners. Fine-grained training costs/soft assignment are a future scientific extension unless their caller contract is actually shared; this extraction does not establish that equivalence. No generic assignment plugin API or new solver dependency is needed.

| Current `src` area | Disposition in this change | Reason / concrete boundary |
| --- | --- | --- |
| `qwen` | Refactor native operations and preserve packed/FA2 adapters | Exact research replay cannot require packing; model-specific image/position behavior remains load-bearing |
| `inference` | Replace duplicated HF mechanics; retain pipeline, workers, vLLM, parsing, scoring and export contracts | Real decode/evaluation consumers and persisted trace/worker identities exist |
| `adapters` | Refine explicit attachment/selection inputs | Eliminate private predicates and synthetic whole configs; preserve checkpoint semantics |
| `losses` | Add aligned scoring; move retained Human13 arm composition; remove unused scaling receipt | Keep reductions and tests of surviving scientific math |
| `optim` | Accept explicit resolved parameter groups | Remove dummy config branches; retain factory/scheduler for actual packed callers |
| `training` / `runtime` | Retain concrete lifecycle; narrow incidental caller setup | Actual Human13 and root training paths, synchronization and raw/prepared optimizer ownership |
| `supervision` / `packing` | Retain packed alignment/serialization | No blanket move; native research simply stops depending on these incidentally |
| `rollout_calibration` | Retain existing state-bank/config behavior; new exact replay does not depend on it | Persisted event/candidate meanings are not generic trajectory inputs |
| `data` / root `coordinate_targets.py` | Retain geometry/token/image semantics; own shared pixel IoU | Shared consumers; no vocabulary/coordinate conversion change |
| `templates` / `augmentation` | Retain current encoding/image callers | Used by data preparation and packed/native flows; no proven benefit from rewriting |
| `config` | Reuse resolution/value validation through `load_research_infer_config`; retain canonical production loader and accepted hash domains | Package profiles keep identical effective values without using debug flags to evade production authoring rules |
| `artifacts` | Keep leaf publication, journal, admission, lazy facade | Existing recovery and public/persisted consumers; make strict paths explicit opt-ins |
| `eval` / `vis` | Add public global assignment; preserve distinct greedy/COCO semantics | Tests distinguish algorithms; presentation is not evaluation authority |
| `analysis` | Historical cleanup candidate only after retained import closure is removed | Base survey found six script consumers; no assumption that scientific helpers are dead |
| `common` and root train/infer entrypoints | Retain | Small shared errors and real supported entrypoints; no independent reason to remove |

This map covers the inherited base but does not schedule speculative rewrites. Source history and dynamic references remain part of exact cut verification. Moving files without removing caller knowledge is not accepted as entropy reduction.

### 5. Four initial direction packages and explicit profiles

The following are the initial maintained executables. A profile is a direction-local config/module specifying a scientific protocol; it is not a global class, registry, strategy hierarchy or separate package per run. Independent units can coexist in one direction without sharing objectives or defaults.

| Package / proposed modules | Selected original entry and closure | Profile boundary |
| --- | --- | --- |
| `probes/dora_owner_learning/`: `prepare.py`, `train.py`, local objective modules, configs/tests | DORA `prepare_source256_ce_rloo_round.py`, `train_source256_ce_rloo_round.py`, selected seeded plan/scoring helpers; retain self owner-outcome coordinate/full-action objective as a separate local module and test consumer | CE, RLOO and owner-outcome keep named configs, masks, credits and denominators; common forward/scoring is in `src`, not sibling script imports |
| `probes/source_rweak_row_cross/`: `prepare.py`, `run.py`, `reduce.py`, local checkpoint checks, configs/tests | Committed accepted `eng_beta/run.py`, scientific manifest preparation and `reduction/reduce.py`; preserve original identities separately | Source/Rweak checkpoint×action cells, selected panel, per-row budgets and direct/suffix attribution; agent benchmark is not an execution profile |
| `probes/logit_lens/`: base capture, causal/radius/continuation entries, configs/tests | `probe_image2299_logit_lens.py` and retained causal-transfer, radius/direction, natural-continuation successors | Layer site, patch formula, fixed-prefix versus natural continuation stay explicit; dynamic base-script import becomes package-local import |
| `probes/human13/`: panel helpers, output-QP and magnitude-DoRA entries, local objectives/runtime, configs/tests | `run_human13_output_qp_same_panel.py`, magnitude finite-overfit/QP runners, required panel/capture/optimizer helpers | Fixed-panel interventions remain separate units; do not merge output-QP and DoRA mathematics or N256 solver identity |

Each entry gets a documented `python -m probes.<direction>.<entry>` command, a minimal real config example and explicit tests. Do not create empty folders for every potential helper. Shared package exports and test discovery are lead-integrated. The first model-free package tracer is row-cross reduction consuming the new matcher; the subsequent generation tracer completes the same package's model route.

Initially historical-only: C trainers, completed COCO trainers/dedup experiments, N256 hash-bound solvers, unfinished row-feedback producer, old unselected support/owner audits and superseded benchmark candidates. Their records all return. Their effective code/config and necessary outputs remain recoverable; running an old route requires restoring its historical version. This deliberately gives up a maintained default-tree launch path for those old producers, not their provenance or evidence. Do not create eight or nine packages merely because that many worktrees exist.

Narrow dependency closures still matter:

- Row-cross runner AND reducer import the COCO worktree at a pinned owner HEAD. The reducer also verifies absolute `sources.code` paths and hashes in its saved manifest. Move required arm/checkpoint validation with this direction, use shared parsing/assignment directly, and explicitly resolve preserved original source bindings without rewriting the manifest or pretending old source executed new code; COCO cannot retire first.
- Self owner-outcome imports DORA absolutely. Migrated objective/scoring consumers must use the new same-checkout owner; archive preserves the original dependency closure.
- N256 imports Human13 helpers and verifies `__file__`/source hashes. Preserve the complete historical closure and receipts; do not reissue old identities for rewritten files.
- DORA/Human13 contain identical Human13 producers; select one source identity rather than duplicate the migration.

### 6. Entropy cuts, rejected cuts and evidence overhead

| Decision | Evidence / net effect | Tradeoff and verifier |
| --- | --- | --- |
| Remove `BackendScalingReceipt`, `validate_planned_step_backend_scaling`, its private divisor helper and dedicated exports/tests | Across all nine in-scope trees, hits are definitions/exports/tests only; no registry or string-dispatch consumer found. Roughly 100 lines plus two dedicated tests, with no replacement runtime layer | Exported unused diagnostic API disappears. Preserve real `validate_accelerator_runtime` neutral-accumulation guard, planned-step math and runtime sequencing tests |
| Replace duplicate native input/position/scoring/padding/hook code | Actual retained callers identified above; migration must delete their old implementations/private imports | Retain exact semantics; compare actual caller outputs/gradients, not only new helpers |
| Simplify optimizer construction | Human13 fills unused embedding optimizer settings to satisfy generic config | Explicit groups replace dummy settings; verify group order/coverage and raw/prepared optimizer identity |
| Keep strict JSON publication and lazy facade | Inference artifact/context/merge consumers; lazy map prevents model/runtime import side effects | Keep occupied-path, strict-value, fsync and import-isolation tests |
| Keep journal and admission as optional advanced capabilities | `resumable_natural_boundary_support_completion.py::{open_slot_journal,execute_slot,materialize_legacy_shard_receipts}` is real recovery; runnable admission vertical and persisted receipts exist | Ordinary probes forgo admission/whole-tree binding/stage recovery unless selected; strict existing paths do not weaken |
| Keep execution-context dual digests in v1 | Byte/value hashes coincide for valid canonical sidecars, but both are in worker transport and persisted records | Defer a schema migration; no new v2 identity system for this cleanup |
| Reject direct config-hash unification | Config hashing accepts tuples and integer-key mappings; artifact hashing rejects them with `artifact.invalid_json_value` | Current hashes and error domains remain; tiny CPU counterexamples confirm non-equivalence |
| Defer generic file-hash centralization and speculative source deletion | Small or uncertain gain relative to caller/compatibility work | Do not grow scope for cosmetic deduplication |

Ordinary producers use explicit config/code revision plus dirty status, input/model locations, seed where relevant, output location and existing JSON helpers. Dirty status does not claim exact replayability; cited reproducible work saves its effective inputs. Do not build a new run-context schema or dynamically select heavyweight machinery behind a profile flag.

Update local branch policy, infra guide, project context and `openspec/config.yaml`: remove the permanent infra lane and implication that every new direction needs clean-tree admission. Shared agent/skill configuration stays untouched; this investigation found no need to weaken its consumer/integrity rules. Correctness checks for masks, positions, gradients, finite values, collision safety and recovery identity remain.

### 7. Knowledge integration and lifecycle

Use existing experiment records, investigation routers, overview, compass and decisions. Save originals, filter inherited copies by ancestry/blob identity, select files by purpose, return results, update navigation, then rewrite current synthesis and necessary route decisions. The compass owns current belief/next-question synthesis; overview owns stable question decomposition and evidence navigation. Remove repeated current prose between them without deleting source facts.

Question groups: finite-panel fit versus transfer; natural owner coverage and training objectives; semantic compression versus matched nulls; row/prefix causality; evaluation/annotation boundaries. Those are reading-path labels, not automatic Python package names. Preserve incompatible populations and metrics; N32 incomplete null evidence, CPU-only feedback preparation and selected row-cross panels cannot become broad scientific conclusions. Separate self-rollout's model-routing benchmark from scientific evidence.

Intake starting records under each worktree's `research/investigations/qwen3-vl-dense-enumeration/`:

| Worktree | Record |
| --- | --- |
| C audit | `experiments/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical/results.md` |
| COCO | `experiments/2026-09-08-coco-owner-recovery-dedup/results.md` and `execution.md` |
| DORA | `2026-09-07-research-frontier-synthesis.md`; `experiments/2026-09-08-row-boundary-continuous-feedback/{unit.md,data-receipt.md}` |
| Human13 | `experiments/2026-09-01-human13-dora-magnitude-finite-overfit/results.md`; direct C/D0 result |
| N256 | `2026-09-01-n256-shared-output-qp-norm-scaling-handoff.md` and its original external receipts |
| Logit-lens | `experiments/2026-09-08-logit-lens-causal-transfer/results.md` and linked successors |
| Self-rollout | `experiments/2026-09-08-owner-outcome-autonomous/results.md`; committed `2026-09-09-source-rweak-row-cross/readout.md` |

Use a short migration disposition section in this change, not a persistent migration database. Preserve relevant dirty source and ignored outputs separately from Git refs. For cited results, verify required evidence can be reached without the retiring directory. Worktree stop, source recovery, execution replay and scientific validity are distinct statuses.

### 8. Parallel implementation ownership

Flat lead plus bounded workers, no subdelegation by default. Reuse worker context only while its responsibility and interfaces remain reliable. Three or four concurrent writers are sufficient; the table describes ownership, not a requirement to spawn every lane at once.

| Package of work | Exclusive write surface | Prerequisite / acceptance dependency |
| --- | --- | --- |
| K: knowledge intake/synthesis | Selected `research/` records and routers; no producer code | Preservation first; can run alongside all code work. One K owner handles compass/index, lead accepts claims |
| E: assignment + offline row-cross reduction | `src/eval/assignment.py`, shared geometry change, row-cross reducer/tests | Frozen match contract and saved inputs; independent of model/GPU work |
| M: native execution | `src/qwen/native.py`, generation/inspection operations, `src/inference/hf_backend.py`, model-route tests | Own all HF extraction. Exact inputs before generation; same owner migrates row-cross runner after E hands back that package's shared files |
| L: learning foundations | `src/losses/`, normalizer cut and tests | Aligned rows/target contract fixed here; can develop in parallel with M; DORA training migration waits for native inputs |
| P: parameter/optimizer inputs | `src/adapters/dora.py`, special embedding attachment entry, `src/optim/` and their tests | Coordinate signatures with M; no second writer in these files. Run alongside L/E or after a slot frees |
| D: direction migrations | DORA, Human13 and logit-lens package files, each one owner | After required M/L/P APIs. Distinct packages may migrate in parallel; their owners do not edit shared `src` |
| Lead integration | Public `__init__` exports, `.gitignore`, `pytest.ini`, local operator docs/config, OpenSpec, Git integration and retirement | Integrate accepted slice deltas; shared files are never concurrently staged/edited by workers |

M may share a prepared-input signature with L without forcing one to wait for all implementation. It must not edit a DORA scorer while L/D owns that producer. E and M may work on disjoint reducer/runner modules only after package-local shared files have an explicit owner; otherwise serialize the handoff. A helper-only pass remains candidate until its real retained consumer is accepted.

## Risks / Trade-offs

- Silent semantic changes from shared reductions or matching → preserve exact existing counterexamples and caller/gradient parity; no default denominators or algorithm substitution.
- HF-derived integer positions created under inference mode can fail differentiable use despite equal values → exercise autograd through the extracted native path.
- Batch padding, EOS, budgets, tracing and RNG interact → real row-cross consumer plus tiny-model sensitivity tests, then bounded model evidence where required.
- Retiring old trainers removes convenient default-tree launches → preserve exact versions and dependency closures; document historical recovery instead of adding shims.
- Strict infrastructure remains sizeable → existing recovery/persisted obligations justify it; ordinary callers avoid the burden without a parallel framework.
- CPU checks cannot establish GPU numerical parity or speed → report baseline scope and require a bounded real-model check only for the changed boundary; no broad experiment rerun.
- Other dirty work or outputs can be lost → explicit save/disposition, protected unrelated changes and final holder/accessibility checks.

## Migration Plan

1. Preserve effective code/config/output dependencies and reconcile overlapping unfinished records.
2. Start K plus E; establish the offline row-cross reducer and assignment consumer without model work.
3. M/L/P proceed on disjoint surfaces after the interface boundaries above are fixed. Update local defaults and remove the unused scaling receipt with its real guard preserved.
4. M completes row-cross generation; migrate DORA, Human13 and logit-lens with explicit profiles and tests. Remove replaced source implementations after each consumer passes.
5. Finalize current knowledge; remove preserved unselected historical producers/configs/tests with exact closure checks; retire individually eligible worktrees and former infra lane.

Rollback uses scoped commits or saved historical versions in isolated checkouts. Do not edit old receipts to identify new code. Performance acceptance compares model-load/forward count, selected-logit payload and actual timing only when measured; net-negative lines alone do not prove success.

## Investigation Verification

These are current-code baselines and design falsification evidence, not implementation acceptance:

- Lead: `python -m pytest -q tests/research/test_compare_clean_rollout_owner_coverage.py` — 6 passed; contains the global-versus-greedy cardinality counterexample.
- Learning worker: `python -m pytest -q tests/losses/test_context_and_terms.py tests/losses/test_normalizers.py tests/research/test_human13_adamw_runtime_ownership.py` — 28 passed.
- Lead: targeted actual accumulation, runtime order and real Accelerate CPU AdamW ownership checks — 3 passed; source and all-nine-tree unused-receipt references verified independently.
- Model worker at self-rollout `d6de155fb`: `python -m pytest -q -p no:cacheprovider .../source-rweak-row-cross/candidates/eng_beta/test_run.py` with bytecode disabled — 6 passed. Covers helpers and saved parser/matcher parity, not model execution.
- Artifact worker: four existing facade/isolation/publication/absent-context checks — 4 passed. Lead inspected the concrete resumable journal caller and replayed config/artifact hash-domain counterexamples.
- No GPU model, new scientific run, or implementation edit occurred during the investigation. OpenSpec validation is reported separately after this revision.

## Deferred Decisions

Stronger sampled invariance across batch sizes/order, a model-family abstraction, v1 execution-context schema simplification and new soft/fine-grained assignment semantics are outside this implementation contract. Exact final internal filenames may change without adding public concepts or changing the above boundaries. A needed real-model smoke receives an explicit checkpoint/input/forward/token/memory/time bound before launch; this plan grants no GPU budget.

## Design Review and Implementation Authorization

The user requested one independent reviewer and explicitly authorized implementation after convergence. The independent Astra high review found no blocking design issue; it confirmed native-gradient, assignment, generation, writer-ownership and preservation boundaries. The lead accepted the design after carrying forward the row-cross absolute source-binding check and fresh clean-source state. This is the sole delegated design review for this target; scoped corrections are checked by the lead. No implementation or scientific acceptance is implied by this design receipt.

## Implementation refinements within the accepted boundary

Real package entry checks exposed `config.noncanonical_infer_path`: a copied Source YAML with debug disabled could not load outside the production directory. The public research loading entry shares the existing resolver/schema/fingerprint implementation and omits only production namespace/leaf-authorship rules. Canonical inference remains unchanged; tests demonstrate equal effective configuration/fingerprint and unchanged debug flags, inherited research profiles and continued unknown-field rejection. This replaces per-package parsing or debug-mode workarounds.

New Human13 finite receipts use an explicit v2 producer marker to permit newly executed N2 artifacts under their content/payload/stage/count contract. Legacy v1/schema-absent N2 keeps its original pinned digest, even if a stage field is added. New producer source metadata is descriptive: the loader does not require that source path to still exist. This implements the already accepted separation of preserved historical identities from new executions.
