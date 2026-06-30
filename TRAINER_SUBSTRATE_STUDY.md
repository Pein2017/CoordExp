---
doc_id: docs.architecture.coordexp-swift-trainer-substrate-study
layer: docs
doc_type: architecture-research
status: proposal
domain: architecture
summary: Evidence-grounded study — HF Trainer vs manual/local training loop for the CoordExp-swift Stage-1 Qwen3-VL packed trainer.
updated: 2026-06-27
---

# CoordExp-Swift: HF `Trainer` vs Manual/Local Training Loop — Architecture Study

Read-only research study for the `codex/coordexp-swift` worktree. It independently
re-derives the trainer-substrate question already recorded in
[`DECISIONS.md`](DECISIONS.md) ("Training Loop, Checkpointing, And Trainer
Substrate") and pressure-tests it against the actual installed framework source.
This is proposal-scoped: not current-behavior authority and not an implementation
checklist by itself.

## 0. Scope, Method, and Evidence Base

The study compares three concrete strategies for the Stage-1 packed Qwen3-VL
trainer, under the 15 design constraints in the request:

- **A — HF `Trainer` subclass:** keep `transformers.Trainer`/`Seq2SeqTrainer` as
  the production base class, override `compute_loss`, `get_train_dataloader`, the
  collator, and callbacks. (This is materially what the current CoordExp stack
  does today *through* ms-swift.)
- **B — Manual loop (bare-ish PyTorch):** hand-roll the loop, DDP, autocast,
  grad-scaling, clipping, optimizer/scheduler, checkpointing.
- **C — Hybrid: manual semantic loop on Accelerate/DeepSpeed primitives
  (recommended).** CoordExp owns the loop, loss, packing, supervision, metrics,
  and checkpoint metadata; Accelerate (and, when configured, DeepSpeed) owns
  device placement, distributed wrapping, mixed precision, grad scaling,
  backward, and clipping; `transformers` is reused *only* for the Qwen3-VL model
  and processor.

**Evidence scope: `partial` (source-grounded, no training/parity run).** All
framework claims are cited to installed source, not memory:

- `transformers==4.57.1` at
  `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`
- `accelerate==1.10.1`, `torch==2.9.1+cu128` (conda env `ms`)
- Current CoordExp trainer stack: `ms-swift` (`swift==3.10.0.dev0`)
  `Seq2SeqTrainer` + an 8-mixin dynamic MRO (`src/bootstrap/trainer_setup.py`,
  `src/sft.py`, `src/trainers/`).

Citations use `path:line`. No GPU, no smoke run, and no FA2 build were exercised;
numeric correctness of any proposed loss is asserted from source semantics, not
measured.

### The single most important framing

A large part of the "Trainer vs manual" debate dissolves once you separate two
layers that the question tends to conflate:

1. **Model-input construction** (packing, `position_ids`, `cu_seq_lens_q/k`,
   `pixel_values`, `image_grid_thw`, FA2 varlen kwargs). In `transformers`,
   training feeds the model via `model(**inputs)` — `trainer.py:4110`. Whatever
   the collator places in `inputs` flows straight through, **identically under
   HF Trainer or a manual loop.** Packed varlen is a *model-input* concern, not a
   *training-loop* concern.
2. **Loss + reduction + step orchestration** (who computes the loss, what the
   denominator is, how gradient accumulation scales it, when the optimizer
   steps). This *is* where HF Trainer imposes opinions — and where CoordExp's
   constraints (segment-balanced length-invariant reduction, canonical
   `TokenSequence`, multiple token-wise auxiliary terms, no `labels`-as-truth)
   collide with those opinions.

The verdict below turns almost entirely on layer 2.

---

## 1. Executive Verdict

**Recommendation: Strategy C — a manual/local training loop built on Accelerate
primitives (DeepSpeed via Accelerate's plugin when configured), reusing
`transformers` only for the Qwen3-VL model/processor. Do *not* adopt
`transformers.Trainer` as the production base class.** In the request's binary
this is the "manual/local Trainer" answer; it is "hybrid" only in that it
deliberately *imports* Accelerate/DeepSpeed/PyTorch systems-layer machinery
instead of reinventing it. This confirms the resolved direction in `DECISIONS.md`.

**Main reason (one paragraph).** HF `Trainer` is organized around a contract that
CoordExp-swift has explicitly rejected: a padded batch of independent examples,
a single dense `labels` tensor as the source of loss truth, the model owning the
loss, and a token-count denominator. Every one of those is load-bearing in
`Trainer`'s hot path. `compute_loss` only diverts `labels` away from the model if
a `label_smoother` or `compute_loss_func` is set (`trainer.py:4101-4104`);
gradient-accumulation correctness is gated on a three-way interaction between
`model_accepts_loss_kwargs`, `num_items_in_batch`, and `compute_loss_func`
(`trainer.py:4059-4064`); and `num_items_in_batch` is computed as
`sum(labels.ne(-100))` with an explicit "we don't support object detection"
caveat (`trainer.py:5590-5640`). CoordExp's golden rule — segment-balanced,
length-invariant reduction over supervised *sites/segments*, with `labels` demoted
to a derived debug artifact — is therefore not a clean override of `Trainer`; it
is a *negation* of `Trainer`'s defaults that must be re-fought on every loss term.
The repo already pays this tax: `src/trainers/metrics/coord_losses.py:20-160` is a
~140-line `compute_loss` whose main job is to recompute `num_items_in_batch`
(72-87) and *manually re-divide by `gradient_accumulation_steps`* (133-158) to
undo the very branch at `trainer.py:4060-4064`. By contrast, the systems-layer
machinery people fear hand-rolling — DDP/FSDP/DeepSpeed wrapping, bf16 autocast,
grad scaling, `backward`, grad clipping — is *not* owned by `Trainer`; it is owned
by **Accelerate**, which a manual loop calls directly. The manual loop reimplements
only the ~100-line orchestration skeleton (`trainer.py:2618-2756`), and because
the project explicitly drops exact optimizer/RNG resume (constraint 14), it sheds
the single hardest part of that skeleton. Net: the manual loop makes the
*non-standard* parts (which are the whole point of CoordExp-swift) first-class and
cheap, while delegating the *standard* parts to the same upstream primitives
`Trainer` itself uses.

---

## 2. Comparison Matrix

Columns: **A = HF `Trainer` subclass**, **B = Manual loop (bare PyTorch)**,
**C = Hybrid: manual loop on Accelerate/DeepSpeed (recommended)**.

| Dimension | A — HF `Trainer` subclass | B — Manual loop (bare) | C — Manual loop on Accelerate (recommended) |
|---|---|---|---|
| **Customization** | Override points exist but are shallow seams over an opinionated batch/labels core; rich per-token metadata cannot reach `compute_loss_func` (only `(outputs, labels, num_items)`, `trainer.py:4117-4127`). You fight defaults. | Total freedom; nothing in the way. | Total freedom at the semantic layer; Accelerate constrains only systems mechanics you *want* delegated. **Best fit.** |
| **Work effort (initial)** | Low to first run *if* standard CE; rises sharply per non-standard loss term (see current `coord_losses.py`). | High — must build precision/distributed correctness yourself. | Medium — ~100-line loop skeleton + Accelerate calls; no precision/DDP reinvention. |
| **Distributed / precision safety** | High (delegated to Accelerate *inside* Trainer) but **opaque** — failures surface as Trainer-internal behavior. | **Low / dangerous** — bf16 autocast, grad scaler skip handling, ZeRO, FSDP state-dict are easy to get subtly wrong. | High **and** legible — same Accelerate primitives, called explicitly (`accelerator.backward`, `clip_grad_norm_`, `prepare`). |
| **Packing support (FA2 varlen)** | Works (kwargs flow via `model(**inputs)`), but column-stripping and `labels`-centric collation get in the way; varlen is *your* code anyway. | You own it fully. | You own it fully and explicitly via `PackedLayout → QwenForwardInputs`; identical model call. **No Trainer penalty avoided here either way.** |
| **Custom loss support** | **Poor fit.** `labels`/`num_items`/GA tri-conditional actively fights segment-balanced reduction and multi-term token-wise losses. | Free. | Free; `LossRunner` consumes logits + `TokenSequence`, owns the causal shift and reduction. **Best fit.** |
| **Checkpoint / eval / logging** | **Free and mature** — rotation, `save_steps=0.2` ratio (`training_args.py:357,429`), FSDP/DeepSpeed save branches (`trainer.py:4205-4224`), callbacks, TB/W&B. Biggest single argument for A. | All reimplemented from zero. | Reimplement a *thin* slice: weights-only save + metadata, milestone scheduler, logger fan-out. Smaller because resume is out of scope (constraint 14). |
| **Debugging** | Hard — stack traces dive through Trainer + ms-swift; behavior emerges from flags and MRO. | Easy in principle, but precision/distributed bugs are silent. | **Easiest end-to-end** — one readable loop; systems bugs still possible but localized to explicit Accelerate calls. |
| **Agent-readability** | Low — execution flow hidden behind framework dispatch, callbacks, signature-introspection (`model_accepts_loss_kwargs`, `trainer.py:636-639`). | High but cluttered with boilerplate. | **High and intentional** — `python -m src train` → explicit `TrainStep` → `LossRunner`; matches the request's "expose the full execution flow." |
| **Future Stage-2 support** | Rollout-aware supervision must be smuggled through `compute_loss` overrides and collator hacks (current `stage2_rollout_correction_impl.py:5603+` overrides `compute_loss`, bypasses `training_step`). | Free but unstructured. | **Best** — Stage-2 enters as another supervision producer feeding the same `TokenSequence`/`LossRunner`; loop untouched (per `DECISIONS.md` "Stage-2 Readiness"). |

**Reading the matrix:** A wins exactly one row outright — checkpoint/eval/logging
maturity — and that advantage is blunted by (a) the project explicitly not needing
exact resume and (b) `save_steps=0.2`-style cadence being a dozen lines to
reproduce. B loses on the one row that matters most for trust: distributed/
precision safety. C wins or ties everywhere that the CoordExp constraints are
non-standard, which is most of them.

---

## 3. Critical Blockers

### 3a. Hardest issues for **HF `Trainer`** under these constraints

1. **`labels` is the loss substrate; CoordExp says it must not be.** `Trainer`
   computes loss from the model's own `labels`-driven loss unless you set a
   `label_smoother` or `compute_loss_func` (`trainer.py:4101-4104`, `4128-4147`).
   `DECISIONS.md` demotes dense `labels` to "derived compatibility and debugging
   artifacts" and makes `TokenSequence` canonical. To honor that inside Trainer
   you must override `compute_loss` *and* stop the model from computing loss
   (don't pass `labels`) *and* still satisfy the parts of Trainer that read
   `labels` (e.g. `num_items`). You end up maintaining two parallel truths.

2. **The gradient-accumulation normalization trap.** `training_step` divides the
   loss by `current_gradient_accumulation_steps` **only** when
   `(not model_accepts_loss_kwargs or num_items_in_batch is None) and
   compute_loss_func is None` (`trainer.py:4059-4064`). Qwen3-VL's `forward`
   accepts `**kwargs`, so `model_accepts_loss_kwargs` auto-detects `True`
   (`trainer.py:636-639`) and the division is **skipped**, on the assumption your
   loss already divided by a token count. A segment-balanced reducer does *not*
   divide by a token count, so the scaling silently disagrees with the GA window.
   This is not hypothetical — `coord_losses.py:133-158` re-implements the missing
   division by hand, with a five-line comment explaining the 4.57 behavior. Every
   future research loss inherits this trap.

3. **`num_items_in_batch` assumes standard token labels.**
   `_get_num_items_in_batch` is literally
   `sum((batch["labels"].ne(-100)).sum() ...)` and requires a `labels` key
   (`trainer.py:5590-5640`), with the candid comment "For now we don't support
   object detection." This is the token-weighted denominator CoordExp's golden
   rule forbids; using it reintroduces length bias.

4. **Metadata starvation at the official seam.** The clean, documented hook —
   `compute_loss_func(outputs, labels, num_items_in_batch)` — receives only model
   outputs, one `labels` tensor, and a scalar (`trainer.py:4117-4127`). It cannot
   see `segment_ids`, `role_ids`, `span_ids`, provenance, or soft targets. To get
   `LossContext` metadata in, you must either (a) pack it into `labels`/extra
   tensors and smuggle through the collator (the model would reject unknown
   kwargs unless `remove_unused_columns=False` and the collator is bespoke), or
   (b) abandon the official hook and override `compute_loss` reading `inputs`
   directly. Both mean you are no longer using the supported customization story.

5. **One-packed-sequence-per-step fights the batch abstraction.** `Trainer`
   builds a `RandomSampler`/`LengthGroupedSampler` and a `DataLoader` with
   `batch_size=_train_batch_size` (`trainer.py:1053-1146`). Forcing the "real unit
   is one pack" requires `per_device_train_batch_size=1` + a passthrough collator
   + packing pushed into the dataset, and `IterableDataset` packing needs you to
   own rank-sharding because `accelerator.prepare(DataLoader(...))`
   (`trainer.py:1117`) shards differently for iterable inputs. You can do it (it's
   what ms-swift does), but the batch concept never stops being load-bearing
   underneath you.

6. **It is a deep stack to debug, and you do not own it.** Today's production
   trainer is ms-swift `Seq2SeqTrainer` + 8 mixins assembled by dynamic
   `type(...)` MRO (`src/bootstrap/trainer_setup.py:64-111`,
   `src/sft.py:4167-4187`). Behavior emerges from flag/MRO interactions; this is
   precisely the "monolithic trainer override" `DECISIONS.md` wants to escape and
   the opposite of "agent-readable execution flow."

### 3b. Hardest issues for the **manual/local loop**

1. **Mixed-precision + grad-scaling correctness.** bf16 vs fp16 differ: fp16 needs
   a `GradScaler` and a *skip-aware* scheduler step. `Trainer` guards
   `lr_scheduler.step()` behind `not accelerator.optimizer_step_was_skipped`
   (`trainer.py:2747-2750`). A naive manual loop that steps the scheduler on
   skipped fp16 steps drifts the LR schedule. **Mitigation:** prefer bf16 (no
   scaler) and route fp16 through Accelerate, which exposes the same skip flag.

2. **Gradient clipping under ZeRO/FSDP.** `nn.utils.clip_grad_norm_` on raw
   parameters is wrong under DeepSpeed/FSDP sharding; `Trainer` calls
   `accelerator.clip_grad_norm_` (`trainer.py:2704-2717`) which dispatches to the
   engine. A manual loop must do the same — **not** call the torch util directly.

3. **DeepSpeed/FSDP config + sharded save.** ZeRO-3 partitions weights; saving
   needs `stage3_gather_16bit_weights_on_model_save` or `zero_to_fp32` recovery
   (`trainer.py:4205-4224`). Accelerate's `DeepSpeedPlugin`/`FsdpPlugin` +
   `accelerator.get_state_dict(model)` reproduce this, but you must wire and
   *test* it; getting it wrong yields silently truncated checkpoints.

4. **Gradient-accumulation loss scaling — the same trap, your side now.** A
   manual loop must pick one explicit, documented convention (e.g. "each
   `TrainStep` produces a segment-balanced mean; accumulate `sum(step_losses) /
   accumulation_steps`; `backward` per micro-step under
   `accelerator.accumulate(model)`"). The danger is mismatched denominators
   between the logged loss, the backward-ed loss, and the eval loss — exactly the
   inconsistency `coord_losses.py` had to patch.

5. **Distributed metric reduction.** CE/top-k accuracy must be reduced across
   ranks with correct weighting (`average_tokens_across_devices`-style gather,
   `trainer.py:4149-4154`, `5618-5624`). Hand-rolled all-reduces with the wrong
   denominator give plausible-but-wrong numbers — a silent-correctness risk.

6. **No callback ecosystem.** TB/W&B, early stopping, instability monitors
   (`src/trainers/monitoring/instability.py`) come "free" with Trainer's callback
   system. A manual loop needs a minimal hook surface or inline calls. This is
   small but real recurring effort.

**Crucial asymmetry:** blockers 1–3 and 5 are all *delegated to Accelerate* in
Strategy C, which is why C — not B — is the recommendation. The residual manual
burden is blocker 4 (a design decision you make once and test once) and blocker 6
(a thin hook surface). The project's drop of exact-resume (constraint 14) removes
the worst manual-loop blocker entirely: no dataloader fast-forward, no
RNG/optimizer-state restore, no `_load_rng_state` (`trainer.py:2604,2660,3231`).

---

## 4. Recommended Architecture

This refines, and is consistent with, the module responsibilities already in
`DECISIONS.md`. Interfaces are **deep**: a small calling surface over substantial
behavior. Names are provisional (subject to the worktree's approval-card
discipline).

### 4.1 Module hierarchy (delta over `DECISIONS.md` layout)

```text
src/
  __main__.py            # dispatch only: train / inspect-* / trace
  commands/              # thin command handlers
  config/                # typed config, YAML inheritance, resolution, fingerprints
  data/                  # RawExample, JSONL+image+geometry validation
  templates/             # RenderedExample (pure text/messages + semantic spans)
  qwen/                  # Qwen3-VL encoder: processor, tokenization, span align,
                         #   EncodedExample, QwenForwardInputs, FA2 varlen tensors
  packing/               # PackedSequenceBuilder -> PackedSequence + PackedLayout
  supervision/           # TokenAtom / TokenSpan / TokenSequence, remap, validate
  losses/                # LossContext, LossRunner, BaseTokenCE, LossTerm, LossResult
  training/              # TrainStep loop, Engine (Accelerate), checkpoint sched.
  metrics/               # typed metric/event records, CE + top-1/top-5
  artifacts/             # run traces, manifests, fingerprints
  cache/                 # staged caches (raw/rendered/encoded/packplan)
  rollouts/              # reserved for Stage-2 (no impl in V1)
```

The only structural addition beyond `DECISIONS.md` is naming an explicit
**`training/engine.py`** seam (the Accelerate boundary) so that "where do
distributed/precision mechanics live" has one obvious home, and the loop itself
stays pure CoordExp semantics.

### 4.2 Deep interfaces (signatures are illustrative, not approval-final)

**(a) Data example encoding — `qwen/encoder.py`**
```python
class QwenStage1Encoder:
    def encode(self, rendered: RenderedExample) -> EncodedExample: ...
    # owns: processor(do_resize=False), placeholder expansion, image_grid_thw,
    #   span->token alignment, pack-cost fields. Fails fast on mismatch.
```
Caller surface = one method. Behind it: processor calls, placeholder/grid
validation, span alignment, `effective_pack_cost`.

**(b) Packed sequence construction — `packing/builder.py`**
```python
class PackedSequenceBuilder:
    def plan(self, examples: Iterable[EncodedExample]) -> Iterator[PackPlan]: ...
    def build(self, plan: PackPlan) -> PackedSequence: ...
    # PackedSequence carries: PackedLayout (model-agnostic) + TokenSequence
    #   (physical positions) ; greedy admission under global_max_length.
```
`PackedLayout` owns segment boundaries, `segment_ids`, `logical_example_ids`,
`physical_position`, segment ownership (for length-invariant reduction). It is
**model-agnostic** — no MRoPE/grid/varlen state (per `DECISIONS.md`).

**(c) Model input assembly — `qwen/forward_inputs.py`**
```python
class QwenForwardInputsBuilder:
    def build(self, packed: PackedSequence) -> QwenForwardInputs: ...
    # derives input_ids, pixel_values, image_grid_thw, 3D/4-row MRoPE
    #   position_ids, and FA2 varlen: cu_seq_lens_q/k (int32), max_length_q/k (int)
    #   from PackedLayout; validates against it.
```
This is where the **only** real packed-varlen work lives — and it is identical
whether or not you use HF Trainer. Qwen3-VL threads `cu_seq_lens_q/k` and
`FlashAttentionKwargs` through every decoder layer
(`modeling_qwen3_vl.py:416-423,784-796`); its `position_ids` are 3D/4-row MRoPE
(`modeling_qwen3_vl.py:321-324,823-830`), so you **must** pass explicit
`cu_seq_lens` (the position-id-inference path
`modeling_flash_attention_utils.py:318-357` does not apply to MRoPE). Mirror
`transformers`' own invariants: int32 cumulative lengths, Python-int max lengths.

**(d) Loss computation — `losses/runner.py`**
```python
class LossRunner:
    def __call__(self, surfaces: ModelSurfaces,
                 packed: PackedSequence) -> LossResult: ...
    # installs BaseTokenCE; compiles LossContext (target_positions,
    #   logits_positions = target-1, primary_token_ids, segment_ids, role_ids,
    #   span_ids, optional sparse aux targets); runs LossTerms; segment-balanced
    #   reduction; emits total + per-term + metrics + diagnostics.

class LossTerm(Protocol):
    name: str
    def __call__(self, ctx: LossContext) -> TermOutput: ...   # tensor selections only
```
This is the interface HF Trainer cannot offer cleanly: the runner sees full
logits *and* full physical supervision metadata, owns the causal shift, and owns
the denominator. No `labels` round-trip, no `num_items` guessing.

**(e) Metrics — `metrics/`**
```python
class MetricSink:
    def record(self, event: MetricEvent) -> None: ...
class StepMetrics:        # CE, top-1, top-5, per-role/per-span slices
    def reduce_across_ranks(self, accelerator) -> dict[str, float]: ...
```
First-class CE + top-1/top-5 (constraint 12), computed from the same forward
logits the loss uses, reduced with explicit, correct denominators.

**(f) Training step — `training/loop.py` + `training/engine.py`**
```python
class TrainEngine:                      # the Accelerate boundary (deep)
    def prepare(self, model, optimizer, scheduler, dataloader): ...
    def backward(self, loss): ...        # -> accelerator.backward
    def clip_grad_norm(self, max_norm): ...   # -> accelerator.clip_grad_norm_
    def optimizer_step(self, optimizer, scheduler): ...  # skip-aware sched step
    @property
    def is_step_skipped(self) -> bool: ...    # optimizer_step_was_skipped

def run_training(cfg, engine, data, loss_runner, sched) -> RunResult:
    for step in train_steps(data):          # TrainStep == one PackedSequence/rank
        with engine.accumulate():
            surfaces = model(**step.qwen_inputs.as_kwargs())   # no labels
            result  = loss_runner(surfaces, step.packed)
            engine.backward(result.total)
        if engine.sync_step:
            engine.clip_grad_norm(cfg.max_grad_norm)
            engine.optimizer_step(optimizer, scheduler)
            metrics.flush(step, result)
        scheduler_ckpt.maybe_save_eval(step)   # milestone-relative
```
The loop body is the *entire* Stage-1 execution flow, visible in one screen —
the explicit analogue of `trainer.py:2618-2756`, minus resume bookkeeping.

**(g) Checkpoint / eval scheduling — `training/schedule.py`**
```python
class MilestoneSchedule:
    @classmethod
    def resolve(cls, max_steps: int, fractions: list[float]) -> "MilestoneSchedule": ...
    def is_save_step(self, step: int) -> bool: ...
    def is_eval_step(self, step: int) -> bool: ...
    # materializes absolute steps from e.g. [0.2,0.4,0.6,0.8,1.0] BEFORE training
    #   and writes them to run artifacts (DECISIONS.md requirement).
```
This is *better* than Trainer's implicit `save_steps=0.2`: it emits a concrete
`save_eval_schedule.json` up front, satisfying the reproducibility requirement
that Trainer only resolves internally.

---

## 5. Implementation Effort Estimate

Buckets: **S ≈ 0.5–2 person-days, M ≈ 3–6, L ≈ 7–12.** Assumes one experienced
engineer, bf16, exact-resume out of scope (constraint 14), single dataset JSONL.

| Deliverable | Strategy C (recommended) | For contrast: A (HF Trainer) |
|---|---|---|
| **Stage-1 single-GPU smoke trainer** (data→pack→forward→BaseTokenCE→step) | **M** — most effort is `qwen/` encoding + `packing/` varlen, which A also needs. | M — similar, *plus* fighting `labels`/`num_items` to get a custom denominator even for v1. |
| **Multi-GPU / DeepSpeed** | **S–M** — `accelerator.prepare` + a `DeepSpeedPlugin`; the loop is unchanged. Cost is config wiring + one sharded-save test. | S — "free" from Trainer, *if* you accept its save/loss conventions. |
| **Checkpoint / eval / logging** | **M** — weights-only save + metadata, `MilestoneSchedule`, logger fan-out, rank-0 guards. Smaller without resume. | S — mostly free (rotation, ratio cadence, callbacks). **A's real advantage.** |
| **Robust tests** | **M** — deterministic segment-balanced 1-step test (unequal segments), FA2 varlen `cu_seq_lens` validation + fail-fast, tiny CE parity vs an HF reference, schedule-artifact test. (These are exactly the V1 checks `DECISIONS.md` already lists.) | M — plus tests that pin down which Trainer flags you depend on, because upgrades move them. |
| **Net first milestone** | **L total (~8–11 pd)** | "L-minus" up front, but with a *recurring* per-loss tax and version-fragility. |

The decisive long-run number is **cost of research loss #3, #4, #5**: under C a new
token-wise term is a `LossTerm` selecting on `role_ids`/`span_ids` from
`LossContext` (S, often <0.5 pd). Under A each new term re-enters the
`labels`/`num_items`/GA-scaling minefield (today's `coord_losses.py` is the
existence proof that this is M-each and bug-prone).

---

## 6. Risk Register

### Silent-correctness risks (highest priority — these produce plausible-but-wrong numbers)

| Risk | Where | Mitigation |
|---|---|---|
| **GA loss-scaling denominator mismatch** between backward-ed, logged, and eval loss | the manual analogue of `trainer.py:4059-4064` | One documented convention in `LossRunner`/loop; a deterministic 2-microstep test asserting `loss(GA=2) == mean(loss(GA=1) pair)`. |
| **Length bias re-entering** via a token-count denominator | reduction policy | Default segment-balanced reducer; assert length-invariance in a unit test with unequal segments; never divide by raw/padded length. |
| **FA2 varlen segment leakage** (later pack segments attending to earlier) | `QwenForwardInputs` cu_seq_lens | Validate `cu_seq_lens_q/k` against `PackedLayout`; fail fast if multi-segment pack lacks varlen tensors; parity test vs unpacked single-example forward (logits within tolerance). |
| **Cross-rank metric mis-weighting** | `metrics.reduce_across_ranks` | Gather token/segment counts, not pre-averaged means; test 2-rank reduction equals single-rank over concatenated data. |
| **Image placeholder / `image_grid_thw` mismatch** | `qwen/` encode | Strict count check before forward (constraint), with example-id-rich error. |
| **bf16/fp16 scheduler drift** on skipped steps | `engine.optimizer_step` | Use Accelerate's `optimizer_step_was_skipped`; prefer bf16. |

### Performance risks

- **Pack admission on text length alone** under-counts visual burden → OOM or
  under-fill. Use the multi-axis `effective_pack_cost` (`DECISIONS.md`:
  `physical_input_length` + `vision_feature_row_count`).
- **Single-pack-per-step with small packs** under-utilizes the GPU; rely on
  `global_max_length` growth + greedy admission, not micro-batching.
- **DataLoader/rank-sharding** for an iterable packed stream done wrong →
  duplicated or skipped data across ranks; test disjoint coverage.

### Future-refactor risks

- **`transformers` API drift** (the model side you keep): Qwen3-VL forward
  kwargs, `FlashAttentionKwargs`, processor behavior, and MRoPE
  (`get_rope_index`) can change between minor versions. **Mitigation:** isolate
  all of it in `qwen/`; pin `transformers`; one "model contract" test.
- **Stage-2 pressure on `PackedLayout`/`TokenSequence`:** keep them
  source-agnostic and provenance-aware now (cheap) so rollout supervision is
  additive, not a refactor (per `DECISIONS.md` "Stage-2 Readiness").
- **The `LossRecipe`/`LossContext` boundary** ossifying: keep `LossContext`
  tensor-first and `LossTerm` a thin protocol; resist a public recipe ontology in
  V1 (`DECISIONS.md` agrees).
- **Reject-but-credible fallback:** if the manual loop ever proves more fragile
  than an adapter, a *thin* HF `Trainer` subclass remains a documented retreat —
  but only as a CE-only/parity oracle, never owning the data path, `labels`
  semantics, or loss denominator.

---

## 7. Final Recommendation

**Do this:** Build Strategy C — a local explicit Stage-1 loop on Accelerate,
reusing `transformers` for Qwen3-VL only. This matches and validates the resolved
position in `DECISIONS.md`; this study's contribution is the source-level evidence
that the decision is correct *and* the precise delegation boundary below.

### Concrete next steps
1. Write the first `training/` **approval card** stating the delegate / mimic /
   omit split (below) — `DECISIONS.md` already names this as the gate.
2. Land the **vertical smoke slice first**: one JSONL row →
   `RawExample → RenderedExample → EncodedExample → PackedSequence +
   QwenForwardInputs → forward → BaseTokenCE → one `accelerator.backward` → step`,
   single GPU, bf16. This de-risks `qwen/` + `packing/` (the genuinely hard,
   substrate-independent part) before any distributed work.
3. Add the four V1 verification tests verbatim from `DECISIONS.md` (deterministic
   segment-balanced loss; tiny CE parity vs an HF reference forward; FA2 varlen
   `cu_seq_lens` validation + fail-fast; resolved milestone schedule artifact).
4. Only then add the `DeepSpeedPlugin`/multi-GPU path and the sharded-save test.

### What to **mimic** from HF `Trainer` (copy the behavior, own the code)
- The GA-window structure: accumulate over micro-steps, `sync` on the boundary,
  clip-then-step-then-sched (`trainer.py:2618-2756`).
- **Skip-aware** scheduler stepping (`trainer.py:2747-2750`).
- Milestone cadence semantics of `save_steps`/`eval_steps` as a fraction of
  `max_steps` (`training_args.py:357,429`) — but materialize absolute steps to an
  artifact up front.
- `save_model`'s distributed-save discipline (rank guards, FSDP/DeepSpeed
  state-dict gather, `trainer.py:4205-4227`) — reproduce via Accelerate.
- The instability/early-stop hook idea (a minimal callback surface, not Trainer's
  full `TrainerCallback` system).

### What to **import / reuse** (do not reinvent)
- **Accelerate** for `prepare`, `backward`, `clip_grad_norm_`, autocast/precision,
  DDP/FSDP/DeepSpeed wrapping, `optimizer_step_was_skipped`, cross-rank gather.
- **DeepSpeed via Accelerate's `DeepSpeedPlugin`** for ZeRO (no hand-rolled ZeRO).
- **`transformers`** for `Qwen3VLForConditionalGeneration`, `AutoProcessor`,
  `get_rope_index`, and the FA2 attention path
  (`attn_implementation="flash_attention_2"`).
- **`transformers` FA2 varlen invariants** as a spec to match (int32 `cu_seq_lens`,
  Python-int max lengths, `modeling_flash_attention_utils.py:337-357`) — and
  `DataCollatorWithFlattening` (`data/data_collator.py:2092`) as a *reference
  implementation* of per-segment `position_ids`/`cu_seq_lens` to read, not to use
  (it is `labels`/2D-RoPE shaped, not MRoPE).
- Standard PyTorch optimizer/scheduler builders.

### What to **intentionally leave out** (V1)
- `transformers.Trainer`/`Seq2SeqTrainer` as a base class, and the entire
  `compute_loss` / `compute_loss_func` / `num_items_in_batch` /
  `model_accepts_loss_kwargs` contract.
- Dense `labels` as loss truth (keep only as a derived debug/parity artifact).
- Exact optimizer/RNG/dataloader **resume** (`_load_rng_state`, fast-forward).
- ms-swift entirely as a training dependency (reference only).
- The `TrainerCallback`/integrations ecosystem, the padded multi-example batch,
  `group_by_length` sampling, and Hydra-style config composition.

### Bottom line
The systems-layer fear ("we'll have to reinvent distributed/precision") is
misplaced: HF `Trainer` doesn't own that layer — Accelerate does, and a manual
loop calls Accelerate directly. The semantic-layer reality ("our loss/packing/
supervision are deliberately non-standard") is exactly where `Trainer`'s defaults
become recurring liabilities, as the current `coord_losses.py` mixin proves line
by line. Owning a ~100-line Accelerate-backed loop buys first-class non-standard
supervision, an agent-readable execution flow, and cheap future research losses,
at the cost of a thin, well-scoped checkpoint/eval/logging slice — a trade that is
clearly correct for this project.

---

## Appendix — Evidence Index (exact handles)

**transformers 4.57.1** (`/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`)
- `trainer.py:4075-4156` `compute_loss` — `labels` popped only with
  label_smoother/compute_loss_func (4101-4104); `model(**inputs)` (4110);
  `compute_loss_func(outputs, labels, num_items_in_batch)` (4117-4127); reads
  model `loss` otherwise (4140-4147); `average_tokens_across_devices` rescale
  (4149-4154).
- `trainer.py:3981-4073` `training_step` — GA-normalization tri-conditional
  (4059-4064); DeepSpeed `scale_wrt_gas=False` (4068-4069);
  `accelerator.backward` (4071).
- `trainer.py:5590-5640` `_get_num_items_in_batch` — `sum(labels.ne(-100))`,
  requires `labels`, "we don't support object detection".
- `trainer.py:636-639` `model_accepts_loss_kwargs` auto-detected from signature.
- `trainer.py:1053-1146` sampler/dataloader — `RandomSampler`/`LengthGroupedSampler`
  (1053-1080); `batch_size`, `collate_fn`, `accelerator.prepare` (1099-1117);
  `IterableDataset` → no sampler (1107).
- `trainer.py:2353+` `_inner_training_loop` — `get_batch_samples` (2618);
  `current_gradient_accumulation_steps` (2621); `do_sync_step` (2624-2626);
  `training_step` (2674); `accelerator.clip_grad_norm_` (2715); `optimizer.step`
  (2740); skip-aware `lr_scheduler.step` (2747-2750); `_maybe_log_save_evaluate`
  (2756).
- `trainer.py:4177-4227` `save_model` (FSDP/DeepSpeed/TP branches); `4305` `_save`;
  `3312` `_save_checkpoint`; `3231` `_load_rng_state` (resume).
- `training_args.py:357,429-431,991,1152` `save_steps`/`eval_steps`/`logging_steps`
  float-in-[0,1) ratio of `max_steps`; validation `1675-1698`.
- `data/data_collator.py:2092-2158` `DataCollatorWithFlattening` — per-segment
  `position_ids` (2150), `cu_seq_lens_q/k` (2158), `separator_id=-100`.
- `modeling_flash_attention_utils.py:318-357` `prepare_fa_kwargs_from_position_ids`
  (int32 cu_seqlens, Python-int max_length); `_get_unpad_data` (208).
- `models/qwen3_vl/modeling_qwen3_vl.py` — vision attn `cu_seq_lens_q/k` (216-217);
  decoder/attention `FlashAttentionKwargs` threading (416-423, 488-510, 784-796);
  3D/4-row MRoPE `position_ids` (321-324, 823-830).

**Current CoordExp stack** (`/data/CoordExp/src/`)
- `trainers/metrics/coord_losses.py:20-160` — `compute_loss` mixin that recomputes
  `num_items_in_batch` (72-87), manually re-divides by `gradient_accumulation_steps`
  (133-158), restores `labels` (159-160).
- `bootstrap/trainer_setup.py:64-111` `compose_trainer_class` mixin stack;
  `sft.py:139-157` `resolve_trainer_cls`; `sft.py:4167-4187` dynamic `type(...)`
  MRO; `sft.py:48-54` ms-swift `SwiftSft`/`TrainerFactory` imports.
- `trainers/stage2_rollout_runtime.py:35` `from swift.trainers import
  Seq2SeqTrainer`; `1982/1999` toggling ms-swift template packing/padding-free.
- `trainers/stage2_rollout_correction_impl.py:5603+` Stage-2 `compute_loss`
  override; `trainers/metrics/batch_contract.py:200-227` `num_items_in_batch`
  plumbing notes.

**Companion:** [`DECISIONS.md`](DECISIONS.md) — "Training Loop, Checkpointing, And
Trainer Substrate", "Qwen Packing And FlashAttention Boundary", "Stage-2
Readiness". This study supplies the source-level evidence behind those decisions.
