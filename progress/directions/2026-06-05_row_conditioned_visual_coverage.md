# Row-Conditioned Visual Coverage for Autoregressive VLM Enumeration

## 0. One-sentence summary

Explore **row-conditioned visual coverage** as an explicit feature-level state
for autoregressive VLM object enumeration: the image is encoded once, and each
object row is conditioned on visual features marked by previously enumerated
objects, using standard SFT with CE as the first objective surface.

Evidence scope: `unit_smoke` after the 2026-06-06 prototype pass. This note
records resolved research-design decisions from a grill discussion plus the
first implementation evidence. It is not a stable contract or production
result.

2026-06-07 refinement: treat the prototype as a **Direct Feature Painting
Oracle**. The immediate purpose is correctness and scientific readability of
the feature-painting mechanism, not cache-efficient decoding. Faithful
row-boundary re-prefill remains the interpretation path until a separate
cache-compatible variant is explicitly compared.

Companion protocol:

```text
docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md
```

## 1. Problem framing

The target failure is broader than duplicate suppression alone. In dense scenes
with many same-description objects, autoregressive VLMs can show both:

- duplicate bursts, where the model re-emits the same or local-neighbor
  instance;
- low recall, where the model appears to stop or drift despite evidence for
  remaining instances.

The working hypothesis is that text history is a weak memory of which visual
instances have already been enumerated. The model may perceive many instances,
but lack an explicit visual coverage state that marks what has already been
counted while preserving evidence for overlapping or nearby objects.

Use the canonical term **coverage** for this idea. Do not introduce a new term
borrowed from non-AI domains.

## 2. Resolved decisions so far

### 2.1 Scope

The first research framing is:

```text
current row-conditioned visual coverage idea + standard SFT with CE
```

Ignore residual-set, trie, multi-positive, and fallback paths for the first
conceptual grill. Those may become later composition surfaces, but they should
not define the core mechanism.

### 2.2 Coverage semantics

Coverage means:

```text
this visual extent or region has already been enumerated
```

Coverage does not mean:

```text
this region is background
this region is negative
this region should be hidden
this region should not be attended
the next object must be elsewhere
EOS should be more likely
```

Coverage is a visual state marker, not a hand-coded selection, suppression, or
attention policy. The model should learn the behavioral meaning of coverage
implicitly through standard SFT/CE.

### 2.3 Primary mechanism family

Use **visual feature tuning** as the primary mechanism family.

Attention-bias routing can remain a diagnostic or control family, but it is not
the preferred first claim path because it more directly imposes where the model
should look. Visual feature tuning better matches the intended "marked visual
scene" analogy: coverage becomes part of the visual representation, and the
decoder learns how to use it.

Conceptual form:

```text
V = vision_encoder(image)
M_k = coverage descriptor from objects enumerated before row k
V'_k = Tune(V, M_k)
predict row k with standard shifted CE
```

The same `Tune(V, M_k)` operation must be available in teacher-forced training
and rollout/decode. The allowed stage difference is the source of the prefix:

```text
training: teacher, sampled, corrupted, or rollout-derived prefix
inference: model-committed prefix
```

### 2.4 Minimal coverage descriptor

For v1, keep the descriptor minimal:

```text
M_k(i) = [
  boundary_mass_i,
  interior_mass_i,
]
```

Do not include class labels, text summaries, confidence, object count, or
recency in the first conceptual version. Those additions would make the
mechanism harder to interpret.

### 2.5 Boundary-primary, interior-weak

Boundary coverage is the primary signal. Interior coverage is weaker auxiliary
state.

Rationale:

- boundary marks help indicate the footprint or extent of already enumerated
  instances;
- interiors often still contain useful evidence for overlapping, nested,
  held, occluded, or nearby objects;
- hard interior suppression would trade duplicate reduction for recall loss.

### 2.6 Additive density, not normalized probability

Interior coverage may be additive and composable across previously enumerated
objects:

```text
raw_interior_i = sum_j soft_inside_overlap(box_j, visual_token_i)
```

This allows the model to sense local coverage density in crowded regions.

Do not normalize interior coverage into a global probability distribution over
the image in v1. Global normalization can erase absolute local density and
cause unrelated regions to change when a new object is added elsewhere.

Preferred interpretation:

```text
soft accumulated coverage density over visual tokens
```

not:

```text
probability mass over image patches with sum_i = 1
```

The exposed value should be bounded or saturated so dense coverage remains a
state signal rather than a hard mask.

### 2.7 Observability invariant

Coverage must never destroy observability.

The original visual feature for every region must remain available. Coverage is
an additional residual state signal, not a replacement for visual content.

Safe conceptual form:

```text
V'_i = V_i
     + alpha_b * boundary_mass_i * e_boundary
     + alpha_i * interior_mass_i * e_interior
```

with:

```text
alpha_b > alpha_i
interior contribution weak
coverage contribution bounded
initial behavior near identity
```

Alpha caps are not sufficient by themselves to guarantee a small perturbation:
the residual magnitude also depends on `||e_boundary||` and `||e_interior||`.
Coverage-active smokes should report `||delta_F_i|| / ||F_i||`, max/mean token
ratios, and coverage embedding norms before any loss or rollout result is
interpreted as a clean mechanism signal.

Avoid:

```text
V'_i = coverage_only_i
V'_i = 0 for covered tokens
attention_to_i = -inf for covered tokens
```

### 2.8 Boundary on the visual-token lattice

After patchification and visual-token merging, exact bbox boundaries are not
available as precise pixel edges. Boundary coverage must therefore be defined
on the visual-token lattice.

Each visual token should be treated as having an image-space footprint:

```text
P_i = footprint of visual token i
B_j = previous bbox j
```

Interior coverage:

```text
inside_mass_i(B_j) = area(P_i intersect B_j) / area(P_i)
```

Boundary coverage uses a soft bbox-edge band rather than an infinitely thin
line:

```text
boundary_band(B_j, w) = pixels within distance w of bbox rectangle boundary
boundary_mass_i(B_j) = area(P_i intersect boundary_band(B_j, w)) / area(P_i)
```

The band width `w` should be tied to the effective visual token footprint, such
as the 32x32 patch-scale intuition for Qwen3-VL-style visual encoding, rather
than the exact precision of coord tokens.

The bbox coordinates remain precise in the language/coord-token sequence. The
visual coverage state is coarse, soft, and resolution-aware.

Boundary marks bbox extent on the post-merge visual-token lattice. It is not a
true object contour unless segmentation masks are later introduced.

### 2.8.1 Union memory caveat

V1 coverage is union memory, not instance memory. All previously enumerated
boxes accumulate into shared boundary/interior channels. It does not encode
which prior object covered a token, how many objects covered it, or the
class/text identity of those objects.

This limitation is acceptable for the first proof because it keeps the
mechanism readable. Future ablations can add count/log-count channels,
class-summary coverage, confidence-weighted coverage, or instance-memory
variants after the direct feature-painting oracle is tested.

Interior coverage means:

```text
already enumerated
```

It does not mean:

```text
visually irrelevant
directly suppressed
safe to ignore
```

Any duplicate reduction, attention shift, or recall change must be learned
through CE and measured, not assumed from the residual design.

### 2.9 Teacher-forcing row states

Use **prefix-state expansion** as the first teacher-forcing formulation.

Each image can yield multiple row-state training examples:

```text
state 0: coverage(empty) -> predict object row 1
state 1: coverage(object 1) -> predict object row 2
state 2: coverage(object 1, object 2) -> predict object row 3
state N: coverage(all objects) -> predict assistant stop
```

For each state:

```text
V'_k = Tune(V, coverage(objects before row k))
```

The target is the next object row under standard shifted CE. The current row's
bbox and future rows must not be included in the coverage descriptor.

The terminal state is part of the same standard CE sequence. It is required so
rollout stopping is trained under the same row-conditioned visual state used at
decode time, but it must not introduce a custom EOS bonus, EOS forcing rule, or
coverage-specific stop loss.

Rejected first-version alternatives:

- one static visual prefix for the whole teacher-forced sequence, because it
  cannot express row-specific coverage;
- one long sequence with multiple tuned visual blocks, because it is
  length-heavy and awkward;
- hidden row-routing sidecars, because they are more custom and easier to
  misalign between training and rollout.

Prefix-state expansion is more expensive, but it is the cleanest first
research formulation for train/decode tensor equivalence and leakage control.

### 2.10 Rollout alignment

Use **row-boundary re-prefill with directly tuned visual features** as the
first-principles rollout formulation.

The intended rollout loop is:

```text
V = vision_encoder(image)  # once

for row k:
  M_k = coverage(committed rows before k)
  V'_k = Tune(V, M_k)
  re-prefill prompt + V'_k + committed text prefix
  decode the next object row
  if the row is parse-valid, commit it and update coverage
```

This deliberately prioritizes mathematical/tensor-flow correctness over
efficiency:

- every row is decoded under the same kind of row-conditioned visual state used
  in teacher-forced prefix-state training;
- previous text-prefix hidden states are recomputed under the current `V'_k`
  rather than mixed with a stale visual cache;
- the image encoder still runs once, but the decoder prefix is recomputed at
  row boundaries.

Cache-preserving alternatives, such as appending coverage tokens after each
object or patching cached visual K/V, are deferred. They may become efficiency
improvements later, but they should not define the first evidence claim.

If this faithful formulation fails to improve the target behavior under clean
evaluation, the core coverage story is weakened at its origin; efficiency
variants should not be used to rescue a mechanism that fails in this oracle-like
setting.

### 2.11 Tensor-flow correctness criterion

For any prefix state `S_k`, teacher-forced training and rollout should compute
the same conditional form:

```text
p(row_k | prompt, text_prefix(S_k), Tune(V, coverage(S_k)))
```

The only allowed stage difference is how `S_k` is obtained:

```text
training: S_k comes from a teacher, sampled, corrupted, or rollout-derived prefix
rollout:  S_k comes from committed model predictions
```

Everything else should be equivalent in tensor meaning:

```text
same base image features V
same coverage descriptor construction
same Tune(V, M_k)
same text-prefix rendering convention
same autoregressive next-token factorization
```

This criterion does not require rollout prefixes to be correct. It requires
that, for any given prefix state, the conditional computation has the same
form in training and rollout.

### 2.12 First tuning operator

Use a **simple additive residual coverage embedding** as the first `Tune`
operator:

```text
V'_i = V_i
     + alpha_b * boundary_mass_i * e_boundary
     + alpha_i * interior_mass_i * e_interior
```

with:

```text
alpha_b > alpha_i
coverage contribution bounded
initial behavior near identity
```

Rationale:

- it directly tests whether boundary/interior coverage marks help;
- it is easy to explain and ablate;
- it is less likely than a larger adapter to produce an ambiguous gain from
  extra capacity alone.

A gated residual adapter remains a later candidate:

```text
delta_i = Gate(V_i, M_i) * Adapter(V_i, M_i)
V'_i = V_i + delta_i
```

Use it only after the additive form establishes whether the core coverage idea
has signal, or if the additive form appears too weak or too blunt.

### 2.13 First boundary-band definition

Use a **single soft bbox-edge band** at roughly one effective visual-token
footprint width for v1.

Conceptual definition:

```text
boundary_band_width ~= one visual token footprint
boundary_mass_i(B_j) = area(P_i intersect boundary_band(B_j)) / area(P_i)
interior_mass_i(B_j) = area(P_i intersect bbox_interior(B_j)) / area(P_i)
```

Accumulate across previous objects additively, then bound or saturate:

```text
boundary_mass_i = saturate(sum_j boundary_mass_i(B_j))
interior_mass_i = saturate(sum_j interior_mass_i(B_j))
```

Do not globally normalize these values over the image.

Rationale:

- one-token-width bands respect the coarse visual-token lattice instead of
  pretending exact pixel-level boundaries survive patchification;
- anti-aliased area overlap makes partial-token boundary contact meaningful;
- additive bounded accumulation preserves local density without masking visual
  content.

Multi-scale rings or signed-distance fields remain possible later, but they are
not the first descriptor.

### 2.14 Safety clarifications for v1

The available supervision is bbox-level. Therefore the boundary channel means:

```text
claimed bbox extent / bbox-edge band
```

It must not be described as a true object contour or instance mask unless a
future mask source is added.

The v1 accumulation rule is bounded union-style coverage:

```text
coverage_i = clamp(sum_j coverage_ij, 0, 1)
```

This preserves whether a visual token footprint has been covered, but it does
not preserve how many previous bboxes overlap that token. A separate count or
log-count channel remains a later ablation, not part of the current CE-only
claim.

Alpha caps are not by themselves a residual-norm guarantee. Production and
smoke runs should monitor:

```text
||delta V|| / ||V||
max_token ||delta V_i|| / ||V_i||
mean_token ||delta V_i|| / ||V_i||
boundary/interior alpha and embedding norms
boundary/interior mass density
```

If these ratios grow too large, the safer follow-up is a normalized residual
parameterization. For the first production attempt, the implementation keeps the
simple additive residual and adds explicit residual-safety metrics.

## 3. Consequences

- The first mechanism should be evaluated as learned visual-state conditioning
  under ordinary CE, not as a new loss.
- Any future implementation must keep training and inference tensor flow
  aligned: same coverage descriptor, same visual feature tuning operation, and
  same row-conditioned semantics.
- If decode uses updated coverage after a committed row, training must expose
  the analogous row-conditioned visual state instead of one static visual
  prefix for the whole sequence.
- First evidence should come from the faithful re-prefill formulation before
  optimizing for KV-cache efficiency.
- Coverage gains should not be interpreted as success unless duplicate behavior
  and recall/overlap guardrails are both checked.
- Attention-bias routing remains useful as a causal probe or ablation, but its
  hand-coded routing semantics are not the main research claim.

## 4. Evidence gate

Use **staged joint evidence** as the first proof standard.

Stage 1: mechanism gate.

```text
coverage off vs coverage on
same image and comparable text prefix
check whether row-conditioned binding changes at the intended surface
```

The most important target surface is early bbox binding, especially `x1/y1`,
because prior CoordExp diagnostics repeatedly identify early coordinate
commitment as a hard point for duplicate/local-basin escape.

Stage 2: rollout guardrail.

Only interpret the mechanism as useful if rollout evidence shows that coverage
does not buy duplicate reduction by harming:

```text
recall
overlap / crowded-scene cases
parse validity
EOS / row-count behavior
```

Working rule:

```text
coverage is alive only if:
  (1) it changes the row-conditioned binding surface in the intended direction;
  and
  (2) it does not reduce duplicates by simply suppressing hard objects,
      truncating output, or destabilizing parsing.
```

## 5. Remaining open forks

No major conceptual fork is currently resolved beyond the decisions above.
Next work should turn these decisions into a concrete experimental plan only
after the user approves moving from grill to planning.

## 6. 2026-06-06 Prototype Evidence

Implementation branch:

```text
/data/CoordExp/.worktrees/row-conditioned-visual-coverage
codex/row-conditioned-visual-coverage
```

Implemented surfaces:

- Strict row-coverage config parser and training smoke configs for
  `random_sft` and `sorted_sft`.
- Visual-token lattice painter with boundary/interior coverage channels on the
  post-merge visual-token lattice.
- Row-state teacher-forcing dataset expansion with prefix coverage excluding
  the target row while preserving rendered target rows for shifted CE.
- Trainer/collator sidecar path carrying coverage state into forward.
- Additive visual-feature residual tuner with boundary and interior channels,
  identity behavior at zero alpha, bounded alpha gates, and no masking or
  feature deletion.
- Row-boundary re-prefill rollout helper with committed-prefix semantics and
  no cross-row KV reuse in the faithful prototype.
- Research-only mechanism/guardrail smoke scripts under
  `scripts/analysis/row_conditioned_visual_coverage/`.

Verification commands run:

```bash
env -u CODEX_CI python -m pytest \
  tests/detection/coverage \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_teacher_forcing_target_builder.py \
  tests/test_teacher_forcing_config_contract.py \
  -q -p no:cacheprovider
# 216 passed

env -u CODEX_CI python -m pytest \
  tests/test_training_surface_resolver.py \
  tests/test_objective_profile_resolution.py \
  tests/test_compact_full_encoding_contract.py \
  tests/test_training_architecture_golden_thread.py \
  -q -p no:cacheprovider
# 113 passed

python -m py_compile \
  src/detection/coverage/config.py \
  src/detection/coverage/types.py \
  src/detection/coverage/geometry.py \
  src/detection/coverage/painting.py \
  src/detection/coverage/prefix_rendering.py \
  src/detection/coverage/row_state_dataset.py \
  src/detection/coverage/forward.py \
  src/detection/coverage/rollout.py \
  src/detection/coverage/artifacts.py \
  scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py \
  scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  scripts/analysis/row_conditioned_visual_coverage/report.py
# passed

env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py --help
env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/run_rollout.py --help
env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/report.py --help
# all passed
```

Smoke generation commands:

```bash
rm -rf temp/row_coverage_smoke
mkdir -p temp/row_coverage_smoke

env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py \
  --config configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml \
  --output-root temp/row_coverage_smoke/mechanism_primary \
  --limit 2 \
  --coverage-enabled true \
  --control-mode primary_coverage

env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  --config configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml \
  --output-root temp/row_coverage_smoke/no_coverage \
  --limit 2 \
  --coverage-enabled false \
  --control-mode no_coverage_reprefill

env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  --config configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml \
  --output-root temp/row_coverage_smoke/primary_coverage \
  --limit 2 \
  --coverage-enabled true \
  --control-mode primary_coverage

env -u PYTHONPATH python \
  scripts/analysis/row_conditioned_visual_coverage/report.py \
  --input-root temp/row_coverage_smoke \
  --output-root temp/row_coverage_smoke/report
```

Smoke artifacts:

```text
temp/row_coverage_smoke/mechanism_primary/mechanism_rows.jsonl
temp/row_coverage_smoke/no_coverage/rollout_guardrail_rows.jsonl
temp/row_coverage_smoke/primary_coverage/rollout_guardrail_rows.jsonl
temp/row_coverage_smoke/report/mechanism_summary.json
temp/row_coverage_smoke/report/guardrail_metrics.json
temp/row_coverage_smoke/report/report.md
```

Observed smoke summary:

```text
mechanism_rows = 2
mechanism_orderings = random_sft, sorted_sft
guardrail_rows = 4
guardrail_orderings = random_sft, sorted_sft
guardrail_modes = no_coverage_reprefill, primary_coverage
claim_eligible_count = 0 for all smoke rows
```

The smoke scripts are intentionally research-only stubs. They validate CLI,
artifact, grouping, malformed-row, empty-input, and no-claim guardrails. They do
not launch a production model, do not load a checkpoint, and do not provide
checkpoint-backed rollout evidence.

Latest pure-CE checkpoint targets identified for the next real model smoke:

```text
fullobj_random_pure_ce_ckpt3668
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
epoch = 4.0
global_step = 3668
training_ordering = random_permutation
template_contract_id = compact_full_no_newline_native_v1

fullobj_sorted_pure_ce_ckpt3668
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
epoch = 4.0
global_step = 3668
training_ordering = sorted
template_contract_id = compact_full_no_newline_native_v1
```

Those checkpoint targets were added to
`configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml` with
`evidence_use: future_checkpoint_backed_smoke`. The YAML was parsed and both
paths were verified to exist.

Residual limitations:

- No full Qwen3-VL checkpoint-backed row-coverage rollout was launched in this
  pass. A later tiny GPU Stage-1 SFT smoke is recorded below.
- The faithful rollout helper intentionally re-prefills at row boundaries and
  does not solve KV-cache efficiency yet.
- Reported smoke guardrails are not model-quality metrics; they are prototype
  contract checks for the mechanism/reporting path.

### 2026-06-06 Tiny GPU Stage-1 SFT Smoke

Scope:

- `tiny_gpu_smoke`
- single A100 GPU (`gpus=0`)
- Stage-1 detection teacher-forcing only
- random SFT row ordering
- 8 optimizer steps
- `train_sample_limit=16`, expanded to 88 row-conditioned training states
- `val_sample_limit=2`, expanded to 23 row-conditioned validation states
- no eval, no checkpoint save

Smoke config:

```text
configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_gpu_smoke_8step_20260606.yaml
```

Launch command:

```bash
TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
config=configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_gpu_smoke_8step_20260606.yaml \
gpus=0 \
train_log_dir=temp/row_coverage_gpu_smoke/logs \
bash scripts/train.sh
```

Artifact root:

```text
temp/detection_teacher_forcing/output/row_coverage_random_sft_gpu_smoke_8step_20260606/smoke-row-coverage-random-sft-gpu-8step-20260606/v0-20260606-114912
```

Launcher log:

```text
temp/row_coverage_gpu_smoke/logs/row_coverage_random_sft_gpu_smoke_8step_20260606-20260606T114805Z.log
```

Pre-launch contract fixes discovered by the smoke:

- A temp-directory config caused launcher precheck path resolution to anchor
  `public_data/*` under `temp/row_coverage_gpu_smoke/`; the reproducible smoke
  config was moved under `configs/` so repo-root path resolution applies.
- The clean-break Stage-1 config did not set `model.model_type`; ms-swift
  rejected the local Qwen3-VL directory as ambiguous among `qwen3_vl`,
  `qwen3_vl_emb`, and `qwen3_vl_reranker`. The smoke config pins
  `model.model_type: qwen3_vl`, matching `configs/base.yaml`.

Confirmed by the successful run:

- JSONL and image-size prechecks passed for train and val.
- Qwen3-VL 2B loaded on GPU 0 with `model_type=qwen3_vl`.
- LoRA and coord-offset hooks attached.
- Row-conditioned dataset expansion was active.
- Row coverage forward context was installed on the PEFT-wrapped model:
  `source=attached_model`.
- The trainer completed 8/8 optimizer steps with nonzero gradients.
- Run artifacts present:
  `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`,
  `experiment_manifest.json`, `run_metadata.json`,
  `train_data_provenance.json`, `eval_data_provenance.json`,
  `config_source.yaml`, `logging.jsonl`, and `train_heartbeat.rank0.jsonl`.

Learning-trend smoke values:

```text
step 1: loss=12.96523571 grad_norm=163.57023621 lr=0.0
step 2: loss=10.38784218 grad_norm=102.89086151 lr=5.0e-05
step 3: loss=12.72720814 grad_norm=163.68154907 lr=4.752e-05
step 4: loss=11.03848839 grad_norm=274.69403076 lr=4.059e-05
step 5: loss=12.56134892 grad_norm=171.14416504 lr=3.056e-05
step 6: loss=11.14252663 grad_norm=118.75360107 lr=1.944e-05
step 7: loss=11.63170528 grad_norm=57.83621216 lr=9.41e-06
step 8: loss=12.26512146 grad_norm=64.22734833 lr=2.48e-06

train_loss=11.83993459
train_runtime=7.256s
train_steps_per_second=1.103
max_memory=7.3 GiB
```

Interpretation:

- This run proves the row-conditioned SFT path can execute forward,
  backward, and optimizer steps on a real Qwen3-VL model without falling back to
  baseline teacher forcing.
- The loss curve is noisy over 8 single-sample steps. It is not monotonic and
  should not be interpreted as quality improvement evidence.
- The smoke does not prove checkpoint serialization/resume/export of the
  row-coverage tuner. `pipeline_manifest.json` is also absent from the
  artifact root, although the available manifest set is listed in
  `experiment_manifest.json`.

### 2026-06-06 Production Training Gate

Planned production config:

```text
configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_ce_cont_from_random_pure_sft_1epoch_eval512_8gpu.yaml
```

Planned 8-GPU preflight config:

```text
configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_ce_cont_from_random_pure_sft_8gpu_preflight.yaml
```

Production policy:

- start from the trained random pure-SFT adapter checkpoint-3668;
- keep random-permutation SFT with standard teacher-forced CE;
- lower LoRA and coord-row learning rates from `5.0e-5` to `1.0e-5`;
- keep the original high-utilization 8-GPU global batch shape
  (`per_device_train_batch_size=16`, `effective_batch_size=128`);
- use one epoch with `training.eval_steps: 1000` and
  `debug.val_sample_limit: 512` for lower-cost eval while monitoring trend;
- use `training.save_model_only: true` so checkpoints are restartable;
- persist `row_coverage_tuner.pt` in restartable checkpoints so the learned
  visual coverage tuner is auditable outside the PEFT adapter payload;
- log residual-safety metrics under `row_coverage/*`.

The 8-GPU preflight may be relaunched if memory or checkpoint serialization
fails. The final 1-epoch production launch should only remain running after the
preflight proves forward/backward, memory shape, row-coverage metric logging,
and row-coverage tuner sidecar checkpointing.

Preflight audit result:

- Earlier 8-GPU preflights proved model load, backward, and checkpoint writing
  but exposed a tensor-flow gap: `row_coverage_state` was present while
  `row_coverage/*` forward metrics were empty. This meant the real Qwen3-VL
  visual feature path was not being tuned.
- The installed Qwen3-VL tensor flow is:
  `Qwen3VLForConditionalGeneration.forward -> self.model(...) ->
  Qwen3VLModel.forward -> self.get_image_features(...) -> self.visual(...)`.
  PEFT/SWIFT wrapping can expose `get_image_features` at several layers, so
  the production hook now patches all discovered visual owners during the
  standard forward and restores them afterward. A call-count guard prevents
  double tuning if an outer owner delegates to an inner patched owner.
- Passing preflight artifact root:
  `temp/detection_teacher_forcing/output/row_coverage_random_sft_ce_cont_random_pure_sft_8gpu_preflight/preflight-row-coverage-random-sft-ce-cont-random-pure-sft-8gpu/v5-20260606-135710`.
- Passing launcher log:
  `temp/row_coverage_prod/logs/row_coverage_random_sft_ce_cont_from_random_pure_sft_8gpu_preflight-20260606T135559Z.log`.
- Config and data prechecks passed for train and val JSONL/image contracts.
- The trainer MRO included
  `RowCoverageMetricsLogMixin -> TeacherForcingObjectiveMixin`.
- One 8-GPU optimizer step completed with global effective batch size 128.
- First-step metrics:
  `loss=2.35817766`,
  `row_coverage/visual_feature_hook_call_count=1.0`,
  `row_coverage/patched_visual_owner_count=4.0`,
  `row_coverage/visual_token_count=61068.0`,
  `row_coverage/delta_to_feature_norm_ratio=0.00019109`,
  `row_coverage/max_token_delta_to_feature_norm_ratio=0.00150319`,
  `row_coverage/boundary_alpha=0.01000977`,
  `row_coverage/interior_alpha=0.00300598`.
- Peak reported trainer memory was `51.66 GiB` on 80-GiB A100s, leaving
  production headroom for this batch shape.
- `checkpoint-1` contains `row_coverage_tuner.pt`, and
  `coordexp_checkpoint_state.pt` records it under `artifacts.row_coverage`
  with `present=true`, `context_source=attached_model`, and tuner hash
  `e1632998172f075f258d07ddc5b440c1bc0660a9b89b166365db544fe7ffbe88`.
- `coordexp_checkpoint_state.pt` also records trainer restart state under
  `artifacts.trainer_state` with `world_size=8` and `rng_state_0.pth` through
  `rng_state_7.pth`.

Production launch is now allowed under the gate, with continued monitoring of
loss, gradient norm, `row_coverage/*` residual ratios, and GPU memory.

### 2026-06-06 Production Launch

Status update: this launch is superseded. It was stopped before continuation
because the compact-full prompt surface still resolved to newline-delimited
instructions even though the checkpoint/template contract is the native
compact-full no-newline surface. The next launch target is the 1-epoch
`eval_steps=1000`, `debug.val_sample_limit=512` config listed above.

Production run launched in tmux session:

```text
rowcov_random_2ep_20260606
```

Launcher log:

```text
/data/CoordExp/outputs/stage1_2b/row_conditioned_visual_coverage/logs/row_coverage_random_sft_ce_cont_from_random_pure_sft_2epoch_8gpu-20260606T140303Z.log
```

Artifact root:

```text
/data/CoordExp/outputs/stage1_2b/row_conditioned_visual_coverage/row_coverage_random_sft_ce_cont_random_pure_sft_2epoch_8gpu/row-coverage-random-sft-ce-cont-random-pure-sft-2epoch-8gpu/v0-20260606-140416
```

Launch-health observations:

- Prechecks passed for train and val JSONL/image contracts.
- Full row-conditioned dataset sizes:
  `train=965903`, `val=41224`.
- Trainer MRO includes
  `RowCoverageMetricsLogMixin -> TeacherForcingObjectiveMixin`.
- Total production train steps: `15094`.
- Step 1:
  `loss=2.29788017`,
  `grad_norm=36.76087189`,
  `row_coverage/visual_feature_hook_call_count=1.0`,
  `row_coverage/patched_visual_owner_count=4.0`,
  `row_coverage/delta_to_feature_norm_ratio=0.00014252`,
  `row_coverage/max_token_delta_to_feature_norm_ratio=0.00136685`,
  `memory=54.0 GiB`.
- Step 10:
  `loss=2.29026074`,
  `grad_norm=22.82148743`,
  `learning_rate=1.1e-07`,
  `row_coverage/visual_feature_hook_call_count=1.0`,
  `row_coverage/patched_visual_owner_count=4.0`,
  `row_coverage/delta_to_feature_norm_ratio=0.00015667`,
  `row_coverage/max_token_delta_to_feature_norm_ratio=0.00150997`,
  `memory=54.07 GiB`,
  `train_speed=15.373083 s/it`.
- After first-batch warmup, visible step cadence in the launcher log was about
  `11-12 s/step`.
- Live GPU memory during early production training stayed in roughly the
  `53-59 GiB` range on 80-GiB A100s.

Initial interpretation:

- The production tensor flow is active: row coverage feature tuning is being
  applied once per standard visual feature path, with four visual owner layers
  patched and restored around the forward.
- Residual magnitude is small relative to feature norm at launch. This satisfies
  the residual-safety gate but still needs trend monitoring as the learned tuner
  updates.
- Loss and gradient norms are in the same healthy range as the successful
  preflight. Step 10 is not enough evidence for quality improvement, only launch
  health.
