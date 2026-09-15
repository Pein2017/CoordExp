# DoRA Source Study

## Scope

Wave 1A read-only study for OpenSpec tasks 2.1 and 2.2, with probe planning
evidence for task 2.3. Wave 1B later added the executable DoRA round-trip
probe receipt recorded below. No production `src/` code was implemented by this
study/probe gate.

Authoritative contracts:

- `openspec/changes/rebuild-coordexp-swift-training-infra/tasks.md`
- `openspec/changes/rebuild-coordexp-swift-training-infra/specs/coordexp-swift-adapters-embeddings-optim/spec.md`
- `docs/superpowers/plans/2026-06-30-coordexp-swift-execution-charter.md`
- `docs/superpowers/plans/2026-06-30-coordexp-swift-src-rebuild-roadmap.md`
- `DECISIONS.md`
- `BLUEPRINT.md`

## Verdict

The public V1 adapter schema uses `adapter.type: dora`, implemented with
PEFT DoRA through `peft.LoraConfig(..., use_dora=True)`.

No separate local, MS-Swift, or PEFT mechanism named `dlora` was found. MS-Swift
exposes DoRA through `use_dora` as a LoRA option and passes that option into
PEFT. The legacy CoordExp config surface also used LoRA with `use_dora: true`,
not a distinct `dlora` adapter family.

Therefore, V1 rejects `adapter.type: dlora` rather than treating it as a public
alias. The historical/provisional term "dLoRA" in earlier planning maps to the
approved public `dora` mechanism for V1.

## Public Naming Decision

Decision recorded on 2026-06-30:

- use `adapter.type: dora`;
- map directly to PEFT `LoraConfig(use_dora=True)`;
- do not keep `adapter.type: dlora` as a CoordExp-specific alias in V1.

Do not silently let `dlora` mean standard LoRA, DoRA, an unnamed variant, or a
novel research mechanism invented during implementation.

## Local Environment Evidence

Verified from `/data/CoordExp/.worktrees/CoordExp-swift`:

```text
peft 0.17.1
transformers 4.57.1
torch 2.9.1+cu128
flash-attn 2.8.3
ms-swift 4.2.2
```

Installed source roots:

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/peft/`
- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/`
- `/data/ms-swift/swift/`

Local Qwen3-VL model path:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

## PEFT DoRA Evidence

PEFT `LoraConfig` defines `use_dora` as Weight-Decomposed Low-Rank Adaptation:
ordinary LoRA learns direction while an additional magnitude parameter learns
weight magnitude.

Relevant installed source:

- `peft/tuners/lora/config.py`
  - `use_dora` config field and documentation.
  - validation rejects incompatible `lora_bias=True` with DoRA.
  - validation rejects DoRA with unsupported Megatron-style config.
- `peft/tuners/lora/layer.py`
  - `LoraLayer` owns `lora_A`, `lora_B`, `use_dora`, and
    `lora_magnitude_vector`.
  - linear targets resolve to `DoraLinearVariant` when `use_dora=True`.
- `peft/tuners/lora/variants.py`
  - `DoraLinearVariant` adds `lora_magnitude_vector` to adapter layer names.
  - DoRA forward path adds magnitude-vector behavior on top of the base result.
- `peft/tuners/lora/dora.py`
  - `DoraLinearLayer.update_layer` creates the magnitude vector as a trainable
    parameter.
- `peft/utils/save_and_load.py`
  - save includes LoRA tensors and DoRA magnitude-vector tensors.
  - save/load rewrites `lora_magnitude_vector.<adapter>` keys to and from the
    live `.weight` parameter form.

Implication: V1 DoRA support must treat magnitude vectors as a first-class
trainable/checkpoint/optimizer surface, not as an optional PEFT detail.

## MS-Swift Evidence

MS-Swift exposes DoRA as a LoRA option:

- `/data/ms-swift/swift/arguments/tuner_args.py`
  - `use_dora` appears as a tuner argument.
  - no separate `dlora` adapter family was found.
- `/data/ms-swift/swift/pipelines/train/tuner.py`
  - adapter preparation builds LoRA kwargs including `use_dora`.
  - LoRA config is passed to PEFT.
- `/data/ms-swift/swift/utils/transformers_utils.py`
  - `find_all_linears` excludes output heads and LoRA internals.
  - multimodal target discovery filters language, vision, and aligner modules.
- `/data/ms-swift/swift/model/model_arch.py`
  - Qwen3-VL module taxonomy includes language model, `lm_head`, visual tower,
    and visual merger/aligner-style modules.

Implication: MS-Swift is useful as a reference for target discovery and
exclusions, but it does not justify a distinct `dlora` mechanism name.

## Legacy CoordExp Evidence

Legacy Stage-1 config used:

```yaml
tuner:
  train_type: lora
  use_dora: true
```

See:

- `configs/stage1/sft_base.yaml`

Legacy vLLM adapter filtering already preserved both `lora_` and
`lora_magnitude_vector` tensors:

- `src/infer/backend_vllm_server.py`

Implication: the existing codebase already treated DoRA magnitude vectors as
part of adapter payloads, but did not call the mechanism `dlora`.

## Qwen3-VL Target Discovery Notes

Installed Transformers Qwen3-VL exposes:

- `visual`
- `language_model`
- top-level `lm_head`
- visual `merger`
- `deepstack_merger_list`

V1 target discovery should be CoordExp-owned and receipt-backed:

- `target_towers: [language]` plus `target_modules: all_linear` should match
  language tower linear modules only.
- `lm_head` must be excluded from adapter targets.
- vision and aligner target choices should remain unsupported until target
  discovery, forward/backward, optimizer grouping, receipt, and save/load
  evidence exist for those choices.

Do not blindly delegate the full selection policy to PEFT global `all-linear`.
Use PEFT/MS-Swift as references, then emit CoordExp receipts.

## Required Wave 1B Probe Assertions

The DoRA round-trip probe must verify:

1. **Config construction**
   - `LoraConfig(use_dora=True, r=..., lora_alpha=..., lora_dropout=...,
     bias="none", target_modules=<resolved_targets>)`.
   - reject unsupported DoRA combinations such as `lora_bias=True`, Megatron
     DoRA path, and unsupported target types.
2. **Target discovery**
   - for language-only `all_linear`, list matched Qwen3-VL language modules;
   - exclude `lm_head`;
   - record unsupported/missing targets.
3. **Trainable parameters**
   - after PEFT injection, trainable names include `lora_A`, `lora_B`, and
     `lora_magnitude_vector` for every DoRA target;
   - every trainable DoRA parameter matches exactly one optimizer group;
   - a tiny backward check proves at least one `lora_magnitude_vector`
     parameter receives a finite gradient, while frozen base parameters do not
     unexpectedly require gradients.
4. **Persistence**
   - saved adapter payload includes LoRA A/B tensors and at least one
     `lora_magnitude_vector` tensor;
   - `adapter_config.json` preserves `use_dora: true`;
   - probe checks both saved-key and reloaded-state-dict key forms.
5. **Reload**
   - fresh base plus saved adapter reconstructs magnitude-vector parameters;
   - shapes/dtypes are preserved;
   - a tiny deterministic forward produces finite logits;
   - original adapter model and reloaded adapter model match logits within a
     fixed tolerance in eval mode.
6. **Receipt**
   - record package versions, base model path, public adapter type, resolved
     PEFT mechanism, target towers, target policy, matched module names, A/B and
     magnitude tensor counts, trainable parameter counts, optimizer group
     matches, save path, reload success, finite/equivalence result, and
     magnitude-vector gradient result.

## Findings

- **P0 resolved by Wave 1B:** `adapter.type: dora` now has PEFT
  `LoraConfig(use_dora=True)` round-trip probe evidence. It still must not be
  wired into production validation until implementation consumes this gate and
  source-study review task 2.10 is accepted.
- **P0:** `adapter.type: dlora` is not a V1 schema value and must be rejected
  rather than silently mapped to DoRA.
- **P1:** DoRA magnitude vectors are part of the trainable adapter surface.
  Optimizer grouping, checkpoint writing, and receipts must not only match
  `lora_A` and `lora_B`.
- **P1:** Public naming drift is resolved by using `dora`; implementation and
  receipts must not reintroduce `dlora` as an accepted adapter type.
- **P1:** Target discovery must be CoordExp-owned, tower-scoped, and
  receipt-backed; `lm_head` exclusion must be proven.
- **P2:** Config validation should reject known PEFT DoRA-incompatible options
  before model mutation.

## Wave 1B Probe Evidence

Executed from `/data/CoordExp/.worktrees/CoordExp-swift` on 2026-06-30:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/probes/coordexp_swift/dora_roundtrip.py \
  --device cuda --dtype bf16 --max-targets 1
```

Receipt:
`outputs/probes/coordexp_swift/dora_roundtrip/receipt.json`.

Observed contract:

- public adapter type: `dora`;
- PEFT mechanism: `LoraConfig(use_dora=True, r=2, lora_alpha=4,
  lora_dropout=0.0, bias="none")`;
- matched Qwen3-VL language linear targets: 196;
- selected tiny probe target count: 1;
- `lm_head` excluded;
- trainable tensors: one `lora_A`, one `lora_B`, one
  `lora_magnitude_vector`;
- optimizer duplicate parameter count: 0;
- tiny backward produced finite logits and finite magnitude-vector gradient;
- saved adapter payload contained LoRA A/B tensors plus one DoRA
  magnitude-vector tensor;
- `adapter_config.json` preserved `use_dora: true`;
- fresh base plus saved adapter reloaded the magnitude vector;
- eval-logit reload equivalence passed with `max_abs_diff: 0.0` in bf16.

## Status

Wave 1A evidence supports a PEFT DoRA-backed implementation path, and the
public adapter name is approved as `adapter.type: dora`. Tasks 2.1 and 2.2 have
source-study evidence. Wave 1B probe evidence completes task 2.3 for the
language-tower DoRA gate. Production implementation remains blocked by task
2.10 review and the later OpenSpec implementation approval gate.
