# Special-Token Embedding Source Study

## Scope

Wave 1A read-only study for OpenSpec tasks 2.4 and 2.5. Wave 1B later added
the executable special-token embedding round-trip probe receipt recorded below.
No production `src/` code was implemented by this study/probe gate.

This study compares:

- a custom Qwen wrapper pair;
- PEFT `TrainableTokensConfig` / `TrainableTokensModel`;
- LoRA `trainable_token_indices`.

## Verdict

Use a custom CoordExp-owned Qwen wrapper pair for V1.

The wrapper pair should implement selected-token full embedding tuning as a
zero-initialized additive delta over frozen base rows:

- tied Qwen models use one `shared_embed_delta`;
- untied models use `input_embed_delta` and `output_embed_delta`;
- checkpoints save compact additive deltas plus metadata;
- optimizer group is `token_embeddings`;
- this surface remains separate from PEFT LoRA/DoRA adapter payloads.

PEFT trainable-token mechanisms are useful references but store absolute
selected row replacements, not additive deltas. Using PEFT as-is while claiming
additive `shared_embed_delta` semantics would create a checkpoint/runtime
mismatch.

## Local Model And Tokenizer Facts

Verified local model:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

Facts from local config/tokenizer:

```text
architectures: Qwen3VLForConditionalGeneration
model_type: qwen3_vl
text_config.hidden_size: 2048
text_config.vocab_size: 152670
tie_word_embeddings: true
text_config.tie_word_embeddings: true
```

Selected token ids:

```text
<|object_ref_start|> = 151646
<|object_ref_end|>   = 151647
<|box_start|>        = 151648
<|box_end|>          = 151649
<|coord_0|>          = 151670
<|coord_999|>        = 152669
```

The selected token group contains 1004 unique single-token ids: four wrappers
plus 1000 coordinate tokens. Coordinate ids are contiguous from 151670 through
152669.

A meta-init check showed the local config path ties input embeddings and
`lm_head` through normal Transformers `tie_weights()`. Production code must
still assert runtime parameter identity after real model load.

## Candidate Comparison

| Mechanism | Fit | Payload Semantics | Notes |
| --- | --- | --- | --- |
| Custom Qwen wrapper pair | Best V1 fit | Additive delta | Keeps selected-token embedding tuning separate from PEFT adapters; matches approved `DECISIONS.md` design. |
| PEFT `TrainableTokens` | Feasible backup | Absolute selected row replacement | Compact, but standalone trainable tokens are not mixed-compatible and semantics differ from CoordExp additive-delta contract. |
| LoRA `trainable_token_indices` | Feasible backup when using PEFT LoRA/DoRA | Absolute selected row replacement | Integrates with PEFT LoRA config but couples special-token surface to adapter payloads. |

## Custom Wrapper Design

The custom wrapper pair should live under future
`src/qwen/special_token_embeddings.py`.

Input wrapper:

```text
base_embedding(input_ids) + selected_delta[input_ids] only for selected ids
```

Output wrapper:

```text
base_lm_head(hidden) with selected-column correction hidden @ delta.T
```

For tied models:

```text
shared_embed_delta: [num_selected_tokens, hidden_size]
```

For the local 2B model:

```text
shared_embed_delta shape: [1004, 2048]
semantics: additive_delta
```

The trainable tensor should be zero-initialized. Do not initialize the delta
with base rows and then add it to base rows, because that would double the base
embedding.

## PEFT Evidence

PEFT `TrainableTokensConfig` is intended to train select token indices without
training the full embedding matrix. It can install tied output wrappers when
`tie_word_embeddings` and `_tied_weights_keys` are available.

However:

- PEFT trainable-token payloads are absolute selected row values;
- source comments describe replacing values rather than adding/subtracting
  deltas;
- standalone `trainable_tokens` is not mixed-compatible;
- `save_embedding_layers=auto` can full-save embeddings if vocab identity
  appears mismatched.

If PEFT is used in the future, CoordExp must either convert absolute selected
rows into additive deltas at save time or explicitly approve absolute-payload
semantics with strict base-row validation.

## Installed Qwen Hook Points

Installed Qwen3-VL source shows stable hook points:

- text model owns `embed_tokens`;
- when `inputs_embeds` is absent, Qwen calls `embed_tokens(input_ids)`;
- multimodal wrapper calls `get_input_embeddings()(input_ids)` before image
  feature scatter;
- top-level conditional-generation forward computes logits with `lm_head`;
- Transformers ties input/output weights when `tie_word_embeddings=true`.

This supports a custom wrapper without using `inputs_embeds`, preserving the V1
Qwen visual replacement contract.

## Legacy CoordExp Evidence

Legacy `src/tokens/row_offsets.py` already demonstrates the feasibility of:

- freezing base embedding/head weights;
- applying selected input offsets;
- scatter-adding selected output-logit column corrections.

That file is reference-only for V1. It hardcodes old defaults and uses an old
`token_embeddings_adapter` surface, so the new implementation should borrow the
invariant, not the API.

## Required Wave 1B Probe Assertions

The special-token embedding round-trip probe must verify:

1. tokenizer identity:
   - all 1004 selected token strings resolve to expected ids;
   - all selected tokens are single-token;
   - no duplicate ids;
   - coordinate ids are contiguous.
2. real model tie mode:
   - load the local Qwen3-VL model in intended dtype/device;
   - assert config tie mode and actual parameter identity;
   - select `shared_embed_delta` for tied models and split deltas for untied
     models.
3. trainable surface:
   - freeze base `embed_tokens.weight` and `lm_head.weight`;
   - only special-token delta params require grad;
   - optimizer group `token_embeddings` owns them exactly once.
4. input behavior:
   - perturb one selected delta row;
   - matching input embedding lookup changes exactly by that delta;
   - non-selected token lookups remain unchanged.
5. output behavior:
   - controlled hidden states change only selected logit columns;
   - selected-column correction equals `hidden @ delta.T`;
   - non-selected output columns remain unchanged.
6. gradient behavior:
   - tiny loss on coordinate and wrapper token yields nonzero selected-delta
     gradient;
   - base/full embedding/head gradients are absent or zero.
7. save payload:
   - write `special_token_embeddings.safetensors`;
   - write `special_token_embeddings.json`;
   - record tensor key, shape, dtype, token ids/strings, base identity,
     tokenizer identity, tie mode, and `semantics: additive_delta`.
8. reload:
   - load `base + optional adapter + special-token delta`;
   - selected input/output behavior matches pre-save perturbation;
   - non-selected rows/columns remain frozen.
9. negative cases:
   - fail on changed tokenizer ids, token strings, base model identity, tie mode,
     shape/dtype, missing coordinate token, duplicate token id, or accidental
     full embedding/head payload.

## Findings

- **P1:** Additive-vs-absolute semantics must be fixed before implementation.
  V1 should standardize on zero-initialized additive deltas saved as additive
  deltas.
- **P1:** PEFT trainable-token paths are absolute selected-row replacements.
  They cannot be used as-is while claiming `shared_embed_delta` additive
  semantics.
- **P1:** Tied behavior must be asserted after real model load, not inferred
  only from config.
- **P2:** PEFT full-embedding save behavior is a footgun if base/tokenizer
  identity drifts.
- **P2:** LoRA `trainable_token_indices` couples the special-token surface to
  adapter config and payloads, which conflicts with the V1 separation.

## Wave 1B Probe Evidence

Executed from `/data/CoordExp/.worktrees/CoordExp-swift` on 2026-06-30:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/probes/coordexp_swift/special_token_embeddings_roundtrip.py \
  --device cuda --dtype bfloat16
```

Receipt:
`outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json`.

Observed contract:

- selected-token count: 1004;
- selected tokens include four schema wrapper tokens plus `<|coord_0|>` through
  `<|coord_999|>`;
- all selected ids are tokenizer-validated, single-token, unique, and coordinate
  ids are contiguous;
- local Qwen3-VL runtime has tied input embedding and `lm_head` parameter
  identity;
- probe mechanism is `semantics: additive_delta`;
- trainable selected sidecar is `shared_embed_delta`;
- selected input lookup changes by the delta within BF16-aware tolerance;
- non-selected input lookup remains unchanged;
- selected output-logit columns receive `hidden @ delta.T`;
- non-selected output-logit columns remain unchanged;
- selected delta receives gradient while frozen base embedding/head do not;
- compact payload writes `special_token_embeddings.safetensors` and
  `special_token_embeddings.json`;
- in-process reload preserves selected input/output behavior;
- fresh base-model reload validates metadata, installs a fresh wrapper pair,
  loads `shared_embed_delta`, preserves tied embedding/head identity, and
  preserves selected/non-selected input and output behavior.

## Status

Wave 1A evidence supports the custom Qwen wrapper pair as V1 recommendation.
Task 2.4 has source-study evidence for the mechanism comparison. Task 2.5 has
an accepted mechanism decision for V1: custom Qwen wrapper pair, zero-initialized
additive delta payloads, runtime tied/untied detection, compact sidecar
checkpoints, and base-plus-adapter-plus-delta loading. Wave 1B probe evidence
proves the selected mechanism in execution for the local tied Qwen3-VL model.
Production implementation remains blocked by task 2.10 review and the later
OpenSpec implementation approval gate.
