# Untied special-token embeddings: production preparation

Owner: this investigation. Base checkout: `coordexp-infras`, starting at
`4b97a5daa03aabbe434682b30cae9aeacd330a23`.

## Current authorized production objective (2026-09-18)

The user subsequently authorized **untied deltas + CE + low-weight axis validity**
in one production run. Use `configs/train/geo_sorted_xy/untied_axis.yaml`.
This supersedes the untie-only contrast below; the original tied/untied configs
remain unchanged as controls. CE weight is 1, axis weight is 0.01, and margin is
1/999 in normalized coordinates. Token-type gate stays at zero. Exact
same-world-size training state is enabled from the first checkpoint.

For each complete box, coordinate-only FP32 softmax yields expected coordinates
mu. Its loss is `(relu(margin + mu_x1 - mu_x2) +
relu(margin + mu_y1 - mu_y2)) / 2`. Average boxes within each supervised segment,
then segments across the global optimizer step, including zero for no-complete-box
segments. Group by pack/segment/example/object identity and declared slot, not
contiguous token positions. Incomplete groups contribute no box term and are
counted in diagnostics. This is teacher-forced expected-axis regularization;
it does not guarantee valid greedy boxes or model-quality improvement.

Reuses the research-probes axis kernel with current infra loss bindings, global
denominators and DDP gradient compensation. FP32 elementwise expectation
reduction avoids BF16 autocast lowering the one-bin margin calculation.
No additional model forwards, parameters, checkpoint payload or inference loss
are introduced.

The original untie-only acceptance remains in `acceptance.md`; the new combined
run acceptance and launch packet are recorded separately in `axis-production.md`.

## Frozen contrast

The user selected the easiest implementation while preserving the usual training
surface: language DoRA plus the 1,004 selected coordinate/wrapper tokens. The
single intervention is independent input/output deltas. This does **not** enable
full-vocabulary embedding or head training.

Both effective matrices start at the same pretrained weights:

```
input  = frozen_pretrained_weight + selected_rows(input_delta)
output = frozen_pretrained_weight + selected_rows(output_delta)
input_delta = output_delta = 0 at initialization
```

The frozen base storage can remain shared; only the trainable deltas must have
different parameter/storage identities. No additional base-model copy is needed.
The base HF config is left intact. The experiment flag belongs to
`model.special_token_embeddings.tie_word_embeddings`, and describes effective
delta tying rather than mutation of the pretrained snapshot's architecture.
Added trainable parameters: `1004 * 2048 = 2,056,192`; parameter, gradient, and two
FP32 Adam moments add approximately 31.4 MiB per rank, excluding runtime buffers.
The actual pretrained safetensors index contains the input embedding key and no
independent `lm_head.weight` key. Setting HF's tying flag to false before loading
would therefore not constitute the requested copy initialization.

## Matched configuration

The authority is the archived **resolved configuration**, not the old YAML
filename (which mentions EBS64) or old preparation notes:

`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/resolved_config.json`

Its completed run used world size 8 and reached step 2444. The requested new run
starts from the pretrained base, not from this trained checkpoint.

| Setting | Value |
|---|---|
| Base | `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent` |
| Epochs / effective batch | 4 / 24 |
| Topology | 8 ranks, 3 accumulated packed micro-steps per rank per full update |
| DoRA | Language tower, all linear targets, r16/a32, dropout 0 |
| LR | Language adapter 2e-4; both token deltas 1e-4 |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0 |
| Schedule | Cosine, warmup ratio 0.1, max grad norm 1 |
| Packing | Source-order next fit, global max length 12000 |
| Objective | Segment-balanced CE; auxiliary losses disabled |
| Precision / attention | BF16 / FlashAttention2 |
| Data | Original `rescale_32_1024_bbox_len12000_xy_sorted` train/val |
| Ordering / prompt | `geo_sorted_xy`, desc-first, exact archived full prompt |
| Eval / checkpoint | Every 0.4 of the schedule; final checkpoint enabled |
| Seed | 17 |

The current generic production template differs in EBS, ordering and prompt.
Use the matched overlays:

```
configs/train/production.yaml
  -> configs/train/geo_sorted_xy/tied.yaml
    -> configs/train/geo_sorted_xy/untied.yaml
      -> configs/smoke/eight_gpu_geo_sorted_xy_untied.yaml
```

Resolved tied and untied production configs differ only in run name and the
special-token tying flag. Current strict-replay, cache, and runtime implementation
are newer than the historical run: matched hyperparameters do not imply a
bitwise replay of historical training. The archived packing algorithm was also
source-order next fit. Current train/val file hashes match archived provenance.

## Checkpoint and inference contract

Tied checkpoints retain their original `shared_embed_delta` tensor. Untied
checkpoints contain **both** `input_embed_delta` and `output_embed_delta`, each
FP32 `[1004, 2048]`, in `special_token_embeddings.safetensors`; metadata records
`tie_word_embeddings: false`. Shape, dtype, key set and base/tokenizer identity
are checked. Both tensors are validated before either runtime tensor is copied.
The existing atomic checkpoint-directory publication and manifest hashing cover
the complete two-tensor file. Exact training state enumerates both trainable
parameters and their optimizer state.

HF dynamic inference reads the payload's tying flag, installs two independent
deltas, then loads both. Supply both the `adapter` and `embedding_delta` paths.
Do not use the adapter directory alone or generic PEFT `save_pretrained` alone:
those do not include this separately owned embedding payload.

Dense materialization / vLLM remains unsupported for this untied format and fails
closed. It is outside this minimal HF route. Switching to full-vocabulary trainable
matrices or replacing the existing wrappers with PEFT Trainable Tokens would
change scope and is unnecessary here. Upstream context:
[PEFT Trainable Tokens](https://huggingface.co/docs/peft/main/package_reference/trainable_tokens).

## Launch commands

Run from `/data/CoordExp/.worktrees/coordexp-infras` using the configured `ms`
Python environment. The commands below are preparation instructions, not evidence
that full production has run:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1

python -m src.prepare_train_cache --config configs/train/geo_sorted_xy/untied.yaml
python -m src.prepare_train_cache --config configs/train/geo_sorted_xy/untied.yaml --require-all-hit
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 --local-addr=127.0.0.1 \
  -m src.train --config configs/train/geo_sorted_xy/untied.yaml
```

The matched production config retains `resume.mode: disabled` from the generic
template; inference payload saving is independent of exact optimizer/RNG saving.
To publish resumable training state from the beginning, use an overlay:

```yaml
schema_version: 1
extends: /data/CoordExp/.worktrees/coordexp-infras/configs/train/geo_sorted_xy/untied.yaml
resume:
  mode: exact_same_world_size
  checkpoint_dir: null
```

Use that same authored overlay for cache preparation and training. A continuation
sets `resume.checkpoint_dir` to a nonterminal checkpoint and retains the total
schedule and world size; it does not restart the epoch budget.

For HF inference, the tested authored example is
`/data/CoordExp/outputs/infra_base/untie-20260918/infer-one-row.yaml`.
For a production checkpoint, copy that YAML and change the run name, input JSONL,
`adapter.path` to `<checkpoint>/adapter`, and `embedding_delta.path` to
`<checkpoint>/special_token_embeddings`. Keep its exact native prompt, BF16/FA2,
and explicitly authored generation/scoring fields. The current inference loader
requires the leaf YAML to author input JSONL, batch size, maximum new tokens,
temperature, top-p and scoring; inheriting all those fields is rejected.

```bash
python -m src.infer --config /absolute/path/to/the-authored-inference.yaml
```

When placing a subset JSONL outside its original directory, rebase image paths
relative to the new JSONL while preserving the resolved image files. The tested
example does this and retains the original object rows.

The bounded smoke uses 256 training rows, 16 eval rows, two updates, saves/evaluates
at steps 1 and 2, and publishes exact training state. Its follow-up resumes step 1
and compares final adapter and both delta tensors against uninterrupted training.
HF reload and generation use separate fresh processes. Evidence root:

`/data/CoordExp/outputs/infra_base/untie-20260918/`

Final acceptance, timing, resource observations and any remaining blockers are
recorded in `acceptance.md` after the live checks finish.
