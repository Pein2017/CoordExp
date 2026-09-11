---
doc_id: docs.standards.porting
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Porting guidance and compatibility notes.
updated: 2026-06-07
---

# Porting notes between Qwen3-VL and CoordExp

## Qwen-VL Family Differences

Use [`UPSTREAM.md`](UPSTREAM.md) and
[`upstream/QWEN_VL.md`](upstream/QWEN_VL.md) for detailed upstream handles.
Do not assume Qwen2-VL, Qwen2.5-VL, and Qwen3-VL share processor, video, RoPE,
or attention behavior.

| Surface | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| Model type | `qwen2_vl` | `qwen2_5_vl` | `qwen3_vl` |
| Vision path | older Qwen-VL baseline | windowed/full-attention vision patterns | DeepStack visual feature injection |
| Video timing | older video flow | `second_per_grid_ts` path | text timestamp/frame-block path |
| Position IDs | 3D MRoPE | 3D MRoPE with Qwen2.5 variants | 3D MRoPE; optional 4-row tensor where row 0 is text positions |
| Packing risk | attention/position reset handling | attention/window/grid handling | row-0 text boundaries plus rows 1-3 MRoPE geometry |
| CoordExp rule | inspect exact installed source | inspect exact installed source | inspect exact installed source |

For CoordExp, the stable rule is to preserve image order, `image_grid_thw`,
video grids, token order, and coordinate geometry before crossing the upstream
model/template boundary. Any port that changes processor class, model family,
or `attn_implementation` must rerun placeholder-count, grid, position-id, and
loss/logit availability checks.

## Supervision and metric ownership

Current CoordExp supervision is owned by
[`src/supervision/`](../../src/supervision/) and loss assembly by
[`src/losses/runner.py`](../../src/losses/runner.py). Use the
[supervision/loss contract](../../openspec/specs/coordexp-infras-supervision-losses/spec.md)
for token types, causal alignment, reductions, and emitted metrics, and the
[implementation map](../IMPLEMENTATION_MAP.md) for verification owners.
Porting a model family does not replace those local semantics with an upstream
trainer's token labels, dataset defaults, or mismatch handling.
