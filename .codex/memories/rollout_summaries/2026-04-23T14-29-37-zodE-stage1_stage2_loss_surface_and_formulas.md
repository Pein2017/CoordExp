thread_id: 019dbabf-1321-72b2-90d9-c445da0b8b06
updated_at: 2026-04-23T14:35:25+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/23/rollout-2026-04-23T14-29-37-019dbabf-1321-72b2-90d9-c445da0b8b06.jsonl
cwd: /data/CoordExp
git_branch: main

# Explored the current Stage-1 and Stage-2 loss surfaces, then expanded them into explicit formulas

Rollout context: The user asked for a summary of all current losses in the Stage-1 and Stage-2 training pipelines for both coordinate tokens and standard text tokens, and then asked to show the loss computation formulas as well. The repo root was `/data/CoordExp`. The assistant used the canonical training docs plus the live trainer/loss code to ground the summary in the current implementation.

## Task 1: Summarize current Stage-1 and Stage-2 losses for coord tokens and standard text tokens

Outcome: success

Preference signals:
- The user asked for “all my current losses” across Stage-1 and Stage-2, for both coordinate tokens and standard text tokens -> future answers should cover the full active loss surface rather than only the coord branch or only the text branch.
- The user then asked to “show out those loss computation formulas as well” -> future responses should include the actual math / formulas when discussing training losses, not just names and prose.

Key steps:
- Read the canonical training docs first: `docs/training/README.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, and `docs/training/METRICS.md`.
- Verified the active config surfaces in `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml`, `configs/stage1/profiles/2b/coord_ce_soft_ce_gate.yaml`, `configs/stage1/profiles/2b/raw_text_xyxy_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml`, and `configs/stage2_two_channel/_shared/objective_tuned.yaml`.
- Checked the live implementations in `src/trainers/teacher_forcing/modules/token_ce.py`, `coord_reg.py`, `bbox_geo.py`, `bbox_size_aux.py`, `loss_duplicate_burst_unlikelihood.py`, `src/trainers/stage2_two_channel/objective_runner.py`, and `src/trainers/losses/coord_soft_ce_w1.py` / `bbox_geo.py` / `src/trainers/teacher_forcing/geometry.py`.
- Confirmed the current Stage-2 two-channel pipeline is channelized, with Channel-A and Channel-B logged separately, and that the rollout-aligned Stage-2 path uses a smaller default objective set (`token_ce + bbox_geo + coord_reg`) unless explicitly extended.

Failures and how to do differently:
- `rtk` was not available in the shell, so exploration fell back to raw `sed`/`rg`/`nl` reads. Future similar sessions in this shell should not assume `rtk` is present.
- Serena’s configured project root did not resolve `src/sft.py` during symbol lookup, so the assistant had to rely on local file reads for code verification. Future similar work should be prepared to fall back to raw reads if Serena’s project binding is mismatched.
- The assistant initially consulted an older memory entry to avoid stale Stage-1 assumptions; that guardrail was useful, but the current rollout still needed direct code reads to verify the live loss contracts.

Reusable knowledge:
- Stage-1 standard coord-token training still uses the `coord_soft_ce_w1` family for coord positions: hard coord CE, softCE, W1, coord gate, text gate, and optional adjacent repulsion, while base CE is masked to non-coord tokens.
- Stage-1 raw-text benchmark disables coord tokens entirely (`coord_tokens.enabled: false`, `coord_soft_ce_w1.enabled: false`), so it is plain CE over serialized text/numeric coordinates.
- Stage-1 can also add `bbox_geo` and `bbox_size_aux` on top of coord-token supervision.
- In the active `stage2_two_channel` path, the main objective families are `token_ce`, `loss_duplicate_burst_unlikelihood` (Channel-B only), `bbox_geo`, `bbox_size_aux`, and `coord_reg`, plus `coord_diag` diagnostics.
- `token_ce` is the shared text loss in Stage-2; it masks coord tokens and splits into structure and description components.
- `coord_reg` contains the coord-token CE / softCE / W1 / gating family, and `text_gate` is a penalty on supervised text positions that leak coord-vocab mass.
- The supported `stage2_rollout_aligned` path reuses the same shared module semantics for text CE, bbox geometry, bbox size aux, and coord regularization, but its default manifest is smaller and does not include duplicate-burst unlikelihood unless explicitly authored.

References:
- [1] `docs/training/STAGE1_OBJECTIVE.md` and `docs/training/METRICS.md` for canonical Stage-1 loss families and metric keys.
- [2] `configs/stage1/_shared/coord_soft_ce_gate_4b.yaml:1`, `configs/stage1/profiles/2b/coord_ce_soft_ce_gate.yaml:15`, and `configs/stage1/profiles/2b/raw_text_xyxy_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml:44` for the current Stage-1 configuration split between coord-token and raw-text modes.
- [3] `configs/stage2_two_channel/_shared/objective_tuned.yaml:7` for the current Stage-2 two-channel objective order and default weights.
- [4] `src/trainers/stage2_two_channel/objective_runner.py:177` for how Stage-2 logs the objective atoms into `loss/text/*`, `loss/B_rollout_text/*`, `loss/coord/*`, and `loss/B_coord/*` namespaces.
- [5] `src/trainers/teacher_forcing/modules/token_ce.py:20`, `coord_reg.py:54`, `loss_duplicate_burst_unlikelihood.py:12`, `bbox_geo.py:44`, `bbox_size_aux.py:51`, and `src/trainers/teacher_forcing/geometry.py:417` for the concrete formula implementations.

## Task 2: Expand the loss formulas explicitly

Outcome: success

Preference signals:
- The user specifically requested the formulas after the conceptual summary -> future responses on this topic should include the mathematical form alongside the narrative description.

Key steps:
- Pulled the exact math from `src/coord_tokens/soft_ce_w1.py`, `src/trainers/losses/coord_soft_ce_w1.py`, `src/trainers/teacher_forcing/geometry.py`, `src/trainers/losses/bbox_geo.py`, and `src/trainers/losses/bbox_size_aux.py`.
- Wrote the formulas in a pipeline-oriented form, separating Stage-1 and Stage-2 and making explicit which tokens/objects each term applies to.
- Captured the shared expectation decode path used by bbox losses: coord logits are softmaxed over 1000 bins, then the expected bin index is used to build continuous box coordinates.

Reusable knowledge:
- The expectation decode path is `softmax(coord_logits / τ)` over 1000 bins, then the expected bin index is normalized by 999.
- `softCE` uses a truncated Gaussian target distribution over ordered bins; `W1` is computed by CDF differences on the discrete line.
- `coord_gate` is `-log(p_coord)`, and `text_gate` is `-log(1 - p_coord)`.
- `bbox_geo` is the sum of SmoothL1 and CIoU terms, with CIoU computed from canonicalized `xyxy` boxes.
- `bbox_size_aux` is `log_wh` plus an optional oversize hinge penalty.
- `loss_duplicate_burst_unlikelihood` is the mean of `-log(1 - p_bad)` over targeted bad continuation tokens at duplicate-burst boundaries.

Failures and how to do differently:
- None material; the main risk was over-compressing the formulas. The better pattern here was to read the implementation and then restate the math directly from the code rather than infer from docs alone.

References:
- [1] `src/coord_tokens/soft_ce_w1.py:58-280` for Gaussian target construction, softCE, W1, and combined coord loss.
- [2] `src/trainers/teacher_forcing/geometry.py:45-480` for expectation decode, bbox canonicalization, bbox regression, oversize penalty, and CIoU.
- [3] `src/trainers/losses/bbox_geo.py:44-125` and `src/trainers/losses/bbox_size_aux.py:51-187` for how the bbox formulas are assembled into Stage-1/Stage-2 loss modules.
- [4] `src/trainers/teacher_forcing/modules/loss_duplicate_burst_unlikelihood.py:12-130` for the duplicate-burst unlikelihood math.
- [5] `src/trainers/teacher_forcing/modules/token_ce.py:20-259` and `src/trainers/teacher_forcing/modules/coord_reg.py:54-340` for the text CE and coord regularizer decomposition.

