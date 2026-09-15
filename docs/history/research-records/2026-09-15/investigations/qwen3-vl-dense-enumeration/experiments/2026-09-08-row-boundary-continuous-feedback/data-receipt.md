# Row-boundary continuous-feedback data receipt v1

Status: **candidate complete (CPU data only; no quality conclusion)**.

- Manifest: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/manifest.json` (SHA256 `63b89eed7cf05947ccb68facbf4ef245b13517c76ec6b97a44210fd63f21249b`; content `71e01ad9bccebc0d949dbcd84205486b2c876e46d6bd15c731b402e9c7b81a40`)
- Producer: `/data/CoordExp/.worktrees/dora-prox-linear-n2/scripts/research/prepare_row_boundary_feedback_data.py` (SHA256 `f94c477e75e984b1cd8551de756f3beae8168fa0d12b3999729dc7d2a1f86304`) at starting Git HEAD `e4f932faa80718fc487fc56744e9e16f1abf62a7`.
- Train: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/train.jsonl` (SHA256 `a68e7a324d2573dc901842791c2e5c91432f0ee0a994f971b9a9cfa6671f3dc9`; 128 images)
- Dev: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/dev.jsonl` (SHA256 `6f3cf6c8fa9080c5218b27fa82d288693022c9a2fb2d3d421bbe42404c39ce12`; 64 untouched raw image/GT rows)
- Source checkpoint: `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`; adapter tensor `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`; adapter config `0781d04d0579c57a45949193ca3b4afeedbadf92ff237873b070450e3b262739`.
- Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`; Source config `/data/CoordExp/.worktrees/dora-prox-linear-n2/configs/coordexp_infras/infer/source256-policy-source-v1.yaml` (SHA256 `50464b58a407e59ffa54ba088a69529f2c3f4d62c539b4c08f6b7712cf6e4184`, resolved fingerprint `f81a0d9325a30e1ba62c7b75ab8ff80a7af4965e9c25eeeec203f8f733c5c121`).
- Source sampled bank: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/ce-rloo-v1/rloo/round-1/plan.json` (SHA256 `21ab60fa6aeecde863894101f7865b3cf11174196695b563fb79af3fffd3a07f`), raw-softmax T1/top-p1/top-k0/RP1, seeds `[2026090601, 2026090602, 2026090603, 2026090604]`, FP32/SDPA; eight artifact paths/hashes are in the manifest.

Train denominator: 256 candidates -> 236 eligible -> 128 frozen; 108 train-only replacements. Attrition: `{"excluded_reserved_identity": 0, "missing_source_rollout": 0, "no_legal_completed_prefix": 0, "no_remaining_after_legal_prefix": 20}`. Prefix rows: `{"1": 50, "2": 25, "3": 31, "4": 22}`; target alternatives: `{"1": 23, "2": 105}`.

Selected accounting: `{"covered_annotated_owners": 189, "legal_prefix_rows": 281, "matched_prefix_rows": 189, "remaining_annotated_owners": 876, "source_rollout_cells_available": 1024, "supervised_target_rows": 233, "target_geometry_valid": 233, "unmatched_prefix_rows": 92}`.

Every train prefix is an exact contiguous Source-generated token slice ending at a legal `<|box_end|>` after 1--4 complete rows. Existing category-consistent global one-to-one IoU>=0.50 matching defines covered owners. Each distinct target is a remaining annotated owner rendered as the complete standard opener/description/xyxy/box-end row; no target includes EOS. Prefix labels are ignored.

Dev64 was selected from the existing dev128 by ID-only seeded order before reading any model output. It contains no prefix or target and does not affect train eligibility, owner choice, loss, or stopping. This split had prior development-evaluation exposure and is not confirmation. The selected train/dev sets have zero overlap with each other, Human13, the 512-image val confirmation file `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-weak-correction-dose/dose-v1/confirmation-input/confirmation512.jsonl`, and the legacy 128-image held-out artifact `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/clean-rollouts-transition-step36-transfer-max3084-matched-b4-v1/qwen3-vl-2b-transition-step36-transfer-heldout-source-max3084-b4-hf/gt_vs_pred.jsonl`. Exact selected IDs and reserve order are in the manifest.

Verification: `{"assertions": "all panel/provenance/grammar/geometry/hash assertions passed", "dev_rows": 64, "sensitivity": {"bad_provenance_rejected": true, "overlap_rejected": true}, "status": "PASS", "train_rows": 128}`. Re-run:

```bash
conda run -n ms python scripts/research/prepare_row_boundary_feedback_data.py check --manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-row-boundary-continuous-feedback/data-v1/manifest.json
```
