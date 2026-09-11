# CoordExp Notion local source recovery manifest

Date: 2026-08-26
Scope: local-source recovery only; no new training, inference, probe, or Notion mutation.
Authority: provenance handoff, not a new scientific result or ontology.

## Decision summary

- Tight-outline: the exact committed research notes were recovered, but their referenced Step-0/Step-1 checkpoint, input JSONL, and primary artifacts are no longer present locally. The defensible claim is a bounded teacher-prefix visual-steering result, not physical-owner identification or future free-running causal use.
- Multi-rollout: the broad K=16 three-checkpoint artifact is present and reproducible at the stored metric surface. It supports only that sampling exposes additional strict-matched owner support.
- Ordering: **PARTIAL**. There are matched within-checkpoint prefix-order interventions and several directional training-lineage comparisons, but no publication-grade training comparison that changes only `y->x` versus `x->y` or only sorted versus random order.
- Address/Commit: retrospective commit-state readout is recovered and bounded. A separate future-use route was executed and failed its target-specific causal-use and safety gates; decodability does not imply future causal use.
- Evaluation: stable parser, artifact, category, matching, malformed, EOS, native-greedy, and provenance primitives are recoverable. They are distributed across production and research-local evaluators, not one universal scientific protocol.
- Industrial polygon/line: no local metric-bearing historical artifact was recovered.

## Compact evidence manifest

| Claim | Notion page | Repo source | Artifact | Source SHA | Scope | Evidence surface | Recovery verdict |
|---|---|---|---|---|---|---|---|
| A. A structured tight boundary cue steers the next target row/geometry | Tight-outline claim `3c79d9ce-3f59-8118-8c7e-e8c4624ff7ad` | `research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-08-pvci-step0-preparation-results/unit.md`; Step-1 sibling | Referenced `/data/CoordExp/outputs/painted_gt/pvci_step0/...` and `pvci_step1/...`; currently absent | `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce` | Held-out val100/val32, teacher-prefix, painted step-484 adapter, target-row evaluator | Committed result note with exact paths, controls, and metrics; no surviving primary artifact | `RECOVERED_BOUNDED` |
| B. The cue proves physical-owner identity selection | Same page | Same Step-0/Step-1 notes | Primary artifacts absent | `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce` | Wrong-mark and wrong-aspect controls | Mark following is geometry-dominant and does not isolate identity from supplied geometry | `CLAIM_TOO_STRONG` |
| C. A future free-running policy causally uses the same tight-outline route | Same page | Step-0/Step-1 notes explicitly exclude production inference | None | `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce` | Teacher-prefix only | No same-route free-running causal-use evidence | `CLAIM_TOO_STRONG` |
| K=16 sampling exposes owners omitted by native greedy | Multi-rollout claim `3c79d9ce-3f59-81ef-83a0-dc6ce23a436a` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084` | `e9614de255c1db446dc77f1c08dd7ab5c1dad6b9` | 12 human-refined images, 346 owners, three step-4887 checkpoints, K=16 | Stored sampled shards plus byte-reproduced class-aware union metrics | `RECOVERED_SUPPORTED` |
| Union owners are jointly compilable into one valid trajectory | Same page | Same source explicitly limits union interpretation | Same artifact | `e9614de255c1db446dc77f1c08dd7ab5c1dad6b9` | Cross-trajectory set union | No compilation or single-trajectory construction evidence | `CLAIM_TOO_STRONG` |
| Earlier serialized-row order can change the next route at fixed tested states | Ordering claim `3c79d9ce-3f59-8123-8a2a-dc75e82aaa65` | `.../2026-07-20-matched-random-sorted-prefix-order-screen/results.md` | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen` | `c3ce45468bdcbbd4a84096d3b9bad284532917be` | Selected images, fixed checkpoint/prompt/prefix tokens; only earlier row order changes within a pair | Paired greedy/sampled owner-route and candidate-row-score interventions | `RECOVERED_BOUNDED` |
| Existing training evidence isolates ordering-only gain | Same page; Serialization owner `3c79d9ce-3f59-81c0-8a6c-dbac1a9f2254` | Eight-coordinate closeout, July-20 screen, June-04 smoke | Eight-coordinate closeout and July-20 roots present; June-04 root absent | `1521772db8a05f8b82d25de8a18d4078503a5475`; `c3ce45468bdcbbd4a84096d3b9bad284532917be`; `15a705ef88d1e0788a9a13a0b794b682724adf6d` | Different checkpoints, epochs, batch size, objective, arity, caps, source commits, or cache versions | Directional lineage plus bounded within-checkpoint interventions | `CLAIM_NEEDS_SPLIT` |
| Commit hidden state contains retrospectively decodable owner-local information | Architecture owner `3c79d9ce-3f59-810c-9fea-cc18e721270a` | `openspec/changes/probe-owner-commit-frozen-set-controller/`; frozen P0 config/scripts | `.worktrees/owner-commit-binding/outputs/probes/coordexp_swift/frozen_owner_set_probe/a3_step2445/finalize/` | `093b4a2d0b8983c4786fb32ba9cdde3202996b6e` | Frozen A3 step-2445, 13 images, generated-state replay | 2,046 matched observations; retrieval and adjacent-state controls | `RECOVERED_BOUNDED` |
| Retrospective commit readout proves that future generation causally consumes the state | Same page | Frozen P0 explicitly lacks an intervention on future generation | Same P0 artifact | `093b4a2d0b8983c4786fb32ba9cdde3202996b6e` | Readout only | No future-token intervention in P0 | `CLAIM_TOO_STRONG` |
| The tested split-layer proposal pulse safely and target-specifically controls future rows | Same page | `.../2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md` | `/data/CoordExp/outputs/probes/coordexp_swift/pvci_own_prefix_sharded_panel_ABC_320_v3_20260712/` | `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce` | 320 held-out images, 16 matched cells, RP1.10/RP1.00 | C-on/C-off and negative-control own-prefix rollouts; gate HOLD | `CLAIM_TOO_STRONG` |
| Stable dense-enumeration evaluation primitives can be recovered | Causal Evaluation owner `3c79d9ce-3f59-8173-ba20-c5cade809d97` | `docs/eval/CONTRACT.md`, `src/inference/{parsing,backend,artifacts}.py`, `src/eval/detection_consumer.py`, `src/vis/matching.py` | Contract-bound JSONL, trace, provenance, metrics, receipt artifacts | File SHAs listed below | Current Swift official evaluator plus bounded research matcher | Source-backed stable primitives; not one universal experimental protocol | `RECOVERED_BOUNDED` |
| Richer industrial polygon/line supervision caused near-full AR detection without ordering | Industrial claim `3c79d9ce-3f59-8157-b68f-ceadaf12a16c` | Only legacy geometry/schema design lineage recovered | No checkpoint/run/evaluator/metric artifact found | None | Historical claim has no locally recoverable experiment identity | Schema support is not model-quality evidence | `SOURCE_MISSING` |

## P0. Tight-outline / painted-mark visual steering

### Exact recovered identity

- Research source:
  - `/data/CoordExp/.worktrees/research-probes/research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-08-pvci-step0-preparation-results/unit.md`
  - Step-1: sibling `2026-07-08-pvci-step1-precommit-probes/unit.md`
- Commit: `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce`.
- Recorded checkpoint: painted stepwise teacher-prefix adapter and embedding delta at `.../checkpoints/step-484/`.
- Recorded input: `coco_val200_len12000.rebased_images.coord.jsonl` from the coordexp-infras worktree.
- Decode: HF, temperature `0`, top-p `1`, repetition penalty `1.10`, max new tokens `96`, batch size `2`.
- Evaluator: target-row F1/precision/recall and step validity; wrong-mark follow analysis compares the first parsed prediction against intended and marked boxes.
- Slices:
  - held-out val100: 825 target rows;
  - held-out val32: 256 rows per coarseness variant;
  - train256 appears only as an overfit reference, not a held-out control.

### Metrics and controls

| Control | Metric/result | Interpretation |
|---|---:|---|
| correct tight mark, val100 | F1 `0.748340` | bounded steering result |
| unpainted, same prompt | F1 `0.143558` | visual mark has large effect |
| wrong-object mark, scored against intended target | F1 `0.017857` | intended target is not retained |
| wrong-object mark, first prediction overlaps marked object | `636/653 = 0.973966` at IoU >= 0.5 | mark steers geometry/row rather than merely disrupting output |
| outline only, val32 | F1 `0.757282` | center point is unnecessary |
| center blob, val32 | F1 `0.126160` | coarse central cue is weak |
| 1.5x outline, val32 | F1 `0.208897` | coarse boundary rapidly degrades |
| 2.0x outline, val32 | F1 `0.050193` | large region cue is ineffective for this teacher |
| Step-1 wrong-aspect outline, val100 | F1 about `0.7275`; closer-mark about `0.9805`; prediction-to-mark IoU about `0.8937` | recorded verdict `geometry_leakage_dominant`; supplied mark geometry dominates clean owner identity |
| Step-1 background outline | F1 about `0.0461` | arbitrary rectangle is not sufficient |

The referenced Step-0/Step-1 artifact roots, step-484 checkpoint, and input JSONL are currently absent. The exact result note is therefore the surviving evidence surface. This is enough to restore provenance and bounded wording, not to claim artifact-level reproducibility.

### Required three-way status

- A. Structured boundary cue causally steers row/geometry: **supported, bounded to the recorded teacher-prefix checkpoint and held-out slices**.
- B. Cue proves physical-owner identity selection: **not supported**. Wrong-aspect/closer-mark behavior is geometry-dominant.
- C. Future free-running policy uses the same route: **not supported**. The units explicitly exclude production/free-running inference.

## P0. Broad multi-rollout union

### Exact recovered identity

- Checkpoints: Sorted, Random, and Permutation arms, each nominal step `4887`; their exact sampled-shard identities are pinned below.
- Evaluation data: `human-refined-12.coord.jsonl`, 12 images, 346 trusted owners.
- Sampler: K=`16`, seeds `21001..21016`, temperature `0.4`, top-p `0.95`, repetition penalty `1.0`, max new tokens `3084`, HF fp32.
- Native greedy baseline: separate stored manifests; do not infer it from the sampled configs. The configs contain temperature `0` and RP `1.10`, while sampled shards explicitly record RP `1.0`.
- Evaluator: deterministic class-aware complete-link clustering at IoU `0.5`, medoid detection, then class-aware one-to-one matching at IoU `0.5`.
- Config fingerprints:
  - Sorted `d44ab5a6112e3fc05b5df6f1d03d6dbc7e36a4c192dce796a28310d93feee43e`;
  - Random `ed8d677f...`;
  - Permutation `1a862866...`.
- Key artifact hashes:
  - Sorted `f1-metrics.json`: `0ed6a8e440a12c0e3b15aff43e6c7c073ff4e93e3afa26f178d9e75317c48688`;
  - Random: `b7f27790e68a4ea9c34253de849df601505f795be00843c2736e04fe81f955d5`;
  - Permutation: `4f7a39387c96d197236237e6c262720f3214df7868b14b3a720ca1f0aa2a5768`.

### Complete sampled-shard `model_identity`

The value is identical in `sampled/shard-0.json` and `sampled/shard-1.json` for each arm. `Identity SHA256` is over `jq -c '.model_identity'`, including backend/effective settings, base, adapter, embedding delta, processor, tokenizer, and generation identity. The exact source field plus this reproducible digest pins the complete value; the checkpoint-defining paths are expanded here so the three arms are not represented by truncated display names.

| Arm | Exact sampled-shard source | Adapter | Embedding delta | Base | Generation fingerprint | Complete identity `jq -c` SHA256 |
|---|---|---|---|---|---|---|
| Sorted | `outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/sorted/sampled/shard-{0,1}.json` → `.model_identity` | `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/adapter` | `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/special_token_embeddings`; additive F32 `[1004,2048]`; base-config SHA256 `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de` | `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent` | `14faa4a4a69a1e5a2647d61943fedd79b10fe9f30fc18324834f0b436ce1cdd6` | `ad69be21580d4d3dbf1f521c7d83e8e6fae1248cbbbc88984d5aef7251c85a7c` |
| Random | `outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/random/sampled/shard-{0,1}.json` → `.model_identity` | `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_random_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1-20260719T070043Z/checkpoints/step-4887/adapter` | `/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_random_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1-20260719T070043Z/checkpoints/step-4887/special_token_embeddings`; additive F32 `[1004,2048]`; base-config SHA256 `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de` | `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent` | `14faa4a4a69a1e5a2647d61943fedd79b10fe9f30fc18324834f0b436ce1cdd6` | `b748e613c7b43706ead1d5ade4d5ffb2b9278a6b2dc45c6312de96b34ca8c74d` |
| Permutation | `outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/permutation/sampled/shard-{0,1}.json` → `.model_identity` | `/data/CoordExp/outputs/prod/coordexp_swift/permutation_bundle_coordinate_noise/single_arm_probe/qwen3_vl_2b_random_step4887_permutation_noise_same_image_bundle_single_arm_probe_k8_n1_b24-20260728T151102Z/checkpoints/step-4887/adapter` | `/data/CoordExp/outputs/prod/coordexp_swift/permutation_bundle_coordinate_noise/single_arm_probe/qwen3_vl_2b_random_step4887_permutation_noise_same_image_bundle_single_arm_probe_k8_n1_b24-20260728T151102Z/checkpoints/step-4887/special_token_embeddings`; additive F32 `[1004,2048]`; base-config SHA256 `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de` | `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent` | `14faa4a4a69a1e5a2647d61943fedd79b10fe9f30fc18324834f0b436ce1cdd6` | `c317d6a94c96534b3dab52ea203307bdc983077b77c4dcb493a595f3a41784ec` |

Shared complete-record properties: HF `generate`, Transformers `4.57.1`, SDPA, patch-embed linearization enabled, active DoRA adapter `default`, tied embeddings, Qwen3-VL-2B (`text_hidden_size=2048`), and contiguous coordinate token IDs `151670..152669`. These are descriptive fields from the recorded identity, not reconstructed assumptions.

| Checkpoint | Greedy strict owner count | K16 union owner count | Union TP / FP / FN | Greedy-missed recovered owners | FP taxonomy: class-absent / mis-grounded / duplicate / loose |
|---|---:|---:|---:|---:|---:|
| Sorted | 107 | 191 | 191 / 476 / 155 | 88 | 13 / 184 / 65 / 214 |
| Random | 132 | 162 | 162 / 431 / 184 | 44 | 4 / 127 / 70 / 230 |
| Permutation | 121 | 169 | 169 / 423 / 177 | 60 | 2 / 136 / 67 / 218 |

All cluster descriptions mapped to known COCO names (`unknown_category_prediction_cluster_count=0`). The broad artifact does **not** contain human-adjudicated entity-level `unsupported` counts; do not relabel all FP clusters as unsupported hallucinations.

Allowed conclusion: sampling exposes additional owner support.
Forbidden conclusion: the recovered owners can be jointly compiled into one valid trajectory.

## P0. Ordering lineage

| Lineage | Order | Checkpoint / epoch | Data | Recipe | Decode | Evaluator | Metric/result | Isolation status |
|---|---|---|---|---|---|---|---|---|
| 2026-06-04 mechanism smoke | sorted vs random | both named ckpt-3668; epoch not recovered | smoke panel; exact data artifact absent | full-object pure CE, no-newline | free-text greedy temp0 | diagnostic non-metric parser | different boundary/FN profiles | Not an accuracy comparison; artifact absent |
| 2026-07-20 prefix-order screen | geometry-sorted vs random-trained | both step-4887, one checkpoint/seed per policy | selected dense images; same nominal source and length | same base/prompt/row format/DoRA/optimizer-scale; different source commits and packing caches | paired greedy plus T0.4 samples; fixed prefixes; one row | physical-owner matching and complete-row score decomposition | both path-dependent; earlier-order-only intervention changes next route at selected states | Matched within-checkpoint prefix intervention; training comparison remains confounded |
| 2026-07-29 broad union | Sorted | step-4887 | human-refined12 | geo-sorted type-gated arm | K16 T0.4/top-p.95/RP1.0; separate greedy | class-aware union matcher | greedy 107, union 191 | Not ordering-only |
| 2026-07-29 broad union | Random | step-4887 | same | random-order arm | same | same | greedy 132, union 162 | Not ordering-only |
| 2026-07-29 broad union | Permutation | step-4887 | same | random + permutation-bundle arm | same | same | greedy 121, union 169 | Not ordering-only |
| 2026-08-05 closeout | `y->x` legacy `geo_sorted` | step-917, 4 epochs | val200 | 4-coord, pure CE, gate0, EBS64 | retained eval; local/operator evaluator identity differs | COCO AP | `0.411179` operator; `0.414944` local | Comparator differs in EBS/evaluator provenance |
| 2026-08-05 closeout | `y->x` legacy `geo_sorted` | step-4887, 8 epochs | val200 | 4-coord, CE + type gate0.2, EBS24 | max 3084 family | COCO AP | `0.415552` | Differs in epoch/objective from xy rows |
| 2026-08-05 closeout | `x->y` `geo_sorted_xy` | step-5529, 8 epochs | val200 | 8-coordinate clockwise, gate0.2, EBS24 | max 3084 | COCO AP | `0.430679` | Ordering and arity differ |
| 2026-08-05 closeout | `x->y` `geo_sorted_xy` | step-2444, 4 epochs | val200 | 4-coordinate, pure CE, gate0, EBS24 | max 512 | COCO AP | `0.434394` | Ordering and EBS differ from step-917; cap/evaluator lineage not fully matched |

Final answer to “is there a truly matched ordering-only comparison?”: **PARTIAL**.

- YES only for bounded within-checkpoint prefix-order interventions at selected states.
- NO for a clean training-effect estimate of `x->y` versus `y->x`, and NO for a publication-grade sorted-versus-random retraining comparison.

## P1. Address / Commit

### Retrospective readout

Frozen P0 source: `/data/CoordExp/.worktrees/owner-commit-binding/openspec/changes/probe-owner-commit-frozen-set-controller/`, source SHA `093b4a2d0b8983c4786fb32ba9cdde3202996b6e`.

Artifact: `/data/CoordExp/.worktrees/owner-commit-binding/outputs/probes/coordexp_swift/frozen_owner_set_probe/a3_step2445/finalize/`.

- 13 images, one greedy plus 16 sampled trajectories each, RP `1.10`, 221/221 cells.
- Legacy12: greedy `114/346`; K16 union `172/346`; 59 owners absent from greedy.
- Image2299: greedy `14/46`; K16 union `34/46`; 21 owners absent from greedy.
- 2,046 strict-matched generated commit observations.
- Own-owner retrieval@1 `75.27%`; same-class retrieval@1 `71.30%`.
- Commit state exceeds box-end and last-coordinate controls for all 2,046 observations; exceeds cyclic next-image same-geometry control in `93.65%`.

### Exact representation definition for retrieval@1 `75.27%`

- **Layer:** the replay hooks the unique `model.language_model.norm` module. The query is the final post-norm language hidden state, not an intermediate layer. Visual candidates come from the unique `model.visual.merger` output.
- **Commit token/boundary:** every parsed generated row is aligned, in forward row order, to the exact generated-token subsequence `raw_span_text + <|commit|>`. `<|commit|>` is token ID `151669`; if it is generated step `k`, the captured full-sequence column is `prompt_width + k`, after consuming the commit token. The companion boundaries are `<|box_end|>` at `k-1` and the last coordinate at `k-2`.
- **Replay and query construction:** exact prompt plus complete generated IDs are replayed once with the same model/image tensors, `use_cache=False`, and no supplied `inputs_embeds`, `position_ids`, cache, or `past_key_values`; the final-norm hook captures detached commit, box-end, and last-coordinate vectors.
- **Owner prototype/readout construction:** for each strict-matched row, use the row's parsed predicted `coord_bins`—never the GT box—to select executed-image main-merger rows. The prototype is their bbox-overlap-fraction-weighted mean, then L2-normalized. Query and prototype are detached and converted to FP32.
- **Candidate pool and search scope:** retrieval is computed independently within each image. For a query, the positive is its strict-matched physical owner's prototype; candidates are all other strict-matched observations pooled across that image's greedy/sampled trajectories, excluding every observation with the same `owner_id`. Thus the reported `1540/2046 = 0.7526881720430108` (`75.27%`) is the aggregate of same-image, different-owner decisions over 13 images. It is neither cross-image nor global retrieval. Same-class retrieval uses the same pool further restricted to the same normalized description.
- **Similarity metric:** FP32 cosine similarity, implemented as dot product after L2 normalization; retrieval@1 passes only when own-owner cosine is strictly greater than the maximum eligible other-owner cosine.
- **Controls:** (1) same row's final post-norm `<|box_end|>` state against its own prototype; (2) same row's final post-norm last-coordinate state against its own prototype; (3) a cyclic next-image control that projects the same normalized predicted geometry onto the next sealed-panel image's verified merger matrix. The third is a control comparison, not part of the retrieval candidate pool.

Exact source: `scripts/research/frozen_owner_set_probe/{replay.py,representation.py,prototypes.py,finalize.py}` and `src/qwen/generated_commit_replay.py` under source SHA `093b4a2d0b8983c4786fb32ba9cdde3202996b6e`; exact metric receipt: `outputs/probes/coordexp_swift/frozen_owner_set_probe/a3_step2445/finalize/evidence/representation.json` in the same worktree.

Supported: owner-local information is retrospectively readable under the frozen generated-state replay.
Not supported: attribution to the auxiliary objective, a deployable controller, or causal future use.

### Future causal use

The separate PVCI proposal-bridge own-prefix panel is the correct future-use evidence surface, not the retrospective P0 readout.

- Source: `.../2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md`, SHA `cbd9fa4f7ed2f2c1cb1ffca2597a7f5ca14e3fce`.
- Artifact root exists: `/data/CoordExp/outputs/probes/coordexp_swift/pvci_own_prefix_sharded_panel_ABC_320_v3_20260712/`.
- 320 held-out images; A/B/C-off/C-on plus four negative controls; RP1.10 and RP1.00; 16 cells.
- Gate: `hold`, `primary_pass=false`, `negative_controls_pass=false`, `safety_pass=false`, ordered label `shortcut_or_row_prior`.
- At RP1.10, C-on versus C-off: precision `-0.224702`, recall `+0.018276`, duplicate rate `+0.180594`, invalid rate `+0.306250`, natural closure `0.6594` versus `0.9906`.
- Another-image and token-permutation feedback reproduce the harmful signature; it is not target-specific uncovered-owner use.

This is evidence that the tested pulse changes future behavior, but negative evidence for safe, target-specific causal use. It must not be merged with retrospective readout into one positive Address/Commit claim.

## P1. Stable evaluation primitives

These are stable implementation surfaces, not experiment-specific Control/Treatment/dose:

| Surface | Stable recovered behavior | Source |
|---|---|---|
| Parser | Compact rows require object-ref wrapper, box wrapper, exactly four coord tokens, and box closure. Valid spans are salvaged; malformed/unsupported JSON/unmatched text are retained as dropped records with reasons. | `src/inference/parsing.py` at `e505d87225e0d967f9272766fb001dcba0e89372` |
| EOS | Terminal `<|im_end|>` is ignored only as trailing stop text. It is not a row separator. | same parser |
| Native greedy | HF generation uses `do_sample=False`, `<|im_end|>` as EOS, a separately resolved pad token, explicit RP, and token traces that distinguish stop from pad. | `src/inference/backend.py` at `3cd40f5f091e774fca1ed4eba56244dc7b4f9630` |
| Decode/RP provenance | Checkpoint, prompt, template, decode policy, model, processor, parser, and score policy are fingerprint-bound. RP is an experimental factor that must be pinned; it is not a universal metric constant. | artifact/provenance writer and evaluator |
| Research one-to-one matcher | Exact normalized description plus pixel-space IoU; current defaults IoU `0.50`; deterministic greedy assignment sorted by `(-IoU, gt_index, pred_index)`. | `src/vis/matching.py` at `53d3e84798aae3f7f115e17f941d7fc2140d7898` |
| Duplicate hint | Same normalized description plus pair IoU; current default `0.30`. It is a hint/taxonomy, not automatic hallucination adjudication. | same matcher |
| Category | Closed COCO-80 normalization is lower-case + collapsed whitespace. Current official Swift consumer counts unknown-category predictions and excludes them from COCO predictions. | `src/eval/detection_categories.py` at `e6e508594442cfccfd7749c46fdbced90ccb90d8`; consumer at `1f262e52623d2a5751ba677067628e160bd47f3c` |
| Unknown-neutral | In incomplete-label research, unmatched/unknown predictions remain neutral pending adjudication; they are neither automatic success nor unsupported hallucination. This is separate from the official COCO consumer's closed-vocabulary exclusion behavior. | research governance and result contracts |
| Unsupported | “Unsupported physical entity” requires explicit evidence/adjudication. Parser `unsupported_format` is a syntax classification and must not be conflated with unsupported scene content. | parser + research contracts |
| Malformed/empty | Invalid spans, empty outputs, dropped rows, and truncation remain visible in counters/artifacts; valid siblings are not erased. | parser/artifact contract |
| Artifact schema | Required current Swift set: `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, score provenance sidecar, token trace, parse diagnostics, image plan, summary, run manifest; evaluator emits `metrics.json`, `evaluation_receipt.json`, `coco_gt.json`, `coco_predictions.json`. | `docs/eval/CONTRACT.md` at `0d6e3427971627f56768dffa968d50f779c1ac0a`; `src/inference/artifacts.py` at `2de5b79f96898ed5b7bd76450423a78ef0a5187c` |

Important split: the current official COCO evaluator and the research F1/owner matcher are different evaluator families. Their unknown, score, duplicate, and IoU behavior must not be flattened into one permanent protocol.

## P2. Industrial polygon/line evidence

The local search covered maintained `research/`, legacy `progress/`, OpenSpec, `docs/history`, registered worktrees, Git history, outputs path names, memory summaries, and local session text.

Recovered only:

- historical schema/design support for `poly` and older `line`/`line_points`;
- a January 2026 clean break that removed line/polyline from the maintained runtime contract;
- later bbox-only architecture notes.

Not recovered:

- dataset/slice identity;
- checkpoint or training recipe;
- inference artifact;
- evaluator identity;
- metric table;
- commit-bound experiment result.

Therefore the Notion industrial claim remains Inconclusive with verdict `SOURCE_MISSING`. Schema capability cannot support historical model success, and no surviving evidence supports annotation-richness causality.

## Canonical Paths audit

| Notion owner/page | Canonical path | Live state | Assessment |
|---|---|---|---|
| Representation / tight-outline | `/data/CoordExp/.worktrees/permanent-owner-bridge/research/` | exists, clean | Current architecture/research owner, but not the exact Step-0 artifact source |
| Representation / tight-outline | `/data/CoordExp/.worktrees/research-probes/research/` | exists, clean | Exact committed PVCI result notes exist; referenced primary artifacts/checkpoint/input are missing |
| Serialization | `/data/CoordExp/src/templates/renderer.py` | exists, committed | Current renderer source; does not alone establish ordering gain |
| Serialization | `/data/CoordExp/.worktrees/coordexp-infras/research/decisions/prefer-x-then-y-object-ordering.md` | exists, clean | Current bounded decision; correctly records compound evidence |
| Architecture Address/Commit | `/data/CoordExp/.worktrees/permanent-owner-bridge/research/` | exists, clean | Stale/incomplete for historical owner-commit P0; exact source is the unlisted clean `owner-commit-binding` worktree |
| Causal Evaluation | `/data/CoordExp/docs/PROJECT_CONTEXT.md` | exists, committed | High-level current context, not the full evaluator implementation |
| Causal Evaluation | `/data/CoordExp/.worktrees/research-probes/research/` | exists, clean | Research-local contracts exist; current production primitives also live under unlisted `src/inference`, `src/eval`, and `src/vis` |
| Industrial/Data Regimes | `/data/CoordExp/.worktrees/human13-nk-factorial-probe/` | exists, clean | Current human13 worktree; not a recovered source for the historical industrial polygon/line claim |
| Image2299 | image2299 research directory | exists; current worktree has two uncommitted 2026-08-26 experiment directories | The prior 2026-08-25 units are now committed at `60a0b25a...`; any Notion statement that those units remain uncommitted is stale, while the directories listed below are genuinely uncommitted |
| Image2299 | research-probes dense-enumeration directory | exists, clean | Canonical historical research authority; not all newest image2299 work is synchronized here |
| val200 Evaluation Gate | `docs/eval/WORKFLOW.md`, `docs/coordexp_infras.md`, `configs/` | all exist; tracked sources clean | Exists; current production evaluator code should be linked in addition to the docs |

### Image2299 commit coverage and current uncommitted directories

Commit `60a0b25a12861785cb319c580fcba7ef7fc02471` touches the following exact experiment directories:

- `2026-08-25-image2299-xy-adapter-embedding-composition`
- `2026-08-25-image2299-xy-certified-anchor-iteration-2`
- `2026-08-25-image2299-xy-certified-anchor-iteration-3`
- `2026-08-25-image2299-xy-gt17-debt-ledger-iteration`
- `2026-08-25-image2299-xy-gt23-relative-barrier-training`
- `2026-08-25-image2299-xy-gt7-debt-repayment`
- `2026-08-25-image2299-xy-prefix-safe-deficit-compilation`
- `2026-08-25-image2299-xy-same-owner-serialization-sentinel`
- `2026-08-25-image2299-xy-single-edge-owner-compilation`
- `2026-08-25-image2299-xy-step1-delta-backtracking`
- `2026-08-25-image2299-xy-two-edge-debt-repayment`

The currently truly uncommitted 2026-08-26 experiment directories in `/data/CoordExp/.worktrees/image2299-mechanism-microscope` are:

- `2026-08-26-image2299-full-root-detached-margin/` (`unit.md` untracked)
- `2026-08-26-image2299-set-level-compilation/` (`unit.md` untracked)

This is a live Git-status statement, not an attribution of ownership or readiness. Separate modified tracked files outside these two directories are not reclassified as 2026-08-26 experiment directories.

## Recovery boundary

No existing Notion page was edited. No model was loaded and no training, inference, evaluator replay, or scientific probe was run. Hashing, source inspection, and Git/path-state checks were read-only.
