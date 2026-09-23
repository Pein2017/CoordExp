# Corrected paired2048 conditional-order-gate candidate

Status: **worker candidate, lead acceptance pending**. This is the user-authorized four-epoch rerun with CE1, token-type gate0.2 and the new conditional bbox-order gate0.2; the old mean-coordinate hinge and Gaussian terms are zero. The gate penalizes coordinate-family mass at `x2 <=` the teacher-prefix `x1` token and `y2 <=` the teacher-prefix `y1` token. It does not hard-mask generation. The old mean-hinge group is preserved in the designated archive; its fitted outputs are not corrected-arm evidence.

Both arms started fresh from the same mature promoted-live untied+axis001 source and used the exact 2048-image, 944-pack/epoch cache and seed1729 schedule. Early and late each completed 472 finite/applied optimizer calls, 3776 global packs and 8192 image presentations over four epochs. Four ranks per arm trained concurrently on disjoint GPUs, with accumulation2 and the frozen LR groups; no checkpoint was selected by validation. The final step472 payloads have 904 early and 903 late named tensors. CPU safetensors readback finds them finite and binds the codebook mode, adapter and independent selected-row payloads. Official HF run manifests record 900 adapter source/runtime tensor hashes equal and the selected input delta loaded from each final checkpoint. The exercised composition route resolves `model_composition.json` and the codebook payload from the checkpoint parent. A separate post-load codebook tensor hash or full-logit reload parity was **not** measured.

The matched mature source is the 200-row official val200 run accepted in the archived predecessor, reused after exact data, prompt, runtime and decoder identity checks; it is not a new replication. Early and late each produced 200 new benchmark-eligible rows, with no missing row. All three conditions use `geo_sorted_xy`, HF FP32/SDPA promoted-live composition, batch4, greedy decoding, max_new_tokens3084 and repetition penalty1.0. The maintained detection evaluator completed on both new runs. Its configured 3084-token benchmark is not unrestricted native completion.

| val200, 1600 known targets | Reused source | Corrected early472 | Corrected late472 |
|---|---:|---:|---:|
| Official AP / AP50 / AP75 | 0.462946 / 0.623166 / 0.497391 | 0.448846 / 0.606585 / 0.481639 | 0.446772 / 0.612554 / 0.477867 |
| Official AR100 | 0.514020 | 0.498796 | 0.494708 |
| Class-consistent IoU50 / IoU80 matches | 976 / 666 | 946 / 651 | 945 / 643 |
| Clean known-positive IoU50 images | 73 / 200 | 73 / 200 | 73 / 200 |
| Accepted predictions / annotation-unmatched proxy | 1691 / 698 | 1798 / 840 | 1838 / 875 |
| Pre-drop invalid geometry spans | 567 | 40 | 332 |
| Equality-invalid / reversal-invalid spans | 559 / 8 | 14 / 26 | 308 / 24 |
| Images with invalid geometry / any bad output / cap | 5 / 5 / 3 | 6 / 6 / 2 | 8 / 8 / 2 |
| Malformed spans / images | 238 / 3 | 247 / 2 | 2 / 2 |
| Exact-row revisits / images | 269 / 6 | 401 / 5 | 401 / 6 |
| Annotation-owner IoU50-proxy revisits / images | 25 / 13 | 27 / 18 | 51 / 17 |

The geometry counts are emitted four-coordinate spans **before parser drops**. A separate readback checks all 600 raw and parse-diagnostic rows: the malformed drops contain no complete four-coordinate span, and accepted predictions contain no nonpositive coordinate order. Equality and reversal are recorded separately; an individual span could meet both on different axes, though no overlap occurs in these saved totals. Official evaluator `invalid_pred_bbox_count` remains a separate post-parser consumer counter (source3, corrected arms0). UNKNOWN means annotation-unmatched, not physically false.

Failures are concentrated rather than evenly distributed. Early has 32/40 invalid spans and 246/247 malformed spans on image18380 (capped); late has 316/332 invalid spans on image7511 (capped). Early image632 emits an equal-x box `[896,520,896,630]`; image7511 emits reversed x `[168,548,167,561]`. Late image885 emits `[0,0,0,36]`, and image7511 emits reversed y `[0,571,38,570]`. The raw slices are in the corresponding `gt_vs_pred.jsonl`; the saved reduction retains every image, parser drop, stop reason, run length and annotation-proxy owner. Source-to-early paired IoU50 images improve14, worsen32, tie154; source-to-late improve17, worsen35, tie148. Fewer geometry-invalid spans therefore do not establish better detection or lower recurrence.

Rank0 logged composite training objective, weighted by eligible segments within each epoch, declines early 1.40534→1.28781 and late 1.40360→1.28602. The weighted order-gate term falls approximately 0.00084→0.00055 in both. This shows optimization activity, not train-native fit or convergence; no full train-native evaluation was authorized. Four epochs and the known mature-source SFT exposure limit transfer interpretation. This bundled early/late comparison does not isolate injection placement or the loss's causal effect. The outcome is evidence for lead review, not promotion or a universal codebook conclusion.

Evidence: `qualification/smoke-candidate-v1.json` and `qualification/lead-smoke-verification-v1.json` bind the accepted technical smoke; `qualification/production-first-entry-summary-v1.json` binds the first two genuine updates per rank. `production-launch-v1.json` and `inference-launch-v1.json` bind the source, cache, configs, checkpoints and exact commands. `qualification/source-reuse-v1.json` binds the archived source200. `analysis/saved-paired-v1.json`, `analysis/geometry-readback-v1.json`, `analysis/training-curves-v1.json`, `analysis/final-payload-readback-v1.json` and `analysis/cost-closeout-v1.json` retain saved-only reductions, per-image detail and closeout. The replay caller is `.local/scratch/ordergate/saved_analysis.py` in this checkout:

```sh
PYTHONPATH=. python .local/scratch/ordergate/saved_analysis.py
python scripts/evaluate_detection.py --artifact-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate/inference/paired2048-ordergate-early-val200 --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate/inference/paired2048-ordergate-early-val200/evaluation
python scripts/evaluate_detection.py --artifact-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate/inference/paired2048-ordergate-late-val200 --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate/inference/paired2048-ordergate-late-val200/evaluation
```

All eight owned model jobs are terminal exit0; 16 recorded producer/supervisor PIDs are absent. Training ended before either official inference; no same-GPU intervals overlap. Original-clock model wall is 4057.989163 seconds (1.1272h), allocated cost 21608.313884 GPU-seconds (6.0023 GPUh), artifacts 1,829,351,614 bytes, under the 4h/32GPUh/8GiB envelope. No new model call follows final inference. Fresh knowledge, two-root output-layout and `git diff --check` pass. The focused order-gate smoke/caller tests remain lead-verified; no whole-repository test-suite pass is claimed. The prior invalid group is archived byte-preservingly with one retired old-path compatibility link.
