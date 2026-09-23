# Paired2048 rescue candidate

Status: **worker candidate, lead acceptance pending**. The fixed source, early-edge and late-merger recipes completed the authored val200 comparison. Neither injected arm shows a clear broad benchmark gain over the matched mature-live source at this four-epoch dose. This is a comparison of two bundled rescue recipes, not an isolated injection-timing or codebook-causality result. It does not establish convergence or a universal negative result.

## Frozen execution and technical evidence

The two arms started fresh from the same mature promoted-live untied+axis001 step2444 source. The canonical `geo_sorted_xy` train selection has 2,048 distinct COCO training images and 15,345 annotated objects. Its 944-pack cache was shared without rebuild; each arm consumed 3,776 packs, 8,192 image presentations and 472 finite/applied optimizer calls over four epochs. Early used GPUs0–3 and late GPUs4–7, each four ranks with accumulation2 and effective eight packs/update. Both final step472 checkpoints were saved, and all training ranks terminated before inference started. The three effective losses were CE1, type gate0.2 over all four groups, and positive-width/height hinge0.01 at margin1/999, with Gaussian0 and segment-balanced normalization. The first production updates preserve LR0 then positive-LR movement, exact common initial tensors/pack IDs, global segment denominators and frozen-source hashes in `production-first-entry-v1.json`. Saved logging has 472 finite/applied rows per arm; mean total objective declines early 1.4065 to 1.2884 and late 1.4042 to 1.2875 across epochs1 to4. These are training-objective means, not val teacher CE or proof of generalization.

The final checkpoint contains adapter, independent input/output deltas and codebook payload. The saved training state binds all eight payload hashes per arm; `analysis/final-payload-readback-v1.json` rechecks those hashes, metadata and finite codebook tensors. Early has a `[1024,8192]` P and final softplus gain0.048113; late has final gain0.047865. Both official run manifests record 900 adapter tensors copied into FP32 runtime with source/runtime tensor SHA equality and loaded selected deltas. The bound official `src/inference/hf_backend.py` branch resolves `model_composition.json` from the adapter's checkpoint parent, requires that checkpoint's own selected-delta path, then installs and loads its codebook metadata/tensors. A separate post-load runtime hash of codebook tensors and the actual injection residual RMS were not persisted; payload preservation and caller execution must not be relabeled as that measurement. The two-image source/early/late official smoke was accepted by lead, with max_new_tokens3084/RP1.0 and non-benchmark status; its training checkpoints were not reused for scientific fits.

One initial early smoke failed from a parent-authored shared objective receipt collision. The arm-specific receipt fix and concurrent-caller check preceded passing early smoke v2; late smoke v1 remained valid. The failed bytes and cost remain in the ledger. No scientific endpoint was selected by smoke or validation.

## Official val200 result

The three `src.infer` YAML runs each contain exactly 200 matching IDs, 1,600 known annotations, matched input-row and prompt fingerprints, `geo_sorted_xy`, HF FP32/SDPA promoted-live composition, batch4, greedy decoding, max_new_tokens3084 and RP1.0. All 600 planned rows are saved, scored and benchmark-eligible, with no HOLD or exclusion for cap/parser failures. The maintained `scripts/evaluate_detection.py` produced official COCO detection receipts and sidecars. AP is for this configured benchmark, not unrestricted native completion.

| val200 | Source | Early step472 | Late step472 |
|---|---:|---:|---:|
| AP / AP50 / AP75 | .46295 / .62317 / .49739 | .44453 / .60221 / .48318 | .46153 / .62525 / .49925 |
| AR100 | .51402 | .49376 | .51582 |
| Class-consistent IoU50 / IoU80 matches, of1600 | 976 / 666 | 948 / 654 | 961 / 656 |
| Clean known-positive images, of200 | 73 | 74 | 75 |
| Accepted predictions / annotation-unmatched proxy | 1691 / 698 | 1487 / 521 | 1710 / 732 |
| Any parser/stop bad image / cap image | 5 / 3 | 6 / 2 | 6 / 2 |
| Dropped spans: invalid geometry / malformed | 567 / 238 | 629 / 2 | 721 / 31 |
| Exact-row revisits / affected images | 269 / 6 | 80 / 7 | 200 / 6 |
| Annotation-owner IoU50 revisits / affected images | 25 / 13 | 19 / 11 | 53 / 20 |

Against source, early has 14 images with more class-consistent IoU50 matches and 31 with fewer (155 equal); late has 24 more and25 fewer (151 equal). IoU80 gains/losses are32/37 early and31/36 late. Early versus late has IoU50 gains/losses27/13 for late and IoU80 gains/losses23/27. These are annotation-proxy comparisons, not exhaustive physical-owner judgments. Annotation-unmatched predictions remain UNKNOWN, not verified false objects.

Geometry invalidity is measured on emitted coordinate bins **before parser drops**. Among source/early/late dropped boxes, 559/558/634 have x or y equality and8/71/127 have x or y reversal; categories overlap if the two axes fail differently. The x-equal counts are320/309/318, y-equal524/547/611, x-reversed7/8/81 and y-reversed1/64/46. These results show that the differentiable validity hinge did not strictly constrain greedy outputs. Parser/type malformed spans and cap are separately retained. The largest invalid bursts occur in source rows885/5586, early rows7511/885, and late rows7511/885; late row18380 additionally has112 drops and22 annotation-owner revisits with a max consecutive owner run15. The source row18380 has236 malformed spans, so lower aggregate malformed count in the injected arms does not imply fewer newly affected images. `analysis/saved-paired-v1.json` retains all per-image counts, drop reasons, capped rows, examples and input hashes.

The early arm is lower than source on AP, AP50, AP75, AR100 and known-positive IoU coverage. Late is near source AP, slightly higher AP50/AP75/AR100 but lower AP and known-positive coverage, with more invalid geometry and owner recurrence. Both have failure redistribution. This is not a claim of statistical equivalence; the user-fixed four epochs, limited val200 panel, source SFT exposure uncertainty and lack of a same-data no-injection fit limit attribution. Training curves and val outcomes cannot determine whether additional exposure would change the result. Lead owns the promotion decision.

## Provenance, replay and closure

Authoritative preparation is `2026-09-23-codebook-paired2048-preparation/manifest-v2.json` SHA256 `f80350293d0f8c895078cc2fea83e892f321c77ecc66e7104e2816947e18e0e9`; the val200 derivative is SHA256 `321eac5f5eddfb31747d638f722c4f1e424a07383bdc27a76352138247d2c4d5`. The producer receipts are `qualification/smoke-candidate-v1.json`, `production-first-entry-v1.json`, `training-complete-v1.json`, `inference-launch-v1.json`, each bound in the candidate manifest. Source/early/late official inputs and metrics are under `inference/paired2048-{source,early,late}-val200/`. Run these CPU replays from the research-probes checkout:

```sh
python scripts/evaluate_detection.py --artifact-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-source-val200 --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-source-val200/evaluation
python scripts/evaluate_detection.py --artifact-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-early-val200 --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-early-val200/evaluation
python scripts/evaluate_detection.py --artifact-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-late-val200 --out-dir /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048/inference/paired2048-late-val200/evaluation
PYTHONPATH=. python .local/scratch/paired2048/saved_analysis.py
```

The task-local analysis script has a byte-exact source capture outside outputs. Its saved-only replay is byte-identical, SHA256 `c04d3d81cfab04a3be65dd4b4d7eac9969d00babc2c8cf50bb71c1fe446095ae`. Official metrics and raw payloads remain the primary evidence. The analysis script is task-local, not a new general evaluator.

Fresh focused caller checks pass:100 tests across paired training, inference config and codebook reload. The research knowledge check, both-root output-layout check and `git diff --check` also pass. This is not a whole-repository test claim.

Current-package maintained edits are limited to `src/config/inference.py`, its focused `tests/inference/test_config_runtime.py`, `probes/training_set_completion/coordinate_codebook_alignment/paired2048_train.py` and its focused test, and the three `configs/coordexp_infras/infer/paired2048-*.yaml` leaves. This candidate report and task-local scratch readback are new records. Existing codebook/trainer/checkpoint machinery was reused; concurrent unrelated dirty paths were not changed for this package.

The original model clock ran from1790158092.1078854 to1790165324.3005283: 7,232.193 wall seconds (2.009h) and22,744.579 allocated GPU-seconds (6.318GPUh), including one failed early smoke and the clarification interval. Eleven job intervals are terminal, ten exit0 and one retained failure; training ended before any production inference. All recorded producer PIDs are absent after closing the two dead tmux panes; both one-shot wake monitors fired and no running monitor/job remains. Artifact root occupies about1.92GB, within8GiB. No new model work is needed for this candidate.

Current-package delegation: no Luna child implemented or launched any part of this package. The parent owned the receipt collision and its caller-level repair, both model launches, official evaluation and saved reduction; no child error or model-ranking claim is attributed. The retained lesson from earlier delegated seams is to require exact effective loss values, real caller collision tests and whole payload compatibility before declaring a child contribution integrated.
