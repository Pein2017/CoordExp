## 1. Implementation
- [ ] 1.1 Add F1-ish matching utilities (IoU + greedy assignment w/ deterministic tie-break) to `src/eval/detection.py`.
- [ ] 1.2 Implement segm IoU when polygons exist, including bbox ↔ poly support via rectangle segmentation.
- [ ] 1.3 Add semantic scoring for matched pairs (exact or embedding similarity), without mutating predicted desc strings.
- [ ] 1.4 Add `--metrics coco|f1ish|both` and F1-ish knobs (IoU thresholds, semantic threshold/model/device) to `scripts/evaluate_detection.py`.
- [ ] 1.5 Emit `matches.jsonl` (primary IoU thr) + optional `matches@<thr>.jsonl` (extra thrs), and augment `per_image.json` with per-threshold matched/missing/hallucination counts under a stable `f1ish` field.
- [ ] 1.6 Update `docs/detection_evaluator.md` with the new metric mode and examples.

## 2. Validation
- [ ] 2.1 Run evaluator on a small smoke rollout (`LIMIT=20`) and confirm deterministic outputs (including tie-cases) and reasonable counts.
- [ ] 2.2 Confirm bbox ↔ poly matching works by constructing a tiny fixture where GT is poly and pred is bbox (and vice versa).
- [ ] 2.3 Confirm semantic scoring treats near-synonyms as correct above threshold.
