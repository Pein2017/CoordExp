# Small, checkpointed PAR review recovery

This continues the frozen review, not a new experiment or new population.
The old Luna owner is stopped. Its images/view history survived, but no Luna
per-sample judgments survived. Do not invent or inherit those missing labels.

The dispatch assigns one batch of ten visual groups. Read only its input.json
and bound source images, rows, history, cards and crops. Own only that batch's
output directory. No GPU calls, new candidates, source edits, raw GT edits,
training, root admission, new agents or publication. Reuse existing cards and
crops; do not build a new review platform or regenerate the entire review set.

## Review contract

- Use Luna, ONE sample per view_image call, detail=original. No multi-sample
  thumbnails. The supplied cards include exact representative history; inspect
  a separate existing c/w crop or raw frame when target extent is unclear.
- Each c and immediate w must be a coherent single physical instance, with a
  supported COCO-80 class and plausible full VISIBLE extent. No pixel-perfect
  requirement and no amodal completion requirement. But existence alone does
  not validate a partial-body box, union box, or uncertain class.
- c is not already covered at exact h. w is distinct from c and prior h.
  Check each alias job's own h; visual overlap or low IoU is not owner identity.
- Occluded/truncated but coherent visible instances are allowed. Clearly
  separable instances are atomic. Dense ambiguous books/fruit/vegetables/cups
  or a coherent bunch can be valid coverage, but are HOLD for this frozen
  SINGLETON c/w bank. No negative/hallucination labels from unmatched GT.
- COCO-80 only (canonical names are bound in each source row). Do not force
  DVD to book, rabbit toy to teddy bear, bouquet/vase to potted plant, or create
  generic fruit/vegetable labels. Exact row geometry/text are immutable.
- candidate_accept requires resolved entity/class/visible extent/newness.
  Otherwise HOLD with the specific unresolved axis; uncertainty is neutral.
  Do not tune labels to meet the training-bank floor.

## Mandatory durable workflow

For EACH group: inspect the sample and needed crops, assess all its exact jobs,
then append ONE JSON object to decisions.jsonl BEFORE viewing the next group.
Use apply_patch for this local annotation edit. Never defer all judgments until
the end. A final answer or an image-call log is not a saved decision.

Each line has:

```json
{"visual_group_id":"PAR-0001","reviewer":"Luna-max / assigned agent name",
 "viewed":[{"path":"absolute actual viewed image path","sha256":"actual hash","detail":"original"}],
 "jobs":[{"job_id":"exact ID","c":{"status":"candidate_accept or HOLD","reason":"physical entity, class, visible extent","newness":"not_seen_in_h or covered or uncertain","newness_reason":"exact-history evidence"},
 "w":{"status":"candidate_accept or HOLD","reason":"physical entity, class, visible extent","newness":"not_seen_in_h_or_c or covered or uncertain","newness_reason":"exact-history and c distinction"},
 "pair_distinct":true,"pair_reason":"specific physical distinction or unresolved reason"}]}
```

pair_distinct can be true, false or null if unknown. Record only actual image
calls under viewed; never claim that every supplied crop was viewed. Preserve
HOLD judgments as useful decisions, not failures to conceal. No need to copy
full source rows into decisions.jsonl; input.json provides immutable binding.

After ten saved groups, verify exact assigned group/job coverage once, hashes
of viewed files, and candidate_accept newness/pair consistency. Write compact
completion.json with input hash, decisions hash, group/job counts, positive
candidate counts, HOLD counts, actual viewed-file counts, and remaining gaps.
Proposals only, not lead-accepted labels. Report paths/counts and STOP.
Do not expand to an adjacent batch. On interruption, immediately return saved
decisions and remaining IDs; do not start further review before saving.
