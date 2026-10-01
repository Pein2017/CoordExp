# Manual Audit Pack: Stage1 vs Stage2a (48)

This pack is the checkpoint-extension audit queue.

Scope:
- checkpoints: stage1-coco80-ckpt-1832-merged, stage2a-eff64-softctx2-merged
- temperatures: 0.5 and 0.7
- per run: top 12 unmatched candidates by packaged audit order
- total rows: 48

Label schema:
- real_visible_object
- duplicate_like
- wrong_location
- dead_or_hallucinated
- uncertain

Recommended launch:

```bash
cd /data/CoordExp
conda run -n ms python scripts/analysis/run_manual_audit_reviewer.py   --audit-csv output/analysis/unmatched-proposal-verifier-manual-audit-stage1-stage2a-v1/manual_audit_recommended48.csv   --port 8765
```
