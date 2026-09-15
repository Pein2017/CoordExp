# Permanent owner bridge Stage 1 production launch acceptance

## Disposition

The exact eight-rank Stage 1 production training was launched once through the
single user-authorized append-only recovery successor. The launch is confirmed:
all eight workers initialized, the run identity is durable, and the first ten
training observations at the acceptance snapshot are finite with applied
optimizer updates and empty non-finite fields.

This is an infrastructure and execution disposition, not a model-quality
result. Training is still running. No final production checkpoint, natural
greedy result, recall/duplicate conclusion, or promotion decision exists yet.

The machine-readable identity and hash record is [receipt.json](receipt.json).

## Fixed execution identity

- Repository HEAD: `78df2b87c925677e93080715d42e5a9348dcb05f`
- Production config fingerprint:
  `770f705f5357fdfc08d1d6fb332d828874265d296c5f567ae4c64266cd3c056c`
- Parent checkpoint: four-coordinate `geo_sorted_xy` step 2444
- Parent config fingerprint:
  `89e5af1269c42bdcc28b7175c8c701a4b57673e4b2fa8255c8912ee8800283ae`
- Successor intent key:
  `1c9e834fc83b3fc773746ceab8982a564fbc5f10aa5238333988c3919fc84c03`
- Recovery ordinal: `1` of a maximum of `1`
- Run ID:
  `qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1-770f705f5357`
- Run directory:
  `/data/CoordExp/.worktrees/permanent-owner-bridge/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1`
- Launcher PID/start ticks: `2841385/3256786717`
- Worker PIDs: `2842289` through `2842296`

The intent records the ordered eight GPU UUIDs, exact Accelerate argv, cache
root, source composition, four cache identities, clean tree, and 2444-step
schedule. The successor claim, intent, activation, binding, preflight, and all
eight worker admissions are hash-bound in the receipt.

## First finite, gradient, and artifact heartbeat

At `2026-08-11T06:58:33Z`, `logging.jsonl` contained exact training steps 1
through 10. Every row reported `finite_status=finite`,
`optimizer_update_status=applied`, and an empty `non_finite_fields` list.

The gradient/artifact acceptance is stronger than the scalar loss row alone.
For each owner-bridge step, `SupervisedTrainer` captures optimizer-group
gradient norms before clipping and the optimizer step, requires finite norms
for an applied update, and places them in the typed rank artifact contribution.
The completed-step handler then appends the logging row and synchronously
reduces and accumulates that contribution. It cannot enter and log step 2 until
the step-1 gradient-bearing artifact callback returns. The existence of steps
2 through 10 therefore proves the first finite/applied gradient-artifact path
completed across all ranks. Presentation event JSON is intentionally published
only at the presentation end; its absence during the first steps is not a
missing callback.

## Teacher-forced evaluation schedule

The sealed production schedule has eight fixed `geo_sorted` teacher-forced
events:

| Planned step | Event | Fraction |
|---:|---|---:|
| 306 | `tf_eval:p0:midpoint` | 1/8 |
| 611 | `tf_eval:p0:end` | 1/4 |
| 917 | `tf_eval:p1:midpoint` | 3/8 |
| 1222 | `tf_eval:p1:end` | 1/2 |
| 1528 | `tf_eval:p2:midpoint` | 5/8 |
| 1833 | `tf_eval:p2:end` | 3/4 |
| 2139 | `tf_eval:p3:midpoint` | 7/8 |
| 2444 | `tf_eval:p3:end` | 1/1 |

These are teacher-forced mechanics and loss observations. They are not natural
decode observations and do not decide checkpoint publication.

## Monitoring and failure handoff

Use `logging.jsonl` as the live progress authority until `run.json` receives a
terminal or checkpoint update. Monitor the run directory above, the eight
worker PIDs, and the successor-specific stdout/stderr under:

`/data/CoordExp/outputs/prod/coordexp_swift/.owner_bridge_stage1_launch_ledger/launch_logs/1c9e834fc83b3fc773746ceab8982a564fbc5f10aa5238333988c3919fc84c03.{stdout,stderr}.log`

If a rank exits, `run.json` becomes failed, any update is unsafe/non-finite, an
artifact or checkpoint identity fails, or progress becomes uncertain, retain
the run and ledger evidence and stop. The recovery authority is consumed; no
third activation is authorized.

## Post-final natural greedy HF evaluation

After and only after the run completes and publishes `checkpoints/final.json`
plus its final checkpoint event, materialize a new canonical config from that
completed run:

```bash
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
  conda run -n ms --no-capture-output python -m \
  scripts.coordexp_swift.materialize_owner_bridge_infer_config \
  --training-run-dir outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1 \
  --input-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/val.coord.jsonl \
  --artifact-root outputs/prod/coordexp_swift_owner_bridge_hf \
  --name owner-bridge-stage1-production-final-78df2b8 \
  --mode production
```

Then run the materialized dynamic-HF config:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
  conda run -n ms --no-capture-output python -m src.infer \
  --config configs/coordexp_swift/infer/materialized_owner_bridge/owner-bridge-stage1-production-final-78df2b8.yaml
```

The materializer must revalidate the final adapter, selected embeddings,
permanent bridge, lineage, and checkpoint-event fingerprint before this command
is admitted. Natural greedy output will characterize the completed checkpoint;
it is not a retrospective launch gate.
