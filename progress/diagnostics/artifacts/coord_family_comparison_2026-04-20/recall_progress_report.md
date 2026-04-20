# Coord Family Recall Progress

- Run: `coord-family-recall-progress-2026-04-20`
- Family count: `6`

## base_xyxy_merged

- status: `oracle_and_verifier_complete`
- sample_size: `64`
- recall_loc: baseline `0.1012` -> oracle-k `0.1119`
- fn_loc: baseline `506` | recoverable `6` | systematic `500`
- verifier: pred_count_total `458`, unmatched_count `403`, duplicate_like_rate `0.004366812227074236`
- recall_probe: suppressed `0.0000`, competitive `0.0178`, weak_visual `0.9822`

## center_parameterization

- status: `oracle_and_verifier_complete`
- sample_size: `64`
- recall_loc: baseline `0.6163` -> oracle-k `0.6998`
- fn_loc: baseline `216` | recoverable `54` | systematic `162`
- verifier: pred_count_total `516`, unmatched_count `165`, duplicate_like_rate `0.003875968992248062`
- recall_probe: suppressed `0.0000`, competitive `0.0417`, weak_visual `0.9583`

## cxcy_logw_logh_pure_ce

- status: `verifier_complete_oracle_pending`
- sample_size: `64`
- verifier: pred_count_total `583`, unmatched_count `325`, duplicate_like_rate `0.0017152658662092624`

## cxcywh_pure_ce

- status: `verifier_complete_oracle_pending`
- sample_size: `64`
- verifier: pred_count_total `604`, unmatched_count `322`, duplicate_like_rate `0.006622516556291391`

## hard_soft_ce_2b

- status: `oracle_and_verifier_complete`
- sample_size: `64`
- recall_loc: baseline `0.1066` -> oracle-k `0.1208`
- fn_loc: baseline `503` | recoverable `8` | systematic `495`
- verifier: pred_count_total `411`, unmatched_count `353`, duplicate_like_rate `0.0024330900243309003`
- recall_probe: suppressed `0.0000`, competitive `0.0378`, weak_visual `0.9622`

## raw_text_xyxy_pure_ce

- status: `oracle_and_verifier_complete`
- sample_size: `64`
- recall_loc: baseline `0.4742` -> oracle-k `0.7069`
- fn_loc: baseline `296` | recoverable `134` | systematic `162`
- verifier: pred_count_total `343`, unmatched_count `63`, duplicate_like_rate `0.0`
- recall_probe: suppressed `0.0068`, competitive `0.0135`, weak_visual `0.9797`
