# Root retry grant — corrected CUDA initialization only

The first attempt is technical-invalid:8workers,0model loads/forwards/cells.
Root reproduced the failure and verified default reset initializes CUDA,
then inspected the exact one-line diff and freshly reran all5CPU tests.
Failed source/packet/receipts remain preserved; the research contract is unchanged.

Owner `/root/sol_high_training_recovery` may launch one corrected attempt on
GPUs0–7 under this exact fresh output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross/retry-v2`.
Use `preparation-v2/packet.json` SHA256
`7459968bef4da5d89c3f75d5e63cf50d2fbd7c89c64efc00e85df776678821f2`
and producer SHA256
`344317d4dfa66fec9e864caea95f4abc45ec1857a006e8168184b98498446e88`.
The same8workers/16cells, resource bounds, exact diagonal-before-cross gate,
single launcher, long wait, outer-exit receipts and no-automatic-retry rules
apply. No other research GPU grant is active. Merge and verify after allrc0;
return candidate evidence and process-release confirmation to root.
