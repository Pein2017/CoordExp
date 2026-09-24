# Chair first-y1 history/position crossing: finite candidate

Status: **candidate for lead review**. The corrected [protocol](unit.md) SHA-256 `acd71e7484f18600000dd66343bd4abebba3f7c74ed16f5c08293d2087960efe` and unchanged [manifest](manifest.json) SHA-256 `3410f250e0113f907a8fbd07d8e402a4aca6f41fd987569d48a884c3c65587d8` admit one original **single-request** `refined-04` batch, target index 0, two fixed x1 probes, and exactly 18 forwards. All 18 cells completed in one GPU job, with no generation or extra model call. No successor is proposed.

## Qualification and exact intervention

The original request is `coco2017_train_000000477415`. CPU preflight reproduced its saved prompt/media/input identity, checked the native row boundaries and Stage2 actual-consumer input prefixes, and captured the [runner](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_chair_history_position/run.py) plus 13 directly used maintained imports **before GPU**. The runner SHA-256 is `0e878f1ee66a8618f5842f4e6022eb39265a2339279de4e0dd4d10dc5c70639d`. [Preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/preflight.json) SHA-256 `1b6f79b6805311ea4c02f2f58a55ecae4ab5128f6426d4589c2fccb7fe4128f1` binds the 18 cells, captures and physical/raw/rotary crosswalk. The shape-aware 2× planning estimate was 0.03177 allocated GPU-hours under the 1-hour incremental cap. [Execution plan](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/execution-plan.json) SHA-256 `d56dc4f0bda03b0fe5c1cbcfbc89e87c3a40578158d888e5b3044793a9e6bf5b` freezes commands and the 3,500-second outer wall guard.

The target current-prefix S is five tokens through the supplied x1. At E, S occupies raw offsets 36–40 and physical input indices 1398–1402; its three-axis MRoPE positions are 423–427. At L, S occupies raw offsets 45–49 and physical indices 1407–1411; MRoPE positions are 432–436. Native y1 source offsets are 41 and 50. Crossed cells change **only these five current-prefix position IDs** on all axes, while retaining their historical positions, input IDs, masks, and physical cache order. L_B substitutes exact earlier native chair row1 for appended broad row A; only its historical y1/x2 tokens at physical indices 1403/1404 change. B is a nine-token row already visited in the native prefix, and it changes preceding-box geometry as well as token content.

The actual model-consumer input IDs, mask, position IDs, pixel and image-grid hashes, and actual rotary-consumer positions were recorded for every cell. The [cold reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/reduction.json) (SHA-256 `67cf49b7cb5db778dfad6ab2ea0dcd598cee11ca3a6ecb4a0e59db4324fbcf99`) verifies that same-history E/L-position pairs differ only in S positions, L_A/L_B pairs only in the two declared historical tokens, and matching pairs preserve consumed mask/cache-position hashes. CPU wrong-slot and extra-content-edit mutations were rejected; historical-position preservation was checked against the saved source and the constructed crossing. These are verifier checks, not modified-model ablations.

The two unforced native controls match source chosen logit, top-two and normalization within a maximum `8.107e-6`, below `2e-4`. All four supplied-prefix E/E and L_A/L references match the retained Stage2 branch step-0 winners, top-two logits and normalization **exactly** at saved precision. Four separately forwarded explicit identity-position controls match fresh full reference vectors exactly (`max |Δ| = 0`). These ten controls ran before eight crossed/content cells. Every cell retained a complete vocabulary vector, normalization and raw consumer evidence before reduction; separate-process [cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/cold-readback.json) passed (SHA-256 `09c7b806056f674b39ef5c7959ba71e8a6ee5b92a057e1b5965059b574707f31`).

## Categorical and continuous readouts

Each table entry is `winner y1 bin / top-two gap / m`, where `m = z(early y1) − z(late y1)` from the full vocabulary vector. For owner 1589003 the fixed pair is 768 versus 413; for owner 1586761 it is 873 versus 0. A positive m favors the early member of that pair. All grid top-two gaps exceed `0.0257`; no cell meets the ≤`0.001` near-tie flag.

| Probe | History | S at E positions | S at L positions |
| --- | --- | --- | --- |
| 1589003 | E | **768** / 0.0351 / +0.6609 | **413** / 0.0578 / −4.3339 |
| 1589003 | L_A | **696** / 0.0257 / +3.0272 | **413** / 0.0409 / −4.9704 |
| 1589003 | L_B | **795** / 0.0356 / +4.8797 | **519** / 0.0825 / +0.3385 |
| 1586761 | E | **873** / 0.0743 / +0.7724 | **999** / 0.1486 / −0.8662 |
| 1586761 | L_A | **873** / 0.0819 / +1.7244 | **0** / 0.1107 / −1.3312 |
| 1586761 | L_B | **873** / 0.0519 / +2.4396 | **873** / 0.0962 / +0.7202 |

Signed differences in m (nats of raw-logit difference) retain the frozen `0.0004` numerical guard:

| Probe | Position L−E at E / L_A / L_B histories | Added A history L_A−E at E / L positions | B content L_B−L_A at E / L positions | A-history × position | B-content × position |
| --- | --- | --- | --- | ---: | ---: |
| 1589003 | −4.9948 / −7.9976 / −4.5413 | +2.3663 / −0.6364 | +1.8525 / +5.3088 | −3.0028 | +3.4563 |
| 1586761 | −1.6387 / −3.0556 / −1.7194 | +0.9519 / −0.4650 | +0.7152 / +2.0514 | −1.4169 | +1.3363 |

At fixed history and x1, moving S positions from E to L shifts the fixed-pair margin toward the late y1 for **both probes in all three histories**. At fixed late history length and positions, replacing broad A with earlier same-class B shifts the margin toward early y1 for both probes, more strongly at L positions. Added-history effects change sign across positions. The interactions preclude a simple sum of independent history and position effects on this readout.

The strict categorical position-following prediction fails on both probes: E/L selects 413 for probe 1589003 but **999**, a third mode, for probe 1586761; L_A/E selects 873 for probe 1586761 but **696**, a third mode, for probe 1589003. Strict added-history-package following also fails. B at L positions restores early 873 for probe 1586761 but selects third-mode 519 for probe 1589003, so the frozen two-probe content-restoration prediction fails. The positive B margin shifts are real finite effects and do not rescue that categorical prediction. The finding is a local content-by-position interaction at **first free y1**, not an identified circuit, native owner preference, valid complete box, physical recovery or population estimate.

## Artifacts, cost and stop

The [pilot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/pilot.json) SHA-256 `d4c2132f318d0ac59534a48528f6c96dbfc921b0be0d4252e6b34696810a15a8`, [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-chair-history-position/attempt-001/receipt.json) SHA-256 `951a8c372726e21044c8181facbfc83fec87a4f6b02ac11dd2960655fc9de96b`, cold readback and reduction bind all 18 planned/executed cells, zero held and zero technical-invalid attempts. The one GPU job exited 0 and is terminal; no owned GPU job remains. It charged **19.93185 allocated GPU-seconds = 0.00553662 GPU-hours**, including load/setup, with 18 model forwards, 18 vision forwards and zero free generation. Peak RSS was 11,120,372 KiB; peak GPU allocated/reserved was 9,071,465,472/9,518,972,928 bytes. Including the prior sequence cost 0.05797509 GPU-hours, cumulative spend is **0.06351171 of 8 GPU-hours**. The 80 attempt artifacts occupy 13,910,415 bytes after reduction. The read-only two-root output-layout check and source compilation passed. Only the lead can accept the interpretation; no Git action was taken under this assignment.
