# Next-history prediction: worker candidate

Status: **candidate complete, pending lead acceptance**. The frozen shared affine one-step prediction fails. This is a scientific negative from five technically qualified readouts, not a qualification failure. No successor or further model calls are proposed here.

## Binding and intervention

- [Protocol](unit.md) SHA-256 `7ed5b1a9791c2144d1d79f910e903443f9ece5f623d5a15ccf2b7fd559aab851`; [manifest](prediction-manifest.json) SHA-256 `b4cfac3c89a7170f9ce6c3728c4bf2c88de9aa5f9ad2e1a2ad59db5454adc8ec`.
- Original source is the single-request `refined-04`, batch0, `coco2017_train_000000477415`; raw/source trace/runtime receipt/image SHA-256 respectively `5f8bdcc1eb2bb93c26389ee9c537043645a565ca8d6b1a028ab0f945d595751c`, `fed8aed02e09d24c897f1b9c1e48021e4af878ae0a18d3bed604596b2c22235f`, `627de9146dc632c39593afdd229cab12e46b8a4a84c46cf4b7c4279269545d4d`, `84a91098415dae8ddf0c5c911505df7a1883b9c6363e5d54a42d9217ab3c2596`.
- U is exactly `native[:54] = native[:36]+A+A`. Native row6 begins at54, shares the four-token chair/grammar header, but is a different box, not a third A arrival. Its unforced x1/y1 bins are98/579. The supplied probes replace only U's current x1 at raw offset58 with bins510/418; one y1 distribution follows at offset59. The full original prompt/image is preserved.
- CPU preflight saved all six FP64 log-softmax distributions over **152,670 tokens** before the GPU job. They came from the bound predecessor E/E, L_A/L, L_A/E vectors under `zL`, `2zL-zE`, `2zL-zLE`, with no fitted parameter or coordinate-only normalization. All six frozen top-two pairs matched the manifest. CPU checks bound the accepted L consumer, wrong row/content rejection, U's x1-only edit, full prompt width1362, L/U widths1412/1421, and L/U current-prefix rotary positions432–436/441–445. The actual model, rotary and attention consumers were recorded for every call.

## Five-cell ledger

All **five planned cells executed; zero held**. The three controls completed before either target. Source parity and both full-vector L references passed tolerance `2e-4`.

| Execution | Cell | Full-vocabulary top two tokens | Check | Raw vector SHA-256 |
| --- | --- | --- | --- | --- |
| 1 | Native U, x1=98 | 152249 / 152254 | Original y1=579; chosen/top-two/logsumexp parity, maximum checked error `5.72e-6` | `cc26d38b6e0fb6b561f8f11d27e6794019e5dbed006a1bdc4876bf21be4dfb3d` |
| 2 | Fresh L, x1=510 | 152083 / 152078 | Full-vector maximum absolute error `0` | `938cd57801567d0f1955dea125d09c6c2fa523a07468d57ea01940fc1acd2455` |
| 3 | Fresh L, x1=418 | 151670 / 152669 | Full-vector maximum absolute error `0` | `ff92dfc75440623115192375387714695c94342d5ab859dc42384c4623382b63` |
| 4 | Target U, x1=510 | **152078 / 152083** | Actual winner bin408, gap `0.00493431` | `313a89db8cda843ae8873337c9f8dba3885f2e3360e2c97181f67e9ae325c427` |
| 5 | Target U, x1=418 | **152669 / 151670** | Actual winner bin999, gap `0.05361080` | `a74a4a28bf9732f01dd5051cc02cfeb74dd7d17f60670760481098a90f8840f1` |

The source-control vector, both L vectors and both U vectors are retained in [raw output](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001). Per-cell records bind each raw vector and actual-consumer JSON. The full forecast vectors and their hashes are in [evidence.json](supporting/evidence.json) and [CPU preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/preflight.json).

## Prospective outcome

The table gives `KL(P_actual || Q_forecast)` in nats over the full vocabulary. The affine criterion required its winner to match **and** its KL to improve over persistence by more than0.001 on **each** probe.

| Supplied x1 | Actual winner bin | Persistence predicted / KL | Affine predicted / KL | Position-step predicted / KL | Affine KL improvement over persistence |
| --- | ---: | --- | --- | --- | ---: |
| 510 | **408** | 413 / `0.001789691` | 413 / `0.699423716` | 408 / `1.707691721` | `-0.697634025` |
| 418 | **999** | 0 / `0.006102740` | 0 / `0.345621276` | 0 / `0.733019560` | `-0.339518536` |

The affine forecast misses both winners and is much farther from both actual distributions than persistence. The position-step forecast gets the first winner but is still much farther from that full distribution; it misses the second winner. Persistence has the lowest KL for both probes, though it misses both categorical winners. At x1=418 all three frozen categorical forecasts are bin0, so their category predictions do not discriminate; their distribution errors do. No KL difference is within the declared `0.001` numerical guard. An independent CPU recomputation from the bound vectors matched all six KL values within `1e-12`.

This rejects this **shared local affine logit-step prediction** on these two supplied-x1 conditional readouts. It does not establish physical recovery, natural exit timing, a valid row6 box, population behavior, or a Transformer circuit. The two probes use one image/request and are not independent images. No forecast was revised after target observation.

## Technical and cost evidence

- [Preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/preflight.json) SHA-256 `a1d20cf40bc2f1fe88d73146fc222a3cd914650a238ff298e837f8c77bd7054c`; source captures include the producer and every directly used maintained import. Its measured-shape forecast was `11.143827` allocated GPU seconds with 2× margin, below the 900-second hard cap.
- [Pilot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/pilot.json) SHA-256 `7b0fd6bd6d8b6a59859bbaf5f1412ffcf3a37097bb8c76da8d704cc1cf5313c4`; [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/receipt.json) SHA-256 `aca7f0eb71b93a4ef5e432760eaaeb90991ec3bca221af78bbca34bd84849dd9`. Exactly five model and five vision forwards, zero free tokens, one GPU job. Measured allocated time including model load/setup `10.508065` seconds = `0.002918907` GPU-hours; outer shell wall `17.056627` seconds. Cumulative charged sequence `0.066430621` GPU-hours versus cap8. Peak RSS `11,215,908` KiB; peak GPU allocated/reserved `9,072,883,200` / `9,523,167,232` bytes. Output artifacts after reduction total `11,384,971` bytes.
- [Separate-process cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/cold-readback.json) SHA-256 `325ddc090e7689c16183472cc72de10367f1ecc41c2d83a15eea502fd250582f` passed vector, forecast, input, position, hash and count checks. [Reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-history-prediction/attempt-001/reduction.json) SHA-256 `2a1d2e7dbb7a7e9a528c656ac5077d00eb41683bc2d3eff2f9db2a8357d51f02` contains complete per-probe probabilities assigned to actual winners and all KL results.
- The GPU command exited `0`; launch PID `1319760` is absent from `/proc`. No further job is running. Producer/source captures and the accepted predecessor remain unchanged. No Git action was taken.
