# First-case candidate: latest-y1 V × key phase

Status: **candidate**, for lead acceptance. Only admitted `mature:1584:2` ran. `mature:2299:2` and `mature:4134:7` remain **HOLD**; no automatic scale or retry is authorized.

## Contract and provenance

- [Unit](unit.md) SHA-256 `b665c1ad61ce8930f59a278ac43f9f30104be89a3df2641263232795bcca4190`; [manifest](manifest.json) SHA-256 `68854cbc6c1c81e122d00631b7724ae5891993ea83086014ee9dcffd87d8e015`.
- New producer: `/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_y1_value_phase/run.py`, SHA-256 `6e606141af0efae208151b07c1f662937743891d2be7f52813da1b32860a309e`. Accepted predecessor producers were read, never edited. The producer and all 22 other directly used maintained/dependency sources were captured before the GPU call.
- CPU [preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1/preflight.json) SHA-256 `111c32418d30332e7e31ea215fbd27cd6c768367defef2c31429d47b1b0bce5d`. Original refined-00 request IDs are val2017 1584, 2299, 2685, 4134 in indices 0–3; target is index 0. Four requests, width 1395, 23,396,352 pixel elements; all four rows active at raw offset 23. Input identity SHA-256 `2856b08333a0d9082c4cfeafba868c0c1ca644c4b3f474add443ee48c2198a01`. Native histories, positions, masks, media, and companions remained bound to the source receipt.
- The CPU fixture exercised the **actual** joint patch on 28-layer inference-tensor cache: normal and forced-body-exception restoration passed; wrong target, wrong slot, wrong donor, unrelated K, and unrelated V were rejected. Full source/content/position CPU mutations and independent full-row +9 phase construction qualified. No CPU model load or CUDA call was made.

Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1/d1-1584-2`. The raw [pilot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1/d1-1584-2/pilot.json) SHA-256 is `487876767feff099ecac5a60cdbe65118d1d1bb55b939083cc63738424efd905`; [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1/d1-1584-2/receipt.json) SHA-256 `f403ebc44a8417b925b4cf87bde1e099ef83b6d3f22b9c19db27f87a54d71a66`; full [reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/first-case-v1/d1-1584-2/reduction.json) SHA-256 `af302612611bca67afa02e660eead27cfd41c7ef159bb3373ffd1e5a75a87257`. Raw vocabulary tensors and observed input/cache/rotary consumer records precede reduction.

## Qualification and finite call ledger

| Call | Cell | Qualification / outcome |
| ---: | --- | --- |
| 1 | Original full source | Four original images and rows; applicable trace parity passed at 2e-4. |
| 2 | Native historical prefill | Own native `pre_k`, K, V, and observed rotary blocks captured; all 28 layers passed native replay and scale-aware phase/FP64 oracles. |
| 3 | Native K, native V | Full/cached max absolute errors by batch index: `7.248e-5, 4.536e-5, 4.917e-5, 3.433e-5`; accepted target native vector max error `0`. |
| 4 | Latest-row K+9, native V | Accepted target +9 reference max error `0`. |
| 5 | Latest-row K+9, identity V-slot sham | Full-vector max error to call 4 `0`; actual one-slot patch and all-layer consumer observed. |
| 6 | Native K, earlier-y1 donor V in latest-y1 slot | Raw vector and actual per-layer K/V consumer saved. |
| 7 | Latest-row K+9, same donor V | Raw vector and actual per-layer K/V consumer saved. |

Only the declared latest nine-token K span moved in K+9 arms. Only latest-y1 V physical slot 1386 received the own earlier-y1 V from slot 1377 in donor arms. At every attention consumer, the selected K/V and all unselected target/companion K/V hashes matched their intended native or treatment values; the causal mask, S positions, and rotary consumer were checked. Historical K/V digests held through suffix computation, companion suffix K/V matched native, and the original cache digest returned after every call. Separate-process CPU cold readback passed all 28 layers and all seven-call/2-vision receipts.

The single GPU job exited successfully with terminal `candidate_complete`: **7 model forwards, 2 vision forwards, 0 free generation, 56.5627064 allocated GPU seconds** (`0.015711863` GPU-hour). Peak allocated/reserved GPU memory was `11,197,285,376 / 12,010,389,504` bytes; peak RSS was `9,472,504` KiB. First-case cap was 360 seconds; package cap 900 seconds; sequence prior was `0.189686339` GPU-hour and cumulative measured sequence is `0.205398202` GPU-hour. Artifacts under the first-case root total `16,416,327` bytes, below the 128 MiB planning allowance. The terminal receipt records no failure; the launch command returned exit code 0.

## Frozen prospective readout

`M = z(earlier-y1 token 152285) - z(latest-y1 token 152311)` in raw logits. The native current-y1 token is 152293.

| K / V cell | M (nat) | Winner / runner, gap | Earlier y1 P / rank | Latest y1 P / rank | Current y1 P / rank |
| --- | ---: | --- | --- | --- | --- |
| Native / native | `+4.362648` | 152293 / 152292, `0.021807` | `0.023790 / 13` | `0.0003032 / 35` | `0.113606 / 1` |
| +9 / native | `-3.229649` | 152311 / 152313, `0.049150` | `0.002277 / 49` | `0.057540 / 1` | `0.004685 / 35` |
| +9 / V sham | `-3.229649` | 152311 / 152313, `0.049150` | `0.002277 / 49` | `0.057540 / 1` | `0.004685 / 35` |
| Native / donor | `+4.354855` | 152293 / 152292, `0.025143` | `0.024922 / 13` | `0.0003201 / 35` | `0.112593 / 1` |
| +9 / donor | `-1.822130` | 152308 / 152307, `0.087223` | `0.006892 / 38` | `0.042628 / 4` | `0.012662 / 25` |

`Delta0 = M(native,donor)-M(native,native) = -0.007793` nat. `Delta1 = M(+9,donor)-M(+9,native) = +1.407518` nat. `Interaction = Delta1-Delta0 = +1.415312` nat. Both frozen strict bounds (`Delta1>1.0`, `interaction>0.5`) pass outside the 0.001-nat guard. A separate raw-vector calculation reproduced all margins and both donor-versus-same-K full-vocabulary TV values: `0.008309` under native K and `0.242158` under +9 K. Exact log probabilities and full-precision ranks for earlier/latest/current, as well as raw vectors and all cell hashes, are in the reduction and cell records.

This is **one-case directional amplification**, not the shared three-case prediction. The +9 donor arm changes the winner to token 152308, and its TV is broad. The evidence supports a phase-conditioned effect of the edited contextual V in this model and source batch; it does not establish literal copying, exclusive latest-y1 attention, a single-key mechanism, natural recurrence, or a population rate. The +9 arm shifts all nine latest-row K positions, not only the y1 key.

## Held cases and measured-cost reforecast

The two remaining cases are unchanged in the manifest and **not admitted**. Their accepted predecessor seven-call shapes are four requests, 23,396,352 pixels, widths 1395 and 1445; their old measured case costs were 49.521356 and 50.670438 seconds. With the same 2× shape-aware planning allowance, the conditional forecasts are 99.042711 and 101.340876 seconds, **200.383588 seconds** together. Measured first-case cost plus that forecast would be 256.946294 seconds (`0.071374` GPU-hour) for this prospective three-case unit and 0.261060310 cumulative sequence GPU-hour. This forecast is not admission or a hard bound; setup, failures, and changed V instrumentation remain chargeable. Lead technical/cost review controls whether either held case is run.

No retry, other case, Git action, or successor was launched. The worker makes no acceptance decision.
