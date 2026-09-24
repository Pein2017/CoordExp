# Reverse key-phase candidate results

**Status: worker candidate; lead acceptance pending.** Worker `01a0ce4b-9b55-7392-8a25-6a76f9e12c3a` (`gpt-6-sol/xhigh`) completed the finite admitted unit for lead `01a0c831-b332-7e12-b931-6ebe2359c99f`. Original refined-04 batch index 0 has one request `coco2017_train_000000477415`, one image, two fixed supplied x1 probes, no free generation.

## Frozen reverse prediction

With current S held at native U rotary positions 441:446, the latest repeated A row’s nine post-RoPE keys were moved plus9 from 432:441 to 441:450 at all 28 layers. The equal-size earlier-row comparator moved 423:432 to 432:441. The target was the already qualified crossed U/L full-vocabulary distribution. FP64 full-vocabulary softmax/TV used no coordinate restriction, fitted coefficient or temperature. Both probes pass the frozen strict rule `B>1e-6`, `r_latest<0.5`, `r_earlier-r_latest>0.1`, outside the 1e-6 numerical guard.

| Owner / supplied x1 | B = TV(native,cross) | latest TV-to-cross / ratio / TV-to-native | earlier TV-to-cross / ratio / TV-to-native | earlier - latest ratio | Category |
|---|---:|---:|---:|---:|---|
| 1589003 / 152180 | 0.761939226 | 0.050338502 / 0.066066295 / 0.771737243 | 0.766204133 / 1.005597436 / 0.010122824 | 0.939531141 | selective |
| 1586761 / 152088 | 0.391244291 | 0.048566881 / 0.124134415 / 0.376731660 | 0.413696264 / 1.057386073 / 0.025552503 | 0.933251657 | selective |

This is a successful reverse prediction at the two frozen supplied-x1 endpoints: changing latest-row addressing is sufficient to approach the crossed distribution, while changing the earlier identical-token row is not. The converse rescue and reverse transfer together support a selective relative-key-phase contribution under these artificial interventions. They do not identify a unique circuit or imply natural exit, physical box recovery, or population transfer; other history-relative positions and S contextual trajectories differ across native and crossed arms.

## Categorical and secondary results

Global winner/runner are vocabulary token IDs. Fixed-pair margins are logits for the predeclared 408 versus 413 or 999 versus 0 coordinate pairs. Full 152,670-logit vectors and actual consumer records for all ten cells are individually bound in `reduction.json`.

| Owner | Cell | Winner | Runner | Gap | Fixed-pair margin |
|---|---|---:|---:|---:|---:|
| 1589003 | native | 152078 | 152083 | 0.004930496 | 0.004930496 |
| 1589003 | cross | 152366 | 152378 | 0.092255592 | -0.121979713 |
| 1589003 | sham | 152078 | 152083 | 0.004930496 | 0.004930496 |
| 1589003 | latest | 152366 | 152373 | 0.101524353 | -0.110346794 |
| 1589003 | earlier | 152078 | 152083 | 0.008167267 | 0.008167267 |
| 1586761 | native | 152669 | 151670 | 0.053606987 | 0.053606987 |
| 1586761 | cross | 152543 | 152548 | 0.137765884 | 0.617607117 |
| 1586761 | sham | 152669 | 151670 | 0.053612709 | 0.053612709 |
| 1586761 | latest | 152543 | 152541 | 0.141362190 | 0.661538124 |
| 1586761 | earlier | 152669 | 151670 | 0.000664711 | 0.000664711 |

Latest treatment matches the crossed global winner for both probes (152366 and 152543); earlier treatment retains the native winner for both (152078 and 152669). Winner matches are secondary to the full-distribution criterion.

## Technical qualification

- CPU sensitivity rejected wrong row, wrong sign, a clipped five-position latest phase, and wrong native-S positions. The small inference-tensor cache fixture passed native-S sham, latest+9, earlier+9, and forced-body-exception crop/restore with V and unselected K intact. CPU record SHA-256 `e266e7888a542453b6dae6c49c2e85fdb2bd0f0511314acc486a67f942ea117b`; no model/vision/CUDA calls.
- The loaded maintained rotary module was invoked once outside any language-model forward to construct full nine-position destinations. Complete independent FP64 frequency/layout oracle maximum error `1.4035192e-05`; earlier destination equals observed latest native phase exactly, and the latest destination first five equal both actual native-S anchor phases exactly. Full destination positions are 441:450, not a clipped five-token S or old observed donor.
- Four cached anchors matched bound full-forward vectors exactly; two native-position identity shams differed from native by at most 1.38282776e-05 logits, below 2e-4. All 56 row/layer phase gates passed; maximum measured/bound ratio 0.249699031.
- Actual model/rotary/attention hooks recorded the same supplied tokens, native S positions for shams/treatments, physical slots 1416:1421, causal masks and all-layer historical K/V before and after attention. For each treatment and probe, the selected K hash changes on 28/28 layers; the other A-row K and all V hashes match native on 28/28. The patch scope restores/crops the original cache in `finally`. Separate-process cold readback passed source, vector, consumer, count and destination bindings.
- CPU preflight captured 23 directly used maintained/imported sources, exact command and single-request shape: prompt 1362, history 1416, full 1421, image grid 1 × 52 × 78. Matched predecessor took 20.475287 seconds for 11 model / 1 vision forwards; 2× planning estimate was 40.950574 seconds under the 900-second cap.

## Immutable evidence and cost

- Frozen protocol SHA-256 `5df865e7ba8e65a5725831eadcb0191f96070fd472a1ab08bb7136e368504d08`; manifest `d746713cb34a726742102ac29c0a8cdfcf981ccfe948cf3b24105735c518bb6c`; producer `c598e4a34b1f7f8299682f24313bcd886c8b05265074baf9167cc7a59d9d8850`; preflight `5485bf3476b733f843e8225385370e90bcd395e209939b10d33b500e101e0668`.
- Raw output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-reverse-key-phase/attempt-001`. Destination phase `ab9ec69a3c44bd07b92557f03774f178a00d46d545edd7706255272e80803ff3`; pilot `353063f80b02871af308ed6ee5769ae16d5bb0c9f2c217d0b463bc05fb533123`; terminal receipt `cbe9cccd673d326c3e6acf596565da00f8070a2fe66c2de6bd61b605d6f09a46`; cold readback `aad5038e9a26aca70c83af1bfe83eb25d5ed717ef7586e524951087911e6b5a6`; reduction `50112205f12a53011238e8880402c7c6337ad6fc27df3571856bfabb9fe84795`. Raw vectors and consumer records preceded reduction.
- Attempt001 is terminal candidate-complete; process PID `1452726` is absent. One prefill + ten suffixes = 11 model, 1 vision forwards; allocated GPU time 24.253737561 seconds = 0.006737149 GPU-hours. Cumulative sequence 0.085444429 GPU-hours, below 8. Peak RSS 9481136 KiB; GPU allocated/reserved peaks 9399284224 / 9676259328 bytes. Raw tree 13509296 bytes, below 64 MiB. GPU command, separate cold readback and reduction exited 0. No automatic retry or successor.
