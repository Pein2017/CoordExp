# Native-x1 key-phase candidate results

**Status: worker candidate; lead acceptance pending.** Worker `01a0ce4b-9b55-7392-8a25-6a76f9e12c3a` (`gpt-6-sol/xhigh`) returns the finite six-call unit to lead `01a0c831-b332-7e12-b931-6ebe2359c99f`. The original single-request refined-04 batch (index 0, train477415) supplied U=native[:54] and the original S=native[54:59], including x1=98/token151768. Native row6 is a distinct chair box, not a third repeated-A arrival.

## Frozen outcome

At this native-x1 endpoint the strict full-vocabulary distribution criterion **passes**. With current S held at native U positions 441:446, latest A cached K plus9 approaches the new crossed S/L distribution; the equal-size earlier A edit remains near native. FP64 softmax used all 152,670 vocabulary tokens without fitting, masking or temperature. The criterion was frozen before this new crossed readout.

| B = TV(native,cross) | Latest TV-to-cross / ratio / TV-to-native | Earlier TV-to-cross / ratio / TV-to-native | Earlier - latest ratio | Category |
|---:|---:|---:|---:|---|
| 0.758279824 | 0.078677034 / 0.103757256 / 0.770387986 | 0.771749755 / 1.017763800 / 0.025275905 | 0.914006544 | selective |

The frozen conditions are B>1e-6, latest ratio<0.5, and earlier-minus-latest ratio>0.1; neither threshold is within the 1e-6 numerical guard. This is a conditional native-prefix intervention result at one image and x1. It does not establish natural recurrence onset/exit, detection validity, physical box recovery, a unique circuit or population transfer.

## Five suffix readouts

The original native y1 is token152249 (579). Global winner and native-y1 log probability are separate from the full-distribution decision. The crossed cell is **new evidence with no prior matched-vector reference**. All five raw full-vocabulary vectors and actual model/attention consumer files are bound individually by full hashes in `reduction.json`.

| Cell, in call order | Winner | Runner | Logit gap | Log P(native y1 token152249) |
|---|---:|---:|---:|---:|
| native | 152249 | 152254 | 0.126756668 | -2.710597839 |
| cross | 152311 | 152308 | 0.016891479 | -6.458645418 |
| sham | 152249 | 152254 | 0.126754761 | -2.710598613 |
| latest | 152319 | 152321 | 0.047901154 | -6.648636824 |
| earlier | 152249 | 152254 | 0.127933502 | -2.666191081 |

Native and sham choose token152249; the new cross chooses152311; latest plus9 chooses152319; earlier plus9 chooses152249. Thus latest does not recover the crossed global winner, although its full distribution is close to crossed (TV 0.078677). No categorical parity is claimed for that treatment.

## Qualification

- The native anchor matched the bound saved full vector within 2.74181366e-05 logits. It matched the original source chosen token and top2 exactly. Source chosen-logit, logsumexp and chosen-log-probability errors were chosen_logit=0, logsumexp=1.90734863e-06, chosen_logprob=2.06071871e-06; each is below 2e-4. Native identity sham maximum difference from the fresh anchor was 1.04904175e-05 logits, below 2e-4.
- CPU sensitivity rejected substituted x1, wrong A span, wrong phase sign, clipped five-position latest phase, and wrong native-S positions. An inference-tensor cache fixture passed sham/latest/earlier patch and forced-exception crop/restore with unchanged V and unselected K. CPU record SHA-256 `12dba2b322f1706ebd64a98fdef53bb9173ba6bd92e3fe166892cc9b77c300d1`; zero model/vision/CUDA calls.
- The loaded maintained rotary module generated all nine latest destination positions 441:450 without a language-model forward. Independent FP64 full-destination maximum error 1.4035192e-05; earlier destination matched observed latest phase exactly, and latest first five matched actual native S exactly. All 56 row/layer numerical phase gates passed; maximum measured/bound ratio 0.249699031.
- Actual model/rotary/attention consumers recorded unchanged supplied tokens, native S for sham/treatments, new crossed S 432:437 for the crossed cell, causal mask and physical slots1416:1421. In both treatments selected K hashes change on 28/28 layers while the other A-row K and all V hashes match native on 28/28; native cache crop/digest restoration passed each suffix. Separate-process cold readback passed source, vector, consumer, count and destination bindings.
- CPU preflight froze 24 directly used maintained/import sources, single-request prompt1362/history1416/full1421 and image grid1×52×78, exact commands and a matched-shape 2× planning estimate of 26.458623 seconds.

## Provenance and cost

- Frozen protocol SHA-256 `08376a86e3040e1674a93f6faace138d960aff67a15381809c19d8640c918bd9`; manifest `5ad2be6ae6bb7e6668ccad63137f2cfac548cc3ed31f63ca17b7e031db487d2a`; producer `75f1d7d81785934ff07d978ed4fe56096740ed34b163166a00d54bbbbe48d3b1`; preflight `6d62c09f847f63e686d915feeab4fbf8cd40bce0048349540ab5fabc6e5d9bb5`.
- Raw output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-native-x1-phase/attempt-001`. Destination phase SHA-256 `d2925aa29d3f47722dacac6dec0f98254f9ec943613d71c4e987386ec8b4f663`; pilot `9f43eec7e1b620508839bdc2789e88b6f70897d84d58939e087ae37624d955a5`; terminal receipt `47c9058322dab70215ab80a6e394eb11324c2110dad4bd73447bd818566a4032`; cold readback `219af179378ca35596b01b93baaf27e474e34ea626007ddae8eb4a084e233dd4`; reduction `80092a1103879aea758967656aeb2c53bb942fe513c2b747f453e79b6f670591`. Raw vectors and consumers preceded reduction.
- Attempt001 is terminal candidate-complete, PID 1466743 absent. Exactly 6 model / 1 vision forwards, no generated tokens. Allocated GPU time 15.180712953 seconds = 0.004216865 GPU-hours under the 0.25-hour cap; sequence cumulative 0.089661293 GPU-hours under 8. Peak RSS 11645076 KiB; GPU allocated/reserved peaks 9399192064 / 9676259328 bytes. Raw tree 10178333 bytes below 64MiB. GPU, cold readback and reduction commands exited 0. No retry or successor.
