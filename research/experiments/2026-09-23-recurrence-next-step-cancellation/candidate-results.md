# L/U next-step cancellation: worker candidate

Status: **candidate complete, pending lead acceptance**. The strict shared compensation criterion fails because the x1=418 position-first path has a step TV below its frozen threshold. The same probe has a large, opposing history-first path, so the outcome is mixed rather than uniformly small. No follow-on call is proposed.

## Frozen contrast and qualification

The [protocol](unit.md) SHA-256 is `c9b0a93c8e8665f49f5e83d21d590bf47a2c2ce061ab60ae631e529e27da1629`; the [manifest](manifest.json) SHA-256 is `53864f21ffa06b7a44bc0f3ac4643b6f1bd5131c80502eace2208c7f6fdec468`. The original source is one `refined-04` request, `coco2017_train_000000477415`, batch0. L is `native[:45]`, U is `native[:54]`, each with the original five-token current prefix through the same supplied x1 (510 or418). L's native S positions are432–436; U's are441–445 on all three MRoPE axes. Crossed cells change only those five S rotary positions. History, image, physical order, masks and cache order stay fixed. The position intervention includes its downstream contextual effects.

CPU preflight verified the single source, four accepted full consumer inputs/positions, all four bound reference vectors and the manifest native TVs. The same actual-consumer verifier used by the GPU model hook rejected wrong S slot, extra historical content, replaced supplied x1 and wrong S position. All direct maintained imports and the producer were captured before launch. The shape-aware estimate, including a 2× margin, was `33.625808` seconds versus the 900-second incremental cap.

All **eight planned cells executed; zero held**. The first four diagonal cells matched their accepted complete vocabulary vectors exactly (maximum absolute error `0`, tolerance `2e-4`) before any crossed call. The eight observed full input/rotary vectors, attention-mask/cache-position hashes, and full logits are bound per cell. Separate-process cold readback passed.

| Order | Owner / history / S positions | Global winner / runner | Gap | Raw vector SHA-256 |
| ---: | --- | --- | ---: | --- |
| 1 | 1589003 L/L | 152083 / 152078 | 0.04091835 | `938cd57801567d0f1955dea125d09c6c2fa523a07468d57ea01940fc1acd2455` |
| 2 | 1589003 U/U | 152078 / 152083 | 0.00493431 | `313a89db8cda843ae8873337c9f8dba3885f2e3360e2c97181f67e9ae325c427` |
| 3 | 1586761 L/L | 151670 / 152669 | 0.11068535 | `ff92dfc75440623115192375387714695c94342d5ab859dc42384c4623382b63` |
| 4 | 1586761 U/U | 152669 / 151670 | 0.05361080 | `a74a4a28bf9732f01dd5051cc02cfeb74dd7d17f60670760481098a90f8840f1` |
| 5 | 1589003 L/U | 152083 / 152078 | 0.03663826 | `0b536eb9c08de05145811a4868314fcbada96e3067cadedefee8fb783c647706` |
| 6 | 1589003 U/L | 152366 / 152378 | 0.09226418 | `b9c56062fda6ff0750cef217d50ccd7ad007fae7017f1982e5b91382650b33d6` |
| 7 | 1586761 L/U | 151670 / 152669 | 0.04886436 | `179be6b0a95dfc420767f96bf70c42e2417ae72c6d0caed17b8b90a45054c145` |
| 8 | 1586761 U/L | 152543 / 152548 | 0.13776493 | `aa63cb4a24f8083b5b42bf19817c190e63740b1fa2197bdfc5ac13ecb89bf1b1` |

## Full-vocabulary decision

For each owner, `a=P_LL`, `b=P_LU`, `c=P_UL`, `d=P_UU` are FP64 full-vocabulary softmax distributions. The native net is `d−a`. Position-first steps are `b−a, d−b`; history-first steps are `c−a, d−c`. The frozen factor threshold is `max(2×TV_native, 0.001)`. Strict compensation requires **both** step TVs strictly above that threshold and cosine below `−0.5` on **both** paths and **both** probes. Uniformly small requires all four step TVs at or below the threshold. No result lies within `1e-8` of a decision boundary.

| Owner | TV native / threshold | Position-first TV1, TV2; cosine; length/net | History-first TV1, TV2; cosine; length/net | Interaction L1 / L2 | Category |
| --- | --- | --- | --- | --- | --- |
| 1589003, x1=510 | 0.02434872 / **0.04869744** | 0.07271153, 0.08245666; −0.98787954; 6.3727 | 0.76500289, 0.76193671; −0.99933888; 62.7113 | 1.45678902 / 0.10292257 | **compensation** on both paths |
| 1586761, x1=418 | 0.04610341 / **0.09220682** | **0.08285628**, 0.10312148; −0.89931939; 4.0339 | 0.42413288, 0.39123894; −0.99819098; 17.6857 | 0.68481013 / 0.05259337 | **mixed**: first position step below threshold |

The x1=418 position-first first step is `0.00935054` below its threshold. Its second position step and both history-first steps exceed the threshold, so the frozen uniformly-small criterion also fails. Both paths sum to the native net algebraically; that identity was verified and is not evidence of compensation by itself. An independent CPU computation from the eight retained vectors reproduced every TV and cosine within `1e-10`.

The secondary winning-pair margins (408 versus413 for x1=510; 999 versus0 for x1=418) are in the [reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/reduction.json). The U/L crossed readouts have different global winners from either winning pair, which is why the decision uses full vocabulary distributions and retains third modes. These conditional cross effects cannot be assigned a unique additive percentage. The result supports substantial opposing local responses for the first probe and for one path of the second, but **does not meet the shared two-path criterion**. It identifies no attention circuit, inevitable long-sequence degradation, valid physical box, or natural exit. The probes share one image and are not independent images.

## Artifacts, cost and terminal state

- [CPU preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/preflight.json) SHA-256 `416557956186a4cd9e41aa25065fc89180ce8b1635fc05fa3d80de2b6b06c5a0`; [pilot](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/pilot.json) SHA-256 `90de488f15a3a218b32e8524faf9315f2a3abaa1f1b43f58baf68ff6f1cf5a99`; [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/receipt.json) SHA-256 `157828602d0196597717255cb799ef8a61275dac312865bb494b724775e28265`.
- [Cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/cold-readback.json) SHA-256 `0fadbdc729d515218b229e3a51a33ff5473db39a943704db7634a249b1083daa`; [reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-next-step-cancellation/attempt-001/reduction.json) SHA-256 `9aa2299a5b1c64888a4690ccc1c665f3aaeff495b8a5de0e3aeb3624637243d3`; [machine evidence](supporting/evidence.json) binds all eight raw vectors and actual consumers.
- Exactly 8 model and 8 vision forwards, zero free tokens. Allocated GPU time including load/setup `11.670776` seconds = `0.003241882` GPU-hours; outer job wall `18.099504` seconds. Charged sequence cumulative `0.069672503` GPU-hours versus cap8. Peak RSS `11,550,532` KiB; GPU peak allocated/reserved `9,072,963,584` / `9,648,996,352` bytes. Raw output after reduction `6,316,577` bytes.
- One GPU job exited `0`; PID `1352989` is absent from `/proc`. Accepted predecessors, the existing staged index and unrelated work were not modified. No Git action was taken.
