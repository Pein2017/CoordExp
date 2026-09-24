# Three-call native historical-read candidate

Status: **worker candidate; lead acceptance pending**. This is the exact first-case admission only: mature313465, original untied `fresh-18` four-request batch, target index 3, before row1 x1. The seven proposed bowl-row2/cow cells remain HOLD. No alternative x1 was supplied and no token was generated.

The frozen [protocol](native-history-read-protocol.md) SHA-256 is `9ccece16a05d07aca61da9311ffea954382d306d23ad85f106baebe16ebd955c`; [admission](lead-native-read-admission-v1.json) SHA-256 is `887875d1ccf7376881e2faddf74645ed560828a3d7a854df1bf96896a8576dab`. The pre-result [interpretation note](lead-interpretation-note-v1.md) SHA-256 is `98a827ea1e2147c7b40c20e627a0dc748da34136eb97ef046695d2974c93d5a9`. The accepted CPU candidate MD/JSON remained byte-identical at SHA-256 `66dc59b1ece316a079ad5432232a1fcf8f003ffb6935751ec3a2607ee604152b` / `dcdc2f20a6dc64b6ec1110faf7cc2a7c4e4cf3dcbd4355c9ee2338b4c07996c4`.

## Frozen call and mask evidence

The [producer](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_native_history_read/run.py) SHA-256 is `0a76ecd3d37ad005ce6c6c905567e381ee1153842d6fa0c710dca9e74139db54`. Its [CPU preflight](supporting/native-read-preflight-001.json) SHA-256 `0b43febe1a3e4ab0e229d8b5c7a11434ead97658c29ecdff358161f4a0191ade` binds 15 direct source captures, all four source requests and images, exact processor identity, batch width 1377, prompt width 1362, and raw-to-physical spans. The 170.81-second shape-aware estimate fit the 360-second cap; the [output cost check](supporting/native-read-cost-check-001.json) projected 34.55 MB under 64 MiB. The actual caller's CPU fixture rejected wrong target, query/key spans, prompt width, changed source mask, and companion/historical/other-key edits. The installed SDPA path uses a boolean causal mask (`True`=readable); the identity 4D mask exactly matched its native construction on this source shape.

Every model call used original prompt/media, tokens, and MRoPE positions. Native target raw history is row0 `[0,10)` plus current header `[10,15)`; the selected keys are row0 `[0,10)`. In padded physical slots these are target row 3, queries `[1372,1377)`, keys `[1362,1372)`. The target header is five tokens; row0 is ten tokens. The ended third request retained its native 11 tokens **including EOS**, followed by four pad tokens. No after-EOS source trace parity was claimed for that request.

The native call used the original 2D input mask. The identity sham used its equivalent installed 4D boolean causal mask. Treatment changed exactly the 5×10 target query/key rectangle from readable to blocked, broadcast over all heads on all 28 text layers. Every attention-module entry was checked against the expected mask; no Q/K/V, positions, tokens, parameters, historical-query rows, other keys, or companions were patched. The [mask audit](supporting/native-read-mask-audit-001.json) SHA-256 `4b1cf8d743b579aec49731470f28a80f32202461c7834b18ae624a776c6a3c77` records 50/50/0 readable selected entries for native/sham/treatment and one identical complement hash. Current-row Q/K/V and later states were allowed to respond.

| Admitted cell | Source/consumer result | Cumulative internal GPU interval |
| --- | --- | ---: |
| Native | Target and active companion chosen/top-two/log-normalizer source parity passed; maximum recorded source error `1.34e-5` (<`2e-4`). | 9.716 s |
| Identity-mask sham | All four full-vocabulary endpoint vectors **exactly** equal native; historical row0 hidden states equal at all 28 layers. | 11.978 s |
| Latest-row mask | All three companion full vectors exactly equal native; historical target row0 states equal at all 28 layers. | 14.237 s |

Raw full-vocabulary vectors and per-layer historical/companion states were saved separately for each call before reduction. A separate CPU process passed [cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-attempt-001/readback.json) SHA-256 `da0ea01e80cb0cf166213342effcf879a06affcff0a0e202ecbce1361d1a48dd`: it reconstructed source inputs and masks, checked actual input hashes and all 28 layer observations, reran source parity, and recomputed the full-vector reductions. The [terminal receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-attempt-001/receipt.json) SHA-256 is `ca4af87c3b134bb13bfc70ae44f1d84005b50a9be11981fabd4aba2bb35aef2e`.

## Signed result and extent ambiguity

The frozen primary endpoint uses B x1 `151827` minus **current revisit A1** x1 `151675` in raw logits. The same physical A bowl was previously emitted at historical A0 x1 `151670`. The latter is a secondary extent check from the same vectors, with no new model call.

| Readout | Native | Identity sham | Latest-row mask | Mask minus native |
| --- | ---: | ---: | ---: | ---: |
| Primary `z(B)-z(A1)` | −8.603378 | −8.603378 | −7.957462 | **+0.645916** |
| Secondary `z(B)-z(A0)` | −8.410620 | −8.410620 | −9.266048 | **−0.855429** |

The primary positive delta exceeds the frozen `0.01`-nat deadband and matches the **local A1-coordinate** directional prediction. The opposing A0 delta makes the result **extent dependent**. In fact the global x1 winner changes from A1 `151675` (P `0.14491`, rank 1) to A0 `151670` (P `0.32306`, rank 1), which denotes the **same physical bowl**. B remains low probability: P `2.66e-5`→`3.06e-5`, rank 203→162; it is not the global winner. A1 falls to P `0.08729`, rank 5, while A0 rises from P `0.11951`, rank 4. Full-vocabulary TV(native, mask) is `0.232054`. The pairwise positive delta and global redistribution are distinct observations; neither a new-owner choice nor reduced physical copying follows. No complete box was decoded, and this assay cannot distinguish owner-level copying from history-conditioned localization refinement or other surviving contextual pathways.

## Cost, terminal state, and boundary

The producer's internal timer recorded 14.243317 seconds including model load and three forwards. The enclosing GPU job took 20.247578 seconds; the [cost audit](supporting/native-read-cost-audit-001.json) SHA-256 `e7358148c76567ca9010993c223d9c3c711c2290f50a17909e0bd847a1968f2e` conservatively **charges the full outer interval**, including Python startup. That is `0.005624327` GPU-hours; charged sequence cumulative is `0.243810170` GPU-hours from the frozen prior `0.238185842`. Counts are exactly **3 model / 3 vision / 0 free tokens**. Peak RSS was 11,637,528 KiB; peak allocated/reserved GPU memory 9,933,522,432 / 10,659,823,616 bytes. Final raw-root payload is 17,078,563 bytes, below 64 MiB. The GPU process is terminal and its PID is absent. There was no failure, diagnostic model call, retry, or held-cell launch.

Commands were `python -B -m probes.training_set_completion.recurrence_native_history_read.run preflight`, then `run`, then a **separate process** `readback`. This finite result is ready for lead technical/scientific acceptance. It does not admit the seven held cells, a full-row continuation, a physical-recovery claim, or a successor.
