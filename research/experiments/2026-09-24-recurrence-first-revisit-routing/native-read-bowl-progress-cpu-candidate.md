# Bowl progress: saved-evidence CPU candidate

Status: **candidate, CPU-qualified for three finalized bowl-row2 cells only**. The seven-cell GPU attempt remains terminal technical-invalid. The [lead failure ruling](lead-native-read-failure-ruling-v1.json) (SHA-256 `2d6d85de592a5eb06fa807e0f303900f22e5e78deb2ee11c151d215287c544ea`) supersedes launch authority. No model, vision, CUDA, or GPU call was made in this continuation.

## Binding and qualification

The immutable [CPU readback](supporting/native-read-bowl-progress-cpu-readback-v1.json) (SHA-256 `f25d4f15fe3dcf6e99ed9b6a1000e2694b8cd3125aafbf8a24c05ffdc029c0a9`) was produced by the new [CPU-only reader](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_native_history_read/bowl_progress_cpu.py) (SHA-256 `5be964b3921510a979445c4d28222624ad6b6c64f128cef0647955ca4920456b`). It bound the failed [receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-revisit-routing/native-read-remaining-v1/receipt.json) (`087c476dc60ec0fcfb6991d296016657617f5c67f5062a371062e1b84d7e7248`), preflight (`307b3c4fbcfc13451a383c22051b69f333d4d1491e910fbcb2032c241e52eb67`), all 15 direct maintained/captured source pairs, and original fresh-18 raw (`5c3758e2d1e677aa6af6b789d4850012f030c652bca8c826ca8fd57a7f14719d`), trace (`eea6a5b2d5ea16f9c3ce63cefdc31331b399c0e61ec7432b36256644fb83b5f7`), runtime receipt (`23e219bab7f87097f1f6342e723b32b2c2e03c8f57f5d0559fc4ffa55b9a5875`) and target image (`bc28ebbcd6ad8015d7097959539c5069d4c7f92fa4431665d893b3d4b74c7f4b`). The original batch has four requests; target index 3 is `coco2017_train_000000313465`.

At row2, the complete native prefix ends before B x1: raw current header `[20,25)`, physical query `[1382,1387)`. The selected latest A1 row is raw `[10,20)`, physical keys `[1372,1382)`; earlier A0 keys `[1362,1372)` remain readable. Full input tokens, 3-axis positions, attention mask and cache positions match preflight hashes and the accepted row1 common prefix. Companion 2 has its original EOS `151645` followed by 14 pad `151643` tokens; its after-EOS source trace is undefined. Native source chosen/top2/log-normalizer parity passed on active batch rows 0, 1 and 3 (maximum observed error `3.0518e-5`, gate `2e-4`).

All three finalized cells have recorded all-28-layer actual-mask checks. Native and sham expose all 50 selected query/key entries; latest-row treatment blocks those 50. Their same-rectangle complement SHA-256 is `d67ae8026916f6d20b91ed5823df19abdd2ca3f07d43e7e6be65747ea888f169`. Sham full-batch logits equal native exactly. Treatment leaves all saved target prior-row layer states, companion last-layer states and all three companion full-vocabulary vectors unchanged exactly. Raw vector SHA-256 values: native `f33347b28afdb143643b440ff0481968a521b99421f1f3a7763f3faa4f7d8d80`, sham `1288f8618963d7a13b185762bfd6acadd8605a88b81df8365bc1ac9dc571b113`, latest mask `0c9a24fec667c96861a4be6115e847edf9ceff5dc8b3e7c6cb878dde65c0e70c`.

## Full-vocabulary FP64 endpoint

Here B is token `151827`, latest A1 extent `151675`, and earlier A0 extent `151670`; A0 and A1 refer to the same physical bowl. Margins are raw-logit differences in nats. P/rank come from the full vocabulary, with no coordinate renormalization.

| Row2 state | Global winner / runner | B P/rank | A1 P/rank | A0 P/rank | B−A1 | B−A0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native = sham | 151827 / 151829 | 0.0610505 / 1 | 0.00236285 / 68 | 0.0333698 / 7 | +3.251831 | +0.604050 |
| Latest-row mask | 151671 / 151670 | 0.000145367 / 192 | 0.121060 / 3 | 0.128437 / 2 | −6.724782 | −6.783935 |

Latest mask minus native is **−9.976613 nats** for the frozen primary B−A1 margin and **−7.387984 nats** for secondary B−A0. Full-vocabulary TV is **0.8477649094**. Both deltas favor earlier extents at row2 under this intervention; the global masked winner is a third token, `151671`, so neither margin denotes an owner prediction.

For the accepted bowl-row1 revisit, the same latest-row mask gave primary **+0.6459159851** and secondary **−0.8554286957**. Thus row2 minus row1 is **−10.6225290298** primary and **−6.5325555801** secondary. The primary sign reverses; the secondary sign stays negative and grows in magnitude. This is a signed coordinate influence contrast across distinct native histories and positions, not a matched causal estimate of coveredness, physical copying, or recovery. No completed treatment box exists.

## Seven-cell denominator and cost

Three bowl-row2 native/sham/latest cells are CPU-qualified candidates. The executed earlier-row cell remains **technical-invalid/unanswered** because it lacks a finalized consumer receipt; its saved raw vector was not reduced. The three cow-row1 cells are **HOLD/unrun**. The failed attempt's four model/four vision calls and **20.853504562** allocated GPU seconds remain charged. Including the accepted first case, panel charge is **41.101082854 seconds** and sequence charge **0.24960280990891084 GPU-hours**. This CPU continuation adds zero model/vision calls and zero GPU time. Lead acceptance is pending.
