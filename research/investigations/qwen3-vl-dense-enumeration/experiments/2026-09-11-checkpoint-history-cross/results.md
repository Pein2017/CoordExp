# Two distinct ways of entering or escaping repetition

Status: **completed, lead-accepted, no promotion**. The frozen four-image,
two-checkpoint by two-first-divergent-row-history cross is technically valid.
All8 on-diagonal actions and stops reproduced exactly before their cross cell;
root freshly replayed the16-cell consumer. These are exposed conditional
interventions, not new independent generalization evidence.

## Decision-bearing outcomes

Each entry below is **free IoU50 owners / strict later-free-row repeats / stop**.
Forced history is excluded from free-owner credit. These owner counts are
annotation-relative; unmatched rows are not automatically hallucinations.

| Image | Stable model, Stable history | Stable model, A history | A model, Stable history | A model, A history |
|---|---|---|---|---|
|39654|10 / 0 / EOS|0 / 156 / cap|11 / 0 / EOS|0 / 194 / cap|
|351017|0 / 136 / cap|8 / 55 / EOS|11 / 56 / EOS|10 / 52 / EOS|
|417044|0 / 291 / cap|0 / 291 / cap|10 / 0 / EOS|10 / 0 / EOS|
|477415|1 / 4 / cap|2 / 6 / cap|15 / 1 / EOS|15 / 0 / EOS|

**39654: local history seeding is causally supported.** Supplying A's tiny
banana row makes even Stable50 enter the cap/owner-loss basin; supplying the
old broad row makes even A terminate without strict repeats and freely recover
11 annotated owners. Thus A did not create a wholly new inability to continue
normally on this image: changing entry exposes a bad conditional continuation
already available in Stable50. Exact later loop tokens differ between models,
but that does not refute the history-controlled cap/coverage contrast.

Root re-viewed the original1248x832 fruit-market image. The tiny banana box
`[0,0,66,93]` is over dark round fruit, not visibly a banana; the old branch
`[0,0,1247,303]` is broad/group-like. Neither row is thereby admitted as a
correct instance training target. Both forced prefixes match0 GT owners;
the10/11 successful owners above are freely generated, not forced credit.

**417044 and477415: changed continuation behavior is supported.** Changing
only the first divergent row does not rescue Stable50, whereas A succeeds
under either supplied history. In477415/Stable-history the first free chair
row is even token-identical across checkpoints, yet later outcomes sharply
diverge. Stable50's low strict-repeat counts here are misleading in isolation:
331/329 complete geometry-invalid free rows plus one fragment dominate its
two capped continuations. A has no drops in these two cells.

**351017: both routes matter, with residual damage.** Supplying A's table row
to Stable50 enables useful continuation and EOS; changing the model also
escapes the original cap under the old history. All three EOS cells still
have52–56 strict repeats and21–51 free parser drops. Ending the cap is not a
clean owner-enumeration repair.

## Inference and boundary

There is no single established repetition mechanism across these four cases.
Separate **entry selection** from **conditional continuation quality**. A
shared model can improve the latter on trained cases while shifting another
image into a pre-existing bad-history basin. The strongest remaining
alternative for broader regression is diffuse parameter interference, which
the separate56-reference greedy-preservation microscope addresses.

The literal first free token is always a wrapper and is non-discriminating.
First complete free rows differ in6/8 same-history checkpoint pairs; full
suffixes differ in8/8. These are useful localization facts, not the primary
basin outcome and not proof of a KV circuit, copying mechanism, physical-owner
absence or training-distribution origin. No further cases or intervention
variants are authorized by this completed unit.

## Reproducibility and cost

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-checkpoint-history-cross`.
Accepted packet `preparation-v2/packet.json` SHA256
`7459968bef4da5d89c3f75d5e63cf50d2fbd7c89c64efc00e85df776678821f2`;
producer `run_cross.py` SHA256
`344317d4dfa66fec9e864caea95f4abc45ec1857a006e8168184b98498446e88`.
Accepted `retry-v2/consumer.json` SHA256
`18f530d2af1f68fbe45fc30615bbf9b577773369c670e131098d9c5fe65d7bde`;
reduction SHA256
`ed5062fe242199bc942d12ef852afb119eccbf49b9e299a5475e4be770095c48`.

8loads,16continuations/image forwards,26,602 generated/model-forward tokens;
2,291.772 allocated GPU-seconds (0.636603 GPU-hours), maximum500.474s/worker.
CUDA allocated9,956,440,576/reserved10,632,560,640 bytes and RSS12,367,589,376
bytes remain within frozen bounds. All8outer exits0; owned processes released.

The original attempt failed before all model loads/forwards because explicit
device peak-reset preceded CUDA initialization. Root reproduced the failure
and minimal default-argument correction; the exact one-line diff and all5CPU
tests were freshly checked. Original producer/packet and all failed receipts
are preserved. This technical failure is not a scientific null.
