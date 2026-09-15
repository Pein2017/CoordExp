---
title: Checkpoint versus first-divergent-row history
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-11-checkpoint-history-cross
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: verified
updated: 2026-09-11
---

## Question and contrast

Executed disposition: [results](results.md). The corrected single retry is
lead-accepted; the original pre-model technical failure remains preserved.
This unit is closed and does not authorize further cases or training.

Between unchanged Stable50 and the accepted positive32 checkpoint, does the
first divergent completed row seed the subsequent loop/recovery, or do the
changed model parameters alter continuation even under the same row history?

The overnight goal authorizes this bounded inference experiment. Use four fixed
exposed images:39654(new catastrophic loop),351017(residual loop),417044 and
477415(resolved old loops). For each, cross two checkpoints with two literal
histories: common prefix h plus the first divergent complete row from either
checkpoint. No row is labeled correct merely because a model generated it.

This yields16 cells:4images x2models x2histories. The8 on-diagonal cells must
reproduce the retained full natural action exactly; the8 cross-history cells
are new conditional interventions. The existing natural observations remain
the baseline, not independent new generalization evidence.

First-difference / common-row-boundary / Stable row-end / positive32 row-end
token offsets, zero-based with ends exclusive:

|image|difference|h end|Stable end|positive32 end|
|---|---:|---:|---:|---:|
|39654|15|9|18|18|
|351017|10|9|19|20|
|417044|15|9|19|19|
|477415|1|0|9|9|

Bind exact IDs to the prior frozen endpoint packet and actual endpoint-A
consumer. All these branch rows are complete and contain no EOS; earlier
malformed history, if present, remains literal. No GT-based branch selection,
new positive label, or manual geometry repair.

## Evidence and interpretation

Each worker first executes its on-diagonal history as the real-entry/parity
gate, then its cross-history cell only after exact match. Preserve complete
raw output and all parser drops. Report first-free action, strict class-blind
native-pixel later-row IoU>0.95 repeats, invalid/malformed rows, token length,
EOS/cap, and annotation-relative owner sets. Separate prefix/forced/free rows
and owners; forced history is never credited as freely recovered.

If both models follow the supplied history, this supports local state-seeding.
If one model behaves differently under the same history, changed conditional
dynamics remain necessary. Mixed results are admissible. A coordinate/class
change can itself alter visual support; this is not proof of a KV circuit,
owner ledger or universal copying mechanism. Existing negative results on
simple history multiplicity remain closed, not rerun.

## Runtime bounds and stop

Same base, special-token embeddings, prompt/image geometry, unmerged FP32 DoRA,
SDPA and native greedy/RP1/EOS151645 as the accepted endpoint. Only the adapter
and literal row history vary. Original total action cap3084; free allowance is
3084 minus history length. No sampling, gradients, training or checkpoint writes.

Eight independent1GPU workers, one checkpoint/image pair each, one model load
and two continuations each. Global16image forwards, at most49,082 generated
tokens/model forwards; per-worker at most6,150tokens,1500seconds,12GiB CUDA
allocated/reserved and16GiB RSS. Lifecycle peaks include loading. Preserve exit
status and failures; no automatic retry/backfill/dose or prefix expansion.

Prepare CPU contract/identity/counter checks, then request root's exact launch
grant; this unit is not that grant. After the16-cell frozen reduction and root
visual/technical acceptance, stop. Root owns interpretation and any next phase.
