---
title: Source versus Rweak first-row crossed continuation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-09-09-source-rweak-row-cross
status: complete
updated: 2026-09-09
---

# Current context and authority

Root completed this bounded investigation. Results and the no-promotion boundary
are in `readout.md`; the completed same-input routing comparison and its bounded
cost snapshot are in `benchmark/readout.md`. All study GPUs are released; no
further model execution or training is planned. The user approved the revised Source /
Rweak64 four-cell continuation study, at most 32 cases, 64 new cross continuations,
necessary diagonal/implementation qualifications, no optimizer steps, and **4
allocated GPU-hours total**, including benchmark qualification and recovery.
All eight A100 80GB GPUs may be used. The user specifically requests efficient
batching and parallel inference, not a long serial decode. No new training,
architecture, annotation mutation, protected confirmation read, or scope sweep.

The user also authorizes explicit same-input model comparisons:
- Engineering: `gpt-5.6-sol/xhigh` versus `gpt-6-astra/low`.
- Scientific interpretation: `gpt-5.6-sol/max` versus `gpt-6-astra/medium`.

Temporary removal of Sol from default routing is a possible recommendation,
not a foregone result or permission to delete global model configuration.
The existing skill preference against unsolicited duplicate tasks is superseded
for these explicitly requested paired benchmarks. No benchmark worker may spawn
helpers, inspect the competitor, or change its assigned model/effort.

# Scientific question

From the two frozen original Source and Rweak64 checkpoints, does crossing the
first naturally divergent complete row expose a better complete outcome and
distinguish direct action-owner changes from later-owner changes and fixed-visible-
history checkpoint dependence?

The strongest alternatives are direct replacement of an owner by the intervened
row, bbox/category/matching changes, and interaction between row and checkpoint.
Path sensitivity alone is already known and is not a new recovery result.
Bad greedy continuation does not prove no correct continuation exists.

For image I, p is the common generated prefix through complete row boundaries;
a0/a1 are the first divergent complete Source/Rweak rows, or an actual EOS action.
Y00 and Y11 are the stored native-from-prompt diagonals. Y01 uses Source to
consume p+a1 and continue; Y10 uses Rweak to consume p+a0 and continue. Build
all recipient states with recipient forward passes, never donor caches.

All four complete outputs and owner sets remain in the result. Do not change
the primary denominator or silently drop direct-action losses. Separately report
which owners are attributable to the forced row/prefix and which are realized
only by the suffix. Global matching ambiguity must remain explicit; do not infer
physical equivalence just because a matched GT ID is the same.

# Fixed evidence population and selection

Existing evidence root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/`.
Input: `inputs-v1/holdout512.jsonl`, 512 images / 3759 annotations, SHA256
`62aff40429cfc10f0a640a6e86d298b560ab79d25ce0a776393757a12d15accd`.
Diagonal run roots:
- `evaluation/Source/holdout/Source-holdout512-native-v1/`
- `evaluation/Rweak/holdout/Rweak-holdout512-native-v1/`
Reference reduction: `evaluation/owner-focus-reduction-v1.json`, SHA256
`418bfeea25d07399917b644bca3361477373bb0d38a20f9da8cfc94c40f08491`.

Reconstruct generated IDs from ordered non-pad prediction token traces and verify
them against original text, row identity, image plans and source manifests. Do not
decode/re-tokenize to invent an alternative historical token route. Preserve
original stopped/capped/malformed outputs as population evidence.

Outcome-stratified, deterministic case selection: at most 16 images with any
Source owner loss; at most 8 with gains and no losses; at most 8 with identical
matched owner sets but changed generated tokens. Rank within each stratum by
SHA256 of `20260909:<row_id>`. A case must have a definable common row-boundary
prefix and two complete legal row/EOS actions. Report every exclusion reason;
if a stratum has fewer eligible cases, take fewer without cross-stratum refill.
Never select using new cross-continuation outcomes or visual attractiveness.
EOS cases are not excluded merely for being terminal. This selected panel cannot
estimate population mechanism frequencies or independent generalization.

The data owner returns a CPU-verifiable manifest and source bindings before
engineering qualification or scientific GPU execution. Root owns its freeze.
Protect confirmation512: neither outputs nor selection input may be read from it.

# Frozen decoding and efficient execution

Live original manifests confirm HF greedy, **RP1.0**, temperature0, top_p1,
max_new_tokens3084, native Qwen im_end stop, historical per-device batch4.
Do NOT inherit RP1.10 from the later A16/feedback experiments. Original Source
adapter/selected embeddings and Rweak64 model-composition identities own loading;
verify actual runtime dtype/attention/geometry/prompt from their receipts/configs.

The whole generated trajectory cap includes p and the intervened action. A
cross continuation may use only its remaining token budget. For EOS actions,
terminal semantics are explicit, never reopen generation after forced EOS.

Execution-only permitted changes: independent per-GPU data parallelism up to8,
larger per-device batches after real qualification, length/shape bucketing,
model-load reuse, and omission of unused score/attention/hidden-state payloads.
No altered precision, attention algorithm, logits policy, media geometry, prompt,
EOS handling or cap to obtain speed. Do not bypass required model composition.
Batching must preserve per-request continuation budgets and exact token histories.

Measure actual batch throughput, generated tokens, peak allocated/reserved memory,
host RSS, image/model forward counts, batch padding, initialization and decode
time. Pick batch size from measured throughput and memory, not largest B alone.
Single-request/B>1 mixed-prefix and same-arm replay checks must cross the real
save/reload/parser/matcher consumer. Report a genuine batch sensitivity mismatch
rather than silently changing the original scientific or numerical tolerance.
Use at most four shared diagonal fixture cases (up to8 diagonal outputs) plus
their necessary cross checks for the two implementations; reused qualification
cross outputs may seed the final table only if their exact final code/config and
artifact identities are accepted. All qualification compute counts toward4h.

# Outcomes and scope

Primary: original category-consistent global one-to-one IoU50 annotated-owner
sets/counts on complete outputs. Carry IoU60/80, valid predictions, macro/micro
recall and annotation-relative F1, gained/retained/lost IDs, pixel-IoU>0.95
category/GT-independent later repeats, parser drops, EOS/caps and all case counts.
Unmatched predictions are annotation-relative, not automatically hallucinations.
Report direct-action versus later-owner explanations without replacing primary
metrics. No unique causal contribution percentage in the presence of interaction.
Off-diagonal outputs are conditional interventions, not native deployment scores.

Potentially better crossed outcomes motivate a subsequent learning question; they
do not authorize training or establish transferable owner-state representations.
Mixed or negative patterns are valid bounded outcomes. Technical invalidity is
not scientific failure. Stop after the one fixed panel and final interpretation.

# Same-input benchmark and ownership

Both engineering candidates receive identical brief bytes, immutable source
bindings, schema and acceptance fixtures, with fresh context and the same tools.
Candidate write directories are isolated by their own agent leaf name; the shared
checkout baseline is `f8c5f1f5100159db62dd7ff4090555dff63b99d8` in
`/data/CoordExp/.worktrees/self-rollout-behavior`. Other named worktrees are
read-only source providers. Root is the only shared record/launch owner.

Both candidates must complete the same real qualification for benchmark acceptance.
Root selects one accepted implementation for the single formal cross panel;
an already accepted implementation need not wait for the competitor's readiness.
Scientific candidates
receive the same frozen results and question, including direct-row/suffix and
identifiability counterexamples; no private engineering history or peer answers.
Their harder task is to derive what these four observations identify, demonstrate
non-identifiable alternatives, and recommend whether any next learning objective
is justified—not to invent another GPU arm.

Acceptance cost covers initial work, substantive review, rework, qualification,
and attributable lead intervention. Preserve explicit dispositions and cached /
uncached / output tokens; report queue/runtime waits separately. Prices, if used,
are the user's estimate basis, not an invoice. Missing usage or rates are unknown,
not zero. Quality/correctness precedes time and cost in the routing decision.

No pushes, unrelated edits, blanket staging, branch/worktree retirement, global
configuration edits, or new Notion publication are part of this investigation.

# Lead data freeze — 2026-09-09

Root independently replayed `data/prepare.py --verify` and the eight data tests;
both exited0. Full source reconstruction reproduced the exact manifest bytes,
all512 images/3759 annotations and original threshold totals. Frozen manifest:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-source-rweak-row-cross/data-v1/manifest.json`,
SHA256 `af3fd69e05bea3c148c293e528a2ea738ac49e45631bb0b71a94d7eb56d3fcd8`.
Actual selection is16/8/8,32 cases. Four illegal first actions,25 identical-token
cases and451 stratum-cap exclusions are individually recorded. No refill.

Important scope observation:25/32 common prefixes are empty; the remaining
prefix lengths are9 (3 cases),10 (2),11 (1),20 (1). All64 selected actions are
rows; no EOS branch was selected. Thus this panel predominantly examines first-
row choice/continuation, not a late-history recovery mechanism. Do not expand or
reselect to manufacture the latter. EOS implementation eligibility remains tested
on a synthetic fixture, not a new scientific EOS result.

Original entry YAMLs are no longer present, but original run `configs/resolved.yaml`
and `resolved.json` are retained, agree and are hash-bound. The persisted YAML is
wrapped under top-level `config`; use the manifest's actual config mapping or the
proper loader, never a nonexistent entry path. Historical entry path/hash remain
explicitly marked unavailable. This is not a missing checkpoint or data identity.

The data package is lead-accepted; engineering GPU qualification and scientific
cross results remain unproven. No GPU grant has yet been issued at this freeze.
