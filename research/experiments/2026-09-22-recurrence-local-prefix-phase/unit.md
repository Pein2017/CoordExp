# Target-query-only local prefix phase coherence

2026-09-22. Root leads and accepts; existing Luna-max worker implements and
executes. The user authorizes autonomous continuation and dataset expansion
when necessary. This unit first resolves one intervention confound; the parallel
saved-cohort inventory is read-only and does not expand this unit's population.

## Frozen question and decision

At val7511 row42 x2, with actual late layer0 output and late current Q/self-K
phase in layers1..27, does restoring the five current-row prefix K phases ONLY
for this query remove the third-coordinate579 result under baseline and identity
recomputation gates?

The accepted predecessor is recurrence-downstream-query-phase attempt002.
Matched late incoming state and query/self-key phase produced579 rather than
early38 or late999. Its historical keys stayed at early phase, including the
five current-row prefix keys. That intervention disrupted local relative phase.
This unit tests that precise alternative; no extra layers, heads or doses.

Full-vocabulary argmax at target batch2, physical query1703 is primary. Root's
advance tentative prediction is restored38. Restored38 supports a local-coherence
explanation for579 without explaining native exit.999 supports sufficiency of
late incoming state/current phase/local alignment at this landmark without the
whole extra history package. Retained579 or another third winner weakens the
categorical local-coherence explanation and closes this branch without another
restoration scan. Report full probabilities/ranks of38,999,579 and top10, plus
z38-z999; a binary margin cannot rescue a failed categorical prediction.
No outcome establishes physical recovery, natural onset/exit timing, axis
confusion, population generality, or the origin of training failures.

## Inputs and cells

Reuse exact mature untied+axis step2444, original FP32 SDPA batch4,width2127,
native input/model/media hashes and selection in the predecessor. Recipient
query1703 (row42, action384), donor2126 (row89, action807), actual late layer0
head_output[1], saved late query cos/sin. Local prefix positions1698:1703 map
one-to-one to2121:2126. Read LIVE normalized pre-K for those five positions and
actual model position_embeddings from the full original inputs; do not synthesize
phases from physical index arithmetic or substitute late contextual key content.
Verify equal prefix token IDs and saved donor/recipient query phase parity.

Three fresh model cells, each otherwise identical to accepted held_late_phase:
1. query_only: original full SDPA path; reproduce accepted579 and full logits.
2. identity_recompute: original full SDPA, then an extra target-query-only SDPA
   using exactly its existing Q/K/V/mask/scale; replace only that output slice.
3. prefix_coherent: same target-query-only recomputation, but K entries for the
   five local prefix tokens use their LIVE normalized pre-K and corresponding
   late prefix cos/sin. All other selected-query K, all V, Q, mask and scale stay
   identical to the live call. This includes unchanged older history0:1698.

The full-attention K must NEVER be globally rephased. Its outputs at all other
queries must be exactly unchanged by the local replacement at that layer.
Later layers may adapt at the target and future positions. Use causal native
row mask [batch2,:,1703:1704,:], not an implicit one-token causal mask. Post-GQA
Q/K/V have16 heads; live pre-K has8, repeated twice after rotation. Selected
query Q shape[1,16,1,128], K/V[1,16,2127,128]. Layer0 is unchanged beyond the
accepted head-output substitution; text-module identity filters leave vision
untouched. No generation/cache reuse. Future-query treated logits are not a
scientific endpoint.

## Qualification and acceptance

Reuse accepted producer collect_cell and its actual SDPA/o_proj attestors without
editing that source. A task-local outer SDPA wrapper can be captured by its
inner observer; scoped text hooks capture live prefix pre-K/phases. Prove the
composed output actually reaches o_proj. Restore hooks/functions/output overrides
in finally; no generic hook framework or copied producer.

CPU positive qualification goes through installed sdpa_attention_forward, real
F.sdpa and the SAME selected-query output replacement used by production, with
unequal dimensions, GQA and a nontrivial causal mask. Compare independent paired
coordinate rotation plus explicit attention calculation. Identity recomputation
must pass. A wrong-query output replacement and a global-prefix-K mutation must
fail at the consumer boundary; do not reject solely via assumed patch constants.

Save actual pre-K, early/late prefix phases, consumed prefix K, query vectors,
selected attention output, and numerical errors before gates. Independent FP64
rotation readback must match within2e-4. For fixed live query/key content, the
coherently shifted query-prefix scores must equal original-relative-phase scores
within2e-4. Check all27 layers, self-score/norm gates from predecessor, exact
unchanged historical K/V in the full call, older K and all V in the selected
call, unchanged original full-call inputs, and exact off-target output replacement.

Fresh query_only full logits must match predecessor held_late_phase within2e-4
and winner579. Identity full logits must match fresh baseline within2e-4 and
winner579; stop before treatment if either fails. Every cell must match the
actual late native incoming layer1 state within2e-4. All companion logits match
baseline within2e-4. Bind native masks/cache/selected LM positions exactly.
Persist raw cell evidence before numerical gates; technical invalidity leaves
the scientific question unanswered. Root independently reads tensors, source
hashes and consumer evidence; only root marks lead-accepted.

## Ownership, cost and stop

Worker owns only probes/training_set_completion/recurrence_local_prefix_phase.py
and outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-local-prefix-phase/attempt-001.
Root owns this unit, selection, independent readback and knowledge records. Bind
predecessor acceptance/selection/tensors, producer and imported/installed sources
before first model call. Preserve source through the maintained source owner.

Exactly3 model/3 vision calls on GPU4,10minutes,48MiB total tensor payload. Each
cell has28 full text-attention calls; identity/treatment each add27 single-query
kernels, which are counted separately and do not count as model forwards.
No training, rollout, scan, peer codebook work or unrelated cleanup. Run CPU
qualification then the three cells automatically. On failed attempt preserve
evidence and return promptly without retry. Stop after stable candidate.
