# Which part of coordinate normalization changes recurrent continuation?

## Release and predecessor

Unit: `2026-09-18-readout-common-component`. Lead/root owns release and acceptance;
916-worker owns bounded execution. This inference-only successor is released under
the user's September18 overnight delegation to investigate duplication causes.
No physical-owner label gate, training, checkpoint-stage sweep or architecture
change is introduced. Children, if useful, are Luna with max or lower effort only,
at most two disjoint owners and no descendants; persistent worker stays Astra-low.

The accepted recent-coordinate experiment found directional local responses but
rare substitution copying, with similar responses at nonrecurrent/pre-onset proxy
boundaries. It does not identify a failure-specific copying circuit. Its optional
saved-state arithmetic found that both the shared and centered coordinate-logit
components can contribute to norm-induced winner changes. Those are overlapping
descriptive counts, not module attribution. This successor tests their behavioral
consequences under an identical intervention duration.

**From the same25 frozen completed-row prefixes, does the shared coordinate-logit
component of norm equalization account for its effects on numerical recurrence,
or are the centered coordinate differences and their interaction necessary?**

The strongest alternatives are a generic decision-boundary perturbation, changing
coordinate-versus-noncoordinate competition, and path-dependent interactions of
the two terms. No single component winning here proves training origin, a native
network module, physical novelty, or a universal loop mechanism.

## Fixed sources and four operators

Reuse the exact mature tied/untied checkpoints, prompt, FP32/SDPA runtime, image
bytes, batch companions, serializer, input tokens and25 selected boundaries from
`2026-09-18-numerical-recurrence-feedback/selection.json`. Bind all hashes in a
new execution manifest; preserve predecessor evidence and snapshots unchanged.
There are11 recurrent-onset boundaries and14 nonrecurrent/pre-onset proxy rows,
not25 independent images or a globally healthy control cohort. All seven scene
identities remain represented, including non-recurrent birds. No new case or
success-based selection. No historical coordinate is changed in this unit.

For coordinate effective output rows w_k, freeze
alpha_k=median_j(||w_j||)/||w_k|| and mu=mean_k(w_k), using the existing FP64
convention on actual effective FP32 rows. At each step h is the actual final
head input including the model's own final normalization. Let z_k be the raw
model logit, b=mu dot h, d_k=z_k-b. The four operators are:

1. Original: z_k.
2. Full normalization: alpha_k*z_k.
3. Shared-component only: z_k+(alpha_k-1)*b.
4. Centered-component only: z_k+(alpha_k-1)*d_k.

The last two changes sum to the full change at the SAME state. Compute in FP64
and cast the final coordinate result to the original logit dtype, matching the
qualified norm policy. Leave all noncoordinate logits bitwise unchanged. Do not
substitute mean rounded logits for mu dot h without an explicit precision check.
This centering is a declared analytic convention, not a discovered biological or
network component. Coordinate-family probability and EOS competition can change.

## Primary execution and outcomes

From each unmodified completed-row prefix, rebuild the native history and run
each operator at every subsequent decoding step until natural EOS,32 complete
target rows or512 target tokens. Same intervention duration/cap in all four arms;
no masking, RP change, extra forced prefix, sampling, beam search or withdrawal
arm. Exactly100 target continuations maximum, including fresh original controls.
Do not reuse a source suffix as the only original control for the new operator
implementation. Retain the established target-only continuation semantics;
companion outputs are not evaluated or claimed.

Primary outcomes, reported per image/model/boundary: first free row, exact and
near recurrence duration, native-pattern return, alternative repeated pattern,
invalid/malformed rows, EOS and cap. Use the accepted numerical metrics and
tolerances; do not silently turn neighboring crowded boxes into physical duplicates.
Report overlap in outcome categories. A shorter loop due only to EOS or a switch
to another invalid/repeating pattern is not recovery. Positive annotations may
provide descriptive context but neither zero FN nor new annotation review gates
this package. No population mAP evaluation or policy promotion is requested.

At the initial target slots and every actual first divergence between operator
branches, retain raw/all-four shadow coordinate logits, full-vocabulary winning
competitors, margins, EOS probability, coordinate-family probability, actual head
input and emitted token. Save shadow scores at coordinate positions throughout
the bounded target continuations if inexpensive. Shadow operators on one state
separate immediate readout competition from later divergent histories; do not
compare later unmatched hidden states as if they were a controlled margin effect.
Record when a flip is coordinate-to-coordinate versus a token-family switch.

The question is conditional continuation, not prevention from an empty prefix.
If full normalization does not alter recurrence on this panel, report that result
and stop this late-normalization explanation; do not add empty-prefix or stronger
dose runs to make a component succeed. If either term mimics full normalization,
report where it fails too. If neither does, retain interaction/path dependence
rather than tuning coefficients. No hypothesis-selected subset or coefficient
sweep follows automatically.

## Qualification and evidence

Reuse the accepted loader/target-history path. Before scaling, one actual paired
group must show: identity operator parity; the full operator matches the existing
norm implementation; shared+centered increments reconstruct the full increment;
noncoordinate logits unchanged; actual target history unpadded; correct role and
stop accounting; persistence/readback. Preserve numeric tolerance and a concrete
sensitivity check. One bounded mechanical correction per affected seam is allowed;
faults remain technical, not scientific nulls. Do not retest unrelated infra.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-readout-common-component`.
Preserve exact source snapshots before launch, source/config/prefix/model/operator
hashes, full target tokens, selected tensor artifacts, CPU-recomputable metrics,
selection denominators and exit/cost receipts. Keep failed attempts. No all-layer
all-token KV/attention archive, new service, scheduler, or production refactor.

Cap8 allocated GPU-hours,100000 batch forwards and16GiB tensors including
qualification/failures. This is a ceiling, not a target. Use all8 GPUs for the
independent conditions/cases; keep pair semantics fixed and reuse loaded models
where existing code supports it. Do not repeat completed work merely for GPU
utilization. Expected stress occupancy is not a launch gate. After a measured
pilot, stop/return NEEDS_LEAD if the finite panel cannot fit these bounds.

## Ownership and terminal

Worker owns implementation, this unit state/results and output root. Root owns
index/catalog/question updates, independent acceptance and any successor. Return
one INTEGRATED_CANDIDATE with complete initial reduction, exact reproducer,
limitations and ended/live jobs. A child may return runtime evidence to the worker
but must not publish the package terminal. The persistent worker alone writes
`integrated-terminal.json` and appends `INTEGRATED_CANDIDATE`, `NEEDS_LEAD`, or
`BLOCKED` to `coordination/events.log` when appropriate. Stop after this package;
no automatic successor, new training, labels, Notion or memory writes.
