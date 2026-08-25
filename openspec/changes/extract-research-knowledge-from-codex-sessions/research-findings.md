# Integrated CoordExp research findings

## Scope and authority

This is the lead-integrated view of 6,278 Codex sessions and 9,746 research-document paths captured at `2026-08-25T12:35:18Z`. It is a routing and synthesis artifact, not a replacement for the current research graph. Exact units, results, artifacts, and current decisions remain authoritative.

Package evidence:

- [May–June sessions](/data/CoordExp/openspec/changes/extract-research-knowledge-from-codex-sessions/evidence/session-early/synthesis.md)
- [July sessions](/data/CoordExp/openspec/changes/extract-research-knowledge-from-codex-sessions/evidence/session-july/synthesis.md)
- [August sessions](/data/CoordExp/openspec/changes/extract-research-knowledge-from-codex-sessions/evidence/session-august/synthesis.md)
- [Research documents](/data/CoordExp/openspec/changes/extract-research-knowledge-from-codex-sessions/evidence/research-docs/synthesis.md)

## Current contract and one unresolved conflict

The stable row wrapper is:

```text
<|object_ref_start|>{desc}<|object_ref_end><|box_start|>{x1}{y1}{x2}{y2}<|box_end|>
```

Rows are concatenated without a row separator or newline. Current production code still validates `geo_sorted` as top-to-bottom then left-to-right (y-then-x). The research route prefers explicit x-then-y ordering (`geo_sorted_xy`) because it aligns authored order with `x1 -> y1` decoding and has bounded positive evidence; a future production switch remains possible but is not yet an implemented production decision.

The terminal newline order is not reconciled and must not be silently normalized:

- the user's migration clarification names `\n<|im_end|>` at the final return boundary;
- live code in `src/templates/renderer.py` and `src/qwen/tokens.py` currently requires `<|im_end|>\n`.

This is `needs-adjudication`, not a stale-note overwrite. No code or Notion page was changed in this wave.

## Integrated findings

### 1. Current authority is a research graph

- Current owner: the canonical `research-probes` compass, its linked units/results/decisions, and narrower current owners.
- Evidence: 9,746 document paths collapse to 653 unique byte contents, with 9,093 duplicate paths and 30 divergent same-relative-path groups.
- Decision: branches, snapshots, handoffs, memories, and sessions are provenance. They do not become current authority by being newer or more detailed.
- Not claimed: document recency, branch identity, or Notion text is scientific acceptance.

### 2. A2 remains the stable historical reference; A3/A4 remain probes

- Evidence: May session archaeology identifies compact-full/support2 A2 as the stable reference, while A3/A4 use sampled ground-truth prefix roll-in and are not exposure-matched.
- Decision: retain A2 for historical comparison and require an exposure-matched controlled contrast before treating A3/A4 as replacements.
- Not claimed: superior natural decoding, generalization, or mechanism truth for A3/A4.

### 3. Dense enumeration is an order-sensitive transition, not an established owner ledger

- Evidence: canonical rows can suppress themselves and change later candidate scores; exact same-covered-set order changes next-owner behavior. Static/dynamic crossover contains weak-specificity and technically invalid cells.
- Decision: keep selection, transcription, commit/coverage, and stop as separate capabilities. Require natural, target-specific, source-preserving evidence before training or architecture promotion.
- Not claimed: a cursor, persistent covered-owner memory, object slot, or order-free controller.

### 4. False negatives are heterogeneous

- Evidence: a frozen 12-image panel separates 114 supported native false negatives from 72 persistent owners within 202 eligible owners. More than half of native misses retain local category-conditioned support; STOP is not the leading universal explanation.
- Decision: preserve supported-but-missed, persistent, unknown-neutral, duplicate, unmatched, and invalid cohorts separately. Test owner-specific or apparent-resolution interventions with gained/retained/lost physical-owner accounting.
- Not claimed: every FN is recoverable or that the panel estimates population prevalence.

### 5. Bagging is an object-support probe, not a safe final enumerator

- Evidence: stochastic unions expose additional audited objects but also correlated repeats, fragmented boxes, category disagreement, hallucination, and no safe precision/cardinality gate.
- Decision: use sampling for candidate discovery and mechanism diagnosis only; freeze physical-owner matching and negative controls before using sampled rows as labels or output.
- Not claimed: complete visual capacity or a deployable sampled-union detector.

### 6. Visual designation is a causal probe and teacher, not a product interface

- Evidence: painted/post-scatter visual interventions can change row-level designation. A learned proposal bridge produced duplication, malformed output, precision loss, and closure failure; source-swap and token-permutation controls reproduced the harmful signature.
- Decision: retain visual designation as a privileged falsification/teaching surface and require target-specific source-swap and wrong-object controls for any successor.
- Not claimed: a final cursor renderer, slot, ledger, or architecture promotion.

### 7. Image2299 establishes bounded trajectory sensitivity, not a decoder

- Evidence: near-policy `up-10` expanded strict owners from 11 to 13 without natural-suffix owner loss, while `down-10` fell to 8. Full canonical GT-prefix next-owner success was `0/10`; natural insertion gained only the inserted owner. Same-owner serialization and interpolation changed owner sets at one frozen boundary but failed the zero-debt gate.
- Decision: preserve the dirty Image2299 worktree as candidate evidence and keep the route on HOLD. Any training vertical must retain all natural owners, add an intended owner, improve net unique owners, and avoid new unmatched/duplicate/malformed outcomes.
- Not claimed: an adjacent-owner decoder, neural cursor, generalization, or training authorization.

### 8. Owner-set union does not imply a shared-prefix argmax compiler

- Evidence: the August mathematical audit shows that K sampled trajectories can contain a union of useful owners without admitting mutually compatible strict-argmax targets at one exact prefix. Unknown or unmatched rows cannot automatically be negatives, and owner exchange can masquerade as union gain.
- Decision: any compiler proposal must resolve one-to-one physical identity, conflicting exact-prefix targets, independent negative evidence, and old-greedy-owner preservation.
- Not claimed: that sampled-union reachability is directly compilable into one natural greedy policy.

### 9. Human13 cross-engine trajectory credit was retired before scientific execution

- Evidence: the cross-engine parity gate failed (`22/1573` over-limit tokens; max error `0.16758` nats). The later all-HF surface completed 463 sampling and 463 replay forwards with zero replay parity error but no backward, optimizer step, proposal audit, or post-update result.
- Decision: preserve `24 checked / 5 unchecked`; do not backfill or archive scientifically unexecuted tasks. Any successor needs acquire/replay, one backward/update, private checkpoint, dual-RP audit, rollback, and exact Source reproduction.
- Not claimed: model-quality gain, algorithm failure, or a deployable controller.

### 10. The standalone Human13 owner-credit probe is experiment-local

- Evidence: a fresh HF/AdamW probe has explicit Source-G preservation, atomic artifacts, rollback, and pre/post greedy decode accounting.
- Decision: report `H gain`, `G loss`, and net unique owners; keep it disposable and separate from the retired production runner/OpenSpec lifecycle.
- Not claimed: broad generalization or production promotion.

### 11. OwnerBridge technical failure and research meaning are separate

- Evidence: V1 W8 collective choreography failed before model evidence; repair/recovery receipts only close mechanics. The separate owner-commit-binding successor is proposal/HOLD work. V1 source-preserving behavior remains owned by `permanent-owner-bridge`.
- Decision: require a fresh immutable scientific run after infrastructure repair. Do not merge the V1 HOLD with the V2 proposal or treat green repair tests as recovered evidence.
- Not claimed: OwnerBridge efficacy, training success, or architecture promotion.

### 12. Ordering has bounded positive evidence, not a universal law

- Evidence: July's completed online permutation result separates random, same-image permutation, sorted, and global-shuffle behavior; sorted has the strongest sampled recoverability in the 12-image diagnostic, and global-shuffle beats random on val200 while remaining below same-image permutation. August ordering work includes candidate/uncommitted surfaces.
- Decision: prefer x-then-y for the research route, preserve exact geometry/order/provenance, and require an explicit production migration decision before changing production.
- Not claimed: a causal explanation for the gain, complete-population generalization, or production readiness.

### 13. Several older routes are negative or non-identifying

- Evidence: axis sorting improved materialization but not localization; the tested prefix-denoising objective was inert; the 6k/12k physical-length contrast was confounded by token/count and packing differences.
- Decision: do not scale or cite these routes as localization/length evidence without a new aligned contrast.
- Not claimed: that every sorting, prefix-denoising, or length intervention is ineffective.

### 14. Research validity depends on narrow infrastructure contracts

- Evidence: session findings identify stale FN-rescue merge paths, row-conditioned prefill reconstruction, sidecar/label/normalization ownership, cache materialization bottlenecks, missing runtime authority, and boundary-tail matching/grouping defects.
- Decision: treat these as evidence-validity gates. A technical green receipt does not become a scientific result; affected artifacts require fresh validation or regeneration.
- Not claimed: model-quality change from a code fix or smoke.

### 15. HF remains semantic authority where vLLM/DoRA diverges

- Evidence: dynamic HF/PEFT semantics showed q_proj difference `0.03125`, full-vocabulary difference `1.5`, and a greedy token-13 flip; reviewed vLLM versions rejected `use_dora=true`.
- Decision: keep HF as the current semantic authority and the vLLM route fail-closed until logprob/KL/token/output gates pass.
- Not claimed: vLLM deployment qualification or an efficiency result.

### 16. Memory stores are routing caches, not shared scientific authority

- Evidence: July sessions and current governance separate Codex/Claude writers and formats from formal repository owners.
- Decision: keep memory stores separate, use registry-first navigation, and promote only reviewed evidence into repository/Notion owners.
- Not claimed: memory freshness, conflict-free automatic sync, or launch authority.

## Classifier boundary

The manifests are exhaustive at the source-path level, but semantic classification is not perfect. Initial passes over-promoted generic development because injected AGENTS/assistant text contained research vocabulary. Corrected passes then exposed the inverse risk: encrypted or inherited child tasks sometimes reveal their research purpose only in assistant output. Known examples remain visible in package METHOD files and the integration receipt. Therefore:

- `coverage.tsv` proves source enumeration and preserves package dispositions;
- package syntheses and exact raw sources, not a `high/medium/skip` label alone, own findings;
- no skipped row is used as evidence that a topic does not exist;
- Notion updates are limited to the unique findings above and require owner-side review.
