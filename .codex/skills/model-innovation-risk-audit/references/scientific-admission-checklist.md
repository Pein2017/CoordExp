# Scientific Admission Checklist

Use this before implementing a decision-grade objective or evidence pipeline
whose scientific meaning depends on derived labels, replay, matching, packing,
or several joined artifacts. It is a **failure-mode checklist**, not a security
threat model: cover reachable ways the experiment can silently measure or train
the wrong thing.

## Freeze One Matrix

For each row, record the invariant, executable owner, cheapest counterexample,
closing evidence, and whether it is CPU-contract or live-vertical evidence.
Select only relevant rows; a short section in the existing unit is sufficient.
The categories below are examples, not a requirement to cover every surface.

| Surface | Invariant | Minimal counterexample |
|---|---|---|
| semantic source | derived category, geometry, spans, and owner outcomes come from sealed raw evidence through the canonical parser/matcher | unchanged tokens with substituted bbox/category; two rows competing for one owner |
| policy state | prompt, complete generated history, processor order, RP/temperature/top-p/top-k, stop, and cap semantics match the sampled policy | changed history or processor field with an otherwise valid receipt |
| grouping | image, RP, seed, request order, logical denominator, and terminated-path padding match the estimand | cross-image/RP/seed substitution; trajectory mean used instead of group mean |
| logits and packing | full vocabulary, causal position, prompt/prefix bytes, segment start, and logical-to-physical mapping are exact | truncated vocab hides the true argmax; same-length prefix substitution |
| artifact closure | the published bundle contains all required preimages, is atomic, reloads completely, and revalidates cross-artifact relations | swapped output; missing replay/parity file; mixed but individually valid artifacts |
| public admission | every supported constructor, factory, `from_dict`, loader, and publisher enforces one shared invariant | direct construction or deserialization bypasses the loader-only check |
| runtime claim | mocks do not stand in for installed sampler, tokenizer, model forward, optimizer, or evaluator semantics | fake receipt passes while the live owner cannot emit the required evidence |

Do not enumerate every possible exploit. One counterexample per distinct
invariant class is enough before implementation; add another only when it
exposes a genuinely different scientific failure.

## Prefer One Admission Choke Point

- Re-derive semantic labels from the lowest sealed evidence that owns them.
- Snapshot caller-owned collections before hashing or validation.
- Make all supported construction and reload paths call the same aggregate
  validator.
- Bind cross-artifact relationships, not only each artifact's local schema.
- Re-admit evidence after a process boundary; do not trust a transported label
  merely because it is immutable or content-addressed.
- Keep formula-only fixtures explicitly separate from scientific loss inputs.

These are scientific-evidence rules, not a requirement to defend against an
arbitrary hostile Python process. Object-copy, monkey-patch, proxy, or pickle
hardening is decision-bearing only when the declared runtime can exercise that
path, the object crosses a durable boundary, or the mutation could alter the
primary observation without detection.

## Review Budget And Stop Rule

1. Before implementation, freeze contract/input identities and relevant risks,
   not the hash of an implementation that does not yet exist.
2. Let the implementer check the applicable invariants through the real
   caller/consumer; retain the exact implementation version and evidence.
3. Use independent review only for a named unresolved decision-changing risk
   that it can close more effectively than a local check, or when the governing
   contract requires it. Freeze the implemented target hash before that review.
   Share the same phase/target/risk budget with other skills; do not reopen
   already closed findings under another audit name.
4. Disposition accepted findings as one correction bundle, grouped by invariant
   class, then recheck only those findings and materially changed evidence.
5. If the recheck finds another bypass in the same class, repair the shared
   admission choke point and its counterexample family rather than starting an
   exploit-by-exploit loop.
6. If no new conclusion-changing class remains, stop CPU hardening and move to
   the smallest production-shaped vertical. Record live-only risks there.

Optional reassurance tests, arbitrary-object tamper resistance, and broad
suite reruns stay off the critical path without a demonstrated impact on the
selected condition, gradient, artifact attribution, or decision-owning result.
Do not manufacture P2 findings merely to fill a report.
