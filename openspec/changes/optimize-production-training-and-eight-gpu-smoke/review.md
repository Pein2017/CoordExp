# Independent proposal review — 2026-09-12

Status: lead-accepted for implementation after the corrections below. This is
proposal acceptance only; runtime and implementation tasks remain open.

The user explicitly authorized implementation after proposal and independent
review, with no further permission gate. Two fresh-context Astra high workers
performed one read-only review round. Neither authored the plan or ran GPUs.

## Frozen reviewed target

| Artifact | SHA-256 before review corrections |
| --- | --- |
| proposal.md | 25590518b663a378ee7b7e17f0deb93b733556090c6307c9f709c80a70fa71e6 |
| design.md | f7db4d43d55470182914b80ee365e7923e7b8c103773f56e24c16d1fcb343140 |
| specs/infra-base/spec.md | f8df88f9477c1bbaf2d5259af5a3605ba1049a1fa1d7a546859c92f824e7749a |
| tasks.md | c69ee5debeed077bc673442c5f3ab192d281f56033ecfc46eff05b20e921eebd |

## Findings and dispositions

1. **Semantic/resource reviewer — `astra_high_review_semantics`:** the draft
   confused planner `packing.worker_count=1` with materialization workers.
   Current `src/training/pack_cache.py:45` defaults materialization to 16;
   `cache_workflow.py:1875` resolves that separate argument. Changing the YAML
   planner count does not choose four preprocessing processes and changes the
   cache identity (`tests/training/test_pack_cache.py:341`). **Accepted and
   corrected:** preserve planner count 1; compare the real entry's unchanged
   16 materialization workers and record metadata, with candidate window 32.
   Controlled tests can still select an internal worker count. No new selector.
2. **Lifecycle reviewer — `astra_high_review_lifecycle`:** inherited production
   disables exact resume, so merely adding step-2 checkpoint cadence does not
   publish training state (`src/training/session.py:2093`). **Accepted and
   corrected:** both uninterrupted parents explicitly enable
   `exact_same_world_size` with null checkpoint; fresh continuation names the
   candidate step-2 checkpoint, keeping the four-update schedule and strict
   CUDA replay. Require the actual training-state publication before resume.

Lead rechecked the current source owners and corrected the original
counterexamples in design/tasks. No topology, estimand, compatibility scope,
or acceptance invariant changed. No second delegated review round is needed.
Strict OpenSpec validation is rerun after correction. These reviews do not
establish any runtime success, numeric parity, speedup, or vLLM admission.

## User-accepted numerical contract and bounded review

On 2026-09-12 the user resolved the remaining composition decision:
“差一格无所谓,这个误差可以接受.” They also requested completion of all tasks
and archive. This authorizes the explicit one-grid composition contract in
design section 5, not arbitrary text, geometry or probability drift.

Because this changes the numerical foundation, `astra_high_review_semantics`
performed one separate read-only review of that revised boundary. Result:
**no blocking finding; lead-accepted for implementation**. The review checked
length/EOS/non-coordinate equality, the coordinate bound, forged summaries,
mapping authentication and the distinction between merged-model likelihoods
and dynamic-HF probability parity. It confirmed that canonical coordinate IDs
are ordered `coord_0` through `coord_999`; the existing count/min/max artifact
summary cannot authenticate the full mapping. The implementation must use the
actual validated tokenizer mapping and reject booleans masquerading as IDs.

The existing producer's exact-greedy/allclose gates must be changed consistently
with the new explicit policy while keeping those diagnostic results truthful.
Runtime/concurrency qualification remains operational evidence; forced replay
remains exact within vLLM. The reviewer did not write source, run GPUs or claim
implementation acceptance. Lead-owned RED/GREEN and real current-checkpoint
qualification close those gates.
