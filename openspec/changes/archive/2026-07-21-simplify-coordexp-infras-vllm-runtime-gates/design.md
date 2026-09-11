## Context

The current vLLM path has two different kinds of evidence entangled in one
launch gate:

1. live correctness evidence about the requested model and current decode;
2. historical qualification evidence about a prior checkout, probe, engine
   argument tuple, and HF/vLLM comparison.

The first class protects model meaning. The second class is useful for audits
and precision studies, but it becomes stale after harmless source or resource
changes. The redesigned path keeps the existing backend-owned session and
content-addressed execution-model architecture while changing which evidence
is authoritative for ordinary execution.

## Goals / Non-Goals

### Goals

- Correctly assemble and execute base model A, DoRA adapter B, and selected-
  token embedding delta C.
- Fail before or during startup on missing payloads, incompatible tensors,
  invalid tied-weight structure, corrupt snapshots, engine-load failures, or
  unusable native decode evidence.
- Preserve policy and raw-model log-likelihood trace semantics.
- Keep enough provenance to reproduce and diagnose a result without requiring
  source-code hash equality.
- Make cross-worktree input ownership explicit and deterministic.

### Non-Goals

- Exact HF/vLLM logit or token equality as a normal runtime requirement.
- Refreshing qualification receipts after every source edit.
- Searching sibling worktrees or output directories to repair wrong paths.
- Supporting arbitrary vLLM engine kwargs, server mode, tensor parallelism, or
  stochastic canonical inference.
- Removing the explicit qualification probes used for optional audits.

## Decisions

### Decision: Live operational preflight owns launch authorization

The vLLM backend will record a compact preflight containing the installed vLLM
version, effective engine arguments, execution-model receipt identity, and
whether optional historical evidence is present or stale. The preflight will
not compare application-source hashes or exact historical parameter values as
blocking checks.

Actual engine construction is the compatibility check for the installed vLLM
version. Version `0.14.1` remains the known-working version in documentation;
other versions are recorded as unverified and may proceed until a real API,
model-load, or decode contract fails.

This permissive rule applies to policy inference. Raw-model likelihood depends
on vLLM's ordering of raw logprob capture relative to the forced logits
processor, which ordinary token alignment cannot prove. An unverified version
therefore fails raw tracing unless a version-specific ordering probe is added;
it may still execute policy-only inference.

### Decision: Execution-model receipts remain strict and self-contained

The materializer continues to inspect and hash the current base, adapter, and
delta payloads, merge DoRA safely, fold the delta exactly once, preserve tied
input/output weights, validate target dtype and tensor shapes, reject adapter
residue, publish atomically, and revalidate every cache hit. These checks are
about the model that will execute and remain blocking.

Automatic binding to `coordexp_composition_fidelity.json` or a durable
composition-comparison receipt is removed from normal resolution. Explicit
composition probes may still bind and validate such a receipt when a research
question needs it.

The blocking receipt itself validates the merge/fold evidence before atomic
publication: adapter evidence is required exactly when an adapter identity is
configured, delta evidence is required exactly when a delta identity is
configured, each outcome binds the corresponding source identity, the delta
records one row addition, and tied input/output storage remains explicit.

### Decision: Live replay alignment owns raw likelihood trust

When raw likelihood is enabled, vLLM still closes the policy engine, opens a
fresh raw-logprob engine, forces the authoritative token sequence, and requires
exact native request ids, prompt ids, generated ids, lengths, stop semantics,
finite non-positive values, and per-token alignment. A historical forced-replay receipt may be
reported as optional provenance but cannot block the live replay.

### Decision: Declaring configs own path resolution

Relative paths continue to resolve against the YAML file that authored the
leaf value. The resolved path is always absolute and is the only path passed to
runtime owners. Before materialization or frontend loading, missing inputs fail
with the field name, declared path, declaring config, and resolved path.
Runtime never searches another worktree, output root, checkpoint alias, or
similarly named directory.

Validation happens before CUDA discovery or JSONL loading and verifies the
expected file/directory kind, so an owned-path error retains declaring-config
provenance instead of being replaced by a lower-level loader error.

This preserves portable repository configs while making copied cross-worktree
configs fail at the configuration boundary with enough information to replace
the value with the intended absolute artifact path.

### Decision: First live decode is the operational smoke

No extra model generation is inserted before every run. The first normal
rank-local decode exercises the real image, prompt projection, engine, policy
trace, and output normalization path. Existing fail-closed backend contracts
prevent final benchmark-looking artifacts from publishing when that decode is
empty, prompt-misaligned, image-misaligned, non-finite, post-stop, or otherwise
unusable. Dedicated fixed-fixture probes remain available before expensive
runs but are not runtime authorization receipts.

Each rank keeps its process, CUDA, live-decode, and cleanup observations separate from the
cross-rank semantic session identity. Strict merge compares only semantic
preflight settings and then aggregates one completed live/cleanup observation
per rank, preserving both data-parallel compatibility and auditability.

## Risks / Trade-offs

- A newly installed vLLM version may start and later expose an untested edge.
  The live backend contracts still fail on malformed evidence, and provenance
  records the exact version for diagnosis.
- Removing historical source binding weakens byte-for-byte reproducibility of
  the application implementation. Resolved config, execution-model identity,
  package versions, effective settings, and normal Git provenance remain; a
  dedicated audit can still run strict qualification probes.
- A malformed first model response can be a model-quality event rather than an
  infrastructure defect. The runtime blocks only on backend-neutral evidence
  invariants, not on object count or evaluator quality.
- Relative paths are still portable but can point at the wrong checkout. Early
  absolute-path receipts and no-search behavior make the ownership visible;
  canonical cross-worktree launches should author the shared artifact path
  explicitly.

## Migration Plan

1. Add regression tests that demonstrate source-hash, exact engine-value,
   concurrency-receipt, forced-replay-receipt, and composition-fidelity absence
   no longer block normal sessions.
2. Introduce the compact non-throwing vLLM preflight/provenance collector and
   route normal session creation through it.
3. Stop automatic composition-fidelity binding and remove the frontend launch
   requirement while retaining explicit composition probe APIs.
4. Add early resolved-input checks with path-origin diagnostics.
5. Update stable specs and operator docs, then run targeted tests and one real
   BF16 composed-model multimodal smoke.

## Open Questions

None for this slice. A later change may add an explicit portable artifact
manifest if repeated cross-machine path authoring proves costly; this change
does not introduce that extra surface preemptively.
