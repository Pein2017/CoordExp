## ADDED Requirements

### Requirement: Target research tree is revalidated before model launch

An admission that can reach a model-launch boundary SHALL bind one declared
target research tree in addition to its existing immutable input bindings. The
target binding MUST contain the absolute resolved worktree root, the full Git
commit identity observed at capture, a clean-status assertion, and an ordered
set of named effective source, runtime, configuration, producer, and validator
inputs. Every effective input MUST be bound by the existing typed immutable
binding rules and MUST resolve to the exact input the consumer will use. The
target-tree identity is admission-owned outer metadata, not a new
`BindingManifest` binding kind; only its declared effective inputs enter the
existing closed typed-binding enumeration.

Capture MUST reject a target that is not a Git worktree, is unresolved or a
symlink, has an unresolved merge state, or has staged, modified, deleted,
renamed, or untracked content. A consumer SHALL revalidate the complete target
binding immediately before it invokes a launcher, loads a model, allocates a
GPU, or creates a vertical-stage artifact. Target revalidation MUST reject a
changed commit, dirty status, changed effective input, or changed resolution
with a typed admission failure.

On such a failure, a previously durable CPU preflight record MUST remain
immutable, but the consumer MUST NOT invoke its launcher, load a model,
allocate a GPU, publish vertical evidence, or finalize the admission. The
failure is mechanics-only and MUST NOT reinterpret consumer-owned scientific
fields.

#### Scenario: Target changes after CPU preflight
- **WHEN** a target source, runtime, configuration, producer, validator, Git
  commit, or clean status changes after a CPU preflight was accepted and before
  the vertical consumer reaches its model-launch boundary
- **THEN** target revalidation fails before any launcher or model action, the
  accepted CPU record remains unchanged, no vertical-stage evidence exists,
  and a new admission identity is required for continuation

#### Scenario: Dirty target is proposed for admission
- **WHEN** a consumer captures or revalidates a target worktree containing any
  staged, modified, deleted, renamed, conflicted, or untracked path
- **THEN** admission fails with a typed target-binding error before an admission
  root or model-launch action is created

#### Scenario: Exact clean target reaches a model boundary
- **WHEN** a clean target worktree, its full commit identity, and every declared
  effective input still match the captured target binding immediately before
  launch
- **THEN** the consumer may proceed to its existing launcher and later
  mechanics-only stage evidence without changing its scientific plan or output
  schema

#### Scenario: Shared baseline has planning dirt
- **WHEN** an otherwise valid shared baseline checkout contains planning or
  documentation dirt and is proposed as the target of a reusable admission
- **THEN** the admission fails rather than allowlisting that dirt; a clean
  dedicated probe worktree is required before a model-launch action

## MODIFIED Requirements

### Requirement: Two real consumers cross the same admission seam

Acceptance SHALL exercise the shared capability through both the current
natural-boundary support path and the current K10-H20 crossover path. Each
consumer adapter MUST reuse the same typed binding and CPU-stage interface
while retaining its own plan, projection, finalizer, validator, scientific
vocabulary, and output schema. The selected bounded natural-boundary support
adapter MUST additionally cross the vertical stage and final admission
interface.

The acceptance gate SHALL include deterministic CPU tests for every binding
kind and failure class, a production-path CPU round trip for both consumers,
and one bounded single-GPU smoke through an existing production worker. The GPU
smoke MUST write a fresh root, bind the exact model/config/runtime/device and
consumer identities, publish at least one durable work item, reach the current
bounded terminal/finalizer and its mechanics validator, and remain
mechanics-only. It MUST NOT claim that the natural-boundary legacy merger was
validated unless that merger's complete declared shard and context denominator
was actually supplied.

During admission capture, revalidation, and execution, the active target
research worktree and every sealed historical root MUST remain immutable from
the consumer and admission implementation. Consumer examples MAY read and
digest-bind that worktree but MUST NOT edit, stage, commit, merge, cherry-pick,
push, or rewrite it. This does not forbid a separately user-authorized,
reviewed integration of an accepted branch developed in the fixed
`/data/CoordExp/.worktrees/research-probe-infras` worktree into
`research-probes`: that Git operation occurs outside admission execution,
preserves both fixed directories, and requires a fresh target binding and
revalidation before a later admission may run.

#### Scenario: Both consumers pass through one interface
- **WHEN** support and crossover adapters submit their distinct CPU evidence
  and the selected support adapter also submits bounded vertical evidence
  through the shared admission capability
- **THEN** both produce contract-valid mechanics stage receipts and the support
  path produces a final admission receipt without adding either consumer's
  scientific schema to stable infrastructure

#### Scenario: Consumer semantics differ
- **WHEN** the two consumers use different plans, projections, endpoints, and
  outcome fields
- **THEN** those differences remain visible and caller-owned while their shared
  path, identity, durability, and evidence-closure mechanics are validated by
  the same admission owner
