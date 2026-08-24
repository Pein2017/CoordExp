## Context

See [proposal.md](proposal.md) for motivation. The clean `research-probes`
head is now `67ad6586bcf3ba5583de5d5d1498b57cfa8c75f5`, following explicit
Human13 graph-owner and retirement-closeout commits. The root repository
documentation and OpenSpec context still describe `main` or `coordexp-swift` as
the accepted implementation, while the user has selected `research-probes` as
the canonical research line and has explicitly kept both production lines
separate. `coordexp-swift` is a wholly independent production infrastructure
line, not a shadow research authority or a source that must be reconciled before
this baseline. The target-binding infra change is a prerequisite for the final
baseline tag, not for this planning change.

## Goals / Non-Goals

**Goals:**

- Establish one unambiguous research authority and a clean tagged fork point
  after the infra integration gate.
- Make every probe's source, documentation, code-promotion, artifact, and
  retirement lifecycle explicit.
- Reduce active-tree entropy only through evidence-backed candidate decisions
  that preserve historical reproducibility and research interpretation.

**Non-Goals:**

- Replacing or merging into production `main`.
- Reconciling, merging, renaming, or retiring the independent
  `coordexp-swift` production-infrastructure line.
- Moving, deleting, or recreating `.worktrees/research-probes` or
  `.worktrees/research-probe-infras`; branch names remain operationally free.
- Bulk code deletion, archive of incomplete OpenSpec changes, raw-artifact Git
  migration, TensorFlow installation, model/GPU execution, or automatic probe
  retirement.
- Turning one probe's research semantics into shared infrastructure.

## Decisions

### Canonical research authority is `research-probes`, not root `main`

The baseline change will update only the current routing/configuration surfaces
that direct research work to root `main`, distinguishing production mainline
from the canonical research baseline. The implementation will first identify
the smallest current router set; it will not mass-rewrite historical documents
or archived changes. `research-probes` remains the canonical branch name for
now, while branch naming is otherwise flexible.

The alternative of keeping root `main` as a nominal authority while telling
researchers to use a worktree is rejected because future agents and proposals
would follow the router before conversational intent. A second alternative of
making `research-probe-infras` a parallel permanent authority is rejected: it
is an integration lane that must merge accepted mechanics into the research
baseline.

### `coordexp-swift` remains independent production infrastructure

`coordexp-swift` does not become a second research base, an upstream merge
prerequisite, or a cleanup target. Router guidance will name it as an
independent production-infrastructure line and state that new research probes
start only from the newest immutable `research-base-vN` tag. A future request
to adopt behavior from `coordexp-swift` must be an explicit compatibility
change with its own evidence and approval; this cutover does not silently
choose or discard any of its commits.

### Baseline promotion has two explicit gates and a later tag action

`67ad658` is the clean predecessor for the infra successor. The final
`research-base-v1` tag does not exist before its dedicated lifecycle task. That
task runs only after (1) target-binding infra is merged and revalidated in the
actual `research-probes` tree, (2) this change's authority/retention gate is
accepted, and (3) the user explicitly approves tag creation. It creates an
annotated tag at the exact post-infra commit frozen by task 1.1 and records the
tag object and peeled commit in the implemented lifecycle record. The pre-infra
image-2299 probe remains bound to its declared `9f902d5ab` source and is not
silently rebased.

This avoids claiming that an unverified infra branch or a dirty working tree is
the new baseline. It also preserves the exact base of concurrent research.

After an accepted reusable-code or compatible-infrastructure merge into
`research-probes`, cut the next immutable `research-base-vN` tag and require
new probes to fork from the newest such tag. Documentation-only research-record
returns do not themselves require a new baseline tag.

Here, "immutable" is an operational retention contract: a reserved
annotated-tag namespace (`research-base-vN` for baselines and
`probe-final/<ticket>` for probe finals) and its target SHA must be recorded in
the merged provenance manifest. Local Git alone is not an off-host immutable
store, and this change claims no off-host replication. Generic-ref
movement/deletion, tag deletion, Git garbage collection, and raw-artifact
reclamation remain separate, explicitly user-approved lifecycle actions.

### Probe lifecycle is asymmetric by artifact type

Each new `probe/<ticket>` worktree starts from the newest immutable
`research-base-vN` tag (`research-base-v1` at the initial cutover). Its durable
research unit, result/review, conclusion, config/provenance manifest, and
external-artifact locator return to `research-probes` whether the result is
positive or negative. Its source code returns only when a second real consumer
needs the same behavior and the shared owner can retain caller semantics.
Raw model outputs, caches, checkpoints, and large artifacts remain external to
Git and are represented only by bound locators/checksums/identities.

Before an ephemeral probe is retired, reserve its final annotated-tag namespace
`probe-final/<ticket>`, record that tag name and its final-HEAD SHA in its
merged provenance manifest, and retain the corresponding replay entry.
Retirement, generic-ref deletion or movement, tag deletion, Git garbage
collection, and raw-artifact reclamation are separate decisions; neither is
implied by baseline-tag approval.

The pre-infra image-2299 provenance manifest is created and verified only in
the later implementation: it must record its exact worktree path, currently
resolved branch/ref, source commit `9f902d5ab`, clean-status evidence, and
replay entry, while explicitly stating that it is not evidence for the new
baseline.

### Fixed paths and reachability are captured, not inferred from branch names

The fixed directories are stable paths, not permanent branch-name contracts.
At the acceptance gate, capture each exact absolute fixed path and its
currently resolved ref and commit. Current observed evidence is
`/data/CoordExp/.worktrees/research-probes` ->
`refs/heads/research-probes` @ `67ad658...`, and
`/data/CoordExp/.worktrees/research-probe-infras` ->
`refs/heads/codex/research-probe-infra-foundation` @ `67ad658...`.

The gate must also prove every active worktree HEAD and every probe/final
lifecycle tip resolves from at least one named ref. The detached
`/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation` at
`477b376a3e31a5dbedf5a87ecafcb372e75a73a9` is an explicit HOLD: this change
does not create a ref for it or prune/remove it, so the acceptance gate remains
HOLD unless separately resolved. The stale, unmounted
`research-probe-infras` branch at `62274a97...` is a content-equivalent,
superseded HOLD in the disposition ledger, not a deletion target.

### Entropy reduction uses a provenance ledger, not static reachability

Before a candidate script is removed from the active tree, the change records
its source identity, producer outputs, artifact schema/hash, known configs and
downstream consumers, historical claim owners, and a small replay or
`--help`/fixture discriminator. The record also pins the pre-removal commit,
path, and replay invocation. A candidate that lacks this evidence remains held.
Accepted removals preserve reproducibility through prior Git history and
research/receipt references, not a duplicate long-lived `legacy/` source tree.

### Existing Human13 changes receive an explicit disposition table

The cutover records every active/complete Human13 OpenSpec and relevant closeout
path as retained, retired-but-open, archive-eligible, superseded, or still
active. It never checks incomplete research tasks merely to obtain a clean
dashboard. The table is frozen against the post-infra target so it cannot
silently omit integration changes.

## Risks / Trade-offs

- [Router drift continues to send work to root `main`] → Treat exact
  configuration/router updates as acceptance work before tagging.
- [Cleanup removes a manual historical producer] → Require the ledger and a
  per-candidate replay discriminator; lack of evidence means HOLD.
- [The infra merge changes the candidate inventory] → Freeze the ledger and
  baseline tasks only after the merged target is revalidated.
- [A probe needs code retained before a second consumer exists] → Keep it in
  the ephemeral worktree or return it as explicitly experiment-local code;
  promotion remains a separate review decision.
- [External artifact roots disappear] → Merge locators and identities with the
  documents; do not treat a Git commit as a substitute for raw evidence.

## Migration Plan

1. Keep the two fixed worktree directories in place and use the clean
   `research-probes` predecessor as the infra successor base.
2. Complete and independently review the target-binding infra change. Its own
   separately authorized integration gate alone may merge accepted mechanics;
   this change only consumes the resulting `research-probes` identity. If that
   revalidation fails, stop this cutover with no tag until a fresh accepted
   infra result exists.
3. Identify and update the smallest current authority/router set, then freeze
   the Human13 disposition table and candidate provenance ledger on that merged
   target.
4. Apply only individually accepted quarantine/removal decisions, preserving
   research records and external-artifact locators.
5. Verify clean status, router consistency, ledger receipts, live
   external-artifact locators/checksums, fixed-path-to-current-ref/commit
   capture, and named-ref reachability for every active worktree HEAD and
   probe/final lifecycle tip. The detached cache-validation worktree remains a
   HOLD, so do not request tag approval unless it is separately resolved. Then
   request user approval to create the `research-base-v1` tag.
