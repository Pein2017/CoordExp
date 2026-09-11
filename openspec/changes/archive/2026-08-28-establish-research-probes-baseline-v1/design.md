## Context

See [proposal.md](proposal.md) for motivation. The clean historical
`research-probes` predecessor is
`67ad6586bcf3ba5583de5d5d1498b57cfa8c75f5`, following explicit Human13
graph-owner and retirement-closeout commits. The accepted post-infra
revalidation anchor is exclusively `f337de5d0bd016b79aa012acfc491544e6313333`:
on 2026-08-24 its actual fixed target tree was captured and revalidated clean by
the target-binding contract, with identity fingerprint
`abe39a84025bc08e0a6249fe5415f5688d6a25e982dad58082bbbfddc028ded8`.
Later planning records do not change that frozen source anchor; a later clean
candidate that descends from it is the tag target. The root
repository documentation and OpenSpec context still describe `main` or
`coordexp-infras` as the accepted implementation, while the user has selected
`research-probes` as the canonical research line and has explicitly kept both
production lines separate. `coordexp-infras` is a wholly independent production
infrastructure line, not a shadow research authority or a source that must be
reconciled before this baseline. The target-binding infra change is a
prerequisite for the final baseline tag, not for this planning change.

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
  `coordexp-infras` production-infrastructure line.
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

### `coordexp-infras` remains independent production infrastructure

`coordexp-infras` does not become a second research base, an upstream merge
prerequisite, or a cleanup target. Router guidance will name it as an
independent production-infrastructure line and state that new research probes
start only from the newest immutable `research-base-vN` tag. A future request
to adopt behavior from `coordexp-infras` must be an explicit compatibility
change with its own evidence and approval; this cutover does not silently
choose or discard any of its commits.

The `coordexp-infras-*` prefix in stable OpenSpec capability names identifies
the retained production codebase contracts; it does not make the
`coordexp-infras` branch a research entrypoint.

### Baseline promotion has two explicit gates and a later tag action

`67ad658` is the historical clean predecessor for the infra successor.
`f337de5d0bd016b79aa012acfc491544e6313333` is the only revalidated
post-infra anchor frozen by task 1.1, not the final baseline-tag target: it
predates the route, lifecycle, disposition, and entropy records this change
must deliver. After those records are complete, task 4.2 freezes a clean final
candidate that descends from the anchor and obtains an exact-candidate review.
The final `research-base-v1` tag does not exist before task 4.3's separate user
approval. Task 4.4 creates an annotated tag at that reviewed candidate and
records its tag object and peeled commit in a post-tag lifecycle receipt; that
receipt is necessarily a later commit and not an excuse to move the tag.

This avoids claiming that an unverified infra branch or a dirty working tree is
the new baseline. It also preserves the exact base of concurrent research.

After an accepted reusable-code or compatible-infrastructure merge into
`research-probes`, cut the next immutable `research-base-vN` tag and require
new probes to fork from the newest such tag. Documentation-only research-record
returns do not themselves require a new baseline tag.

Here, "immutable" is an operational retention contract: a reserved
annotated-tag namespace (`research-base-vN` for baselines and
`probe-final/<ticket>` for probe finals) and its target SHA must be recorded in
the merged provenance manifest. The current user-approved durability boundary
is local Git plus native worktree locks for the two fixed directories. Local Git
alone is not an off-host immutable store; this change claims no off-host
replication, and no remote publication occurs. Generic-ref movement/deletion,
tag deletion, Git garbage collection, and raw-artifact reclamation remain
separate, explicitly user-approved lifecycle actions.

The target-binding GPU mechanics smoke remains unexecuted and is explicitly
non-blocking for this governance tag: CPU-only fail-closed capture and replay
are the evidence boundary here. A later GPU authorization may run that smoke,
but neither this tag nor its records may claim GPU execution from CPU receipts.

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

### Fixed paths and reachability are captured, not inferred from names

The fixed directories are stable paths, not permanent branch-name contracts.
At the acceptance gate, capture each exact absolute fixed path, resolved Git
admin directory, currently resolved named ref, and gate-time commit. The admin
directory name is implementation-owned and may not resemble the worktree
basename, so path-only or basename-only protection is insufficient. The two
fixed worktrees are also protected by native `git worktree lock` records; their
reasons are part of the gate evidence, and unlocking them requires a separate
lifecycle decision. Do not make a self-invalidating evergreen claim about the
current head of a moving branch.

The gate must also prove every baseline-owned fixed-worktree HEAD and every
probe/final lifecycle tip recorded by this baseline resolves from at least one
named ref. Independent concurrent worktrees are out of scope and are not
consulted. The detached
`/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation` at
`477b376a3e31a5dbedf5a87ecafcb372e75a73a9` is an explicit HOLD: this change
does not create a ref for it or prune/remove it. Its registered worktree HEAD
is currently a Git root, so the unrelated HOLD does not block baseline-tag
review; the future risk is separate worktree removal/prune followed by reflog
expiry. The stale, unmounted
`research-probe-infras` branch at `62274a97...` is a superseded HOLD in the
disposition ledger, not content-equivalent and not a deletion target.

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

| Item | Disposition | Reason and boundary |
| --- | --- | --- |
| `add-human13-k-union-greedy-overfit-probe` | archive-eligible | 34/34 tasks are complete; retain its evidence unless a separate archive action is approved. |
| `add-human13-on-policy-first-bottleneck-successor` | archive-eligible | 19/19 tasks are complete; it remains a bounded negative/mechanics result. |
| `add-human13-row-contrast-geometry-preservation-successor` | archive-eligible | 14/14 tasks are complete; retain its bounded benchmark record. |
| `add-human13-k-trajectory-rp-crossover-screen` | retired-but-open / superseded | 28/37 tasks; later work does not complete its unchecked matrix path. |
| `add-human13-all-hf-shared-surface-trajectory-credit-vertical` | retired-but-open / superseded | 24/29 tasks; no private update or model-quality result can be inferred. |
| `harden-research-probe-target-binding` | retain active / GPU HOLD | CPU contract integration is complete; task 4.2 is a separately authorized, unexecuted GPU mechanics smoke. |
| `establish-research-probes-baseline-v1` | retain active | This change remains the current cutover owner until its final tag gate. |
| `coordexp-infras` | retain independent production infrastructure | It is neither research authority nor a retirement candidate. |
| `research-probe-infras` @ `62274a97...` | superseded HOLD | Its relevant agent-contract sync is upstream, but it differs materially from the target and has no deletion authorization. |

### Current entropy ledger has no admitted deletion

The candidate ledger intentionally produces no removal in this change:

| Candidate | Disposition | Minimum preservation condition before any later action |
| --- | --- | --- |
| `docs/history/{worktree-cleanup,research-intake,worktree-union}/**/snapshots/**` | quarantine candidate, high risk | Byte-exact manifest, immutable archive, manifest rebinding, and replay proof; current snapshot/path count is insufficient. |
| `reference/legacy_src/**` | keep | It is an intentional reference-only recovery quarantine; an approved replacement recovery source and import/path audit would be required. |
| `scripts/research/analyze_native_sibling_branch_value.py` | HOLD | Retain a replacement capable of replaying every documented result artifact. |
| `scripts/research/run_static_dynamic_owner_interface_experiment.py` | keep | Dynamic and direct successor loading remains active. |
| target-binding admission core and consumer adapters | keep | Two consumer-owned, fail-closed source/receipt identities are load-bearing. |
| fixed `research-probe-infras` integration lane | HOLD | Its same-looking support seam is deliberate two-consumer integration, not a shadow surface. |
| superseded external output/log roots | HOLD | Every locator needs producer, claim-owner, hash, and replay preservation before archival or reclamation. |

## Risks / Trade-offs

- [Router drift continues to send work to root `main`] → Treat exact
  configuration/router updates as acceptance work before tagging.
- [Cleanup removes a manual historical producer] → Require the ledger and a
  per-candidate replay discriminator; lack of evidence means HOLD.
- [The infra merge changes the candidate inventory] → Freeze the ledger and
  baseline tasks only after the merged target is revalidated.
- [A tag points at an unimplemented plan] → Keep the 1.1 infra anchor separate
  from the 4.2 final candidate and tag only the latter after exact review.
- [Local-only retention is mistaken for off-host durability] → Record the
  boundary and two worktree locks; require a separate approval for publication.
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
   the Human13 disposition table and candidate provenance ledger on the
   revalidated lineage.
4. Make no removal in this change: separately scope any archive, quarantine, or
   reclamation candidate only after its preservation/replay proof is accepted.
5. Verify clean status, router consistency, ledger receipts, live
   external-artifact locators/checksums, fixed-path-to-admin-dir-to-current-ref
   capture, native lock state, and named-ref reachability for each baseline-owned
   fixed-worktree HEAD and probe/final lifecycle tip recorded by this baseline.
   Do not consult independent concurrent worktrees. Preserve the cache-validation
   checkout as a separate HOLD without treating its current registered HEAD as
   a baseline-tag blocker. Freeze and exact-review the resulting candidate,
   then request user approval to create `research-base-v1` at that candidate.
