## Why

Research probes currently share a historically accumulated checkout without a
single clean research baseline or a consistent rule for returning durable
knowledge while retiring disposable experiment code. The root `main` branch is
an idle production-training route, not the active research authority. This
change establishes a reviewable baseline-cutover plan without erasing evidence
or prematurely deleting reproducibility-critical producers.

## What Changes

- Establish `research-probes` as the canonical research baseline after the
  accepted target-binding infra change is integrated and revalidated. The
  historical clean predecessor is `67ad6586bcf3ba5583de5d5d1498b57cfa8c75f5`.
  The sole post-infra cutover input is
  `f337de5d0bd016b79aa012acfc491544e6313333`, captured and revalidated as a
  clean actual target tree with identity fingerprint
  `abe39a84025bc08e0a6249fe5415f5688d6a25e982dad58082bbbfddc028ded8`.
  `research-base-v1` does not exist before the final, user-approved lifecycle
  task: that task creates an annotated tag at that frozen commit and records
  the tag object and peeled commit.
- Declare `coordexp-swift` a wholly separate production-infrastructure line:
  it is neither a research authority nor a prerequisite source for this
  baseline, and this change does not reconcile, merge, or retire it.
- Update the smallest current router/configuration guidance that still names
  repository `main` or `coordexp-swift` as the default research implementation,
  while keeping the separate production lines distinct and preserving the fixed
  directory names
  `.worktrees/research-probes` and `.worktrees/research-probe-infras`.
- Define the probe lifecycle: fork an ephemeral `probe/<ticket>` worktree from
  the tagged research baseline; merge research documents, conclusions, and
  provenance manifests back; promote code only after a real second consumer;
  keep raw large artifacts outside Git; then retire the probe worktree through
  an explicit evidence-preserving process. "Immutable" here means a reserved
  annotated-tag namespace (`research-base-vN` for baselines and
  `probe-final/<ticket>` for probe finals) plus its SHA are recorded in the
  merged manifest; it does not claim that local Git is an off-host immutable
  store.
- Build a per-candidate producer-to-artifact-to-consumer-to-replay ledger for
  active-tree simplification. Only after that ledger is accepted may a code
  path be quarantined or removed from the active baseline. Historical research
  records, OpenSpec archives, and raw-artifact provenance are retained.
- Freeze the disposition of the current Human13 closeout and any remaining
  active changes before tagging the new baseline.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. This is governance, lifecycle, routing, and evidence-retention work; it
does not change a supported runtime/API/schema behavior.

## Impact

- This change explicitly opts out of stable spec deltas via `skip_specs: true`.
- Expected surfaces are OpenSpec routing/configuration, the smallest relevant
  authority documents and research indexes, baseline tags/branch policy, and a
  later evidence ledger for cleanup candidates.
- This planning revision takes no tag/ref action. It does not itself delete
  code, move either fixed worktree directory, merge infra, reconcile
  `coordexp-swift`, archive incomplete changes, install TensorFlow, launch
  model/GPU work, push, or publish results. Task 4.4 is the separately explicit,
  post-approval future tag-creation action; off-host replication, generic-ref
  movement/deletion, tag deletion, Git garbage collection, and artifact
  reclamation remain separately user-gated.
