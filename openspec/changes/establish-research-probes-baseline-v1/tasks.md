## 1. Freeze the post-infra baseline input

- [x] 1.1 After the infra change's own approved integration gate has accepted,
  merged, and revalidated `harden-research-probe-target-binding` in
  `research-probes`, record the exact resulting commit as the only infra
  revalidation anchor; do not treat the historical infra branch or a dirty tree
  as baseline evidence. It is not the eventual baseline-tag target: a later
  reviewed final candidate must descend from it. If revalidation fails, stop
  the cutover with no tag until a fresh accepted infra result exists. Recorded
  on 2026-08-24: clean actual target
  `f337de5d0bd016b79aa012acfc491544e6313333`, identity fingerprint
  `abe39a84025bc08e0a6249fe5415f5688d6a25e982dad58082bbbfddc028ded8`.
- [x] 1.2 Inventory the smallest current router/configuration documents and
  OpenSpec context that still direct research work to root `main` or
  `coordexp-swift`; distinguish current authority, the independent production
  infrastructure line, and historical documentation before proposing edits.
  The current router set is `openspec/config.yaml`,
  `docs/PROJECT_CONTEXT.md`, `docs/BRANCH_AND_WORKTREE_POLICY.md`,
  `docs/README.md`, `docs/AGENT_INDEX.md`, and `docs/COORDEXP_SWIFT.md`;
  `docs/IMPLEMENTATION_MAP.md` is a retained production source map, not a
  general research router.
- [ ] 1.3 Freeze a disposition table for every active/complete Human13 change
  and relevant closeout path against that post-infra target: retain active,
  retired-but-open, archive-eligible, superseded, or otherwise held. Do not
  mark an incomplete task complete to simplify the table. Record
  `coordexp-swift` as independently retained production infrastructure, not as
  a research-probe authority or a retirement candidate.
- [ ] 1.4 Record the stale, unmounted `research-probe-infras` branch at
  `62274a97...` as superseded HOLD in that disposition ledger; it is not
  content-equivalent to the current target and is not a deletion target.

## 2. Establish research authority and lifecycle guidance

- [ ] 2.1 Update only the accepted current routing/configuration surfaces so
  `research-probes` is the canonical research baseline, root `main` remains a
  distinct production route, and `coordexp-swift` remains independent
  production infrastructure rather than a research entrypoint; preserve both
  fixed worktree directory names.
- [ ] 2.2 Add concise lifecycle guidance for `probe/<ticket>` worktrees:
  tagged-base creation, source identity, document/result/manifest return,
  second-consumer code promotion, external raw artifacts, and evidence-safe
  retirement. Before retirement, reserve the final annotated-tag namespace
  `probe-final/<ticket>` and record that tag name, final-HEAD SHA, source path,
  and replay entry in the merged manifest; generic-ref movement/deletion and
  tag mutation remain separately user-gated.
- [ ] 2.3 In the later merged research provenance manifest, record and verify
  the pre-infra image-2299 worktree's exact path, currently resolved branch/ref,
  source commit `9f902d5ab`, clean-status evidence, and replay entry. State
  explicitly that it is not evidence for the new baseline; do not create that
  manifest during planning.

## 3. Build the entropy-reduction decision ledger

- [ ] 3.1 For each candidate active-tree script or support surface, record its
  exact pre-removal commit and path, produced artifacts/receipt schema, known
  config and downstream consumers, claim owner, and a small replay, fixture,
  or `--help` discriminator.
- [ ] 3.2 Classify each candidate as keep, hold, quarantine candidate, or
  separately removable. A missing producer-to-artifact-to-consumer proof is a
  HOLD, not a deletion justification.
- [ ] 3.3 Propose each accepted quarantine/removal as an explicit scoped follow-
  up with its preservation and replay plan; do not create a duplicate permanent
  `legacy/` source tree or bulk-remove paths in this change.

## 4. Baseline acceptance gate

- [ ] 4.1 Run strict OpenSpec validation, routing/link checks, clean-status and
  fixed-directory invariants, and residue checks for stale research-authority
  statements in tracked paths only (never rewrite worker snapshots); attach the
  exact post-infra source identity, ledger evidence, and a liveness/checksum
  receipt for every cited external-artifact locator. Capture each fixed absolute
  path -> resolved Git admin directory -> currently resolved named ref ->
  gate-time commit, rather than enforcing a permanent branch name or keeping a
  self-invalidating current-HEAD value in this task. Verify each native
  worktree lock and the explicitly local-only durability boundary; no remote
  publication or off-host replication is implied.
- [ ] 4.1a Prove every active worktree HEAD and every probe/final lifecycle tip
  resolves from at least one named ref. Keep the detached
  `/data/CoordExp/.worktrees/permanent-owner-bridge-cache-validation` at
  `477b376a3e31a5dbedf5a87ecafcb372e75a73a9` as an explicit HOLD; do not
  create a ref, prune, or remove it in this change. Its registered worktree HEAD
  is presently a Git root, so this unrelated HOLD does not block baseline-tag
  review; record that future worktree removal/prune plus reflog expiry requires
  its own preservation decision. Record other detached checkouts and their
  containing refs separately rather than treating all detached paths alike.
- [ ] 4.2 Freeze the complete clean baseline-candidate diff that descends from
  the 1.1 anchor and obtain a `claude-opus-5` leaf, `write:false`
  lifecycle/compatibility review on that exact candidate; lead disposition owns
  one bundled correction round and independently replays decisive checks. The
  user-selected CPU-only target-binding evidence is sufficient for this tag
  gate; `harden-research-probe-target-binding` task 4.2 remains a separate,
  unexecuted GPU mechanics smoke and must not be presented as GPU evidence.
- [ ] 4.3 Submit the verified baseline-cutover packet for user approval. That
  approval may authorize only creation of `research-base-v1`. Any
  quarantine/removal, worktree retirement, branch deletion, or raw-artifact
  reclamation requires a separate explicit approval; the fixed
  `.worktrees/research-probes` and `.worktrees/research-probe-infras`
  directories are never retirement targets.
- [ ] 4.4 Only after the explicit approval from 4.3, create the annotated
  `research-base-v1` tag at the exact final candidate frozen by 4.2, not at the
  1.1 infra anchor; the tag does not exist before this task. Verify
  `git cat-file -t research-base-v1` returns `tag`, verify
  `research-base-v1^{commit}` resolves to that final candidate, and commit a
  post-tag lifecycle receipt that records both resolved values. No remote/off-
  host replication is claimed; generic-ref movement/deletion, tag deletion,
  Git garbage collection, and artifact reclamation remain separately user-
  gated.
